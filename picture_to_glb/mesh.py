"""Loading, reducing, unwrapping and writing meshes."""
import json
import struct
from io import BytesIO

import fast_simplification
import numpy as np
import trimesh
import xatlas
from PIL import Image

FIT_TRIANGLES = 4000  # the camera fit uses a small copy of the mesh, which is plenty for an outline
JPEG_QUALITY = 92


def load(path):
    """One mesh from a GLB, OBJ or similar, with duplicate vertices merged."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"no triangle mesh found in {path}")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)


def reduce(mesh, target):
    """Quadric decimation down to `target` triangles (or leave it if it's already smaller)."""
    if target >= len(mesh.faces):
        return mesh
    points, faces = fast_simplification.simplify(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32),
                                                 target_count=int(target))
    return trimesh.Trimesh(vertices=points, faces=faces, process=True)


def unwrap(mesh, size):
    """Flatten the surface into a texture layout with xatlas. Returns a mesh with UVs."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32))
    chart = xatlas.ChartOptions()
    chart.max_cost = 8.0
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = 3
    atlas.generate(chart, pack)
    vmapping, indices, uvs = atlas[0]
    blank = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), (128, 128, 128)))
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, np.float32)[vmapping], faces=indices, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uvs, material=blank))


def write_model(mesh, image, path, jpeg=True):
    """Write a GLB with the texture inside. Returns (size with PNG, final size) in bytes."""
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, roughnessFactor=1.0, metallicFactor=0.0)
    trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False,
                    visual=trimesh.visual.TextureVisuals(uv=np.asarray(mesh.visual.uv), material=material)
                    ).export(path, include_normals=True)
    png_size = path.stat().st_size
    if jpeg:
        compress_glb(path, JPEG_QUALITY)
    return png_size, path.stat().st_size


def compress_glb(path, quality):
    """Swap the GLB's PNG texture for a JPEG, rebuilding the binary section around it."""
    data = path.read_bytes()
    json_length = struct.unpack("<I", data[12:16])[0]
    tree = json.loads(data[20:20 + json_length])
    binary = data[20 + json_length + 8:]

    views = tree["bufferViews"]
    image = tree["images"][0]
    view = views[image["bufferView"]]
    old = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
    out = BytesIO()
    Image.open(BytesIO(old)).convert("RGB").save(out, "JPEG", quality=quality, subsampling=0, optimize=True)
    new = out.getvalue()

    rebuilt = bytearray()
    for i, item in enumerate(views):
        start = item.get("byteOffset", 0)
        chunk = new if i == image["bufferView"] else binary[start:start + item["byteLength"]]
        rebuilt.extend(b"\0" * ((4 - len(rebuilt) % 4) % 4))
        item["byteOffset"] = len(rebuilt)
        item["byteLength"] = len(chunk)
        rebuilt.extend(chunk)
    rebuilt.extend(b"\0" * ((4 - len(rebuilt) % 4) % 4))
    tree["buffers"][0]["byteLength"] = len(rebuilt)
    image["mimeType"] = "image/jpeg"

    text = json.dumps(tree, separators=(",", ":")).encode()
    text += b" " * ((4 - len(text) % 4) % 4)
    total = 12 + 8 + len(text) + 8 + len(rebuilt)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, total) + struct.pack("<I", len(text)) + b"JSON" + text
                     + struct.pack("<I", len(rebuilt)) + b"BIN\0" + bytes(rebuilt))
