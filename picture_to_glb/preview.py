"""A small software renderer for checking a finished model without opening a 3D program."""
import numpy as np
from PIL import Image

from .camera import Camera, rasterize

VIEWS = {"front": (0, 12), "quarter": (45, 18), "back": (180, 12), "above": (45, 42)}
BACKGROUND = np.array([0.32, 0.31, 0.30])
AMBIENT = 0.45


def render(mesh, texture, az, el, size=480):
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv)
    tex = np.asarray(texture.convert("RGB"), dtype=np.float64) / 255
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    diagonal = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    camera = Camera(center, diagonal, az, el, 3.0, 0.5, 0.5, 1.0)
    pts = camera.project(verts)[0]
    camera.scale = 0.82 / max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    pts = camera.project(verts)[0]
    camera.tx = 1.0 - (pts[:, 0].max() + pts[:, 0].min()) / 2
    camera.ty = 1.0 - (pts[:, 1].max() + pts[:, 1].min()) / 2

    pixels, depth = camera.project(verts)
    ids, _, bary = rasterize(pixels * size, depth, faces, size, barycentric=True)
    hit = ids >= 0
    face = ids[hit]
    texel_uv = np.einsum("ij,ijk->ik", bary[hit], uv[faces[face]])
    h, w = tex.shape[:2]
    tx = np.clip((texel_uv[:, 0] * w).astype(int), 0, w - 1)
    ty = np.clip(((1 - texel_uv[:, 1]) * h).astype(int), 0, h - 1)

    light = camera.to_camera + 0.6 * camera.up - 0.4 * camera.right
    light /= np.linalg.norm(light)
    normals = np.asarray(mesh.face_normals)[face]
    shade = AMBIENT + (1 - AMBIENT) * np.clip(normals @ light, 0, 1)

    image = np.tile(BACKGROUND, (size, size, 1))
    image[hit] = np.clip(tex[ty, tx] * shade[:, None], 0, 1)
    return Image.fromarray((image * 255).round().astype(np.uint8))


def preview_sheet(picture, mesh, texture, out, size=480):
    """The input picture, then the model from the front, three-quarter, back and above, side by side."""
    first = Image.open(picture).convert("RGB")
    first.thumbnail((size, size))
    panels = [first] + [render(mesh, texture, az, el, size) for az, el in VIEWS.values()]
    sheet = Image.new("RGB", (size * len(panels), size), tuple(int(c * 255) for c in BACKGROUND))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * size + (size - panel.width) // 2, (size - panel.height) // 2))
    sheet.save(out, quality=90)
