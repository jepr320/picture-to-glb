"""Paint a picture onto an unwrapped mesh, and fill in what the picture doesn't show.

Every texel of the texture knows its point on the model. That point is projected into the picture
with the fitted camera; if the camera could see it, the texel takes that pixel's color. The rest is
copied from the mirrored side where the shape allows, or borrowed from the nearest painted colors.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .camera import BACKGROUND_TOLERANCE, picture_mask, rasterize

IMAGE_SIZE = 1024  # pictures are sampled at this resolution
FACING_START, FACING_FULL = 0.12, 0.4  # surfaces seen edge-on take less color from the picture
VISIBILITY_TOLERANCE = 0.006  # depth slack for "the camera sees this point", as a share of the model's diagonal
MIRROR_TOLERANCE = 0.02
MIRROR_MIN_NORMAL = 0.3
HEIGHT_WEIGHT = 3.0  # unseen areas prefer colors from the same height on the model
NEAREST_K = 6
FILL_BLUR = 2.0
BLEND_POWER = 4.0  # sheets: higher means each texel leans harder on the view that sees it best
MIN_GAP_PIXELS = 60
DISPLAY_GAIN = 1.08
DISPLAY_DESATURATE = 0.12


def rasterize_uv(uv, faces, size):
    """Per-texel face index and barycentric weights in texture space."""
    pts = np.stack([uv[:, 0] * size, (1 - uv[:, 1]) * size], -1)
    ids = np.full((size, size), -1, int)
    bary = np.zeros((size, size, 3))
    for f, (a, b, c) in enumerate(faces):
        tri = pts[[a, b, c]]
        x0, y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int) + 2, size)
        if x0 >= x1 or y0 >= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        m00, m01 = tri[0, 0] - tri[2, 0], tri[1, 0] - tri[2, 0]
        m10, m11 = tri[0, 1] - tri[2, 1], tri[1, 1] - tri[2, 1]
        det = m00 * m11 - m01 * m10
        if abs(det) < 1e-12:
            continue
        dx, dy = xs - tri[2, 0], ys - tri[2, 1]
        l0, l1 = (m11 * dx - m01 * dy) / det, (-m10 * dx + m00 * dy) / det
        l2 = 1 - l0 - l1
        # A little slack past the edges so texels on seams are covered.
        inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        if not inside.any():
            continue
        yy, xx = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
        ids[yy, xx] = f
        bary[yy, xx] = np.stack([l0[inside], l1[inside], l2[inside]], -1)
    return ids, bary


def bilinear(image, x, y):
    x, y = np.clip(x - 0.5, 0, image.shape[1] - 1.001), np.clip(y - 0.5, 0, image.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def enclosed_background(rgb):
    """Background showing through gaps in the object, like between a fence's posts."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    flat = np.abs(rgb - np.median(border, axis=0)).max(axis=2) < BACKGROUND_TOLERANCE
    labels, _ = ndimage.label(flat)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return ndimage.binary_dilation(sizes[labels] >= MIN_GAP_PIXELS, iterations=4)


class PictureSource:
    """A picture with its fitted camera and a z-buffer of the mesh, for visibility tests."""

    def __init__(self, camera, path, verts, faces, size=IMAGE_SIZE):
        self.camera, self.size = camera, size
        self.rgb, mask = picture_mask(path, size)
        # Pull in from the outline so background pixels can't bleed onto the model's edges.
        self.mask = ndimage.binary_erosion(mask, iterations=3) & ~enclosed_background(self.rgb)
        pixels, depth = camera.project(verts)
        self.ids, self.z_buffer = rasterize(pixels * size, depth, faces, size)

    def sample(self, points, tolerance):
        """Return (visible flag, color) for 3D points."""
        at, depth = self.camera.project(points)
        x, y = at[:, 0] * self.size, at[:, 1] * self.size
        ix, iy = np.clip(x.astype(int), 0, self.size - 1), np.clip(y.astype(int), 0, self.size - 1)
        inside = (x >= 0) & (x < self.size) & (y >= 0) & (y < self.size)
        visible = inside & self.mask[iy, ix] & (self.z_buffer[iy, ix] - depth < tolerance * self.camera.diagonal)
        return visible, bilinear(self.rgb, x, y)

    def weight(self, points, normals, tolerance=VISIBILITY_TOLERANCE):
        """How much each point should trust this picture (0 to 1), and the picture's color there."""
        facing = np.einsum("ij,ij->i", normals, self.camera.view_vectors(points))
        seen, color = self.sample(points, tolerance)
        return smoothstep(FACING_START, FACING_FULL, facing) * seen, color


def surface(mesh, size):
    """The model point and face normal behind every used texel."""
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ids, barycentric = rasterize_uv(np.asarray(mesh.visual.uv), faces, size)
    used = ids >= 0
    texel_face = ids[used]
    points = np.einsum("ij,ijk->ik", barycentric[used], verts[faces[texel_face]])
    return used, texel_face, points, np.asarray(mesh.face_normals)[texel_face]


def fill_from_mirror(source, points, normals, pivot, face_normals):
    """Colors for unseen points, taken from the opposite side where that side is really visible.

    Each point is flipped across the X axis, the Z axis or both, whichever flip faces the camera most.
    The flip only counts if the surface the camera sees there faces the same way, so a chimney on the
    front of a roof isn't copied onto the back of it.
    """
    flips = np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, -1]], float)
    chosen = flips[np.stack([(normals * f) @ source.camera.to_camera for f in flips], axis=1).argmax(axis=1)]
    targets, wanted = pivot + (points - pivot) * chosen, normals * chosen
    visible, color = source.sample(targets, MIRROR_TOLERANCE)
    at, _ = source.camera.project(targets)
    x = np.clip((at[:, 0] * source.size).astype(int), 0, source.size - 1)
    y = np.clip((at[:, 1] * source.size).astype(int), 0, source.size - 1)
    hit = source.ids[y, x]
    agrees = hit >= 0
    agrees[agrees] = np.einsum("ij,ij->i", face_normals[hit[agrees]], wanted[agrees]) > MIRROR_MIN_NORMAL
    return visible & agrees, color


def finish(points, used, size, seen_color, weight, fill, known):
    """Fill unknown texels from their nearest known neighbors, soften the fill, and build the image."""
    fill = fill.copy()
    unknown = np.where(~known)[0]
    if len(unknown):
        if not known.any():
            raise ValueError("the picture didn't paint any of the model; check the camera fit")
        stretch = np.array([1.0, HEIGHT_WEIGHT, 1.0])
        donors = np.where(known)[0]
        _, nearest = cKDTree(points[donors] * stretch).query(points[unknown] * stretch, k=NEAREST_K, workers=-1)
        fill[unknown] = fill[donors[nearest]].mean(axis=1)

    # Soften the patchwork of borrowed colors, inside the used part of the texture only.
    layer = np.zeros((size, size, 3))
    layer[used] = fill
    amount = ndimage.gaussian_filter(used.astype(float), FILL_BLUR)
    soft = np.stack([ndimage.gaussian_filter(layer[..., c], FILL_BLUR) for c in range(3)], -1)
    soft = soft[used] / np.maximum(amount[used], 1e-6)[:, None]
    color = weight[:, None] * seen_color + (1 - weight[:, None]) * soft

    luma = color @ np.array([0.2126, 0.7152, 0.0722])
    color = np.clip((color + (luma[:, None] - color) * DISPLAY_DESATURATE) * DISPLAY_GAIN, 0, 1)

    # Spread colors past the edges of each texture island so seams don't show when the GPU filters.
    texture = np.zeros((size, size, 3))
    texture[used] = color
    nearest_used = ndimage.distance_transform_edt(~used, return_distances=False, return_indices=True)
    texture = texture[nearest_used[0], nearest_used[1]]
    return Image.fromarray((texture * 255).round().astype(np.uint8)), len(unknown)


def bake_picture(picture, mesh, camera, size, use_mirror):
    """Texture from one picture. Returns the image and how much of it came from where."""
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    used, _, points, normals = surface(mesh, size)
    source = PictureSource(camera, picture, verts, faces)
    weight, seen_color = source.weight(points, normals)

    known = weight > 0.5
    fill = seen_color.copy()
    mirrored = 0
    if use_mirror:
        need = np.where(~known)[0]
        pivot = (verts.max(axis=0) + verts.min(axis=0)) / 2
        hit, color = fill_from_mirror(source, points[need], normals[need], pivot, np.asarray(mesh.face_normals))
        fill[need[hit]] = color[hit]
        known[need[hit]] = True
        mirrored = int(hit.sum())

    image, borrowed = finish(points, used, size, seen_color, weight, fill, known)
    total = int(used.sum())
    return image, {"direct": float((weight > 0.5).mean()), "mirrored": mirrored / total, "nearest": borrowed / total}


def bake_views(pictures, cameras, mesh, size):
    """Texture from several views of the same object, each texel leaning on the view that sees it best."""
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    used, _, points, normals = surface(mesh, size)
    weights, colors = [], []
    for name, path in pictures.items():
        w, c = PictureSource(cameras[name], path, verts, faces).weight(points, normals)
        weights.append(w)
        colors.append(c)
    weights, colors = np.stack(weights), np.stack(colors)

    sharpened = weights ** BLEND_POWER
    seen_color = np.einsum("vt,vtc->tc", sharpened / np.maximum(sharpened.sum(axis=0), 1e-9), colors)
    weight = weights.max(axis=0)
    known = weight > 0.5
    image, borrowed = finish(points, used, size, seen_color, weight, seen_color, known)
    total = int(used.sum())
    by_view = np.bincount(weights.argmax(axis=0)[known], minlength=len(pictures)) / max(int(known.sum()), 1)
    shares = {"direct": float(known.mean()), "nearest": borrowed / total}
    return image, shares, dict(zip(pictures, by_view.round(2).tolist()))
