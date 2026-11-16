"""A perspective camera, a numpy z-buffer, and fitting a camera so a mesh's outline matches a picture."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize

BACKGROUND_TOLERANCE = 0.09  # how far a pixel may differ from the border color and still count as background
FIT_SIZE = 256  # pictures are fitted at this resolution


def axes(azimuth, elevation):
    """Camera right, up and toward-camera vectors. Azimuth 0 looks at the model's front (+Z), Y is up."""
    az, el = np.radians(azimuth), np.radians(elevation)
    right = np.array([np.cos(az), 0.0, -np.sin(az)])
    to_camera = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    return right, np.cross(to_camera, right), to_camera


class Camera:
    """Perspective camera. `distance` is in multiples of the mesh diagonal; image coordinates run 0 to 1."""

    def __init__(self, center, diagonal, az, el, distance, tx, ty, scale):
        self.center, self.diagonal = np.asarray(center, float), float(diagonal)
        self.az, self.el, self.distance, self.tx, self.ty, self.scale = az, el, distance, tx, ty, scale
        self.right, self.up, self.to_camera = axes(az, el)

    def project(self, points):
        """Return (positions in a 1.0-sized image, depth with larger meaning nearer)."""
        rel = np.asarray(points) - self.center
        x, y, z = rel @ self.right, rel @ self.up, rel @ self.to_camera
        w = 1 - z / (self.distance * self.diagonal)
        return np.stack([self.tx + self.scale * x / w, self.ty - self.scale * y / w], -1), z

    def view_vectors(self, points):
        """Unit vectors from each surface point toward the camera."""
        eye = self.center + self.to_camera * self.distance * self.diagonal
        v = eye - np.asarray(points)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def to_json(self):
        return {"center": self.center.tolist(), "diagonal": self.diagonal, "az": float(self.az), "el": float(self.el),
                "distance": float(self.distance), "tx": float(self.tx), "ty": float(self.ty), "scale": float(self.scale)}

    @classmethod
    def from_json(cls, d):
        return cls(**d)


def rasterize(pixels, depth, faces, size, barycentric=False):
    """Z-buffer: per-pixel face index (-1 for none) and depth, plus barycentric weights if asked."""
    z_buffer, ids = np.full((size, size), -1e18), np.full((size, size), -1, int)
    bary = np.zeros((size, size, 3)) if barycentric else None
    for f, (a, b, c) in enumerate(faces):
        tri = pixels[[a, b, c]]
        x0, y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int) + 1, size)
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
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        z = (l0 * depth[a] + l1 * depth[b] + l2 * depth[c])[inside]
        yy, xx = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
        nearer = z > z_buffer[yy, xx]
        z_buffer[yy[nearer], xx[nearer]] = z[nearer]
        ids[yy[nearer], xx[nearer]] = f
        if barycentric:
            bary[yy[nearer], xx[nearer]] = np.stack([l0[inside], l1[inside], l2[inside]], -1)[nearer]
    return (ids, z_buffer, bary) if barycentric else (ids, z_buffer)


# ---- Pictures ----------------------------------------------------------------------------------


def load_square(path, size):
    """The picture as floats, padded to a square with its own background color, then resized."""
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64) / 255
    w, h = img.size
    if w != h:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        fill = tuple(int(round(c * 255)) for c in np.median(border, axis=0))
        side = max(w, h)
        square = Image.new("RGB", (side, side), fill)
        square.paste(img, ((side - w) // 2, (side - h) // 2))
        img = square
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float64) / 255


def object_mask(rgb):
    """True where the object is: everything not connected to the border's flat background."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background_like = np.abs(rgb - np.median(border, axis=0)).max(axis=2) < BACKGROUND_TOLERANCE
    labels, _ = ndimage.label(background_like)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    mask = ~np.isin(labels, edge_labels[edge_labels != 0])
    return ndimage.binary_opening(mask, iterations=2)


def picture_mask(path, size):
    rgb = load_square(path, size)
    return rgb, object_mask(rgb)


# ---- Fitting -----------------------------------------------------------------------------------


def silhouette(camera, verts, faces, size):
    pts = camera.project(verts)[0] * size
    canvas = Image.new("1", (size, size))
    draw = ImageDraw.Draw(canvas)
    for tri in pts[faces]:
        draw.polygon([tuple(q) for q in tri], fill=1)
    return np.asarray(canvas)


def fit_box(center, diagonal, az, el, distance, verts, mask):
    """Scale and shift the projection so its bounding box matches the mask's."""
    camera = Camera(center, diagonal, az, el, distance, 0.5, 0.5, 1.0)
    pts = camera.project(verts)[0]
    ys, xs = np.where(mask)
    n = mask.shape[0]
    camera.scale = np.sqrt(((xs.max() - xs.min()) / n / np.ptp(pts[:, 0])) * ((ys.max() - ys.min()) / n / np.ptp(pts[:, 1])))
    pts = camera.project(verts)[0]
    camera.tx = 0.5 + (xs.max() + xs.min()) / 2 / n - (pts[:, 0].max() + pts[:, 0].min()) / 2
    camera.ty = 0.5 + (ys.max() + ys.min()) / 2 / n - (pts[:, 1].max() + pts[:, 1].min()) / 2
    return camera


def iou(camera, verts, faces, mask):
    """Overlap of the mesh's outline and the picture's, from 0 (none) to 1 (exact)."""
    drawn = silhouette(camera, verts, faces, mask.shape[0])
    return (drawn & mask).sum() / max((drawn | mask).sum(), 1)


def fit_camera(verts, faces, mask, azimuths, az_spread, elevations, max_evals, steps):
    """Coarse grid search over direction and lens, then Nelder-Mead on all six camera numbers."""
    if not mask.any():
        raise ValueError("no object found in the picture; it needs a plain background")
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    diagonal = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    best, best_score = None, -1.0
    for az0 in azimuths:
        for az in az_spread(az0):
            for el in elevations:
                for distance in (50.0, 3.0, 1.8):
                    camera = fit_box(center, diagonal, az, el, distance, verts, mask)
                    score = iou(camera, verts, faces, mask)
                    if score > best_score:
                        best, best_score = camera, score

    def objective(x):
        az, el, log_d, tx, ty, log_s = x
        cam = Camera(center, diagonal, az, el, float(np.exp(log_d)), tx, ty, float(np.exp(log_s)))
        return -iou(cam, verts, faces, mask)

    start = np.array([best.az, best.el, np.log(best.distance), best.tx, best.ty, np.log(best.scale)])
    simplex = np.vstack([start] + [start + d for d in np.diag(steps)])
    result = minimize(objective, start, method="Nelder-Mead",
                      options={"maxfev": max_evals, "initial_simplex": simplex, "xatol": 0.02, "fatol": 1e-4})
    az, el, log_d, tx, ty, log_s = result.x
    return Camera(center, diagonal, az, el, float(np.exp(log_d)), tx, ty, float(np.exp(log_s))), -result.fun


def fit_single(verts, faces, mask):
    """Camera for one picture: any direction across the front half, from level to well above."""
    return fit_camera(verts, faces, mask, [0], lambda _: range(-90, 91, 6), range(0, 61, 5), 700,
                      [8.0, 8.0, 0.4, 0.03, 0.03, 0.06])


def fit_near(verts, faces, mask, azimuths):
    """Camera for one view of a turnaround sheet, searched near the directions it should have."""
    return fit_camera(verts, faces, mask, azimuths, lambda a: range(a - 20, a + 21, 5), range(0, 26, 5), 400,
                      [6.0, 5.0, 0.4, 0.03, 0.03, 0.06])
