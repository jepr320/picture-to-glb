"""Turnaround sheets: front, side and back views of one object, side by side in one picture."""
import numpy as np
from PIL import Image
from scipy import ndimage

from . import camera as cam

SPLIT_SEARCH = 0.12  # how far from the exact thirds a cut may move, as a share of the width
VIEW_SIZE = 1024
PAD = 0.08
MAX_ANGLE_DRIFT = 35  # degrees a view may stray from where it should be
MIN_VIEW_FIT = 0.8  # a view fitting worse than this paints in the wrong places, so it is skipped


def split_sheet(sheet, out):
    """Cut the sheet at the emptiest columns near its thirds and center each view on a square."""
    img = Image.open(sheet).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64) / 255
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    filled = ndimage.binary_opening(np.abs(rgb - background).max(axis=2) > cam.BACKGROUND_TOLERANCE, iterations=2)
    width, _ = img.size
    columns = filled.sum(axis=0)
    cuts = [0]
    for third in (1, 2):
        low, high = int((third / 3 - SPLIT_SEARCH) * width), int((third / 3 + SPLIT_SEARCH) * width)
        empty = np.where(columns[low:high] == columns[low:high].min())[0]
        cuts.append(low + int(np.median(empty)))
    cuts.append(width)

    boxes = []
    for left, right in zip(cuts, cuts[1:]):
        ys, xs = np.where(filled[:, left:right])
        if len(xs) == 0:
            raise ValueError("a third of the sheet is empty; it needs three views side by side")
        boxes.append((left + xs.min(), ys.min(), left + xs.max() + 1, ys.max() + 1))
    # All views share one scale, so their sizes stay comparable.
    side = int(max(max(b[2] - b[0], b[3] - b[1]) for b in boxes) * (1 + 2 * PAD))
    fill = tuple(int(c * 255) for c in background)
    paths = {}
    for name, (x0, y0, x1, y1) in zip(("front", "side", "back"), boxes):
        square = Image.new("RGB", (side, side), fill)
        square.paste(img.crop((x0, y0, x1, y1)), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
        paths[name] = out / f"{name}.png"
        square.resize((VIEW_SIZE, VIEW_SIZE), Image.LANCZOS).save(paths[name])
    return paths


def fit_all(paths, fit_mesh):
    """A camera per view. Returns (cameras, scores); a view that matched from the wrong direction scores 0.

    The mesh is built from the front view but can come out turned (an object with no clear front,
    like a boulder, may keep the angle of the original picture). So the front is searched widely, and
    side and back are judged relative to wherever the front landed.
    """
    verts, faces = np.asarray(fit_mesh.vertices), np.asarray(fit_mesh.faces)
    cameras, scores = {}, {}
    for name in ["front"] + [n for n in paths if n != "front"]:
        _, mask = cam.picture_mask(paths[name], cam.FIT_SIZE)
        if name == "front":
            cameras[name], scores[name] = cam.fit_near(verts, faces, mask, [-60, -30, 0, 30, 60])
            continue
        front = cameras["front"].az
        expected = [round(front + d) for d in {"side": [90, -90], "back": [180]}[name]]
        cameras[name], scores[name] = cam.fit_near(verts, faces, mask, expected)
        off_by = min(abs((cameras[name].az - e + 180) % 360 - 180) for e in expected)
        if off_by > MAX_ANGLE_DRIFT:
            scores[name] = 0.0
    return cameras, scores
