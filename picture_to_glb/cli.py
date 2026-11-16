"""picture-to-glb: one picture (or a turnaround sheet) plus a raw mesh in, a textured game-ready GLB out.

Every step saves its result under <out>/<name>/work/, so rerunning skips finished steps.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from . import bake, camera, mesh as meshes, modly, preview, sheet

DETAIL = {"low": 0.004, "medium": 0.0125, "high": 0.05}  # share of the raw triangles
MIN_GOOD_FIT = 0.85


def step(message):
    print(f"\n== {message}", flush=True)


def fail(message):
    print(f"\nSTOPPED: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_saved(path):
    return trimesh.load(path, force="mesh", process=False)


def parse(argv):
    parser = argparse.ArgumentParser(prog="picture-to-glb", description=__doc__.split("\n\n")[0])
    parser.add_argument("picture", type=Path, help="picture of one object on a plain background (or a turnaround sheet with --sheet)")
    parser.add_argument("name", nargs="?", help="short name for the output folder (default: the picture's file name)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mesh", type=Path, help="raw mesh made from this picture (GLB, OBJ, ...) by any image-to-3D tool")
    source.add_argument("--modly", nargs="?", const=modly.DEFAULT_URL, metavar="URL",
                        help=f"make the mesh with a running Modly (default {modly.DEFAULT_URL})")
    parser.add_argument("--out", type=Path, default=Path("output"), help="folder for results (default ./output)")
    parser.add_argument("--detail", default="medium",
                        help="low, medium, high (0.4%%, 1.25%%, 5%% of the raw triangles) or a triangle count (default medium)")
    parser.add_argument("--texture", type=int, default=2048, help="texture size in pixels (default 2048)")
    parser.add_argument("--sheet", action="store_true",
                        help="the picture is a turnaround sheet (front, side, back side by side); paint the back from it")
    parser.add_argument("--no-mirror", action="store_true",
                        help="don't copy the visible side onto the back (use for people, or a face can land on the back of a head)")
    parser.add_argument("--png", action="store_true", help="keep a lossless PNG texture inside the GLB instead of a JPEG")
    parser.add_argument("--no-preview", action="store_true", help="skip the preview image")
    parser.add_argument("--redo", action="store_true", help="ignore saved steps for this detail level")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    name = args.name or args.picture.stem
    if not args.picture.is_file():
        fail(f"picture not found: {args.picture}")
    if args.detail not in DETAIL and not args.detail.isdigit():
        fail("--detail must be low, medium, high, or a number of triangles")
    if not name.replace("_", "").replace("-", "").isalnum():
        fail("the name may only use letters, numbers, - and _")
    if args.mesh and not args.mesh.is_file():
        fail(f"mesh not found: {args.mesh}")

    started, timings = time.time(), {}
    folder = args.out / name
    raw = folder / "work" / "raw.glb"
    work = folder / "work" / f"{args.detail}_{args.texture}"
    if args.redo and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # A folder belongs to one picture; a different picture under the same name would reuse the wrong mesh.
    stamp = folder / "work" / "picture.json"
    signature = {"size": args.picture.stat().st_size, "name": args.picture.name}
    if stamp.exists() and json.loads(stamp.read_text()) != signature:
        fail(f"{folder} was made from a different picture. Use another name, or delete the folder.")
    stamp.write_text(json.dumps(signature))

    if args.sheet:
        shutil.copy(args.picture, folder / "sheet.png")
        try:
            views = sheet.split_sheet(args.picture, folder)
        except ValueError as error:
            fail(str(error))
        front = views["front"]  # the mesh and the preview come from the front view
    else:
        front = folder / f"picture{args.picture.suffix.lower()}"
        if args.picture.resolve() != front.resolve():
            shutil.copy(args.picture, front)

    t = time.time()
    if not raw.exists():
        if args.mesh:
            step(f"Using the mesh {args.mesh}")
            shutil.copy(args.mesh, raw) if args.mesh.suffix.lower() == ".glb" else meshes.load(args.mesh).export(raw)
        elif args.modly:
            step("Making the 3D mesh in Modly (a few minutes)")
            try:
                modly.generate(front, raw, url=args.modly)
            except modly.ModlyError as error:
                fail(str(error))
        else:
            fail("no mesh yet. Pass --mesh <file> (from any image-to-3D tool) or --modly to make one.")
    timings["mesh"] = time.time() - t

    t = time.time()
    if not (work / "mesh.glb").exists():
        step("Reducing triangles")
        full = meshes.load(raw)
        amount = DETAIL.get(args.detail, args.detail)
        target = int(len(full.faces) * float(amount)) if args.detail in DETAIL else int(amount)
        reduced = meshes.reduce(full, max(target, 100))
        reduced.export(work / "mesh.glb")
        meshes.reduce(reduced, meshes.FIT_TRIANGLES).export(work / "fit.glb")
        print(f"{len(full.faces)} triangles -> {len(reduced.faces)}")
    timings["reduce"] = time.time() - t
    fit_mesh = load_saved(work / "fit.glb")
    fit_verts, fit_faces = np.asarray(fit_mesh.vertices), np.asarray(fit_mesh.faces)

    t = time.time()
    if args.sheet:
        step("Fitting a camera to each view")
        cache = work / "cameras.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            cameras = {k: camera.Camera.from_json(v["camera"]) for k, v in saved.items()}
            scores = {k: v["iou"] for k, v in saved.items()}
        else:
            cameras, scores = sheet.fit_all(views, fit_mesh)
            cache.write_text(json.dumps({k: {"camera": cameras[k].to_json(), "iou": float(scores[k])} for k in cameras}, indent=1))
        print("Outline match: " + ", ".join(f"{k} {v:.2f} (angle {cameras[k].az:.0f})" for k, v in scores.items())
              + "  (0.00 means it matched best from the wrong direction)")
        # The mesh is built from the front, so the front always paints.
        usable = {k: v for k, v in views.items() if k == "front" or scores[k] >= sheet.MIN_VIEW_FIT}
        if len(usable) < len(views):
            print(f"Skipping views that fit worse than {sheet.MIN_VIEW_FIT}: " + ", ".join(k for k in views if k not in usable))
    else:
        step("Fitting a camera to the picture")
        cache = work / "camera.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            score = saved.pop("iou")
            fitted = camera.Camera.from_json(saved)
        else:
            _, mask = camera.picture_mask(front, camera.FIT_SIZE)
            try:
                fitted, score = camera.fit_single(fit_verts, fit_faces, mask)
            except ValueError as error:
                fail(str(error))
            cache.write_text(json.dumps({**fitted.to_json(), "iou": float(score)}, indent=1))
        print(f"Outline match {score:.2f}, view angle {fitted.az:.0f} degrees")
        if score < MIN_GOOD_FIT:
            print(f"WARNING: the match is below {MIN_GOOD_FIT}, so color may land in the wrong places.")
    timings["camera"] = time.time() - t

    step("Unwrapping")
    t = time.time()
    if not (work / "uv.glb").exists():
        meshes.unwrap(load_saved(work / "mesh.glb"), args.texture).export(work / "uv.glb", include_normals=False)
    model_mesh = load_saved(work / "uv.glb")
    timings["unwrap"] = time.time() - t

    step("Painting the texture" + (" from every usable view" if args.sheet else ""))
    t = time.time()
    try:
        if args.sheet:
            texture, shares, by_view = bake.bake_views(usable, cameras, model_mesh, args.texture)
        else:
            texture, shares = bake.bake_picture(front, model_mesh, fitted, args.texture, not args.no_mirror)
    except ValueError as error:
        fail(str(error))
    timings["paint"] = time.time() - t
    print("Texture sources: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in shares.items())
          + (f" | by view: {by_view}" if args.sheet else ""))

    model = folder / f"{name}_{args.detail}.glb"
    png_size, final_size = meshes.write_model(model_mesh, texture, model, jpeg=not args.png)

    preview_path = None
    if not args.no_preview:
        step("Rendering a preview")
        t = time.time()
        preview_path = folder / f"preview_{args.detail}.jpg"
        preview.preview_sheet(front, model_mesh, texture, preview_path)
        timings["preview"] = time.time() - t

    print(f"\nDONE {name}: {len(model_mesh.faces)} triangles in {time.time() - started:.0f}s")
    print("Time per step: " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    if not args.png:
        print(f"GLB {final_size / 1e6:.1f} MB (with a PNG texture it would be {png_size / 1e6:.1f} MB)")
    print(f"Model:   {model}" + (f"\nPreview: {preview_path}" if preview_path else ""))


if __name__ == "__main__":
    main()
