# picture-to-glb

Turn a picture of an object into a textured, game-ready 3D model.

Image-to-3D generators are good at shape. Their textures are often missing, or don't look like the picture. This tool keeps the generator's shape and paints the original picture onto it, so the model looks like the concept art:

1. **Reduce** the generator's heavy mesh to a game budget.
2. **Fit a camera:** work out where a camera would have stood to take the picture, by matching the mesh's outline to the picture's.
3. **Unwrap** the surface into a texture layout.
4. **Paint:** every part of the model the camera could see takes its color straight from the picture.
5. **Fill the unseen sides:** mirror the visible side where the shape allows, otherwise borrow the nearest painted colors (preferring the same height).
6. **Write** a GLB with a JPEG texture inside, plus a preview image.

The output works in Godot, Blender, Unity, three.js and anything else that reads glTF.

It's plain Python (numpy, scipy, Pillow, trimesh, xatlas and fast-simplification). No AI model, graphics card or 3D program is needed for this part. Making the mesh in the first place does need one; see [You need a mesh generator](#you-need-a-mesh-generator).

## Install

Python 3.10 or newer:

```
git clone <this repo>
cd picture-to-glb
python -m venv .venv
.venv/bin/pip install -e .        # Windows: .venv\Scripts\pip install -e .
```

## Use

With a mesh you already made from the picture (in any image-to-3D tool):

```
picture-to-glb chair.png --mesh chair_raw.glb
```

Or let a running [Modly](https://github.com/lightningpixel/modly) make the mesh:

```
picture-to-glb chair.png --modly
```

Results go to `output/chair/`:

| File | What it is |
|---|---|
| `chair_medium.glb` | The finished model |
| `preview_medium.jpg` | The picture, then the model from the front, three-quarter, back and above |
| `work/` | Saved steps. Rerunning skips the ones already done, so trying another setting is quick |

A finished run takes about 15 to 50 seconds on a laptop, depending on detail.

### Options

| Option | What it does |
|---|---|
| `--detail low`, `medium`, `high` or a number | Triangles to keep: 0.4%, 1.25% or 5% of the raw mesh, or an exact count like `20000`. Default `medium` |
| `--sheet` | The picture is a turnaround sheet (see below) |
| `--no-mirror` | Don't copy the visible side onto the back. **Use this for people and animals**, or a face can end up on the back of a head |
| `--texture 1024` | Texture size in pixels. Default 2048 |
| `--png` | Keep a lossless PNG texture instead of a JPEG (about twice the size) |
| `--out folder` | Where results go. Default `./output` |
| `--modly URL` | Modly's address, if it isn't `http://127.0.0.1:8765` |
| `--redo` | Ignore saved steps for this detail level |
| `--no-preview` | Skip the preview image |

### What makes a good picture

- **One object, fully in view, on a plain light background.** The background is detected from the picture's border, so it should be one flat color that touches all four edges. Busy or gradient backgrounds break the camera fit.
- **The same picture the mesh was made from.** The texture is lined up against the mesh's outline, so a different picture of "the same" object won't line up.
- The run prints an **outline match** score. Above about 0.85 is good; below it, expect some color in the wrong places.

### Turnaround sheets: real backs instead of invented ones

From a single picture only the side facing the camera is real; the back is a guess, and usually looks blotchy. A **turnaround sheet** shows the same object from the front, side and back, side by side in one image (left to right, on one plain background). With `--sheet`:

```
picture-to-glb chair_sheet.png chair --sheet --mesh chair_raw.glb
```

the sheet is cut into three views, the mesh is made from the front view (with `--modly`, this happens automatically; with `--mesh`, make it from the front view yourself), and each view paints the part of the model it sees best. A view that doesn't line up well (match below 0.8) is skipped rather than smearing color.

Image editing models that can turn an object, such as Qwen-Image-Edit with a multiple-angles LoRA, can draw the side and back views from a single picture. They work best when asked to turn a straight-on front view.

## You need a mesh generator

This tool doesn't make shapes. It needs a raw mesh made from your picture by an image-to-3D model. Anything that outputs a single GLB or OBJ works with `--mesh`, as long as:

- it was made from **the same picture** (or the front view of a sheet),
- the model is **Y-up and faces +Z**, roughly toward where the picture's camera was. This is the glTF convention, and what most generators produce.

Untextured output is fine; only the shape is used.

### Modly (what this was built and tested with)

[Modly](https://github.com/lightningpixel/modly) is a free (MIT) desktop app that runs image-to-3D models locally. `--modly` talks to its local API. What it needs:

- **Windows, Linux or Apple Silicon macOS.** This tool was tested with Modly 0.4.3 on Linux.
- **A graphics card with enough memory for the model you pick.** Hunyuan3D 2 Mini used about 5.6 GB on a 6 GB NVIDIA RTX 4050 and took about 3 minutes per mesh. Modly also supports AMD cards through ROCm. Close other GPU-heavy programs (3D editors, games) first.
- **A model extension.** In Modly's Models page, install one (Hunyuan3D 2 Mini is the one `--modly` asks for by default) and download its weights, a few GB.
- **Modly running** when you start `picture-to-glb --modly`. It listens on `127.0.0.1:8765`. Note that **Modly stops whatever else is using port 8765 when it starts.**

Settings `--modly` sends (in `picture_to_glb/modly.py`): 30 steps, octree resolution 380, guidance 5.5, seed 42. After each mesh it asks Modly to unload the model, to free the graphics memory.

**Check the model's license, not just Modly's.** Hunyuan3D 2 Mini is under the Tencent Hunyuan 3D 2.0 Community License, which **does not apply in the European Union, the United Kingdom or South Korea**, needs a separate license above 1 million monthly users, and forbids using its outputs to improve other AI models. Other generators (TripoSG, TRELLIS and so on) have their own terms.

## Known limits

- **Backs from a single picture are invented**, from mirroring and nearby colors. Use a turnaround sheet if the back matters.
- **Shape errors stay in.** If the generator gets the shape wrong, painting won't fix it.
- **Pale parts that match the background** can be mistaken for gaps and left unpainted.
- **Two separate objects in one picture** (a boat beside a dock) confuse the camera fit. Make them separately.
- **Overlapping views on a sheet** can leave a faint double image where two views both paint the same area.
- **Scale:** models come out at whatever size the generator chose; set real sizes in your engine.

## License

MIT. See [LICENSE](LICENSE). Mesh generators and their models have their own licenses (see above).
