"""Make a raw mesh with a running Modly (https://github.com/lightningpixel/modly) through its local API."""
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_MODEL = "hunyuan3d-mini/generate"
DEFAULT_PARAMS = {"num_inference_steps": 30, "octree_resolution": 380, "guidance_scale": 5.5, "seed": 42}
TIMEOUT_MINUTES = 20


class ModlyError(RuntimeError):
    pass


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _post_form(url, fields, file_field, file_path):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for key, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{Path(file_path).name}"\r\n'
             f"Content-Type: {mimetypes.guess_type(str(file_path))[0] or 'image/png'}\r\n\r\n").encode()
    body += Path(file_path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=bytes(body), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def generate(picture, dest, url=DEFAULT_URL, model=DEFAULT_MODEL, params=None, collection="picture-to-glb"):
    """Send the picture to Modly, wait for the mesh, and save it to `dest`."""
    url = url.rstrip("/")
    try:
        _get(url + "/health")
    except (urllib.error.URLError, OSError) as error:
        raise ModlyError(f"Modly isn't answering at {url} ({error}). Start Modly first, or pass --mesh.") from error
    try:
        job = _post_form(url + "/generate/from-image",
                         {"model_id": model, "collection": collection, "remesh": "none",
                          "params": json.dumps(params or DEFAULT_PARAMS)}, "image", picture)["job_id"]
    except urllib.error.HTTPError as error:
        raise ModlyError(f"Modly refused the job: {error.read()[:400]!r}") from error
    try:
        deadline = time.time() + TIMEOUT_MINUTES * 60
        while time.time() < deadline:
            status = json.loads(_get(f"{url}/generate/status/{job}"))
            state = status.get("status")
            if state in ("error", "failed", "cancelled"):
                raise ModlyError(f"Modly job failed: {json.dumps(status)[:400]}")
            if state in ("done", "completed", "success"):
                output = status.get("output_url") or status.get("output") or status.get("result") or ""
                if isinstance(output, dict):
                    output = output.get("output_url") or output.get("path") or ""
                output = str(output)
                if not output.startswith("http"):
                    output = f"{url}/workspace/{output.split('/workspace/')[-1].lstrip('/')}"
                Path(dest).write_bytes(_get(output))
                return
            time.sleep(4)
        raise ModlyError(f"Modly took more than {TIMEOUT_MINUTES} minutes.")
    finally:
        try:  # free the graphics memory for whatever runs next
            urllib.request.urlopen(urllib.request.Request(url + "/model/unload-all", method="POST"), timeout=30)
        except (urllib.error.URLError, OSError):
            pass
