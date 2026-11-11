# mlr-ex08 — leaf classifier in the browser

**Live page: <https://veneas.github.io/mlr-ex08/>**

Open the link on a phone (or scan the QR code at the bottom of the page), press
**START CAMERA** and allow the camera. A leaf is classified every frame, on the phone.

## What it does

One HTML file, no backend, no build step. `leaf_model.onnx` (the MobileNet exported in Task 2e)
runs inside the browser through **ONNX Runtime Web** with the WebAssembly execution provider.
Nothing is uploaded: the camera frames are drawn onto a canvas in the page and never leave the device.

Preprocessing follows `model_info.json` exactly, which is the same recipe as the notebook's
`transforms.Compose([ToTensor(), Normalize(mean, std)])`:

1. take the central square of the frame: `S = min(W, H)`, starting at
   `(floor((W − S)/2), floor((H − S)/2))`,
2. draw that square onto a 96 × 96 canvas (image smoothing on),
3. read the RGBA pixels and write them into a planar `1 × 3 × 96 × 96` `Float32Array`,
4. per channel `out[c, y, x] = (byte / 255 − mean[c]) / std[c]` with
   `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`,
5. softmax the four logits and print every class to three decimals.

The small panel under the video shows the 96 × 96 tensor's picture, i.e. what the model is actually
being fed.

## Checking it

| Check | How |
| --- | --- |
| Same numbers as the notebook | Press **VERIFY CHECK_IMAGE.JPG**. It classifies `check_image.jpg` and compares every class with `check_expected.json`, the probabilities the notebook printed for that file. A measured agreement of about **6·10⁻⁸** is a PASS (limit 1·10⁻³); the badge and the per-class table appear on the page. |
| Same check without pressing anything | Open <https://veneas.github.io/mlr-ex08/?selftest=1> — the self-test runs by itself once the model is loaded. |
| Live camera on your phone | Press **START CAMERA**, point at a leaf (or at a photo of a leaf on another screen) and read the bars. |
| Nothing leaves the phone | Panel 05 lists every URL the page fetched. There are exactly three kinds: ONNX Runtime Web from the jsDelivr CDN (version pinned to 1.31.1), the model and `model_info.json`, and the reference files you ask for. The page contains no `POST`, no form, no analytics. |
| Upload path | **UPLOAD IMAGE** accepts `image/jpeg`, `image/png`, `image/webp` between 1 byte and 15 MB; anything else is refused with a message in the banner instead of a crash. |

If both `leaf_model.onnx` and `best_leaf_model.onnx` are present, a picker appears in panel 02 and
each model is compared with the reference file that belongs to it
(`check_expected.json` and `best_check_expected.json`).

## Files

| File | What it is |
| --- | --- |
| `index.html` | the whole app: markup, CSS, JavaScript, QR generator |
| `leaf_model.onnx` | the Task 2 model, `mobilenet_v3_small` fine-tuned on the group photos, with `model_info` embedded as metadata |
| `model_info.json` | class names, input size 96, mean and std used by the preprocessing |
| `check_image.jpg` | one validation photo (a cherry leaf) |
| `check_expected.json` | the probabilities the notebook computed for that photo |
| `best_leaf_model.onnx`, `best_model_info.json`, `best_check_expected.json` | the Task 4 ablation winner and its own reference numbers (optional; the page works without them) |
| `REPORT.md` | Task 4 report — the ablation study, written by hand |
| `REVIEW.md` | Task 2f review — the phone check, written by hand |

## Notes

- The QR code at the bottom of the page is generated in the browser by a small byte-mode QR encoder
  written into `index.html`; no QR web service is contacted. It encodes the page's own address.
- Camera access needs https, which GitHub Pages provides. If permission is refused, or the device has
  no camera, the banner explains what to do and the upload path still works.
- The runtime is loaded with the WebAssembly provider on one thread: GitHub Pages cannot send the
  COOP/COEP headers that shared-memory WebAssembly needs, and one thread is the reliable setting on a
  phone. A single 96 × 96 inference takes about 10–20 ms on a laptop and runs in real time on a phone.
