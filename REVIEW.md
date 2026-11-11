## 1. Running on my phone 
- After opening the "*https://veneas.github.io/mlr-ex08/*" on a *Iphone 13* running *iOS 27.0.1* in *Safari*, served over HTTPS by GitHub Pages. 
- After pressing **START CAMERA** and allowing camera access. While pointing it at *check_image.jpg*, the output gave **CHERRY: 0.976** and **CHESTNUT: 0.021** (this output did not fit into the phone screen but was taken right after the screenshot which showed the camera working properly). When the camera was ON and I was moving it across the monitor, the bars were updated continuously.  `shot_phone.png` is the result of a screenshot as the camera was on and pointed at the web of the same website. 

**What did not work at first and how I fixed it:** 
- As I've spent quite a lot of time specifying instructions as the first prompt, and the agent itself took more than 3hrs of work, it surprisingly worked right away. It worked with the notebook off, without any bluetooth connection. Also the **UPLOAD IMAGE** worked flawlessly on both PC and phone. 

## Same numbers as the notebook 
- After pressing **VERIFY CHECK_IMAGE.JPG**, which classified `check_image.jpg` and compared every class with the `check_expected.json`. The probabilities the notebook itself produced for that file with PyTorch. 

| class | notebook (`check_expected.json`) | page | same to 3 decimals |
| --- | --- | --- | --- |
| apple    | 5.597728e-08  | 5.597447e-8 | yes |
| cherry   | 9.999985e-01  | 9.999984e-1 | yes |
| chestnut | 1.500195e-06  | 1.500192e-6 | yes |
| maple    | 5.535953e-08  | 5.535797e-8 | yes |

Largest difference: **6.180e-8**.
Screenshot: `shot_selftest.png`

**Why they are not bit-identical:**
- Both sides compute in **float32**, but the sums themselves are accumulated in a different order (and different code), and the JPEG itself is decoded by the browser rather than Pillow. 
- Floating point addition is not associative -> therefore last bits differ. 
- Important to note that on the `check_image.jpg` is VALIDATION photo which the model has already seen and been evaluated on. So this does not say anything about accuracy, but thanks to it I know the preprocessing and arithmetic agrees with the notebook. 

## The preprocessing matching the model 

| where | what the code does | what it proves |
| --- | --- | --- |
| `index.html:741-746` | the comment above `toTensor()` naming the notebook recipe | Page states which notebook transform it's gonna implement, BEFORE it does |
| `index.html:747-768` | `toTensor()` in full | the whole pipeline is ONE function |
| `index.html:752-754` | `side = Math.min(width, height)`, `x0 = Math.floor((width - side) / 2)`, `y0 = Math.floor((height - side) / 2)` | the central square `S = min(W, H)` at `(⌊(W−S)/2⌋, ⌊(H−S)/2⌋)` |
| `index.html:755-759` | `imageSmoothingEnabled = true`, `imageSmoothingQuality = "high"`, then `drawImage(source, x0, y0, side, side, 0, 0, size, size)` | Resampling the square onto the 96 x 96 work canvas |
| `index.html:761-765` | `input[i] = lut[0][…]`, `lut[1][rgba[rgb]]`, `input[2 * plane + i] = lut[2][…]` | RGBA interleaved becomes **planar CHW**, one pass, `1 × 3 × 96 × 96` - the layout `ToTensor()` produces |
| `index.html:670-672` (in `buildBuffers()`, `655`) | `table[byte] = (byte / 255.0 - mean[c]) / std[c]`, 256 entries per channel | `/255`, `−mean`, `/std` folded into a look-up table: same arithmetic, evaluated 768 times for the whole session instead of 27 648 times per frame |
| `index.html:775-787` | `cropGeometryCheck()` over eight shapes (96×96, 1280×720, 750×1000, 4000×3000, …) | the crop **arithmetic** is tested on the *shapes* a phone really produces - the self-test alone **cannot** test it, because the reference photo is already 96 × 96 |
| `index.html:791-800` | `assertTensor()`: dims must be `1,3,96,96`, buffer a `Float32Array` of the right length, every value finite | the shape and dtype are proven before **every** inference, on camera frames too, not once at load time |
| `model_info.json` | `classes`, `input_size: 96`, `mean [0.485,0.456,0.406]`, `std [0.229,0.224,0.225]` | the page reads its constants from the same file the exporter wrote, so the normalisation cannot silently drift from the notebook's `Normalize(mean, std)` (cell 40) |

- The table itself is not an approximation -> the calculations depend only on the byte, therefore it evaluates once per byte value and gives the same float32 numbers as evaluating it per pixel -> all this for fraction of the work. 

## Which URLs does the page load, and WHY? 

- For this see the `shot_urls.png`

*Why?*

| what | where from | why it exists |
| --- | --- | --- |
| `ort.min.mjs`, its `…jsep.mjs` loader, `ort-wasm-simd-threaded.jsep.wasm` (≈5.6 MB) | `cdn.jsdelivr.net`, fallback `unpkg.com`, both pinned to `1.31.1` (`ORT_VERSION`, `index.html:558`) | Inference engine -> GitHub pages could serve it, but that would result in committing MBs of someone's elses output. A CDN keeps the repo to 10 files and the version means the engine CANNOT change in the demo |
| `leaf_model.onnx` or `best_leaf_model.onnx` | the same folder on Pages | The weights themselves |
| `model_info.json`, `check_image.jpg`, `check_expected.json`, `best_model_info.json`, `best_check_expected.json` | the same folder | The classes which preprocessing must use + reference photo plus the notebook's own numbers for the self-test |

- Exactly **five `fetch` calls**: `index.html:645`, `712`, `1052`, `1080`, `1153`. All are GETs or HEADs on
relative, same-origin paths. The one at `1153` is a HEAD that probes each model in the list, so the picker only offers files which are ACTUALLY there. The CDN downloads are not `fetch` at all -> they come from the pinned
`import(base + "ort.min.mjs")` at `index.html:626`.
- **No** `XMLHttpRequest`, no `WebSocket`, no `navigator.sendBeacon`, no `<form>`, no `.submit()`, and no
`localStorage` **anywhere** in the file. 
- A camera frame's path is `video` -> `drawImage` -> `getImageData` -> a `Float32Array` inside the page
(`index.html:747-768`). This ensured nothing was hard encoded, and there was no way of anything to be uploaded. 
- The QR code is **computed** by the encoder at `index.html:259-553`, not requested from a QR web service. Like this it avoided learning the URL. (It encodes `location.href` with the query string stripped, so
`?selftest=1` never ends up in a printed code: `index.html:1204`.)