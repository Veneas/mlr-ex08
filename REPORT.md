# REPORT
- **CHAPTERS:**
    - Setup
    - How big the mean difference is, and what it means 
    - Results 
    - What changes brought 
    - Which model goes on the phone 
    - Limitations and what could be improved

## **Setup** 
- **DATA:**
    - **107 training / 29 validation** photos with **4 classes** (apple, cherry, chestnut, maple).
    - Per class, train: apple 28, cherry 28, chestnut 15, maple 36
    - Validation: apple 8, cherry 8, chestnut 4, maple 9. 
    - All photos were already 96 x 96 JPEGs. Chestnut was the minority class (15 out of 107). 
- **BASELINE V0** -> TASK 3 `LeafNet`, which was trained from scratch, with no augmentation and no BatchNorm. Each variant changed one thing relative to the one before it. There were seeds 0, 1, 2, for every row. 
- **Two evaluation sets:** 
    - `val` and `val_camera` -> the same 29 images perturbed once, with a fixed seed 42 (random resized cropt from 0.7-1.0, rotation +- 15°, colour jitter), cached to a tensor and hash checked -> therefore all 27 trials were scored on byte-identical images. 
- **LEAKAGE:**
    - Augmentation was applied to the TRAIN split only 
    - `val_camera` was built once and only READ afterwards
    - class weights came from TRAIN labels only
    - no statistic was computed on VALIDATION data
- **V0-7** took a few minutes (on an RTX 4060, 0.12-0.53s per epoch), so all 24 runs + supplementary **V4b** were affordable instead of sampled.

## How big the mean difference is, and what it means
- 29 validation images -> **one image = 3.45% accuracy**, 
- Any gap of one/two images between two rows was **not a result**. 

## **Results** 

| variant | change | val | balanced val | val_camera |
| --- | --- | --- | --- | --- |
| V0 | baseline | 0.655 ± 0.060 | 0.626 ± 0.052 | 0.402 ± 0.072 |
| V1 | + geometric augmentation | 0.552 ± 0.060 | 0.562 ± 0.075 | 0.471 ± 0.040 |
| V2 | V1 + BatchNorm | 0.563 ± 0.111 | 0.564 ± 0.080 | 0.494 ± 0.020 |
| V3 | V2 + ColorJitter(0.3,0.3,0.3,0.05) | 0.690 ± 0.034 | 0.691 ± 0.030 | 0.575 ± 0.020 |
| V4 | V3 + class-weighted cross-entropy | 0.609 ± 0.020 | 0.646 ± 0.021 | 0.506 ± 0.020 |
| V5 | pretrained MobileNetV3-Small, only `classifier[3]` trained | 0.690 ± 0.000 | 0.656 ± 0.000 | 0.575 ± 0.053 |
| V6 | pretrained, all layers trained, Adam lr 1e-3 (the Task 2 recipe) | 0.621 ± 0.060 | 0.594 ± 0.054 | 0.598 ± 0.105 |
| V7 | V6 + geo & colour aug + AdamW(wd 1e-2) + cosine LR + label smoothing 0.1 + best-epoch checkpoint, 25 epochs | **0.724 ± 0.000** | **0.688 ± 0.000** | **0.713 ± 0.020** |
| V4b | *supplementary, outside the matrix:* V3 with a WeightedRandomSampler instead of the weighted loss | 0.586 ± 0.150 | 0.603 ± 0.120 | 0.517 ± 0.034 |

- Trainable parameters for scale: V0-V4b approximately 0.21 M, V5 44 4100 (with backbone frozen), V6/V7 1.52 M.

## **What changes brought**
- Deltas were measured from the table. pp = percentage points. 

| comparison | val | balanced | val_camera |
| --- | --- | --- | --- |
| V1 − V0 (geometric augmentation alone) | −10.3 | −6.4 | **+6.9** |
| V2 − V1 (+ BatchNorm) | +1.1 | +0.1 | +2.3 |
| V3 − V2 (+ ColorJitter) | **+12.6** | **+12.7** | **+8.0** |
| V4 − V3 (+ class weights) | −8.0 | −4.5 | −6.9 |
| V6 − V5 (unfreeze everything) | −6.9 | −6.2 | +2.3 |
| V7 − V6 (full recipe, 25 epochs) | +10.3 | +9.4 | +11.5 |
| V7 − V0 (whole ladder) | +6.9 | +6.1 | **+31.0** |
| V7 − V3 (best row vs best small net) | +3.4 | −0.3 | +13.8 |

 - **Colour jitter was the biggest single change on the small nets:**
    - +12.6 pp `clean val` and +8.0 pp `camera` over V2, and +17.2pp camera over V0. BatchNorm on its own barely moved anything (+1.1/+2.3)
- **Geometric augmentation increases camera accuracy, but decreases clean accuracy**
    - +6.9 pp on `val_camera` but **-10.3**pp on clean val.
- **Imbalance handling hurt:**
    - Class-weighted cross-entropy cost 8.0 pp val, 4.5 pp balanced and 6.9 pp camera against V3
    - It DIDN'T improve the balanced score. 
    - V4b was *worse* and the least stable row of the study (+- 0.150 val). 
    - As the imbalance was not that big, the up-weighting probably mostly amplified gradient noise. 
- **Full fine-tuning is not better than freezing the backbone on clean val:**
    - In comparison it is 0.621 vs 0.690. V6 drives training loss to 0.010 - 0.028  -> which means it memorises all 107 photos and its validation best appears at epoch 2, 4 and 12 for the three seeds, and then slides. The `val_camera` swings +- 10.5 pp between seeds. Freezing gives 0.690 while training **4 100** parameters. 

- **V7 inspection:**
    - V7 keeps the **best of 25 validation epochs** -> therefore 0.724 is a maximum over 25 candidates. While every other row reports only the single last epoch. From the per-epoch log:
        - Across all 75 V7 epochs, the accuracy averaged at 0.639. If I would take a "random" epoch, it would give +- 0.64.
        - Its margin over V3/V5 survived on the camera set, but not on clean val where gap was 3.4 pp which is equal to **exactly one image**. 
        - As there is 0.00 std, I questioned how is it possible. As the three seeds report the SAME maximum, it is what the max-selection produces. 
        - `val_camera` sort of **favours** the colour jitter because of the construction -> this is because it was built **with** a colour jitter.

## **Which model goes on the phone**
- I would recommend the **V7**, which was exported as `best_leaf_model.onnx`, which is selectable in the page's picker. Numbers were all re-verified through ONNX runtime with the page's own preprocessing:
    - Val 21/29 = **0.7241**, with per class being: 
        - apple 4/8
        - cherry 6/8
        - chestnut 2/4
        - maple 9/9 
        - and `val_camera` also being 21/29
    - Most of the mistakes are predicted as *cherry*:
        - `apple_000/001/003/007` -> cherry, `cherry_001/007` -> apple, `chestnut_000/001` -> cherry. Therefore the residual error was a cherry vs the rest of the samples.
    - V0 on the same camera set: 0.402 +- 0.072, i.e. **+31 pp** for the exported model. 
    - Exported ONNX file reproduces the notebook-path probabilities to **1.58e-07**.
    - When on the live page, `check_image.jpg` under V7 resulted in:
        - cherry 0.902
        - maple 0.057
        - apple 0.025
        - chestnut 0.016
        - this was the same TOP-1, which is far less absolute than V0's 0.9999985. 
        - this result was expected as the V7 was trained with label smoothing of 0.1 -> this deliberately kept numbers away from reaching infinities.
        - See the `shot_task4model.png`

## **Limitations and what could be improved** 
- The amount of validation images was pretty low as I tried to work with only the in-class taken photos. This could've been improved by using the last year's photos which would increase the numbers. I did not use it though as I've seen several of them taken incorrectly (more leaves in the background, not nicely isolated ones). 
- no held-out test set implemented
- I did not go outside and find a real-life leaf (from the types of trees we constructed the data from) on which I could test the algorithm on
