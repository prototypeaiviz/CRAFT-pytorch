# CRAFT: Character-Region Awareness For Text Detection

PyTorch implementation of the CRAFT text detector.

> **Paper:** [Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)  
> **Authors:** Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee — Clova AI Research, NAVER Corp.

---

## What CRAFT Does

CRAFT is a **scene text detector**. Given any image (photo, document scan, screenshot, etc.), it finds and draws bounding boxes around every region that contains text. It does **not** read the text — it only locates where the text is.

It works in two stages:

1. **Score map prediction** — a neural network looks at the image and produces two heatmaps:
   - **Text region score** — how likely each pixel is the centre of a character.
   - **Affinity score (link score)** — how likely two adjacent characters belong to the same word.

2. **Post-processing** — the two heatmaps are thresholded, combined, and analysed with connected-component labelling to produce clean bounding boxes (rectangles or polygons) around each detected text instance.

An optional **Link Refiner** network can be run on top to improve the affinity map for curved or arbitrarily-shaped text.

---

## How the Neural Network Works

```
Input image
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  ENCODER  —  VGG16-BN (pretrained on ImageNet)          │
│                                                         │
│  Produces 5 feature maps at different spatial scales:   │
│    relu2_2  →  stride  4  (highest resolution)          │
│    relu3_2  →  stride  8                                │
│    relu4_3  →  stride 16                                │
│    relu5_3  →  stride 32                                │
│    fc7      →  stride 32  (dilated conv, wide context)  │
└─────────────────────────────────────────────────────────┘
    │  (5 feature maps, deepest to shallowest)
    ▼
┌─────────────────────────────────────────────────────────┐
│  DECODER  —  U-Net style upsampling path                │
│                                                         │
│  upconv1: fc7 + relu5_3  →  256 ch                      │
│  upconv2: ↑ + relu4_3    →  128 ch   (bilinear upsample │
│  upconv3: ↑ + relu3_2    →   64 ch    at each step)     │
│  upconv4: ↑ + relu2_2    →   32 ch                      │
│                                                         │
│  Each upconv block = 1×1 conv + 3×3 conv + BN + ReLU   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CLASSIFICATION HEAD  —  conv_cls                       │
│                                                         │
│  32 → 32 → 32 → 16 → 16 → 2  channels                  │
│  Output: 2 score maps (text region + affinity)          │
└─────────────────────────────────────────────────────────┘
    │
    ▼  (optional)
┌─────────────────────────────────────────────────────────┐
│  LINK REFINER  —  RefineNet                             │
│                                                         │
│  Takes CRAFT's raw output + the 32-ch decoder features  │
│  Runs 4 parallel ASPP branches (dilation 6/12/18/24)    │
│  Outputs a refined affinity score map                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Post-processing (CPU, OpenCV)
  → bounding boxes or polygons in original image coordinates
```

The output score maps are at **half the input resolution** (the network has an inherent 2× downsampling). All coordinates are scaled back to the original image size at the end.

---

## Project Structure

```
CRAFT-pytorch/
│
├── test.py             # Entry point — loads model, loops over images, saves results
│
├── craft.py            # CRAFT model definition (encoder + decoder + head)
├── refinenet.py        # Optional LinkRefiner model definition
│
├── craft_utils.py      # Post-processing: heatmaps → bounding boxes / polygons
├── imgproc.py          # Image I/O and pre/post-processing helpers
├── file_utils.py       # File discovery and result saving helpers
│
├── basenet/
│   ├── __init__.py
│   └── vgg16_bn.py     # VGG16-BN encoder wrapper (5-slice feature extractor)
│
├── weights/            # Place downloaded .pth model files here
├── result/             # Output directory (auto-created on first run)
└── requirements.txt
```

---

## Input

### Images

- Place all images you want to process in a single folder (e.g. `./test_images/`).
- Supported formats: **JPG, JPEG, PNG, GIF, PGM**.
- Images can be any size and aspect ratio — the pre-processing step resizes and pads them automatically.
- Images must be **RGB or grayscale** (RGBA images have the alpha channel stripped automatically; grayscale images are converted to RGB).

### Model weights

Three pretrained models are available:

| Model | Trained on | Language | Use case | Download |
|---|---|---|---|---|
| `craft_mlt_25k.pth` | SynthText, IC13, IC17 | English + multilingual | General purpose | [Download](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) |
| `craft_ic15_20k.pth` | SynthText, IC15 | English | IC15 benchmark only | [Download](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf) |
| `craft_refiner_CTW1500.pth` | CTW1500 | — | Link refiner (used with General model) | [Download](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO) |

Place downloaded `.pth` files inside the `weights/` folder.

---

## Output

For each input image, two files are written to `./result/` (or the folder you choose):

| File | Description |
|---|---|
| `res_<filename>.txt` | One detected text region per line, each as comma-separated polygon vertex coordinates: `x1,y1,x2,y2,...,xN,yN` |
| `res_<filename>.jpg` | A copy of the input image with red polygon outlines drawn around every detected text region |
| `res_<filename>_mask.jpg` | Side-by-side JET heatmap of the text-region score (left) and affinity score (right) — useful for debugging thresholds |

---

## Installation

```bash
pip install -r requirements.txt
```

**Minimum versions:**

```
torch>=0.4.1
torchvision>=0.2.1
opencv-python>=3.4.2
scikit-image>=0.14.2
scipy>=1.1.0
```

Python 3.7+ is recommended.

---

## Running Inference

### Basic usage

```bash
python test.py --trained_model weights/craft_mlt_25k.pth \
               --test_folder /path/to/your/images/
```

### With the Link Refiner (better for curved / long text lines)

```bash
python test.py --trained_model weights/craft_mlt_25k.pth \
               --refine \
               --refiner_model weights/craft_refiner_CTW1500.pth \
               --test_folder /path/to/your/images/
```

### CPU-only machine

```bash
python test.py --trained_model weights/craft_mlt_25k.pth \
               --cuda false \
               --test_folder /path/to/your/images/
```

### Output polygons instead of rectangles

```bash
python test.py --trained_model weights/craft_mlt_25k.pth \
               --poly \
               --test_folder /path/to/your/images/
```

---

## All Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| `--trained_model` | `weights/craft_mlt_25k.pth` | str | Path to the CRAFT `.pth` weights file |
| `--test_folder` | `/data/` | str | Folder containing input images |
| `--cuda` | `True` | bool | Use GPU for inference (`true`/`false`) |
| `--canvas_size` | `1280` | int | Maximum side length (px) the image is resized to before inference. Larger = slower but catches small text |
| `--mag_ratio` | `1.5` | float | Magnification applied before capping at `canvas_size`. Values >1 upscale the image first, helping detect small text |
| `--text_threshold` | `0.7` | float | A connected component is accepted as text only if its peak text-region score exceeds this. Raise to reduce false positives; lower to catch faint text |
| `--low_text` | `0.4` | float | Lower binarisation threshold on the text map. Controls how far each character's halo extends. Lower = bigger/merged regions |
| `--link_threshold` | `0.4` | float | Binarisation threshold on the affinity map. Controls whether adjacent characters are merged into one box. Lower = more merging |
| `--poly` | `False` | flag | Output tight polygons instead of rectangles. Automatically enabled when `--refine` is used |
| `--refine` | `False` | flag | Run the LinkRefiner after the main model for improved affinity maps on curved text |
| `--refiner_model` | `weights/craft_refiner_CTW1500.pth` | str | Path to the RefineNet `.pth` weights file |
| `--show_time` | `False` | flag | Print per-image inference and post-processing time |

---

## Understanding the Three Thresholds

The three threshold arguments control a trade-off between **recall** (finding all text) and **precision** (not falsely detecting non-text):

```
text_threshold  — final gate: how confident must the peak of a region be?
                  Too high → misses faint or small text
                  Too low  → background noise becomes text

low_text        — how far around each character do we "grow" the region?
                  Too high → characters get split into fragments
                  Too low  → characters bleed into each other

link_threshold  — how strong must the affinity signal be to merge characters?
                  Too high → words get split into individual characters
                  Too low  → separate words get merged into one box
```

A good starting point for most images is the default (`text=0.7`, `low=0.4`, `link=0.4`). For documents with dense small text, try lowering `canvas_size` or raising `mag_ratio`.

---

## Pre-processing Pipeline (what happens to your image before the network sees it)

1. **Load** — image is read as RGB, edge cases (grayscale, RGBA, multi-layer TIFF) are normalised to 3-channel uint8.
2. **Magnify & cap** — the longer side is scaled to `mag_ratio × original_size`, then capped at `canvas_size`.
3. **Aspect-ratio preserving resize** — both dimensions are computed at the same scale ratio.
4. **Pad to multiples of 32** — the network's VGG encoder has a stride of 32, so both dimensions must be divisible by 32. Zero-padding is added on the right and bottom.
5. **ImageNet normalisation** — pixel values are shifted and scaled using ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
6. **Tensor conversion** — `(H, W, C)` numpy array → `(1, C, H, W)` PyTorch tensor, optionally moved to GPU.

---

## Post-processing Pipeline (what happens to the network output)

1. **Extract score maps** — channel 0 = text region score, channel 1 = affinity score, both as 2-D float arrays.
2. **(Optional) Refine** — pass score maps + decoder features through RefineNet to get an improved affinity map.
3. **Threshold** — `low_text` applied to text map; `link_threshold` applied to affinity map. Both produce binary masks.
4. **Combine** — the two binary masks are OR-ed together: a pixel is foreground if it is text OR a link.
5. **Connected components** — OpenCV's `connectedComponentsWithStats` labels all foreground blobs.
6. **Filter** — blobs smaller than 10 px² or whose peak text score is below `text_threshold` are discarded.
7. **Dilate segmap** — each blob's mask is dilated by an amount proportional to its size, closing small gaps between characters.
8. **Fit bounding box** — `cv2.minAreaRect` fits a minimum-area rotated rectangle. Near-square boxes are replaced with axis-aligned boxes.
9. **(Optional) Polygon** — if `--poly` is set, the rotated rectangle is warped to upright, column profiles find the text boundary, and pivot points are computed along the text spine to form a tight polygon.
10. **Scale back** — all coordinates are multiplied by `(1/resize_ratio) × 2` to return to original image space.

---

## Codebase Map

| File | Responsibility |
|---|---|
| `test.py` | CLI entry point, model loading, inference loop, result saving |
| `craft.py` | `CRAFT` model: VGG encoder + U-Net decoder + classification head |
| `basenet/vgg16_bn.py` | `vgg16_bn`: splits pretrained VGG16-BN into 5 feature-map slices; `init_weights`: weight initialisation helper |
| `refinenet.py` | `RefineNet`: ASPP-based link-score refinement network |
| `craft_utils.py` | `getDetBoxes_core`: heatmaps → boxes; `getPoly_core`: boxes → polygons; `adjustResultCoordinates`: scale back to original image |
| `imgproc.py` | `loadImage`, `normalizeMeanVariance`, `resize_aspect_ratio`, `cvt2HeatmapImg` |
| `file_utils.py` | `get_files`/`list_files`: scan directory for images; `saveResult`: write `.txt` + annotated `.jpg` |

---

## Citation

```bibtex
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```

---

## License

Copyright (c) 2019-present NAVER Corp. — MIT License.  
See full license text at the bottom of the original source files.
