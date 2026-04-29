# -*- coding: utf-8 -*-
"""
Minimal single-image runner for CRAFT.

Usage:
    python run_single_image.py --image path/to/image.jpg \
                               --weights weights/craft_mlt_25k.pth

Download weights from:
    General model  → https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
    Place the .pth file in the weights/ folder next to this script.

Outputs written to ./result/:
    res_<name>.jpg   — input image with red boxes drawn on detected text
    res_<name>.txt   — one box per line as x1,y1,x2,y2,x3,y3,x4,y4
    res_<name>_mask.jpg — side-by-side JET heatmap of both score maps
"""

import os
import time
import argparse

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict

import imgproc
import craft_utils
from craft import CRAFT


# ---------------------------------------------------------------------------
# Utility: strip the "module." prefix added when a model is saved after
# wrapping with DataParallel.  Lets you load a multi-GPU checkpoint into a
# plain single-GPU model.
# ---------------------------------------------------------------------------
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# ---------------------------------------------------------------------------
# test_net: runs the full CRAFT pipeline on ONE image.
#
# Steps inside:
#   1. Resize the image (keep aspect ratio, pad to 32-multiples)
#   2. Normalise with ImageNet mean/std
#   3. Convert to tensor and move to GPU if available
#   4. Forward pass through CRAFT → two score maps
#   5. Post-process score maps → bounding boxes / polygons
#   6. Scale coordinates back to the original image size
#   7. Build a visualisation heatmap
#
# Returns:
#   boxes         — list of (4,2) arrays, one box per detected text region
#   polys         — same but as tight polygons (or same as boxes if poly=False)
#   ret_score_text— side-by-side JET heatmap image for debugging
# ---------------------------------------------------------------------------
def test_net(net, image, text_threshold, link_threshold, low_text,
             cuda, poly, canvas_size, mag_ratio):

    t0 = time.time()

    # ---- Step 1 & 2: resize + normalise ------------------------------------
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    # ratio_w / ratio_h: multiply by these to go from heatmap coords → original image coords
    ratio_h = ratio_w = 1 / target_ratio
    print(f"  Original size  : {image.shape[1]}×{image.shape[0]}  (W×H)")
    print(f"  Resized to     : {img_resized.shape[1]}×{img_resized.shape[0]}  (ratio={target_ratio:.3f})")
    print(f"  Heatmap size   : {size_heatmap}  (half of resized)")

    x = imgproc.normalizeMeanVariance(img_resized)

    # ---- Step 3: numpy → PyTorch tensor ------------------------------------
    # permute: (H,W,C) → (C,H,W), then unsqueeze adds the batch dim → (1,C,H,W)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()
    print(f"  Tensor shape   : {list(x.shape)}  (batch=1, C=3, H, W)")

    # ---- Step 4: forward pass ----------------------------------------------
    print("\n  Running CRAFT forward pass ...")
    with torch.no_grad():
        y, feature = net(x)

    # y shape: (1, H, W, 2)  — BHWC format
    # channel 0 = text region score   (how likely each pixel is a character centre)
    # channel 1 = affinity/link score (how likely two characters belong together)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    print(f"  Score map shape: {score_text.shape}  (H×W, values 0-1)")
    print(f"  Text  score — min={score_text.min():.3f}  max={score_text.max():.3f}")
    print(f"  Link  score — min={score_link.min():.3f}  max={score_link.max():.3f}")

    t0 = time.time() - t0
    t1 = time.time()

    # ---- Step 5: post-process score maps → boxes ---------------------------
    print("\n  Running post-processing ...")
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )
    print(f"  Detected regions: {len(boxes)}")

    # ---- Step 6: scale coordinates back to original image space ------------
    # The heatmap is half the network-input size (ratio_net=2) and the network
    # input was itself a resized version of the original → multiply by both.
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]   # fall back to rectangle when polygon failed

    t1 = time.time() - t1
    print(f"  Inference time : {t0:.3f}s   Post-process time: {t1:.3f}s")

    # ---- Step 7: build heatmap visualisation --------------------------------
    render_img = np.hstack((score_text.copy(), score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':


    cuda = torch.cuda.is_available()
    image_file_name = "/media/mostafahaggag/D/Data/OCR/intern_data/DATASET_OCR/example_maz/Image_0003_20260213114456.bmp"
    text_threshold = 0.7
    low_text = 0.4
    link_threshold=0.4
    canvas_size=1280
    mag_ratio =1.5
    weights = "weights/craft_mlt_25k.pth"   # ← set this to your downloaded .pth file
    poly = False
    # ---- 1. Load image -----------------------------------------------------
    print("=" * 55)
    print("STEP 1 — Load image")
    print("=" * 55)
    if not os.path.isfile(image_file_name):
        raise FileNotFoundError(f"Image not found: {image_file_name}")
    image = imgproc.loadImage(image_file_name)
    print(f"  Loaded: {image_file_name}")
    print(f"  Shape : {image.shape}  (H={image.shape[0]}, W={image.shape[1]}, C={image.shape[2]})")

    # ---- 2. Load model -----------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 2 — Load CRAFT model")
    print("=" * 55)
    net = CRAFT()
    if not weights or not os.path.isfile(weights):
        raise FileNotFoundError(
            f"Weights not found: '{weights}'\n"
            f"  Download: https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ\n"
            f"  Then set:  weights = '/path/to/craft_mlt_25k.pth'"
        )
    print(f"  Loading weights: {weights}")
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(weights)))
        net = net.cuda()
    else:
        net.load_state_dict(copyStateDict(torch.load(weights, map_location='cpu')))
    net.eval()
    print(f"  Device: {'GPU (CUDA)' if cuda else 'CPU'}")

    # ---- 3. Run inference --------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 3 — Run inference")
    print("=" * 55)
    boxes, polys, score_heatmap = test_net(
        net, image,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        cuda=cuda,
        poly=poly,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
    )

    # ---- 4. Save results ---------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 4 — Save results")
    print("=" * 55)
    os.makedirs('./result', exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_file_name))[0]

    # 4a. Score heatmap (text map left, link map right, JET colormap)
    mask_path = f"./result/{filename}_mask.jpg"
    cv2.imwrite(mask_path, score_heatmap)
    print(f"  Heatmap  → {mask_path}")

    # 4b. Annotated image + coordinates .txt
    result_img  = image[:, :, ::-1].copy()   # RGB → BGR for OpenCV
    result_txt  = f"./result/res_{filename}.txt"
    result_jpg  = f"./result/res_{filename}.jpg"

    with open(result_txt, 'w') as f:
        for i, poly in enumerate(polys):
            pts = np.array(poly).astype(np.int32).reshape(-1)
            f.write(','.join(map(str, pts)) + '\r\n')
            cv2.polylines(result_img, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    cv2.imwrite(result_jpg, result_img)
    print(f"  Boxes    → {result_txt}  ({len(polys)} regions)")
    print(f"  Image    → {result_jpg}")

    print("\nDone.")
