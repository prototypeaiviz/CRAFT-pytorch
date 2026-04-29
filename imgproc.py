"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2

# Reads an image file and returns it as a 3-channel (H, W, 3) uint8 RGB array.
# Handles edge cases:
#   - Multi-layer TIF (shape[0]==2): takes only the first layer.
#   - Grayscale image (2-D array): converts to RGB by replicating the channel.
#   - RGBA image (4 channels): drops the alpha channel.
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

# Normalises a float32 RGB image using ImageNet mean and variance.
# The mean and variance match those used when the VGG encoder was pretrained,
# so this is a required pre-processing step before running the model.
# Input pixels are expected to be in [0, 255] range; output is float32 with
# roughly zero mean and unit variance per channel.
def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

# Reverses normalizeMeanVariance: converts a normalised float32 image back to
# a displayable uint8 image in [0, 255], clipped to remove out-of-range values.
def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Resizes an image while:
#   1. Preserving aspect ratio.
#   2. Optionally magnifying it first by mag_ratio (e.g. 1.5× before capping).
#   3. Capping at square_size on the longest side.
#   4. Padding width and height to multiples of 32 (required by the network's
#      stride — VGG downsamples by 32×, so dimensions must be divisible by 32).
#
# Returns:
#   resized      — padded float32 image ready for the network.
#   ratio        — the actual scale factor applied (used later to map detections
#                  back to the original image).
#   size_heatmap — (w/2, h/2) tuple indicating the expected heatmap output size
#                  (the network outputs at half the input resolution).
def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    # Round dimensions up to the nearest multiple of 32
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

# Converts a float score map (values in [0, 1]) to a colour heatmap image (uint8).
# Values are clipped to [0, 1], scaled to [0, 255], and rendered using OpenCV's
# JET colormap (blue=low score, red=high score) for easy visualisation.
def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


# ---------------------------------------------------------------------------
# Example: demonstrates every function in this file on a single image.
#
# What it does step by step:
#   1. loadImage          — read the file from disk as an RGB uint8 array
#   2. resize_aspect_ratio — scale it down (max side = 1280px), pad to 32-multiples
#   3. normalizeMeanVariance — subtract ImageNet mean, divide by ImageNet std
#   4. denormalizeMeanVariance — reverse step 3 (proves round-trip works)
#   5. cvt2HeatmapImg     — treat the red channel of the normalised image as a
#                           fake score map and visualise it as a JET heatmap
#   6. cv2.imwrite        — save all intermediate results to disk so you can
#                           open them and see what each step actually does
#
# Run from the project root:
#   python imgproc.py path/to/any_image.jpg
# ---------------------------------------------------------------------------
def example_usage(image_path):
    # ---- Step 1: Load -------------------------------------------------------
    # Returns a (H, W, 3) uint8 array in RGB order.
    img = loadImage(image_path)
    print(f"[1] Loaded image shape : {img.shape}  dtype: {img.dtype}")
    # Save: convert RGB → BGR because OpenCV saves in BGR
    cv2.imwrite("example_1_loaded.jpg", img[:, :, ::-1])

    # ---- Step 2: Resize & pad -----------------------------------------------
    # square_size=1280 : the longest side will be at most 1280 px
    # mag_ratio=1.5    : first try to upscale by 1.5×, then cap at 1280
    # interpolation    : cv2.INTER_LINEAR is a good general-purpose choice
    resized, ratio, size_heatmap = resize_aspect_ratio(
        img, square_size=1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1
    )
    print(f"[2] Resized shape      : {resized.shape}  ratio: {ratio:.4f}")
    print(f"    Expected heatmap   : {size_heatmap}  (half of resized w×h)")
    # resized is float32 [0,255] — clamp before saving
    cv2.imwrite("example_2_resized.jpg", np.clip(resized, 0, 255).astype(np.uint8)[:, :, ::-1])

    # ---- Step 3: Normalise --------------------------------------------------
    # Output is float32, each channel has ~zero mean and ~unit variance.
    # This is the tensor you would feed into the CRAFT model.
    normalised = normalizeMeanVariance(resized)
    print(f"[3] Normalised  min/max: {normalised.min():.3f} / {normalised.max():.3f}")

    # ---- Step 4: Denormalise (round-trip check) -----------------------------
    # Reverses step 3 — the result should look identical to `resized`.
    restored = denormalizeMeanVariance(normalised)
    print(f"[4] Restored    min/max: {restored.min()} / {restored.max()}  dtype: {restored.dtype}")
    cv2.imwrite("example_4_restored.jpg", restored[:, :, ::-1])

    # ---- Step 5: Fake score map → heatmap -----------------------------------
    # cvt2HeatmapImg expects a 2-D float array with values in [0, 1].
    # We take the red channel of the normalised image and rescale it to [0, 1]
    # just to have something meaningful to visualise.
    red_channel = normalised[:, :, 0]                          # shape (H, W)
    score_map = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min() + 1e-6)
    heatmap = cvt2HeatmapImg(score_map)
    print(f"[5] Heatmap shape      : {heatmap.shape}  dtype: {heatmap.dtype}")
    cv2.imwrite("example_5_heatmap.jpg", heatmap)

    print("\nSaved files:")
    print("  example_1_loaded.jpg   — raw loaded image")
    print("  example_2_resized.jpg  — after resize_aspect_ratio")
    print("  example_4_restored.jpg — after normalize then denormalize (should match #2)")
    print("  example_5_heatmap.jpg  — JET heatmap of the red channel score map")


if __name__ == "__main__":
    image_path = "/media/mostafahaggag/D/Data/OCR/intern_data/DATASET_OCR/example_maz/Image_0000_20260213114455.bmp"
    example_usage(image_path)
