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
