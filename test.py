"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

# Strips the "module." prefix that PyTorch adds when a model is saved after
# being wrapped with DataParallel.  Without this, loading a DataParallel
# checkpoint into a plain model would fail due to key name mismatches.
# If the checkpoint was already saved from a plain model (no "module." prefix),
# the function is essentially a no-op that just copies the dict.
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

# Converts a string argument to a boolean.
# Accepts "yes", "y", "true", "t", "1" as True; anything else as False.
# Used so argparse can accept --cuda true / --cuda false on the command line.
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# --- Command-line arguments ---------------------------------------------------
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
# Collect all image paths from the specified test folder at startup
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

# Runs the full CRAFT inference pipeline on a single image and returns detections.
#
# net            — the loaded CRAFT model (possibly wrapped in DataParallel).
# image          — raw RGB image as a numpy array (H, W, 3).
# text_threshold — minimum peak score to accept a connected component as text.
# link_threshold — binarisation threshold for the link/affinity score map.
# low_text       — lower binarisation threshold for the text score map.
# cuda           — whether to run on GPU.
# poly           — if True, refine boxes into tight polygons.
# refine_net     — optional RefineNet; if provided, it re-estimates the link map
#                  after the main network forward pass.
#
# Pipeline:
#   1. Resize the image (preserve aspect ratio, pad to multiples of 32).
#   2. Normalise with ImageNet mean/variance.
#   3. Convert to a 4-D tensor (batch=1) and optionally move to GPU.
#   4. Forward pass through CRAFT → raw score maps (y) and decoder features.
#   5. Optionally forward through RefineNet to get a refined link map.
#   6. Post-process with getDetBoxes → bounding boxes and polygons.
#   7. Scale coordinates back to original image size.
#   8. Build a side-by-side heatmap visualisation of the two score maps.
#
# Returns:
#   boxes         — list of (4,2) bounding boxes in original image coordinates.
#   polys         — list of polygon arrays (or boxes if polygon refinement failed).
#   ret_score_text— colourised heatmap image for visualisation.
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    # y has shape (1, H, W, 2); channel 0 = text region, channel 1 = link/affinity
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    # Scale detections from heatmap space back to original image space
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # Fall back to the bounding box for any polygon that could not be computed
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    # Place text score and link score side-by-side, then convert to a JET heatmap
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)   # wrap for multi-GPU support
        cudnn.benchmark = False            # disabled because input sizes vary per image

    net.eval()   # switch to inference mode (disables dropout / uses BN running stats)

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True   # polygon output is required when the refiner is active

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        # img[:,:,::-1] converts RGB → BGR because OpenCV expects BGR for display/save
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
