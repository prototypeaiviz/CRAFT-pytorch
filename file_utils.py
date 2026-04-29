# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py

# Thin wrapper around list_files: scans img_dir and returns three lists:
#   imgs  — image file paths (.jpg, .jpeg, .gif, .png, .pgm)
#   masks — mask file paths (.bmp)
#   xmls  — annotation/ground-truth file paths (.xml, .gt, .txt)
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

# Recursively walks in_path and splits every file it finds into one of three
# categories based on file extension:
#   img_files  — common raster image formats (jpg/jpeg/gif/png/pgm)
#   mask_files — BMP files (used as binary mask annotations)
#   gt_files   — text/annotation files (xml, gt, txt)
#   .zip files are silently skipped.
#
# Returns (img_files, mask_files, gt_files) — each a flat list of full paths.
def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

# Saves the detection result for one image:
#   1. Writes a .txt file listing each detected polygon's vertices (comma-separated).
#   2. Draws the polygon outlines on the image in red (0,0,255).
#      - If verticals is provided, vertical text boxes are drawn in blue (255,0,0).
#      - If texts is provided, the recognised string is printed near each box.
#   3. Saves the annotated image as a .jpg alongside the .txt file.
#
# Args:
#   img_file  — original image path (used to derive the output filename).
#   img       — raw image array (H, W, 3).
#   boxes     — list of polygon arrays, each shaped (N, 2).
#   dirname   — output directory (created if it does not exist).
#   verticals — optional list of booleans, True means the text is vertical.
#   texts     — optional list of strings, the recognised text for each box.
def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)
