"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

""" auxilary functions """
# Applies an inverse perspective transform matrix (Minv) to a single 2-D point pt.
# Used to map coordinates from the warped (rectified) word crop back to the
# original image coordinate system.
# Returns a 2-element array [x, y] in the original image space.
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


# Core detection function: converts CRAFT's two heatmaps into bounding boxes.
#
# textmap        — 2-D float array, character region score (0-1) from CRAFT.
# linkmap        — 2-D float array, affinity/link score (0-1) from CRAFT.
# text_threshold — minimum peak score inside a connected component to accept it as text.
# link_threshold — binarisation threshold applied to the link map.
# low_text       — lower binarisation threshold applied to the text map (used to
#                  grow the initial mask and capture faint text edges).
#
# Steps:
#   1. Threshold both maps to binary masks.
#   2. Combine them (OR) and run connected-component analysis.
#   3. For each component:
#       a. Skip if too small (<10 px) or peak text score is below text_threshold.
#       b. Build a segmentation mask, remove pure-link pixels, then dilate slightly
#          to close small gaps (dilation amount scales with component size).
#       c. Fit a minimum-area rectangle (rotated bounding box) around the component.
#       d. If the box is nearly square (ratio ≈ 1), replace it with an axis-aligned box.
#       e. Re-order vertices clockwise starting from the top-left.
#
# Returns:
#   det     — list of (4,2) float32 arrays, one rotated bounding box per text instance.
#   labels  — the full connected-component label image (needed by getPoly_core).
#   mapper  — list mapping each det index to its connected-component label id.
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # Combine text and link scores: a pixel is foreground if it is text OR a link
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        # niter: dilation amount derived from the component's area and aspect ratio
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        # Collect all non-zero pixel coordinates and fit a minimum-area rectangle
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        # If the box is nearly square (ratio close to 1), it may be a diamond artefact;
        # replace with an axis-aligned bounding box for robustness.
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        # Roll the 4 vertices so the first vertex is always the top-left (smallest x+y sum)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

# Refines the rectangular boxes from getDetBoxes_core into tighter polygons that
# follow the actual text shape — useful for curved or tilted text lines.
#
# boxes   — list of (4,2) rotated rectangles from getDetBoxes_core.
# labels  — connected-component label image.
# mapper  — maps each box index to its component label id.
# linkmap — original link score map (not used directly here, kept for API symmetry).
#
# For each box the function:
#   1. Warps the label region into a rectangle (perspective transform).
#   2. Samples vertical column profiles to find top/bottom text boundaries.
#   3. Divides the width into segments and computes pivot points along the text spine.
#   4. Extends the polygon at both ends to fully cover the first/last characters.
#   5. Unwarps all polygon points back to original image coordinates via Minv.
#
# Returns a list of polygon arrays (one per box).  Entries are None when the box
# is too small, the transform is singular, or the polygon cannot be computed.
def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5         # number of pivot points along the text spine
    max_len_ratio = 0.7  # skip if the text spans most of the box height (likely not a word)
    expand_ratio = 1.45  # how much to expand the half-character height for the polygon
    max_r = 2.0          # maximum search radius for the start/end edge points
    step_r = 0.2         # step size for the edge-point search

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        # Map the rotated box region to an upright rectangle for easier processing
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        # Keep only the pixels that belong to this specific connected component
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        # For each column x, find the topmost and bottommost foreground pixel.
        # cp entries: (x, top_y, bottom_y)
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        # If the text height fills most of the box, it's likely a single tall character
        # or an artefact — skip polygon generation for this box.
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        # Divide the width into (2*num_cp+1) equal segments.
        # Odd segments are "active" zones where we pick pivot points (the tallest
        # column in that zone becomes the pivot).
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        # half_char_h is the half-height used to project the top/bottom polygon edges
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        # For each pivot, compute the local text-line slope from neighboring section
        # centres and rotate the top/bottom offsets accordingly so they stay
        # perpendicular to the baseline.
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        # Search outward from the first/last pivot for a line that no longer overlaps
        # the text mask — that line marks the true start/end of the word polygon.
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        # Walk: start-edge top → pivot tops (left→right) → end-edge top
        #       end-edge bottom → pivot bottoms (right→left) → start-edge bottom
        # Each point is unwarped from the rectified space back to image space.
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

# Public entry point for bounding-box/polygon detection.
# Calls getDetBoxes_core to get rectangular boxes, then optionally refines them
# into polygons via getPoly_core (when poly=True).
# If poly=False, polys is a list of None values with the same length as boxes.
#
# Returns:
#   boxes  — list of (4,2) rotated rectangles.
#   polys  — list of polygon arrays (or None values when poly=False or refinement failed).
def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

# Scales polygon/box coordinates back to the original image size.
#
# The network processes a resized image (ratio < 1 when the image is shrunk, and
# the output heatmaps are half the network-input resolution — ratio_net=2 by default).
# Multiplying by (ratio_w * ratio_net, ratio_h * ratio_net) undoes both scalings.
#
# polys    — list of coordinate arrays (each shaped (N,2)).
# ratio_w  — inverse of the horizontal resize ratio used before inference.
# ratio_h  — inverse of the vertical resize ratio used before inference.
# ratio_net— accounts for the network's internal 2× downsampling of the heatmap.
def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # This demo builds fake-but-realistic score maps by hand so you can see
    # exactly what each function in this file does without needing a trained
    # model.  Every example is saved as a JPG so you can open it visually.
    #
    # Examples covered:
    #   1. warpCoord              — un-warp a point through an inverse matrix
    #   2. getDetBoxes_core       — threshold maps → connected components → boxes
    #   3. getDetBoxes (poly=False) — the public wrapper, rect output
    #   4. getDetBoxes (poly=True)  — the public wrapper, polygon output
    #   5. adjustResultCoordinates  — scale boxes back to original image size
    # -----------------------------------------------------------------------

    # ── shared canvas size ──────────────────────────────────────────────────
    H, W = 200, 400   # fake heatmap size (think of it as the network output)

    # -----------------------------------------------------------------------
    # Helper: draw boxes/polys on a colour image and save it
    # -----------------------------------------------------------------------
    def save_visual(filename, canvas, boxes, polys=None, color_box=(0,255,0), color_poly=(0,0,255)):
        img = canvas.copy()
        for box in boxes:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color_box, thickness=2)
        if polys:
            for poly in polys:
                if poly is not None:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=color_poly, thickness=2)
        cv2.imwrite(filename, img)
        print(f"  Saved → {filename}")

    # -----------------------------------------------------------------------
    # EXAMPLE 1 — warpCoord
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 1 — warpCoord")
    print("=" * 60)
    # Imagine a word was cropped and straightened using a perspective transform M.
    # Minv is the inverse of that transform — it maps coordinates in the
    # straightened crop back to where they are in the original image.
    #
    # Here we use the identity matrix so the point comes back unchanged,
    # which makes it easy to verify the math is correct.
    Minv_identity = np.eye(3)
    point_in_crop = (50.0, 30.0)
    point_in_original = warpCoord(Minv_identity, point_in_crop)
    print(f"  Input point (in crop)      : {point_in_crop}")
    print(f"  Output point (in original) : {point_in_original}")
    print(f"  With identity Minv the point is unchanged — as expected.\n")

    # Now use a real translation matrix: shift x+10, y+20
    Minv_translate = np.array([[1, 0, 10],
                                [0, 1, 20],
                                [0, 0,  1]], dtype=float)
    shifted = warpCoord(Minv_translate, point_in_crop)
    print(f"  With translate Minv (+10x, +20y):")
    print(f"  Input  : {point_in_crop}")
    print(f"  Output : {shifted}  (should be [60, 50])\n")

    # -----------------------------------------------------------------------
    # EXAMPLE 2 — build fake score maps with 3 "words" drawn on them
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 2 — getDetBoxes_core  (threshold maps → boxes)")
    print("=" * 60)

    # Create blank score maps (float32, values 0..1)
    textmap = np.zeros((H, W), dtype=np.float32)
    linkmap = np.zeros((H, W), dtype=np.float32)

    # Draw 3 fake "words" as bright rectangles on the text map.
    # Each rectangle represents the character-region score blob for one word.
    # word1: a short horizontal word on the left
    cv2.rectangle(textmap, (20,  60), (100, 90),  0.9, -1)   # filled rect, score=0.9
    # word2: a medium word in the centre
    cv2.rectangle(textmap, (150, 60), (270, 90),  0.85, -1)
    # word3: a longer word on the right
    cv2.rectangle(textmap, (300, 60), (380, 90),  0.8, -1)

    # Draw link blobs between word1-word2 and word2-word3 on the link map.
    # These simulate the affinity signal that connects characters within a word.
    cv2.rectangle(linkmap, (100, 65), (150, 85),  0.6, -1)   # gap between word1 & word2
    cv2.rectangle(linkmap, (270, 65), (300, 85),  0.6, -1)   # gap between word2 & word3

    # Save the raw maps as heatmap images so you can see what we're working with
    text_vis = (np.clip(textmap, 0, 1) * 255).astype(np.uint8)
    link_vis = (np.clip(linkmap, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("demo_textmap.jpg", cv2.applyColorMap(text_vis, cv2.COLORMAP_JET))
    cv2.imwrite("demo_linkmap.jpg", cv2.applyColorMap(link_vis, cv2.COLORMAP_JET))
    print("  Saved → demo_textmap.jpg  (character region score map)")
    print("  Saved → demo_linkmap.jpg  (affinity/link score map)\n")

    # Run getDetBoxes_core with default-like thresholds
    text_threshold = 0.7   # peak inside a blob must exceed this to count as text
    link_threshold = 0.4   # link map binarisation threshold
    low_text       = 0.4   # lower threshold to grow the initial text mask

    (boxes,
     labels,
     mapper) = getDetBoxes_core(
        textmap,
        linkmap,
        text_threshold,
        link_threshold,
        low_text
    )

    print(f"  text_threshold={text_threshold}, link_threshold={link_threshold}, low_text={low_text}")
    print(f"  Number of boxes found      : {len(boxes)}")
    for i, box in enumerate(boxes):
        print(f"  Box {i}: {box.astype(int).tolist()}")

    # Visualise boxes on a grey canvas
    canvas = cv2.cvtColor((text_vis), cv2.COLOR_GRAY2BGR)
    save_visual("demo_boxes_core.jpg", canvas, boxes)
    print()

    # -----------------------------------------------------------------------
    # EXAMPLE 3 — getDetBoxes  (public wrapper, rect mode)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 3 — getDetBoxes  (poly=False → rectangles)")
    print("=" * 60)
    boxes_rect, polys_none = getDetBoxes(
        textmap, linkmap, text_threshold, link_threshold, low_text, poly=False
    )
    print(f"  Boxes  : {len(boxes_rect)}")
    print(f"  Polys  : {polys_none}  ← all None because poly=False")
    save_visual("demo_getdetboxes_rect.jpg", canvas, boxes_rect)
    print()

    # -----------------------------------------------------------------------
    # EXAMPLE 4 — getDetBoxes  (poly=True → polygons)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 4 — getDetBoxes  (poly=True → polygons)")
    print("=" * 60)
    # Polygon mode tries to fit a tight polygon around each word instead of a
    # rectangle.  It may return None for some boxes if they are too small or
    # the polygon algorithm can't converge — in that case the box is used instead.
    boxes_poly, polys = getDetBoxes(
        textmap, linkmap, text_threshold, link_threshold, low_text, poly=True
    )
    print(f"  Boxes  : {len(boxes_poly)}")
    for i, poly in enumerate(polys):
        if poly is not None:
            print(f"  Poly {i} : {len(poly)} vertices  → {poly.astype(int).tolist()}")
        else:
            print(f"  Poly {i} : None  (polygon failed — will fall back to box)")
    save_visual("demo_getdetboxes_poly.jpg", canvas, boxes_poly, polys)
    print()

    # -----------------------------------------------------------------------
    # EXAMPLE 5 — adjustResultCoordinates
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 5 — adjustResultCoordinates")
    print("=" * 60)
    # The heatmap is half the size of the network input (ratio_net=2),
    # and the network input was itself a resized version of the original image.
    #
    # Suppose the original image was 800×1600 px and was resized to fit in
    # canvas_size=1280 — so the longest side (1600) was scaled to 1280.
    # resize_ratio = 1280 / 1600 = 0.8
    # ratio_w = ratio_h = 1 / resize_ratio = 1.25  (to undo the shrink)
    ratio_w = 1.25
    ratio_h = 1.25

    print(f"  Before scaling — box 0 vertices:")
    print(f"  {boxes_rect[0].astype(int).tolist()}")

    # adjustResultCoordinates modifies the array in-place via multiplication
    boxes_scaled = adjustResultCoordinates(list(boxes_rect), ratio_w, ratio_h, ratio_net=2)

    print(f"  After  scaling (ratio_w={ratio_w}, ratio_h={ratio_h}, ratio_net=2) — box 0:")
    print(f"  {boxes_scaled[0].astype(int).tolist()}")
    print(f"  Each coordinate was multiplied by {ratio_w} × 2 = {ratio_w*2}")
    print(f"  → coordinates now live in the original 800×1600 image space")
