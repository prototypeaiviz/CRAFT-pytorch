"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from basenet.vgg16_bn import init_weights


# Optional post-processing network that refines the affinity (link) score map
# produced by CRAFT.  It takes CRAFT's raw output (y) and the 32-ch decoder
# feature map (upconv4) and produces a single improved link score map.
#
# Architecture:
#   last_conv  — three 3×3 conv+BN+ReLU layers that fuse y and upconv4 into a
#                shared 64-ch feature map.
#   aspp1..4   — four parallel ASPP branches with dilation rates 6, 12, 18, 24.
#                Each branch produces 1 output channel.  The four outputs are
#                summed to form the final refined link score.
#
# ASPP (Atrous Spatial Pyramid Pooling) captures context at multiple scales by
# using different dilation rates, which is useful for detecting text of varying
# sizes and orientations.
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        # Fusion block: concatenates CRAFT's score maps (y, 2 ch after permute)
        # with the 32-ch decoder feature (upconv4) → 34 ch total input.
        # Three conv layers refine this into a 64-ch shared representation.
        self.last_conv = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # ASPP branch with dilation=6: captures fine-grained context (~6px gap)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # ASPP branch with dilation=12: captures medium-range context
        self.aspp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # ASPP branch with dilation=18: captures wider context
        self.aspp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # ASPP branch with dilation=24: captures the broadest context
        self.aspp4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    # y       — CRAFT output score maps, shape (batch, H, W, 2) in BHWC format.
    #           Permuted back to BCHW internally before concatenation.
    # upconv4 — 32-ch decoder feature map from CRAFT, shape (batch, 32, H, W).
    #
    # The two tensors are concatenated along the channel dimension (2+32=34 ch),
    # passed through last_conv, then through 4 parallel ASPP branches.
    # The 4 single-channel outputs are element-wise summed to produce the refined
    # link score map, then permuted back to BHWC before returning.
    def forward(self, y, upconv4):
        refine = torch.cat([y.permute(0,3,1,2), upconv4], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)
