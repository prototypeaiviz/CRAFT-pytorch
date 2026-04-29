"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet.vgg16_bn import vgg16_bn, init_weights

# A small building block used in the decoder (U-Net upsampling path).
# It takes two concatenated feature maps (from the decoder path and the skip
# connection from the encoder) and refines them with two convolutions:
#   1. 1×1 conv  — merges the concatenated channels down to mid_ch
#   2. 3×3 conv  — spatial refinement, outputs out_ch channels
# Both convolutions are followed by BatchNorm + ReLU.
class double_conv(nn.Module):
    # in_ch:  channels coming from the previous decoder stage
    # mid_ch: intermediate channel count after the 1×1 merge conv
    # out_ch: output channel count after the 3×3 refinement conv
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    # Passes x through the two-conv block and returns the result.
    def forward(self, x):
        x = self.conv(x)
        return x


# The full CRAFT (Character Region Awareness For Text detection) model.
# Architecture:
#   Encoder: VGG16-BN (vgg16_bn) produces 5 feature maps at different scales.
#   Decoder: U-Net-style path with 4 upconv stages that progressively merge
#            deep (low-res) features with shallow (high-res) skip connections.
#   Head:    conv_cls — a small conv stack that maps the decoder output to
#            2 score maps: character region score and affinity (link) score.
#
# Output shape: (batch, H/2, W/2, 2)  — half the input resolution, 2 channels.
class CRAFT(nn.Module):
    # pretrained: pass True to load ImageNet weights in the VGG encoder
    # freeze:     pass True to freeze the first VGG conv block during training
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        # upconv1: merges fc7 (1024ch) + relu5_3 (512ch) → 256ch
        self.upconv1 = double_conv(1024, 512, 256)
        # upconv2: merges upconv1 output (256ch) + relu4_3 (256ch) → 128ch
        self.upconv2 = double_conv(512, 256, 128)
        # upconv3: merges upconv2 output (128ch) + relu3_2 (128ch) → 64ch
        self.upconv3 = double_conv(256, 128, 64)
        # upconv4: merges upconv3 output (64ch) + relu2_2 (64ch) → 32ch
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        # conv_cls: takes the 32-ch decoder feature map and produces 2 score maps
        # (character region score and affinity/link score) via a series of 3×3
        # and 1×1 convolutions that gradually reduce channels: 32→32→32→16→16→2.
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        # Initialize all decoder and head weights (encoder weights come from VGG pretrained)
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    # Forward pass:
    #   1. Run encoder (VGG) to get multi-scale feature maps.
    #   2. Decode bottom-up: at each step, upsample the current feature map to
    #      match the next skip connection's spatial size, concatenate, then run
    #      double_conv.
    #   3. Run the classification head on the final feature map.
    #
    # Returns:
    #   y       — score maps (batch, H/2, W/2, 2): channel 0 = text region score,
    #             channel 1 = affinity/link score. Permuted to BHWC for compatibility.
    #   feature — raw 32-ch decoder feature map before the classification head,
    #             used by RefineNet if link refinement is enabled.
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        # sources[0]=fc7 (deepest), sources[1]=relu5_3, ..., sources[4]=relu2_2 (shallowest)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)
