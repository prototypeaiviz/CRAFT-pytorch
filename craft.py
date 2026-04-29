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
    # -----------------------------------------------------------------------
    # Step-by-step shape tracer for the full CRAFT forward pass.
    # Hooks into every stage so you can see exactly how the tensor dimensions
    # change from raw input → encoder slices → decoder upconvs → output maps.
    # -----------------------------------------------------------------------

    def fmt(t):
        # Helper: format a tensor shape as a short readable string
        b, c, h, w = t.shape
        return f"(B={b}, C={c:>4d}, H={h:>3d}, W={w:>3d})"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}\n")

    # Build model (pretrained=False avoids needing weights file for this demo)
    model = CRAFT(pretrained=False, freeze=False).to(device)
    model.eval()

    # Input: one fake RGB image, 640x640
    # In real use this would be a normalised image from imgproc.normalizeMeanVariance
    x = torch.randn(1, 3, 640, 640).to(device)

    print("=" * 65)
    print(" INPUT")
    print("=" * 65)
    print(f"  x (raw image tensor)          {fmt(x)}")
    print(f"  → 3 channels (RGB), 640×640 px\n")

    with torch.no_grad():

        # ---- ENCODER (VGG16-BN slices) ------------------------------------
        print("=" * 65)
        print(" ENCODER  —  VGG16-BN  (basenet)")
        print("=" * 65)
        sources = model.basenet(x)
        # sources is a namedtuple: fc7, relu5_3, relu4_3, relu3_2, relu2_2
        names   = ["fc7      (slice5)", "relu5_3  (slice4)", "relu4_3  (slice3)",
                   "relu3_2  (slice2)", "relu2_2  (slice1)"]
        for name, feat in zip(names, sources):
            print(f"  {name}   {fmt(feat)}")
        print()
        print("  Each slice halves H and W (stride 2 per pool).")
        print("  fc7 and relu5_3 share the same spatial size because")
        print("  slice5 uses stride-1 MaxPool (no extra downsampling).\n")

        # ---- DECODER  (U-Net upsampling path) ------------------------------
        print("=" * 65)
        print(" DECODER  —  U-Net upsampling path")
        print("=" * 65)

        # --- upconv1: fc7 + relu5_3 -----------------------------------------
        cat1 = torch.cat([sources[0], sources[1]], dim=1)
        print(f"  cat(fc7, relu5_3)             {fmt(cat1)}"
              f"  ← {sources[0].shape[1]}+{sources[1].shape[1]} channels merged")
        y = model.upconv1(cat1)
        print(f"  after upconv1                 {fmt(y)}"
              f"  ← double_conv reduces to 256 ch\n")

        # --- upconv2: upsample + relu4_3 ------------------------------------
        y_up = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        print(f"  upsample → match relu4_3      {fmt(y_up)}"
              f"  ← bilinear upsample to {tuple(sources[2].size()[2:])} (relu4_3 size)")
        cat2 = torch.cat([y_up, sources[2]], dim=1)
        print(f"  cat(upsampled, relu4_3)       {fmt(cat2)}"
              f"  ← {y_up.shape[1]}+{sources[2].shape[1]} channels merged")
        y = model.upconv2(cat2)
        print(f"  after upconv2                 {fmt(y)}"
              f"  ← double_conv reduces to 128 ch\n")

        # --- upconv3: upsample + relu3_2 ------------------------------------
        y_up = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        print(f"  upsample → match relu3_2      {fmt(y_up)}"
              f"  ← bilinear upsample to {tuple(sources[3].size()[2:])} (relu3_2 size)")
        cat3 = torch.cat([y_up, sources[3]], dim=1)
        print(f"  cat(upsampled, relu3_2)       {fmt(cat3)}"
              f"  ← {y_up.shape[1]}+{sources[3].shape[1]} channels merged")
        y = model.upconv3(cat3)
        print(f"  after upconv3                 {fmt(y)}"
              f"  ← double_conv reduces to 64 ch\n")

        # --- upconv4: upsample + relu2_2 ------------------------------------
        y_up = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        print(f"  upsample → match relu2_2      {fmt(y_up)}"
              f"  ← bilinear upsample to {tuple(sources[4].size()[2:])} (relu2_2 size)")
        cat4 = torch.cat([y_up, sources[4]], dim=1)
        print(f"  cat(upsampled, relu2_2)       {fmt(cat4)}"
              f"  ← {y_up.shape[1]}+{sources[4].shape[1]} channels merged")
        feature = model.upconv4(cat4)
        print(f"  after upconv4  (feature map)  {fmt(feature)}"
              f"  ← double_conv reduces to 32 ch\n")

        # ---- HEAD (classification) -----------------------------------------
        print("=" * 65)
        print(" HEAD  —  conv_cls  (produces the two score maps)")
        print("=" * 65)
        y_cls = model.conv_cls(feature)
        print(f"  conv_cls output               {fmt(y_cls)}"
              f"  ← 2 channels: text-region + affinity")
        y_final = y_cls.permute(0, 2, 3, 1)
        print(f"  after permute (BCHW→BHWC)     (B={y_final.shape[0]}, H={y_final.shape[1]:>3d},"
              f" W={y_final.shape[2]:>3d}, C={y_final.shape[3]})\n")

        print("=" * 65)
        print(" SUMMARY")
        print("=" * 65)
        print(f"  Input image      : {x.shape[2]}×{x.shape[3]} px")
        print(f"  Score maps       : {y_final.shape[1]}×{y_final.shape[2]} px"
              f"  (= input / {x.shape[2]//y_final.shape[1]}×  — network's inherent downsampling)")
        print(f"  Channel 0        : text region score  (where characters are)")
        print(f"  Channel 1        : affinity score     (links between characters)")
        print(f"  feature map      : {fmt(feature)}  → passed to RefineNet if enabled")
