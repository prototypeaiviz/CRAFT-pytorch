from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}
# Initializes weights for Conv2d, BatchNorm2d, and Linear layers.
# Conv2d uses Xavier uniform (good for deep nets), BatchNorm fills weight=1/bias=0,
# Linear uses a small normal distribution. Called on freshly built layers that have
# no pretrained weights.
def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

# Wraps the pretrained VGG16-BN feature extractor and splits it into 5 slices
# so that the CRAFT decoder can collect intermediate feature maps at different
# spatial scales (similar to a feature pyramid).
#
# Slice layout (VGG16-BN layer indices):
#   slice1  [0..11]   → output after conv2_2  (stride 4,  high resolution)
#   slice2  [12..18]  → output after conv3_3  (stride 8)
#   slice3  [19..28]  → output after conv4_3  (stride 16)
#   slice4  [29..38]  → output after conv5_3  (stride 32)
#   slice5  custom    → dilated conv replacing VGG's fc6/fc7 (stride 32, wider receptive field)
class vgg16_bn(torch.nn.Module):
    # pretrained: load ImageNet weights for slices 1-4
    # freeze:     if True, gradient updates are disabled for slice1 (earliest conv)
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        # Force HTTP download in case the HTTPS cert is blocked
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        # Pull only the convolutional feature part of VGG16-BN (no classifier head)
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        # Replaces VGG's fully-connected layers with dilated convolutions so the
        # network stays fully-convolutional and keeps spatial information.
        # MaxPool with stride=1 avoids further downsampling.
        # dilation=6 gives a large receptive field without extra parameters.
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    # Runs the input image X through the 5 slices sequentially and collects the
    # output of each slice. Returns a named tuple so callers can access outputs
    # by name (fc7, relu5_3, relu4_3, relu3_2, relu2_2) — ordered from deepest
    # (smallest spatial size) to shallowest (largest spatial size).
    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # What this demo shows:
    #
    #  1. What the input tensor looks like (shape, dtype)
    #  2. What each of the 5 slices does to the spatial size (H, W shrink)
    #  3. What the channel count is at each slice output
    #  4. How the named-tuple lets you access each feature map by name
    #  5. Which slice1 parameters are frozen (require_grad=False)
    #  6. What init_weights does to a brand-new Conv2d layer
    # -------------------------------------------------------------------------

    print("=" * 60)
    print("STEP 1 — Build the model (pretrained=False to skip download)")
    print("=" * 60)
    # pretrained=False so the demo runs without an internet connection.
    # In real use (inside CRAFT) pretrained=True loads ImageNet weights.
    model = vgg16_bn(pretrained=True, freeze=True)
    model.eval()  # turn off dropout / use BN running stats (no training here)
    print("Model built successfully.\n")

    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 — Create a fake input image tensor")
    print("=" * 60)
    # Real images come in as (batch, channels, height, width).
    # batch=1 : one image at a time during CRAFT inference
    # channels=3 : RGB
    # 320x320 : small size so this runs fast on CPU
    # Values are random floats — in real use these would be
    # ImageNet-normalised pixel values (roughly -2 to +2).
    fake_input = torch.randn(1, 3, 320, 320)
    print(f"Input shape  : {list(fake_input.shape)}")
    print(f"  batch={fake_input.shape[0]}, channels={fake_input.shape[1]}, "
          f"H={fake_input.shape[2]}, W={fake_input.shape[3]}\n")

    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3 — Run the forward pass and inspect all 5 outputs")
    print("=" * 60)
    # The forward() method returns a named tuple with 5 fields.
    # Each field is a feature map from a different depth in the network.
    with torch.no_grad():
        outputs = model(fake_input)

    # Print a table: name | spatial size | channels | stride vs input
    print(f"{'Name':<12}  {'Shape (B,C,H,W)':<28}  {'Stride (approx)'}")
    print("-" * 58)
    for name, feat in zip(outputs._fields, outputs):
        b, c, h, w = feat.shape
        stride_h = fake_input.shape[2] // h
        stride_w = fake_input.shape[3] // w
        print(f"{name:<12}  {str(list(feat.shape)):<28}  {stride_h}x{stride_w}")

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 4 — Access individual outputs by name (the named tuple)")
    print("=" * 60)
    # Because forward() returns a namedtuple you can use dot notation.
    # CRAFT's decoder starts with the deepest feature (fc7) and works
    # up to the shallowest (relu2_2), merging them one by one.
    print(f"outputs.fc7      shape: {list(outputs.fc7.shape)}")
    print(f"outputs.relu5_3  shape: {list(outputs.relu5_3.shape)}")
    print(f"outputs.relu4_3  shape: {list(outputs.relu4_3.shape)}")
    print(f"outputs.relu3_2  shape: {list(outputs.relu3_2.shape)}")
    print(f"outputs.relu2_2  shape: {list(outputs.relu2_2.shape)}")
    print()
    print("Notice: going from fc7 → relu2_2, the spatial size doubles each")
    print("step while the channel count halves. This is the 'pyramid' that")
    print("the CRAFT U-Net decoder uses to reconstruct fine spatial detail.")

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 5 — Which parameters are frozen in slice1?")
    print("=" * 60)
    # When freeze=True, slice1's parameters have requires_grad=False,
    # meaning they will NOT be updated during training.
    # This protects the low-level edge detectors learned from ImageNet.
    frozen = [(n, p.shape) for n, p in model.slice1.named_parameters() if not p.requires_grad]
    trainable = [(n, p.shape) for n, p in model.slice1.named_parameters() if p.requires_grad]
    print(f"Frozen params in slice1    : {len(frozen)}")
    for name, shape in frozen[:4]:          # show first 4 to keep output short
        print(f"  {name:<40} shape={list(shape)}")
    print(f"Trainable params in slice1 : {len(trainable)}  (0 because freeze=True)")

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 6 — What does init_weights do to a fresh Conv2d?")
    print("=" * 60)
    # Create a plain Conv2d and check its weights BEFORE and AFTER init_weights.
    test_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    print(f"Before init_weights:")
    print(f"  weight mean={test_conv.weight.data.mean():.4f}  std={test_conv.weight.data.std():.4f}")
    print(f"  bias   mean={test_conv.bias.data.mean():.4f}")
    init_weights([test_conv])
    print(f"After  init_weights (Xavier uniform):")
    print(f"  weight mean={test_conv.weight.data.mean():.4f}  std={test_conv.weight.data.std():.4f}")
    print(f"  bias   mean={test_conv.bias.data.mean():.4f}  (zeroed out)")
    print()
    print("Xavier uniform keeps the signal variance stable across layers,")
    print("which helps gradients flow during training without vanishing/exploding.")
