from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls

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
