import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, pyramid_channels=128, segmentation_channels=64, dropout=0.1, final_channels=1):
        super(FPNDecoder, self).__init__()

        self.lateral_blocks = nn.ModuleList([
            FPNBlock(in_ch, pyramid_channels) for in_ch in encoder_channels
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(pyramid_channels, segmentation_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(segmentation_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1)

    def forward(self, features):
        # Step 1: Apply 1x1 conv to each encoder output
        features = [lateral(f) for f, lateral in zip(features, self.lateral_blocks)]

        # Step 2: Upsample to match highest resolution (features[-1] = layer_1)
        target_size = features[-1].shape[2:]
        features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                    for f in features]

        # Step 3: Sum all upsampled features
        x = sum(features)

        # Step 4: Fuse and refine
        x = self.dropout(x)
        x = self.fusion_conv(x)
        x = self.final_conv(x)
        return x
