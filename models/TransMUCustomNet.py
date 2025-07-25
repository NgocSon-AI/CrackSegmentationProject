import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcn import DeformConv2d
from models.mobilevit import MobileViTBlock
from models.fpp_decoder import FPNDecoder


def shortcut(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class DUC(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hdc = nn.Sequential(
                ConvBlock(in_channels, out_channels, padding=1, dilation=1),
                ConvBlock(out_channels, out_channels, padding=2, dilation=2),
                ConvBlock(out_channels, out_channels, padding=5, dilation=5, with_nonlinearity=False)
            )
        self.shortcut = shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.se = SE_Block(c=out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.se(self.hdc(x))
        x   = self.relu(res + x)
        return x


class DownBlockwithVit(nn.Module):
    def __init__(self, in_channels, out_channels, dim, L, kernel_size=3, patch_size=(4,4)):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.convblock  = ResidualBlock(in_channels, out_channels)
        self.vitblock   = MobileViTBlock(dim, L, out_channels, kernel_size, patch_size, int(dim*2))

    def forward(self, x):
        x = self.downsample(x)
        x = self.convblock(x)
        x = self.vitblock(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.bridge = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        return self.bridge(self.downsample(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = DUC(in_channels, in_channels*2)
        self.residualblock = ResidualBlock(in_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.residualblock(x)
        return x    


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x
    

class TransMUCustomNet(nn.Module):
    DEPTH = 4
    def __init__(self, n_classes=1, dims=[144, 240, 320]):
        super().__init__()
        self.n_classes = n_classes

        # --- Encoder --- #
        self.down_blocks = nn.ModuleList([
            ResidualBlock(in_channels=3, out_channels=32),
            DownBlockwithVit(in_channels=32, out_channels=64, dim=dims[0], L=2),
            DownBlockwithVit(in_channels=64, out_channels=128, dim=dims[1], L=4),
            DownBlockwithVit(in_channels=128, out_channels=256, dim=dims[2], L=3),
        ])

        # Bridge
        self.bridge = Bridge(256, 512)

        # BEM (Boundary Enhancement)
        self.boundary = nn.Sequential(
            DeformConv2d(32, 32, modulation=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        )

        self.se = SE_Block(c=32, r=4)
        self.fpn_decoder = FPNDecoder(
            encoder_channels=[512, 256, 128, 64, 32],
            pyramid_channels=128,
            segmentation_channels=64,
            dropout=0.1,
            final_channels=n_classes
        )

    def forward(self, x, isTrain=False):
        stages = dict()
        stages[f"layer_0"] = x

        # Encoder
        for i, block in enumerate(self.down_blocks, 1):
            x = block(x)
            stages[f"layer_{i}"] = x

        # BEM
        stages1 = stages[f"layer_1"]
        B_out = self.boundary(stages1)

        stages[f"layer_1"] = stages1 + B_out.repeat_interleave(int(stages1.shape[1]), dim=1)
        
        # Bridge
        x = self.bridge(x)

        # Decoder
        features = [
            x,
            stages["layer_4"],
            stages["layer_3"],
            stages["layer_2"],
            stages["layer_1"]
        ]

        out = self.fpn_decoder(features)

        if isTrain:
            return out, B_out
        
        else:
            return out
        

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    model = TransMUCustomNet(n_classes=1)

    input_data = torch.randn(1, 3, 256, 256).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)  # istrain=False mặc định

    print("Input shape :", input_data.shape)
    print("Output shape:", output.shape)