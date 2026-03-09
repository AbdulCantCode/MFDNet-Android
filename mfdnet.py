# Final MFDNet implementation for your project (32 channels, 4 FDBlocks)


import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Depthwise separable convolution
# -----------------------------
class SepConv(nn.Module):
    """
    Efficient depthwise separable convolution block.
    
    Reduces parameter count by separating spatial convolution (depthwise) 
    from channel-wise projection (pointwise). Essential for keeping mobile 
    inference under 100ms.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

# -----------------------------
# Feature Distillation Block
# -----------------------------
class FDBlock(nn.Module):
    """
    Feature Distillation Block (FDB) with lightweight channel attention.
    
    Sequentially refines features using depthwise separable convolutions,
    applies a cheap global channel attention gate, and preserves high-frequency 
    image details via a residual skip connection.
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = SepConv(ch, ch)
        self.conv2 = SepConv(ch, ch)
        self.conv3 = SepConv(ch, ch)

        # tiny channel attention (lightweight)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch//4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, ch//4), ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        g = self.attn(out)         # gating per-channel
        out = out * g
        return out + x             # residual connection

# -----------------------------
# Multi-Scale Fusion Block
# -----------------------------
class MultiScaleFusion(nn.Module):
    """
    Aggregates multi-scale features to capture diverse receptive fields.
    
    Extracts broader contextual information via average pooling, processes it, 
    and fuses it back with the original resolution. Kept intentionally light 
    for fast Android execution.
    """
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = SepConv(ch, ch)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x2 = self.pool(x)
        x2 = self.conv(x2)
        x2 = self.up(x2)
        return x + x2

# -----------------------------
# Final MFDNet Architecture
# -----------------------------
class MFDNet(nn.Module):
    """
    Mobile Feature Distillation Network (MFDNet) for Real-Time Image Denoising.
    
    A highly optimized, mobile-first neural network designed to remove image noise
    with sub-100ms inference on modern smartphone CPUs/GPUs.
    
    Args:
        in_ch (int): Number of input image channels (e.g., 3 for RGB).
        base_ch (int): Base feature channel width (keeps memory footprint low).
        num_blocks (int): Number of Feature Distillation Blocks (FDBlocks).
    """
    def __init__(self, in_ch=3, base_ch=32, num_blocks=4):
        super().__init__()
        # Head: initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Stacked Feature Distillation Blocks
        fdbs = []
        for _ in range(num_blocks):
            fdbs.append(FDBlock(base_ch))
        self.fdb_stack = nn.Sequential(*fdbs)

        # Multi-scale fusion
        self.msf = MultiScaleFusion(base_ch)

        # Bottleneck + reconstruction (light)
        self.bottleneck = SepConv(base_ch, base_ch)
        self.reconstruct = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x):
        fea = self.head(x)            # extract shallow features
        fea = self.fdb_stack(fea)     # deeper features via multiple FDBlocks
        fea = self.msf(fea)           # fuse multi-scale info
        fea = self.bottleneck(fea)    # light bottleneck
        res = self.reconstruct(fea)   # predict residual (clean - noisy)
        out = torch.clamp(x + res, 0.0, 1.0)  # residual learning: noisy + predicted residual
        return out