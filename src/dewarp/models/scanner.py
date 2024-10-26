from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseModule


class ResidualBlock(BaseModule):
    """
    Residual block for the encoder network.
    """
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 norm_fn: str = 'group',
                 stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                              padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups, planes)
            self.norm2 = nn.GroupNorm(num_groups, planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups, planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            if not stride == 1:
                self.norm3 = nn.Identity()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(BaseModule):
    """
    Bottleneck residual block with 3 convolutions.
    """
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 norm_fn: str = 'group',
                 stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1,
                              padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3,
                              padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1,
                              padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups//4, planes//4)
            self.norm2 = nn.GroupNorm(num_groups//4, planes//4)
            self.norm3 = nn.GroupNorm(num_groups, planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups, planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            if not stride == 1:
                self.norm4 = nn.Identity()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(BaseModule):
    """
    Basic encoder architecture using residual blocks.
    """
    def __init__(self,
                 output_dim: int = 128,
                 norm_fn: str = 'batch',
                 dropout: float = 0.0,
                 use_bottleneck: bool = False):
        super().__init__()

        self.norm_fn = norm_fn
        self.use_bottleneck = use_bottleneck

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(8, 64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Identity()

        self.conv1 = nn.Conv2d(3, 80, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 80
        self.layer1 = self._make_layer(80, stride=1)
        self.layer2 = self._make_layer(160, stride=2)
        self.layer3 = self._make_layer(240, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(240, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Create a layer with residual/bottleneck blocks."""
        block_class = BottleneckBlock if self.use_bottleneck else ResidualBlock

        layer1 = block_class(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = block_class(dim, dim, self.norm_fn, stride=1)

        layers = (layer1, layer2)
        self.in_planes = dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.conv2(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for visualization or analysis."""
        features = {}

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        features['initial'] = x

        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.conv2(x)
        features['final'] = x

        return features

    def compute_receptive_field(self) -> Tuple[int, int]:
        """Compute the theoretical receptive field size."""
        # Initial convolution
        rf_size = 7
        rf_stride = 2

        # Layer computations
        layers = [self.layer1, self.layer2, self.layer3]

        for layer in layers:
            for block in layer:
                if isinstance(block, (ResidualBlock, BottleneckBlock)):
                    # Each residual/bottleneck block has two 3x3 convs
                    rf_size += 2 * (3 - 1) * rf_stride
                    if hasattr(block, 'downsample') and block.downsample is not None:
                        rf_stride *= 2

        return rf_size, rf_stride

    def get_flops(self, input_shape: Tuple[int, int, int, int]) -> int:
        """
        Calculate approximate FLOPs for the encoder.

        Args:
            input_shape: Input tensor shape (B, C, H, W)

        Returns:
            Approximate number of FLOPs
        """
        batch_size, channels, height, width = input_shape
        flops = 0

        # Initial convolution
        h, w = height // 2, width // 2
        flops += h * w * channels * 80 * 7 * 7

        # Layer 1
        flops += h * w * 80 * 80 * 3 * 3 * 2  # Two residual blocks

        # Layer 2
        h, w = h // 2, w // 2
        flops += h * w * 80 * 160 * 3 * 3 * 2

        # Layer 3
        h, w = h // 2, w // 2
        flops += h * w * 160 * 240 * 3 * 3 * 2

        # Final convolution
        flops += h * w * 240 * self.conv2.out_channels

        return flops * batch_size

    def get_number_params(self) -> Dict[str, int]:
        """
        Get number of parameters in each part of the encoder.

        Returns:
            Dictionary with parameter counts
        """
        params = {
            'initial': sum(p.numel() for p in self.conv1.parameters()) +
                      sum(p.numel() for p in self.norm1.parameters()),
            'layer1': sum(p.numel() for p in self.layer1.parameters()),
            'layer2': sum(p.numel() for p in self.layer2.parameters()),
            'layer3': sum(p.numel() for p in self.layer3.parameters()),
            'final': sum(p.numel() for p in self.conv2.parameters())
        }
        params['total'] = sum(params.values())
        return params


class FlowHead(BaseModule):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(BaseModule):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(BaseModule):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(BaseModule):
    def __init__(self):
        super().__init__()
        self.convc1 = nn.Conv2d(320, 240, 1, padding=0)
        self.convc2 = nn.Conv2d(240, 160, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 160, 7, padding=3)
        self.convf2 = nn.Conv2d(160, 80, 3, padding=1)
        self.conv = nn.Conv2d(160+80, 160-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(BaseModule):
    def __init__(self, hidden_dim=160):  # Changed from 128 to 160
        super().__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=160+160)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=320)

        # Mask architecture must exactly match pretrained weights
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 288, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 64*9, 1, padding=0)
        )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        return net, mask, delta_flow


class DocumentScanner(BaseModule):
    """Document scanner model for document detection and perspective correction."""
    
    def __init__(self):
        super().__init__()
        
        self.hidden_dim = hdim = 160
        self.context_dim = 160
        
        # Renamed from fnet to feature_encoder to match usage
        self.feature_encoder = BasicEncoder(output_dim=320, norm_fn='instance')
        self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize coordinate grids for flow computation."""
        N, C, H, W = img.shape
        
        # Create coordinate grids
        coords_large = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        
        return coords_large, coords0, coords1

    def upsample_flow(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Upsample flow field using convex combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1: torch.Tensor, 
                iters: int = 12,
                flow_init: Optional[torch.Tensor] = None,
                test_mode: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            image1: Input image tensor
            iters: Number of refinement iterations
            flow_init: Initial flow field (optional)
            test_mode: Whether to return only final prediction
            
        Returns:
            Predicted flow field(s)
        """
        image1 = image1.contiguous()

        # Extract features
        fmap1 = self.feature_encoder(image1)
        
        # Split features
        fmap1 = torch.tanh(fmap1)
        net, inp = torch.split(fmap1, [self.hidden_dim, self.hidden_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # Initialize coordinates
        coords_large, coords0, coords1 = self.initialize_flow(image1)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            flow = coords1 - coords0

            # Update prediction
            net, up_mask, delta_flow = self.update_block(net, inp, fmap1, flow)
            coords1 = coords1 + delta_flow

            # Upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(coords_large + flow_up)

            # Update feature maps
            fmap1 = bilinear_sampler(fmap1, coords1.permute(0, 2, 3, 1))

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions


def bilinear_sample(image: torch.Tensor,
                   grid: torch.Tensor,
                   mode: str = 'bilinear',
                   padding_mode: str = 'zeros',
                   align_corners: bool = True,
                   return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample from image using bilinear interpolation.

    Args:
        image: Input image tensor (B, C, H, W)
        grid: Sampling grid tensor (B, H, W, 2) in range [-1, 1]
        mode: Interpolation mode ('bilinear' or 'nearest')
        padding_mode: Padding mode for outside grid values
        align_corners: Whether to align grid corners
        return_mask: Whether to return valid sampling mask

    Returns:
        Sampled tensor and optionally validity mask
    """
    sampled = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    if return_mask:
        mask = ((grid[..., 0] >= -1) & (grid[..., 0] <= 1) &
                (grid[..., 1] >= -1) & (grid[..., 1] <= 1))
        return sampled, mask.float()

    return sampled


def upsample_flow(flow: torch.Tensor,
                 mask: torch.Tensor,
                 scale_factor: int = 8) -> torch.Tensor:
    """
    Upsample flow field using convex combination.

    Args:
        flow: Input flow field tensor
        mask: Combination weights for upsampling
        scale_factor: Factor to upsample by

    Returns:
        Upsampled flow field
    """
    batch_size, _, height, width = flow.shape

    # Reshape mask into attention weights
    mask = mask.view(batch_size, 1, 9, scale_factor, scale_factor, height, width)
    mask = torch.softmax(mask, dim=2)

    # Unfold flow field
    up_flow = F.unfold(scale_factor * flow, [3, 3], padding=1)
    up_flow = up_flow.view(batch_size, 2, 9, 1, 1, height, width)

    # Compute convex combination
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(batch_size, 2, scale_factor * height, scale_factor * width)


def coords_grid(batch: int, ht: int, wd: int) -> torch.Tensor:
    """Generate coordinates grid."""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img: torch.Tensor, 
                     coords: torch.Tensor, 
                     mode: str = 'bilinear',
                     mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Wrapper for grid_sample, uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
