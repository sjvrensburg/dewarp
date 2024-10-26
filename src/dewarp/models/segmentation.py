from typing import Optional, Tuple, Union, Type, List, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseModule

class ConvBlock(BaseModule):
    """Basic convolutional block with normalization and activation."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm_type: str = 'batch',
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0):
        super().__init__()

        # Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=norm_type is None
        )

        # Normalization
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'group':
            num_groups = min(32, out_channels // 8)
            self.norm = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        self.activation = activation(inplace=True) if activation else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SeparableConvBlock(BaseModule):
    """Depthwise separable convolution block."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 norm_type: str = 'batch',
                 activation: Type[nn.Module] = nn.ReLU):
        super().__init__()

        self.depthwise = ConvBlock(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
            norm_type=norm_type, activation=activation
        )

        self.pointwise = ConvBlock(
            in_channels, out_channels, 1,
            norm_type=norm_type, activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvGRUCell(BaseModule):
    """Convolutional GRU cell for spatial features."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim, kernel_size,
            padding=padding
        )

        self.conv_can = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size,
            padding=padding
        )

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width,
                          device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h], dim=1)

        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined = torch.cat([x, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_can(combined))

        h_next = update_gate * h + (1 - update_gate) * candidate

        return h_next


class SeparableConvGRU(BaseModule):
    """GRU cell with separable convolutions for both horizontal and vertical processing."""
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192+128):
        super().__init__()

        # Horizontal processing
        self.conv_gates_h = nn.Conv2d(
            hidden_dim + input_dim, 2 * hidden_dim, (1, 5),
            padding=(0, 2)
        )
        self.conv_can_h = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5),
            padding=(0, 2)
        )

        # Vertical processing
        self.conv_gates_v = nn.Conv2d(
            hidden_dim + input_dim, 2 * hidden_dim, (5, 1),
            padding=(2, 0)
        )
        self.conv_can_v = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1),
            padding=(2, 0)
        )

    def _gru_step(self,
                  x: torch.Tensor,
                  h: torch.Tensor,
                  conv_gates: nn.Module,
                  conv_can: nn.Module) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)

        gates = conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, h.size(1), dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined = torch.cat([x, reset_gate * h], dim=1)
        candidate = torch.tanh(conv_can(combined))

        h_next = update_gate * h + (1 - update_gate) * candidate
        return h_next

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Horizontal processing
        h = self._gru_step(x, h, self.conv_gates_h, self.conv_can_h)

        # Vertical processing
        h = self._gru_step(x, h, self.conv_gates_v, self.conv_can_v)

        return h


class FlowHead(BaseModule):
    """Network head for optical flow prediction."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class MotionEncoder(BaseModule):
    """Encoder for motion features combining correlation and flow information."""
    def __init__(self,
                 corr_channels: int = 320,
                 mid_channels: int = 240,
                 out_channels: int = 160):
        super().__init__()

        # Correlation processing
        self.corr_conv1 = nn.Conv2d(corr_channels, mid_channels, 1)
        self.corr_conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

        # Flow processing
        self.flow_conv1 = nn.Conv2d(2, out_channels, 7, padding=3)
        self.flow_conv2 = nn.Conv2d(out_channels, out_channels//2, 3, padding=1)

        # Final fusion
        self.fusion_conv = nn.Conv2d(out_channels + out_channels//2,
                                   out_channels-2, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # Process correlation features
        corr = self.relu(self.corr_conv1(corr))
        corr = self.relu(self.corr_conv2(corr))

        # Process flow features
        flow_feat = self.relu(self.flow_conv1(flow))
        flow_feat = self.relu(self.flow_conv2(flow_feat))

        # Fuse features
        fusion = torch.cat([corr, flow_feat], dim=1)
        out = self.relu(self.fusion_conv(fusion))

        # Concatenate with original flow
        return torch.cat([out, flow], dim=1)


class UpdateBlock(BaseModule):
    """Update block combining motion encoding, GRU processing, and flow prediction."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.encoder = MotionEncoder()
        self.gru = SeparableConvGRU(hidden_dim=hidden_dim, input_dim=160+160)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=320)

        # Mask prediction for flow upsampling
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1)
        )

    def forward(self,
                net: torch.Tensor,
                inp: torch.Tensor,
                corr: torch.Tensor,
                flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        # GRU update
        net = self.gru(net, inp)

        # Predict flow update and mask
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)

        return net, mask, delta_flow


class BasicMotionEncoder(BaseModule):
    """
    Basic encoder for motion features combining correlation and flow information.
    """
    def __init__(self,
                 corr_levels: int = 4,
                 corr_radius: int = 4,
                 input_dim: int = 320,
                 hidden_dims: List[int] = [240, 160],
                 flow_input_dim: int = 2,
                 flow_hidden_dims: List[int] = [160, 80]):
        super().__init__()

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        # Correlation processing layers
        self.convc1 = nn.Conv2d(input_dim, hidden_dims[0], 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1)

        # Flow processing layers
        self.convf1 = nn.Conv2d(flow_input_dim, flow_hidden_dims[0], 7, padding=3)
        self.convf2 = nn.Conv2d(flow_hidden_dims[0], flow_hidden_dims[1], 3, padding=1)

        # Fusion layer
        self.conv = nn.Conv2d(hidden_dims[1] + flow_hidden_dims[1],
                             hidden_dims[1] - flow_input_dim, 3, padding=1)

        # Initialize weights
        self.apply(self.initialize_weights)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the motion encoder.

        Args:
            flow: Input flow tensor of shape (B, 2, H, W)
            corr: Correlation tensor of shape (B, C, H, W)

        Returns:
            Encoded motion features
        """
        # Process correlation features
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        # Process flow features
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # Concatenate and fuse features
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        # Concatenate with input flow
        return torch.cat([out, flow], dim=1)

    def compute_correlation_features(self,
                                   flow: torch.Tensor,
                                   corr: torch.Tensor,
                                   return_intermediates: bool = False
                                   ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute correlation features with optional intermediate results.

        Args:
            flow: Input flow tensor
            corr: Correlation tensor
            return_intermediates: Whether to return intermediate features

        Returns:
            Processed features or dictionary with intermediates
        """
        intermediates = {}

        # Process correlation
        cor = F.relu(self.convc1(corr))
        if return_intermediates:
            intermediates['corr_features_1'] = cor

        cor = F.relu(self.convc2(cor))
        if return_intermediates:
            intermediates['corr_features_2'] = cor

        if return_intermediates:
            return cor, intermediates
        return cor

    def compute_flow_features(self,
                            flow: torch.Tensor,
                            return_intermediates: bool = False
                            ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow features with optional intermediate results.

        Args:
            flow: Input flow tensor
            return_intermediates: Whether to return intermediate features

        Returns:
            Processed features or dictionary with intermediates
        """
        intermediates = {}

        # Process flow
        flo = F.relu(self.convf1(flow))
        if return_intermediates:
            intermediates['flow_features_1'] = flo

        flo = F.relu(self.convf2(flo))
        if return_intermediates:
            intermediates['flow_features_2'] = flo

        if return_intermediates:
            return flo, intermediates
        return flo


class AdvancedMotionEncoder(BaseModule):
    """
    Advanced motion encoder with multi-scale processing and attention.
    """
    def __init__(self,
                 input_dim: int = 320,
                 hidden_dim: int = 160,
                 num_scales: int = 3):
        super().__init__()

        self.num_scales = num_scales

        # Multi-scale correlation processing
        self.corr_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])

        # Multi-scale flow processing
        self.flow_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, hidden_dim, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_scales, 1),
            nn.Softmax(dim=1)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim-2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self.apply(self.initialize_weights)

    def _process_scale(self,
                      flow: torch.Tensor,
                      corr: torch.Tensor,
                      scale_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process features at a specific scale."""
        # Scale the inputs if needed
        if scale_idx > 0:
            scale_factor = 1 / (2 ** scale_idx)
            flow = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear',
                               align_corners=True) * scale_factor
            corr = F.interpolate(corr, scale_factor=scale_factor, mode='bilinear',
                               align_corners=True)

        # Process features
        corr_feat = self.corr_encoders[scale_idx](corr)
        flow_feat = self.flow_encoders[scale_idx](flow)

        return corr_feat, flow_feat

    def forward(self,
                flow: torch.Tensor,
                corr: torch.Tensor,
                return_intermediates: bool = False
                ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-scale processing and attention.

        Args:
            flow: Input flow tensor
            corr: Correlation tensor
            return_intermediates: Whether to return intermediate features

        Returns:
            Processed features or dictionary with intermediates
        """
        intermediates = {} if return_intermediates else None

        # Process each scale
        corr_feats = []
        flow_feats = []

        for i in range(self.num_scales):
            corr_feat, flow_feat = self._process_scale(flow, corr, i)
            corr_feats.append(corr_feat)
            flow_feats.append(flow_feat)

            if return_intermediates:
                intermediates[f'scale_{i}_corr'] = corr_feat
                intermediates[f'scale_{i}_flow'] = flow_feat

        # Resize all features to the original scale
        for i in range(1, self.num_scales):
            corr_feats[i] = F.interpolate(corr_feats[i], size=corr_feats[0].shape[-2:],
                                        mode='bilinear', align_corners=True)
            flow_feats[i] = F.interpolate(flow_feats[i], size=flow_feats[0].shape[-2:],
                                        mode='bilinear', align_corners=True)

        # Concatenate features
        corr_cat = torch.cat(corr_feats, dim=1)
        flow_cat = torch.cat(flow_feats, dim=1)

        if return_intermediates:
            intermediates['concatenated_corr'] = corr_cat
            intermediates['concatenated_flow'] = flow_cat

        # Compute attention weights
        attention_weights = self.attention(corr_cat)
        if return_intermediates:
            intermediates['attention_weights'] = attention_weights

        # Apply attention and fuse features
        attended_corr = torch.sum(corr_cat * attention_weights.unsqueeze(1), dim=1)
        attended_flow = torch.sum(flow_cat * attention_weights.unsqueeze(1), dim=1)

        if return_intermediates:
            intermediates['attended_corr'] = attended_corr
            intermediates['attended_flow'] = attended_flow

        # Final fusion
        fused = self.fusion(torch.cat([attended_corr, attended_flow], dim=1))
        out = torch.cat([fused, flow], dim=1)

        if return_intermediates:
            intermediates['fused_features'] = fused
            intermediates['output'] = out
            return out, intermediates

        return out

    def get_attention_maps(self,
                          flow: torch.Tensor,
                          corr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.

        Args:
            flow: Input flow tensor
            corr: Correlation tensor

        Returns:
            Dictionary containing attention maps for each scale
        """
        attention_maps = {}

        # Process features and get attention weights
        corr_feats = []
        for i in range(self.num_scales):
            corr_feat, _ = self._process_scale(flow, corr, i)
            corr_feats.append(F.interpolate(corr_feat, size=corr_feats[0].shape[-2:]
                                          if i > 0 else corr_feat.shape[-2:],
                                          mode='bilinear', align_corners=True))

        # Compute attention weights
        corr_cat = torch.cat(corr_feats, dim=1)
        attention_weights = self.attention(corr_cat)

        # Store attention maps for each scale
        for i in range(self.num_scales):
            attention_maps[f'scale_{i}'] = attention_weights[:, i:i+1]

        return attention_maps


class MotionFusion(BaseModule):
    """
    Module for fusing motion features from multiple sources.
    """
    def __init__(self,
                 input_dim: int = 160,
                 hidden_dim: int = 128,
                 output_dim: int = 160):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim * 2, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(hidden_dim)
        self.norm2 = nn.InstanceNorm2d(hidden_dim)

        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self.apply(self.initialize_weights)

    def forward(self,
                feat1: torch.Tensor,
                feat2: torch.Tensor) -> torch.Tensor:
        """
        Fuse two sets of motion features.

        Args:
            feat1: First feature tensor
            feat2: Second feature tensor

        Returns:
            Fused features
        """
        # Concatenate features
        x = torch.cat([feat1, feat2], dim=1)

        # Process through network
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.conv3(x))

        return x


class sobel_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_opx = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_opy = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32').reshape((1, 1, 3, 3))
        sobel_kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32').reshape((1, 1, 3, 3))
        self.conv_opx.weight.data = torch.from_numpy(sobel_kernelx)
        self.conv_opy.weight.data = torch.from_numpy(sobel_kernely)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, im):  # input rgb
        x = (0.299 * im[:, 0, :, :] + 0.587 * im[:, 1, :, :] + 0.114 * im[:, 2, :, :]).unsqueeze(1)  # rgb2gray
        gradx = self.conv_opx(x)
        grady = self.conv_opy(x)

        x = (gradx ** 2 + grady ** 2) ** 0.5
        x = (x - x.min()) / (x.max() - x.min())
        x = F.pad(x, (1, 1, 1, 1))

        x = torch.cat([im, x], dim=1)
        return x


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.edge = sobel_net()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        x = self.edge(x)
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
