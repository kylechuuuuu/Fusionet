import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add sam2 to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(current_dir, 'sam2')
if sam2_path not in sys.path:
    sys.path.append(sam2_path)

try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("Warning: Could not import build_sam2 from sam2. Make sure sam2 is in the path.")

def orthogonality_loss(features_list, eps=1e-8):
    """
    Compute orthogonality loss between feature maps from different branches.
    """
    if len(features_list) < 2:
        return torch.tensor(0.0, device=features_list[0].device if features_list else torch.device('cpu'))

    total_loss = torch.tensor(0.0, device=features_list[0].device, dtype=torch.float32)
    num_pairs = 0

    # Ensure everything is in float32 for stability
    flattened_features = []
    for feat in features_list:
        # Detach or clone if necessary, but here we need gradients
        B, C, H, W = feat.shape
        # Move to float32 before normalization to avoid precision issues
        flattened = feat.float().reshape(B, -1)  
        # Normalize along the feature dimension with a slightly larger eps
        normalized = F.normalize(flattened, p=2, dim=1, eps=1e-10)
        flattened_features.append(normalized)

    # Calculate cosine similarity squared between each pair of feature maps
    # Squared similarity is more stable and has better gradients near zero
    for i in range(len(flattened_features)):
        for j in range(i + 1, len(flattened_features)):
            # Compute cosine similarity between features from branch i and branch j
            sim = torch.mean(torch.sum(flattened_features[i] * flattened_features[j], dim=1)**2)
            total_loss = total_loss + sim
            num_pairs += 1

    if num_pairs > 0:
        total_loss = total_loss / num_pairs

    return total_loss

def noise_consistency_loss(logits, noise_mask, eps=1e-8):
    """
    Compute consistency loss between the model's total vessel prediction 
    and the binary noise mask.
    
    Args:
        logits: Model output logits of shape (B, C, H, W)
        noise_mask: Binary mask of shape (B, H, W) or (B, 1, H, W)
    """
    if noise_mask.dim() == 4:
        noise_mask = noise_mask.squeeze(1)
        
    # Move to float32 for numerical stability in BCE during AMP training
    probs = torch.softmax(logits.float(), dim=1)
    # Sum probabilities of all vessel classes (1 to num_classes-1)
    vessel_prob = torch.sum(probs[:, 1:], dim=1)
    
    # Use manual BCE to bypass AMP blacklist and maintain numerical stability
    # F.binary_cross_entropy is blocked in autocast even if inputs are float32
    target = noise_mask.float()
    vessel_prob_clamped = vessel_prob.clamp(eps, 1-eps)
    bce = -(target * torch.log(vessel_prob_clamped) + (1 - target) * torch.log(1 - vessel_prob_clamped)).mean()
    
    # Dice Loss (also in float32)
    inter = torch.sum(vessel_prob * target)
    union = torch.sum(vessel_prob) + torch.sum(target)
    dice = 1 - (2. * inter + eps) / (union + eps)
    
    return bce + dice

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism to better integrate branch features
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention to help with vessel continuity
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        
        # Apply Channel Attention
        x = x * self.ca(x)
        
        # Apply Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_in = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sa(sa_in)
        
        return self.conv2(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetBranch(nn.Module):
    def __init__(self, n_channels=3, out_feat=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_feat, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        return self.out_conv(x)

class MLPAdapter(nn.Module):
    def __init__(self, original_mlp, dim, adapter_dim=64):
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, dim)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        # Hiera inputs to MLP are (B, H, W, C)
        return self.original_mlp(x) + self.adapter(x)

def apply_adapter(model, adapter_dim=64):
    # model is ImageEncoder, we access the trunk (Hiera)
    trunk = model.trunk
    for block in trunk.blocks:
        # MultiScaleBlock has .mlp which handles (B, H, W, C)
        block.mlp = MLPAdapter(block.mlp, block.dim_out, adapter_dim)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.ps(self.conv(x))))

class PixelShuffleDecoder(nn.Module):
    def __init__(self, in_dim=256, out_feat=32):
        super().__init__()
        # Assuming all input features from neck have in_dim (256)
        # We will handle a variable number of feature levels
        self.up_blocks = nn.ModuleList([
            PixelShuffleUpsample(in_dim, in_dim),
            PixelShuffleUpsample(in_dim, in_dim),
            PixelShuffleUpsample(in_dim, in_dim)
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_dim, out_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_feat),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_size):
        # features is a list of tensors from fine to coarse or coarse to fine
        # For SAM2 FpnNeck, it's usually [1/4, 1/8, 1/16] or [1/4, 1/8, 1/16, 1/32]
        # We process from coarse to fine.
        
        x = features[-1]
        for i in range(len(features) - 1, 0, -1):
            if i - 1 < len(self.up_blocks):
                x = self.up_blocks[len(features) - 1 - i](x)
                # Ensure spatial match if there are discrepancies
                if x.shape[2:] != features[i-1].shape[2:]:
                    x = F.interpolate(x, size=features[i-1].shape[2:], mode='bilinear', align_corners=False)
                x = x + features[i-1]
        
        x = self.conv_out(x)
        # Upsample to the final target size (e.g. original image size)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class TransformerDecoderBranch(nn.Module):
    def __init__(self, in_dim=256, out_feat=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Linear(in_dim * 4, in_dim)
        )
        self.norm2 = nn.LayerNorm(in_dim)
        
        # Reduce channels before large upsampling to save memory and avoid INT_MAX issues
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, out_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_feat),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, target_size):
        f3 = features[-1] # Shape (B, C, H, W)
        B, C, H, W = f3.shape
        x = f3.flatten(2).transpose(1, 2) # (B, L, C)
        
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Use rhn (Residual Homogenized Network) style residual
        x = x + self.mlp(self.norm2(x))
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.reduce(x)
        
        # Now upsample the reduced feature map to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Branch 1: UNet
        self.branch1 = UNetBranch(3, out_feat=32)

        # Config for SAM2
        sam2_ckpt = "sam2/sam2_pth/sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l"

        # Branch 2: SAM2 (Frozen) + Transformer Decoder
        # apply_postprocessing=False to get raw features
        sam2_ckpt = "sam2/sam2_pth/sam2.1_hiera_large.pt"
        sam2_model_frozen = build_sam2(sam2_cfg, sam2_ckpt, device="cpu", apply_postprocessing=False)
        self.encoder_frozen = sam2_model_frozen.image_encoder
        for param in self.encoder_frozen.parameters():
            param.requires_grad = False

        self.decoder2 = TransformerDecoderBranch(in_dim=256, out_feat=32)

        # Branch 3: SAM2 (Fine-tuned with Adapter) + PixelShuffle Decoder
        sam2_model_tuned = build_sam2(sam2_cfg, sam2_ckpt, device="cpu", apply_postprocessing=False)
        self.encoder_tuned = sam2_model_tuned.image_encoder
        # Freeze base trunk weights
        for param in self.encoder_tuned.parameters():
            param.requires_grad = False

        # Add Adapters to the trunk
        apply_adapter(self.encoder_tuned, adapter_dim=64)

        # Unfreeze adapters
        for name, param in self.encoder_tuned.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True

        # PixelShuffle Decoder
        self.decoder3 = PixelShuffleDecoder(in_dim=256, out_feat=32)

        # Fusion Output
        # self.fusion_head = FeatureFusionBlock(32 + 32 + 32, 64, num_classes)
        self.parallel_heads = nn.ModuleList([
            FeatureFusionBlock(32 + 32 + 32, 64, 1)
            for _ in range(num_classes)
        ])

    def forward(self, x, return_branch_features=False):
        target_size = x.shape[2:]
        # UNet Branch
        feat1 = self.branch1(x)

        # SAM2 Frozen Branch
        with torch.no_grad():
            enc2 = self.encoder_frozen(x)
            feat2_fpn = enc2["backbone_fpn"]
        feat2 = self.decoder2(feat2_fpn, target_size)

        # SAM2 Tuned Branch
        enc3 = self.encoder_tuned(x)
        feat3_fpn = enc3["backbone_fpn"]
        feat3 = self.decoder3(feat3_fpn, target_size)

        # Compute orthogonality loss between branch features
        ortho_loss = orthogonality_loss([feat1, feat2, feat3])

        # Fusion
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        
        outputs = []
        for head in self.parallel_heads:
            output = head(combined)
            outputs.append(output)

        # Calculate orthogonality loss between parallel heads to ensure disjoint predictions
        # Use sigmoid to convert logits to probabilities, so that we penalize overlapping activations
        # instead of penalizing anti-correlated logits.
        head_probs = [torch.sigmoid(out) for out in outputs]
        head_ortho_loss = orthogonality_loss(head_probs)
        
        # Combine branch orthogonality and head orthogonality
        total_ortho_loss = ortho_loss + head_ortho_loss

        # Concatenate all outputs along the channel dimension
        out = torch.cat(outputs, dim=1)

        if return_branch_features:
            return out, total_ortho_loss, {'branch1': feat1, 'branch2': feat2, 'branch3': feat3}
        else:
            return out, total_ortho_loss
