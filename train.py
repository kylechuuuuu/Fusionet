import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fusionet2 import FusionModel, noise_consistency_loss
from dataset import VesselDataset
from test import test as run_inference
from calculate_metrics import calculate_metrics as run_eval
import os
import logging
from tqdm import tqdm
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) indices
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class HybridLoss(nn.Module):
    def __init__(self, num_classes, weight=None, dice_weight=0.5):
        super(HybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes)
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (1.0 - self.dice_weight) * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


def calculate_dice(pred, target, num_classes, smooth=1e-5):
    # pred: (B, C, H, W) logits
    # target: (B, H, W) indices
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1) # (B, H, W)
    
    dice_scores = []
    # Calculate Dice for each class, usually skipping background (class 0) if it dominates
    # But let's include all or skip 0 based on preference. 
    # Here we calculate mean Dice over all classes.
    for i in range(num_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())
        
    return dice_scores


def train():
    # Setup logging
    # Reset logging handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("train.log", mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Config
    batch_size = 1  # Small batch size due to large image size
    num_epochs = 350
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7 # Restored noise class, now 7 classes

    # Paths
    data_root = "DSCA_new"

    # Model
    logger.info("Initializing model...")
    model = FusionModel(num_classes=num_classes)
    model = model.to(device)

    # Data
    logger.info("Loading data...")
    train_dataset = VesselDataset(data_root, split="train", augment=True)
    val_dataset = VesselDataset(data_root, split="val", augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Optimizer & Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Use Hybrid Loss (CE + Dice)
    criterion = HybridLoss(num_classes)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Training Loop
    best_mean_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (images, masks_dict, _) in enumerate(pbar):
            images = images.to(device)
            mask_full = masks_dict['full'].to(device)

            optimizer.zero_grad()
            
            # Forward with mixed precision
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                output, ortho_loss = model(images)
                
                # Ensure outputs are finite before loss calculation
                if not torch.isfinite(output).all():
                    logger.warning("Non-finite output detected!")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

                # Loss calculation with orthogonality constraint
                seg_loss = criterion(output, mask_full)
                
                # Consistency loss for the entire vessel structure (all non-background classes)
                # This helps the model maintain better structural integrity
                noise_mask = (mask_full > 0).float()
                noise_loss = noise_consistency_loss(output, noise_mask)

                # Recommended weights:
                # ortho_weight (0.05 - 0.1): Higher forces diversity but might hurt individual branch quality
                # noise_weight (0.1 - 0.2): Higher helps with structural connectivity
                ortho_weight = 0.6
                noise_weight = 0.1
                
                loss = seg_loss + (ortho_weight * ortho_loss) + (noise_weight * noise_loss)
            
            # Backward and Step with scaler
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Check for NaN in loss to prevent train_loss from becoming NaN
            if not torch.isnan(loss):
                train_loss += loss.item()
            else:
                logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {i}")
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "seg": f"{seg_loss.item():.4f}", 
                "ortho": f"{ortho_loss.item():.4f}",
                "noise": f"{noise_loss.item():.4f}"
            })

        avg_train_loss = train_loss / len(train_loader)

        # External Evaluation (Inference + Metrics Calculation) every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}: Starting inference and metrics calculation...")
            run_inference(model)
            eval_results = run_eval(pred_dir='results/overall', gt_dir='DSCA_new/val/masks', label_path='DSCA_new/label.json')
            
            if eval_results:
                avg_class_dice = eval_results['dice_per_class']
                current_mean_dice = np.mean(avg_class_dice)
                mean_dice_fg = eval_results.get('mean_dice_fg', 0.0)
                
                logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}")
                logger.info(f"Validation Results - Mean Dice (All): {current_mean_dice:.4f}, Mean Dice (FG): {mean_dice_fg:.4f}")
                logger.info(f"Dice per class: {avg_class_dice}")

                # Save Best Model
                if current_mean_dice > best_mean_dice:
                    best_mean_dice = current_mean_dice
                    torch.save(model.state_dict(), "best_fusion_model.pth")
                    logger.info(f"👍Saved best model with Mean Dice: {best_mean_dice:.4f}")
        else:
            logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}")
        
        # Flush logs
        for handler in logger.handlers:
            handler.flush()
        
        scheduler.step()

if __name__ == "__main__":
    train()
