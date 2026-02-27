import os
import json
import argparse
import numpy as np
from PIL import Image
from glob import glob

def calculate_metrics(pred_dir='results/overall', gt_dir='DSCA_new/val/masks', label_path='DSCA_new/label.json'):
    # Configuration
    LABEL_PATH = label_path
    PRED_DIR = pred_dir
    GT_DIR = gt_dir
    
    # Load Label Config
    if not os.path.exists(LABEL_PATH):
        # Fallback to root label.json if not in DSCA_new
        LABEL_PATH = 'label.json'
        
    if not os.path.exists(LABEL_PATH):
        # Error out if still not found
        print(f"Error: label.json not found at {label_path}")
        return None

    with open(LABEL_PATH, 'r') as f:
        label_config = json.load(f)
        
    # Map Colors to IDs
    color_to_id = {}
    id_to_name = {}
    
    print(f"{'ID':<5} {'Name':<30} {'Color':<15}")
    print('-' * 50)
    for key, val in label_config.items():
        hex_color = val['color']
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        color_to_id[(r, g, b)] = val['id']
        id_to_name[val['id']] = val['name']
        print(f"{val['id']:<5} {val['name']:<30} {hex_color} {(r,g,b)}")
    
    num_classes = max(id_to_name.keys()) + 1
    
    # Initialize accumulators
    total_tp = np.zeros(num_classes, dtype=np.float64)
    total_fp = np.zeros(num_classes, dtype=np.float64)
    total_fn = np.zeros(num_classes, dtype=np.float64)
    
    # Get file lists
    pred_files = sorted(glob(os.path.join(PRED_DIR, '*.png')))
    
    print(f"\nFound {len(pred_files)} prediction files in {PRED_DIR}.")
    
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(GT_DIR, filename)
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {filename}, skipping.")
            continue
            
        # Load images
        pred_img = np.array(Image.open(pred_path).convert('RGB'))
        gt_img = np.array(Image.open(gt_path).convert('RGB'))
        
        if pred_img.shape != gt_img.shape:
            print(f"Warning: Shape mismatch for {filename} {pred_img.shape} vs {gt_img.shape}. Resizing Pred to GT.")
            pred_img_pil = Image.fromarray(pred_img)
            pred_img = np.array(pred_img_pil.resize((gt_img.shape[1], gt_img.shape[0]), Image.NEAREST))

        # Convert to ID masks
        pred_mask = np.zeros(pred_img.shape[:2], dtype=np.uint8)
        gt_mask = np.zeros(gt_img.shape[:2], dtype=np.uint8)
        
        # Mapping loop
        for color, cid in color_to_id.items():
            color_arr = np.array(color)
            # Check matches
            # pred
            # Using absolute difference is safer for potential compression artifacts, though PNG should be lossless.
            # But here we assume exact match.
            matches_pred = np.all(pred_img == color_arr, axis=-1)
            pred_mask[matches_pred] = cid
            
            # gt
            matches_gt = np.all(gt_img == color_arr, axis=-1)
            gt_mask[matches_gt] = cid
            
        # Calculate Confusion components
        for c in range(num_classes):
            p = (pred_mask == c)
            g = (gt_mask == c)
            
            tp = np.logical_and(p, g).sum()
            fp = np.logical_and(p, ~g).sum()
            fn = np.logical_and(~p, g).sum()
            
            total_tp[c] += tp
            total_fp[c] += fp
            total_fn[c] += fn
            
    # Compute Metrics
    print("\n" + "="*80)
    print(f"{'Class':<30} {'Dice':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    for c in range(num_classes):
        tp = total_tp[c]
        fp = total_fp[c]
        fn = total_fn[c]
        
        denom_dice = 2*tp + fp + fn
        dice = (2*tp) / denom_dice if denom_dice > 0 else 0
        
        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0
        
        denom_prec = tp + fp
        precision = tp / denom_prec if denom_prec > 0 else 0
        
        denom_rec = tp + fn
        recall = tp / denom_rec if denom_rec > 0 else 0
        
        name = id_to_name.get(c, f"Class {c}")
        
        print(f"{name:<30} {dice:.4f}     {iou:.4f}     {precision:.4f}        {recall:.4f}")
        
        all_metrics['dice'].append(dice)
        all_metrics['iou'].append(iou)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)

        if c != 0: # Collect for Mean (excluding background)
            metrics['dice'].append(dice)
            metrics['iou'].append(iou)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            
    print("-" * 80)
    print(f"{'Mean (Foreground)':<30} {np.mean(metrics['dice']):.4f}     {np.mean(metrics['iou']):.4f}     {np.mean(metrics['precision']):.4f}        {np.mean(metrics['recall']):.4f}")
    print("="*80)

    return {
        'dice_per_class': all_metrics['dice'],
        'iou_per_class': all_metrics['iou'],
        'mean_dice': np.mean(all_metrics['dice']),
        'mean_dice_fg': np.mean(metrics['dice'])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate segmentation metrics.")
    parser.add_argument('--pred_dir', type=str, default='results/overall', help='Directory containing prediction masks')
    parser.add_argument('--gt_dir', type=str, default='DSCA_new/val/masks', help='Directory containing ground truth masks')
    parser.add_argument('--label_path', type=str, default='DSCA_new/label.json', help='Path to label.json')
    args = parser.parse_args()
    calculate_metrics(args.pred_dir, args.gt_dir, args.label_path)
