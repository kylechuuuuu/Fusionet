import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random

class VesselDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=1024, augment=False):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Define colors for multi-class segmentation based on label.json
        # ID 1: noise (blue), ID 2: carotid (red), ID 3: vertebral (brown)
        # ID 4: anterior (yellow), ID 5: middle (purple), ID 6: posterior (light_yellow)
        self.color_to_id = {
            (0, 162, 232): 1,    # noise (blue)
            (134, 0, 21): 2,    # carotid_artery (red)
            (185, 122, 87): 3,  # vertebral_artery (brown)
            (255, 242, 0): 4,   # anterior_cerebral_artery (yellow)
            (200, 191, 231): 5, # middle_cerebral_artery (purple)
            (239, 228, 176): 6  # posterior_cerebral_artery (light_yellow)
        }
        
        # For visualization in test.py
        self.colors = [
            (0, 0, 0),         # 0: background
            (0, 162, 232),     # 1: noise
            (134, 0, 21),      # 2: carotid
            (185, 122, 87),    # 3: vertebral
            (255, 242, 0),     # 4: anterior
            (200, 191, 231),   # 5: middle
            (239, 228, 176)    # 6: posterior
        ]

        # Default normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def encode_mask(self, mask):
        # mask: PIL Image RGB
        mask = np.array(mask)
        mask_out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        
        for color, idx in self.color_to_id.items():
            # Check for exact match
            match = np.all(mask == color, axis=-1)
            mask_out[match] = idx
            
        return torch.from_numpy(mask_out).long()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        if not os.path.exists(mask_path):
            base, _ = os.path.splitext(img_name)
            for ext in ['.png', '.jpg', '.jpeg']:
                temp_path = os.path.join(self.masks_dir, base + ext)
                if os.path.exists(temp_path):
                    mask_path = temp_path
                    break
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # Resize both to img_size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        if self.augment and self.split == 'train':
            # Spatial augmentation (Synchronized)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            
        image = T.ToTensor()(image)
        image = self.normalize(image)
        mask_raw = self.encode_mask(mask)
        
        # Branch-specific GTs as requested
        # 1. Main artery GT: Background(0), ID 2 -> 1, ID 3 -> 2. Others -> 0
        mask_main = torch.zeros_like(mask_raw)
        mask_main[mask_raw == 2] = 1
        mask_main[mask_raw == 3] = 2
        
        # 2. Cerebral artery GT: Background(0), ID 4 -> 1, ID 5 -> 2, ID 6 -> 3. Others -> 0
        mask_cerebral = torch.zeros_like(mask_raw)
        mask_cerebral[mask_raw == 4] = 1
        mask_cerebral[mask_raw == 5] = 2
        mask_cerebral[mask_raw == 6] = 3
        
        masks = {
            'full': mask_raw,
            'main': mask_main,
            'cerebral': mask_cerebral
        }
        
        return image, masks, img_name
