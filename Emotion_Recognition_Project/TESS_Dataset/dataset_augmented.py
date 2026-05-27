"""
Enhanced TESS Dataset with Text Augmentation Support

This dataset supports three modes:
1. 'default': Uses original simple text ("Say the word X")
2. 'augmented': Uses randomly sampled augmented text variations
3. 'augmented_all': Returns all augmented variations (for ensemble methods)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import random


class TESS_Dataset_Augmented(Dataset):
    """
    TESS Dataset with optional text augmentation support.
    
    Args:
        csv_file: Path to metadata CSV
        features_dir: Original features directory
        augmented_dir: Augmented features directory (optional)
        mode: 'audio', 'text', 'multimodal', or 'multimodal_augmented'
        augmentation: None, 'random', or 'all'
            - None: Use original text features
            - 'random': Randomly sample one augmentation per epoch
            - 'all': Return list of all augmentations
    """
    
    def __init__(self, csv_file, features_dir, augmented_dir=None, 
                 mode='multimodal', augmentation=None):
        self.data_frame = pd.read_csv(csv_file)
        self.features_dir = features_dir
        self.augmented_dir = augmented_dir
        self.mode = mode
        self.augmentation = augmentation
        
        # Validate augmented directory exists if augmentation is requested
        if augmentation and augmented_dir and not os.path.exists(augmented_dir):
            print(f"Warning: Augmented directory not found. Falling back to original features.")
            self.augmentation = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        base_name = os.path.basename(row['file_path']).replace('.wav', '')
        label = row['label_id']
        
        # ============================================
        # AUDIO ONLY
        # ============================================
        if self.mode == 'audio':
            audio_path = os.path.join(self.features_dir, f"{base_name}_audio.pt")
            return torch.load(audio_path, weights_only=True), label
        
        # ============================================
        # TEXT ONLY
        # ============================================
        elif self.mode == 'text':
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            return torch.load(text_path, weights_only=False), label
        
        elif self.mode == 'text_augmented':
            # Use augmented text if available
            if self.augmentation and self.augmented_dir:
                aug_path = os.path.join(self.augmented_dir, f"{base_name}_text_augmented.pt")
                if os.path.exists(aug_path):
                    augmented_encodings = torch.load(aug_path, weights_only=False)
                    
                    if self.augmentation == 'random':
                        # Return one random augmentation
                        selected = random.choice(augmented_encodings)
                        return selected, label
                    
                    elif self.augmentation == 'all':
                        # Return all augmentations (for ensemble/voting)
                        return augmented_encodings, label
            
            # Fallback to original if augmented not available
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            return torch.load(text_path, weights_only=False), label
        
        # ============================================
        # MULTIMODAL
        # ============================================
        elif self.mode == 'multimodal':
            audio_path = os.path.join(self.features_dir, f"{base_name}_audio.pt")
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            
            audio_features = torch.load(audio_path, weights_only=True)
            text_features = torch.load(text_path, weights_only=False)
            
            return audio_features, text_features, label
        
        elif self.mode == 'multimodal_augmented':
            audio_path = os.path.join(self.features_dir, f"{base_name}_audio.pt")
            audio_features = torch.load(audio_path, weights_only=True)
            
            # Use augmented text if available
            if self.augmentation and self.augmented_dir:
                aug_path = os.path.join(self.augmented_dir, f"{base_name}_text_augmented.pt")
                if os.path.exists(aug_path):
                    augmented_encodings = torch.load(aug_path, weights_only=False)
                    
                    if self.augmentation == 'random':
                        text_features = random.choice(augmented_encodings)
                        return audio_features, text_features, label
                    
                    elif self.augmentation == 'all':
                        return audio_features, augmented_encodings, label
            
            # Fallback to original if augmented not available
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            text_features = torch.load(text_path, weights_only=False)
            
            return audio_features, text_features, label


class AugmentationCollator:
    """
    Custom collate function to handle augmented features when augmentation='all'.
    This converts a batch of lists into properly batched tensors for all variations.
    """
    
    def __init__(self, num_augmentations=8):
        """
        Args:
            num_augmentations: Expected number of augmentations per sample
        """
        self.num_augmentations = num_augmentations
    
    def __call__(self, batch):
        """
        Collate batch with augmented features.
        
        For augmentation='all':
            Stacks all augmentations across the batch
        For standard batches:
            Uses default PyTorch stacking
        """
        # Check if batch contains augmented features (lists of encodings)
        first_item = batch[0]
        
        if isinstance(first_item[1], list):  # Text augmentations
            # Handle augmented text case
            audios = [item[0] for item in batch]
            text_augs = [item[1] for item in batch]
            labels = torch.tensor([item[2] for item in batch])
            
            audio_batch = torch.stack(audios)
            
            # Stack augmentations: (batch_size, num_augmentations, ...)
            text_batch = []
            for aug_list in text_augs:
                aug_input_ids = torch.cat([enc['input_ids'] for enc in aug_list], dim=0)
                aug_attention_mask = torch.cat([enc['attention_mask'] for enc in aug_list], dim=0)
                text_batch.append((aug_input_ids, aug_attention_mask))
            
            return audio_batch, text_batch, labels
        
        else:
            # Default collate
            return torch.stack([item[0] for item in batch]), \
                   torch.stack([item[1] for item in batch]) if isinstance(first_item[1], torch.Tensor) else [item[1] for item in batch], \
                   torch.tensor([item[2] for item in batch]) if len(batch[0]) == 3 else torch.tensor([item[1] for item in batch])
