import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class TESS_Dataset(Dataset):
    # 1. THE SETUP
    def __init__(self, csv_file, features_dir, mode='multimodal'):
        self.data_frame = pd.read_csv(csv_file)
        self.features_dir = features_dir
        self.mode = mode 

    # 2. THE COUNT
    def __len__(self):
        return len(self.data_frame)

    # 3. THE FETCH
    def __getitem__(self, index):
        # Find the specific row in the CSV
        row = self.data_frame.iloc[index]
        base_name = os.path.basename(row['file_path']).replace('.wav', '')
        label = row['label_id']
        
        # Grab only what is requested based on the mode
        if self.mode == 'audio':
            audio_path = os.path.join(self.features_dir, f"{base_name}_audio.pt")
            return torch.load(audio_path, weights_only=True), label
            
        elif self.mode == 'text':
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            return torch.load(text_path, weights_only=False), label
            
        elif self.mode == 'multimodal':
            audio_path = os.path.join(self.features_dir, f"{base_name}_audio.pt")
            text_path = os.path.join(self.features_dir, f"{base_name}_text.pt")
            
            audio_features = torch.load(audio_path, weights_only=True)
            text_features = torch.load(text_path, weights_only=False)
            
            return audio_features, text_features, label