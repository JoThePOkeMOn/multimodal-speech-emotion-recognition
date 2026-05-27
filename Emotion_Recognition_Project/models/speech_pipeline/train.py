import sys
import os
import random
import numpy as np

# ---------------------------------------------------------
# THE DIRECTORY PATH FIX 
# Forces Python to look in the main project folder for imports
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from TESS_Dataset.dataset import TESS_Dataset
from TESS_Dataset.model import AudioClassifier

# ---------------------------------------------------------
# THE ENTERPRISE SEED LOCK
# ---------------------------------------------------------
def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ---------------------------------------------------------

print("--- Starting the AUDIO-ONLY Training Pipeline ---")

# 1. Setup & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5

PROJECT_DIR = '/content/drive/MyDrive/Emotion_Recognition_Project'
CSV_PATH = os.path.join(PROJECT_DIR, 'tess_metadata.csv')
FEATURES_DIR = os.path.join(PROJECT_DIR, 'TESS_Dataset', 'Features')
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'best_audio_model.pth')

# 2. Data Loading & 80/20 Split
full_dataset = TESS_Dataset(csv_file=CSV_PATH, features_dir=FEATURES_DIR, mode='audio')
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Initialization
model = AudioClassifier(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. Training & Validation Loop
for epoch in range(EPOCHS):
    # --- PHASE 1: TRAINING ---
    model.train()
    running_train_loss = 0.0
    for audio, labels in train_loader:
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
    avg_train_loss = running_train_loss / len(train_loader)
    
    # --- PHASE 2: VALIDATION ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = (correct / total) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

# 5. Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Training Complete! Saved best_audio_model.pth")