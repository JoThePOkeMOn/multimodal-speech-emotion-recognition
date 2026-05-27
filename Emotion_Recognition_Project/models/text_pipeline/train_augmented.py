"""
Enhanced Text-Only Pipeline Training with Augmented Features

This script trains the TextClassifier using augmented text variations
to improve robustness and accuracy.
"""

import sys
import os
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from TESS_Dataset.dataset_augmented import TESS_Dataset_Augmented
from TESS_Dataset.model import TextClassifier


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

print("=" * 60)
print("TEXT-ONLY PIPELINE WITH AUGMENTED FEATURES")
print("=" * 60)

# ============================================
# Configuration
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATIENCE = 10

PROJECT_DIR = '/content/drive/MyDrive/Emotion_Recognition_Project'
CSV_PATH = os.path.join(PROJECT_DIR, 'tess_metadata.csv')
FEATURES_DIR = os.path.join(PROJECT_DIR,'TESS_Dataset', 'Features')
AUGMENTED_DIR = os.path.join(PROJECT_DIR,'TESS_Dataset', 'Features_Augmented')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'Checkpoints')
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'best_text_model_augmented.pth')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nPaths:")
print(f"  CSV: {CSV_PATH}")
print(f"  Features: {FEATURES_DIR}")
print(f"  Augmented: {AUGMENTED_DIR}")
print(f"  Checkpoints: {CHECKPOINT_DIR}")

# ============================================
# Load Dataset with Augmentation
# ============================================
print("\n[1/4] Loading dataset with augmented text features...")

full_dataset = TESS_Dataset_Augmented(
    csv_file=CSV_PATH,
    features_dir=FEATURES_DIR,
    augmented_dir=AUGMENTED_DIR,
    mode='text_augmented',
    augmentation='random'  # Random augmentation per epoch
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")

# ============================================
# Initialize Model
# ============================================
print("\n[2/4] Initializing TextClassifier...")

model = TextClassifier(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# Training Loop
# ============================================
print("\n[3/4] Training...")
print("-" * 60)

best_val_loss = float('inf')
patience_counter = 0
epoch_history = []

for epoch in range(1, NUM_EPOCHS + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (text, labels) in enumerate(train_loader):
        input_ids = text['input_ids'].squeeze(1).to(device)
        mask = text['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    train_loss /= len(train_loader)
    train_acc = (train_correct / train_total) * 100
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for text, labels in test_loader:
            input_ids = text['input_ids'].squeeze(1).to(device)
            mask = text['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= len(test_loader)
    val_acc = (val_correct / val_total) * 100
    
    scheduler.step(val_loss)
    
    epoch_history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })
    
    # Logging
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:6.2f}% | "
              f"VaLoss: {val_loss:.4f} | VaAcc: {val_acc:6.2f}%")
    
    # Checkpoint: save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print("-" * 60)

# ============================================
# Final Evaluation on Test Set
# ============================================
print("\n[4/4] Evaluating on test set...")

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for text, labels in test_loader:
        input_ids = text['input_ids'].squeeze(1).to(device)
        mask = text['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids, mask)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = (correct / total) * 100

print(f"\n{'=' * 60}")
print(f"FINAL TEXT-ONLY TEST ACCURACY (with augmentation): {final_accuracy:.2f}%")
print(f"Best model saved to: {MODEL_SAVE_PATH}")
print(f"{'=' * 60}")
