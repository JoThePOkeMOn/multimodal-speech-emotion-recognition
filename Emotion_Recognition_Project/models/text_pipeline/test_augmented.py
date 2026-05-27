"""
Test Text Model Trained with Augmented Features

This script evaluates the TextClassifier model that was trained using
augmented text data. It loads augmented features (same as training).
"""

import sys
import os
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader, random_split
from TESS_Dataset.dataset_augmented import TESS_Dataset_Augmented
from TESS_Dataset.model import TextClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE


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
print("TESTING TEXT MODEL (Trained with Augmented Features)")
print("=" * 60)

# 1. Setup & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}\n")

BATCH_SIZE = 16

PROJECT_DIR = '/content/drive/MyDrive/Emotion_Recognition_Project'
CSV_PATH = os.path.join(PROJECT_DIR, 'tess_metadata.csv')
FEATURES_DIR = os.path.join(PROJECT_DIR,'TESS_Dataset', 'Features')
AUGMENTED_DIR = os.path.join(PROJECT_DIR,'TESS_Dataset', 'Features_Augmented')
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_DIR, 'best_text_model_augmented.pth')

print(f"Paths:")
print(f"  CSV: {CSV_PATH}")
print(f"  Features: {FEATURES_DIR}")
print(f"  Augmented: {AUGMENTED_DIR}")
print(f"  Model: {MODEL_WEIGHTS_PATH}\n")

# 2. Load Dataset with Augmented Features (Same as Training)
print("[1/3] Loading augmented test dataset...")

full_dataset = TESS_Dataset_Augmented(
    csv_file=CSV_PATH,
    features_dir=FEATURES_DIR,
    augmented_dir=AUGMENTED_DIR,
    mode='text_augmented',
    augmentation='random'  # Random variation for evaluation
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

_, test_dataset = random_split(full_dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Test samples: {len(test_dataset)}")
print(f"  Test batches: {len(test_loader)}\n")

# 3. Load the Trained Model
print("[2/3] Loading trained model...")

model = TextClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()

print(f"  Model loaded: {MODEL_WEIGHTS_PATH}\n")

# 4. The Testing Loop with Data Collection
print("[3/3] Evaluating...")

correct = 0
total = 0

all_true_labels = []
all_predictions = []
all_text_vectors = []

with torch.no_grad():
    for text, labels in test_loader:
        input_ids = text['input_ids'].squeeze(1).to(device)
        mask = text['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)
        
        # Extract text encoder output (before classification head)
        text_vectors = model.encoder(input_ids, mask)
        outputs = model.classifier(text_vectors)
        _, predicted = torch.max(outputs, 1)
        
        all_true_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predicted.detach().cpu().numpy())
        all_text_vectors.append(text_vectors.detach().cpu().numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = (correct / total) * 100
print(f"\nFINAL TEXT-ONLY TEST ACCURACY (Augmented): {final_accuracy:.2f}%\n")

# 5. Generate Metrics & Plots
target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
plots_dir = os.path.join(PROJECT_DIR, 'Results', 'plots')
os.makedirs(plots_dir, exist_ok=True)

y_true = np.concatenate(all_true_labels, axis=0)
y_pred = np.concatenate(all_predictions, axis=0)
text_vectors = np.concatenate(all_text_vectors, axis=0)

# Classification Report
report = classification_report(y_true, y_pred, target_names=target_names)
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Text-Only Confusion Matrix (Augmented Training)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'text_cm_augmented.png'))
plt.close()
print(f"✓ Confusion matrix saved: text_cm_augmented.png")

# t-SNE Plot
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(text_vectors)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap='tab10', s=12, alpha=0.8)
colorbar = plt.colorbar(scatter, ticks=range(len(target_names)))
colorbar.ax.set_yticklabels(target_names)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Text Encoder Feature Space (t-SNE) - Augmented Training')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'text_tsne_augmented.png'))
plt.close()
print(f"✓ t-SNE plot saved: text_tsne_augmented.png")

print(f"\n{'=' * 60}")
print(f"Plots saved to: {plots_dir}")
print(f"{'=' * 60}")
