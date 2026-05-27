"""
Test Multimodal Fusion Model Trained with Augmented Text Features

This script evaluates the MultimodalFusion model trained using augmented text.
It loads both audio and augmented text features (same as training).
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
from TESS_Dataset.model import MultimodalFusion
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
print("TESTING MULTIMODAL FUSION (Trained with Augmented Text)")
print("=" * 60)

# 1. Setup & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}\n")

BATCH_SIZE = 16

PROJECT_DIR = '/content/drive/MyDrive/Emotion_Recognition_Project'
CSV_PATH = os.path.join(PROJECT_DIR, 'tess_metadata.csv')
FEATURES_DIR = '/tmp/Features_Local'
AUGMENTED_DIR = '/tmp/Features_Augmented_Local'
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_DIR, 'best_multimodal_model_augmented.pth')

print(f"Paths:")
print(f"  CSV: {CSV_PATH}")
print(f"  Features: {FEATURES_DIR}")
print(f"  Augmented: {AUGMENTED_DIR}")
print(f"  Model: {MODEL_WEIGHTS_PATH}\n")

# 2. Load Dataset with Augmented Features (Same as Training)
print("[1/3] Loading augmented multimodal dataset...")

full_dataset = TESS_Dataset_Augmented(
    csv_file=CSV_PATH,
    features_dir=FEATURES_DIR,
    augmented_dir=AUGMENTED_DIR,
    mode='multimodal_augmented',  # ← Audio + augmented text
    augmentation='random'           # ← Random variation per epoch
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

_, test_dataset = random_split(full_dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Test samples: {len(test_dataset)}")
print(f"  Test batches: {len(test_loader)}\n")

# 3. Load the Trained Model
print("[2/3] Loading trained model...")

model = MultimodalFusion(num_classes=7).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=False))
model.eval()

print(f"  Model loaded: {MODEL_WEIGHTS_PATH}\n")

# 4. The Testing Loop with Data Collection
print("[3/3] Evaluating...")

correct = 0
total = 0

all_true_labels = []
all_predictions = []
all_fused_vectors = []

with torch.no_grad():
    for audio, text, labels in test_loader:
        audio = audio.to(device)
        input_ids = text['input_ids'].squeeze(1).to(device)
        mask = text['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)
        
        # Extract intermediate features from encoders
        audio_vector = model.audio_branch(audio)
        text_vector = model.text_branch(input_ids, mask)
        
        # Project text to match audio dimensions
        text_vector_projected = model.text_projection(text_vector)
        
        # Prepare for Cross-Attention
        audio_query = audio_vector.unsqueeze(1)
        text_key_value = text_vector_projected.unsqueeze(1)
        
        # Cross-Attention
        attn_output, _ = model.attention(query=audio_query, key=text_key_value, value=text_key_value)
        attn_output = attn_output.squeeze(1)
        
        # Fused vector before classification
        fused_vector = torch.cat((attn_output, text_vector_projected), dim=1)
        
        # Classification
        outputs = model.classifier(fused_vector)
        _, predicted = torch.max(outputs, 1)
        
        all_true_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predicted.detach().cpu().numpy())
        all_fused_vectors.append(fused_vector.detach().cpu().numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = (correct / total) * 100
print(f"\nFINAL MULTIMODAL TEST ACCURACY (Augmented): {final_accuracy:.2f}%\n")

# 5. Generate Metrics & Plots
target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
plots_dir = os.path.join(PROJECT_DIR, 'Results', 'plots')
os.makedirs(plots_dir, exist_ok=True)

y_true = np.concatenate(all_true_labels, axis=0)
y_pred = np.concatenate(all_predictions, axis=0)
fused_vectors = np.concatenate(all_fused_vectors, axis=0)

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
plt.title('Multimodal Fusion Confusion Matrix (Augmented Training)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fusion_cm_augmented.png'))
plt.close()
print(f"✓ Confusion matrix saved: fusion_cm_augmented.png")

# t-SNE Plot
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(fused_vectors)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap='tab10', s=12, alpha=0.8)
colorbar = plt.colorbar(scatter, ticks=range(len(target_names)))
colorbar.ax.set_yticklabels(target_names)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Multimodal Fused Feature Space (t-SNE) - Augmented Training')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fusion_tsne_augmented.png'))
plt.close()
print(f"✓ t-SNE plot saved: fusion_tsne_augmented.png")

print(f"\n{'=' * 60}")
print(f"Plots saved to: {plots_dir}")
print(f"{'=' * 60}")
