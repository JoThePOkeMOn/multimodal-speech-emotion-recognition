import sys
import os
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader, random_split
from TESS_Dataset.dataset import TESS_Dataset
from TESS_Dataset.model import AudioClassifier
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

print("--- Running Final Evaluation: AUDIO-ONLY Pipeline ---")

# 1. Setup & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

PROJECT_DIR = '/content/drive/MyDrive/Emotion_Recognition_Project'
CSV_PATH = os.path.join(PROJECT_DIR, 'tess_metadata.csv')
FEATURES_DIR = os.path.join(PROJECT_DIR, 'TESS_Dataset', 'Features')
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_DIR, 'best_audio_model.pth')

# 2. Recreate the Exact 80/20 Split
full_dataset = TESS_Dataset(csv_file=CSV_PATH, features_dir=FEATURES_DIR, mode='audio')
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

_, test_dataset = random_split(full_dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Load the Trained Model
model = AudioClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval() 

# 4. The Final Testing Loop
correct = 0
total = 0

all_true_labels = []
all_predictions = []
all_audio_vectors = []

print("Commencing final evaluation on unseen acoustic data...")

with torch.no_grad():
    for audio, labels in test_loader:
        audio, labels = audio.to(device), labels.to(device)

        audio_vectors = model.encoder(audio)
        outputs = model.classifier(audio_vectors)
        _, predicted = torch.max(outputs, 1)

        all_true_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predicted.detach().cpu().numpy())
        all_audio_vectors.append(audio_vectors.detach().cpu().numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = (correct / total) * 100
print(f"FINAL AUDIO-ONLY TEST ACCURACY: {final_accuracy:.2f}%")

target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
plots_dir = os.path.join(PROJECT_DIR, 'Results', 'plots')
os.makedirs(plots_dir, exist_ok=True)

y_true = np.concatenate(all_true_labels, axis=0)
y_pred = np.concatenate(all_predictions, axis=0)
audio_vectors = np.concatenate(all_audio_vectors, axis=0)

report = classification_report(y_true, y_pred, target_names=target_names)
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Audio-Only Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'audio_cm.png'))
plt.close()

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(audio_vectors)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap='tab10', s=12, alpha=0.8)
colorbar = plt.colorbar(scatter, ticks=range(len(target_names)))
colorbar.ax.set_yticklabels(target_names)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Audio Encoder Feature Space (t-SNE)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'audio_tsne.png'))
plt.close()
