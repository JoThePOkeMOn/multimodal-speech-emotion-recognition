# 🎭 Multimodal Emotion Recognition System

A comprehensive emotion recognition system using the **Toronto Emotional Speech Set (TESS)**. This project implements three pipelines (Audio-only, Text-only, and Multimodal Fusion) to detect emotions from speech, utilizing an advanced **Cross-Attention gating mechanism** to resolve Modality Interference.

---

## 🏗️ Project Structure

```
├── data_preprocessing.ipynb        # Feature extraction notebook
└── Emotion_Recognition_Project/
    ├── TESS_Dataset/               # Model definitions & DataLoaders
    ├── models/                     # Pipeline code (speech, text, fusion)
    │   ├── speech_pipeline/        # Audio-only pipeline
    │   ├── text_pipeline/          # Text-only pipeline
    │   └── fusion_pipeline/        # Multimodal fusion pipeline
    ├── Results/                    # Confusion matrices & t-SNE plots
    ├── text_augmentation.py        # Augmentation generation script
    ├── requirements.txt            # Project dependencies
    └── README.md                   # This file
```

---

## 🚀 Environment Setup

Choose the method that matches your current workflow.

### Method A: Google Colab (Recommended)

This method utilizes your Google Drive for persistent storage.

#### Step 1: Setup Environment

1. Open your Colab notebook
2. Click the 🔑 **Secrets** icon on the left sidebar
3. Add your `KAGGLE_USERNAME` and `KAGGLE_KEY`

#### Step 2: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Emotion_Recognition_Project
```

#### Step 3: Run Preprocessing

1. Open `data_preprocessing.ipynb`
2. Run all cells sequentially
3. This will automatically:
   - Download the TESS dataset from Kaggle
   - Generate the metadata
   - Save extracted features to your `Features/` folder

#### Step 4: Run Pipeline Scripts

Execute training/testing scripts using the `!python` command:

```python
!python models/speech_pipeline/train.py
```

---

### Method B: Local Machine

#### Step 1: Clone Repository

```bash
git clone https://github.com/JoThePOkeMOn/multimodal-speech-emotion-recognition.git
cd multimodal-speech-emotion-recognition
```

#### Step 2: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Run Preprocessing

Ensure the raw TESS dataset is in `TESS_Dataset/tess_raw_audio/`, then run:

```bash
jupyter notebook data_preprocessing.ipynb
```

Run all cells sequentially to extract features.

#### Step 4: Run Pipeline Scripts

Simply run the scripts directly:

```bash
python models/speech_pipeline/train.py
```

---

## ▶️ Execution Order

To replicate the project results, follow this sequence:

| Step | Task | Script/Notebook |
|------|------|-----------------|
| 1 | Feature Extraction | `data_preprocessing.ipynb` |
| 2 | Data Augmentation | `text_augmentation.py` |
| 3 | Train Audio-Only | `models/speech_pipeline/train.py` |
| 4 | Train Text-Only | `models/text_pipeline/train_augmented.py` |
| 5 | Train Multimodal | `models/fusion_pipeline/train_augmented.py` |
| 6 | Test All Models | `models/*/test_augmented.py` |

---

## 📊 Results Summary

| Pipeline | Accuracy | Key Insight |
|----------|----------|-------------|
| **Audio-Only** | 99.64% | Excellent baseline; prosody is sufficient |
| **Text-Only** | 78.12% | Achieved via Template-Based Augmentation |
| **Multimodal** | 77.68% | Uses Cross-Attention to refine acoustic features |

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key Packages:**
- **PyTorch** 2.0.1 - Deep learning framework
- **Transformers** 4.31.0 - DistilRoBERTa for text encoding
- **Librosa** 0.10.0 - Audio feature extraction (MFCC)
- **Scikit-learn** 1.3.1 - ML metrics & evaluation
- **Seaborn & Matplotlib** - Data visualization


---

## 🎯 Pipeline Overview

### 1. Audio-Only Pipeline
- **Input:** 40 MFCCs (Mel-Frequency Cepstral Coefficients)
- **Model:** BiLSTM (256-dim hidden state)
- **Output:** Emotion classification

### 2. Text-Only Pipeline
- **Input:** Transcribed text from speech
- **Model:** DistilRoBERTa + Linear classifier
- **Output:** Emotion classification

### 3. Multimodal Fusion Pipeline
- **Input:** Audio (MFCC) + Text (transcription)
- **Model:** Cross-Attention fusion mechanism
- **Architecture:**
  - Audio Encoder: BiLSTM → 256-dim vector
  - Text Encoder: DistilRoBERTa → 768-dim vector
  - Text Projection: 768 → 256 dimensions
  - Cross-Attention: Audio attends to Text
  - Classifier: Fused vector → 7 emotion classes
- **Output:** Emotion classification

---

## 🎵 Emotion Classes

The system recognizes **7 emotions**:

1. 😠 **Angry** - High arousal, negative valence
2. 🤢 **Disgust** - High arousal, negative valence
3. 😨 **Fear** - High arousal, negative valence
4. 😊 **Happy** - High arousal, positive valence
5. 😐 **Neutral** - Low arousal, neutral valence
6. 😤 **Sad** - Low arousal, negative valence
7. 🎤 **Pleasantly Surprised (PS)** - High arousal, positive valence

---

## 📈 Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 30 |
| Early Stopping | Yes (patience=5) |

### Data Augmentation

- **Text Augmentation:** Template-based paraphrasing
- **Audio:** No augmentation (synthetic speech handled separately)
- **Fusion Training:** Uses both augmented text and original audio

---

## 🧪 Testing & Evaluation

Each pipeline generates:

1. **Classification Report** - Precision, Recall, F1-Score per emotion
2. **Confusion Matrix** - Heatmap visualization
3. **t-SNE Plot** - 2D feature space visualization (encoder output)

Results saved in `Results/plots/` directory:
- `audio_cm.png` - Audio confusion matrix
- `audio_tsne.png` - Audio feature space
- `text_cm.png` - Text confusion matrix
- `text_tsne.png` - Text feature space
- `fusion_cm.png` - Multimodal confusion matrix
- `fusion_tsne.png` - Multimodal feature space

---

## 🔍 Key Features

✅ **High Accuracy** - Audio-only achieves 99.64% accuracy  
✅ **Multimodal Fusion** - Cross-Attention mechanism for effective combination  
✅ **Data Augmentation** - Template-based text augmentation for improved robustness  
✅ **Comprehensive Evaluation** - Classification reports, confusion matrices, t-SNE visualizations  
✅ **GPU Support** - Automatic GPU detection and utilization  
✅ **Easy Setup** - One-command installation and execution  

---



## 📚 References

- **TESS Dataset:** https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- **PyTorch:** https://pytorch.org/
- **Transformers (HuggingFace):** https://huggingface.co/transformers/
- **Librosa:** https://librosa.org/
- **Scikit-learn:** https://scikit-learn.org/
