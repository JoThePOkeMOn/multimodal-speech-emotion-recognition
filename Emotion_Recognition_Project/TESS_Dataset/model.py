import torch
import torch.nn as nn
from transformers import AutoModel

# ===========================================================================
# 1. THE CORE ENCODERS (The Extractors)
# ===========================================================================

class AudioEncoder(nn.Module):
    """Processes the 40 MFCCs into a dense 256-dimensional acoustic vector."""
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=40, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, audio_features):
        lstm_out, (hidden, cell) = self.lstm(audio_features)
        # Concatenate the final forward and backward states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        final_audio_vector = torch.cat((hidden_forward, hidden_backward), dim=1)
        return final_audio_vector

class TextEncoder(nn.Module):
    """Processes the tokenized text using DistilRoBERTa into a 768-dimensional semantic vector."""
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.roberta = AutoModel.from_pretrained("distilroberta-base")

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the [CLS] token representation (the sentence summary)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token


# ===========================================================================
# 2. THE MASTER PIPELINE (Multimodal Fusion)
# ===========================================================================

class MultimodalFusion(nn.Module):
    """The master architecture using Cross-Attention to fuse audio and text."""
    def __init__(self, num_classes=7):
        super(MultimodalFusion, self).__init__()
        
        self.audio_branch = AudioEncoder()
        self.text_branch = TextEncoder()
        
        # Maps RoBERTa's 768 dimensions down to match Audio's 256 dimensions
        self.text_projection = nn.Linear(768, 256)
        
        # Cross-Modal Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # The Final Classification Head (512 inputs: 256 attention + 256 projected text)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) 
        )

    def forward(self, audio_features, input_ids, attention_mask):
        # 1. Extract features
        audio_vector = self.audio_branch(audio_features)
        text_vector = self.text_branch(input_ids, attention_mask)
        
        # 2. Project text to match audio dimensions
        text_vector_projected = self.text_projection(text_vector)
        
        # 3. Prepare for Attention (Adding a sequence dimension)
        audio_query = audio_vector.unsqueeze(1)
        text_key_value = text_vector_projected.unsqueeze(1)
        
        # 4. Cross-Attention: Audio attends to Text
        attn_output, _ = self.attention(query=audio_query, key=text_key_value, value=text_key_value)
        attn_output = attn_output.squeeze(1)
        
        # 5. Fuse the attention output with the original projected text
        fused_vector = torch.cat((attn_output, text_vector_projected), dim=1)
        
        # 6. Final prediction
        logits = self.classifier(fused_vector)
        return logits


# ===========================================================================
# 3. STANDALONE PIPELINES (For Ablation Testing / Single Modality)
# ===========================================================================

class AudioClassifier(nn.Module):
    """Wrapper to train and test ONLY the audio data."""
    def __init__(self, num_classes=7):
        super(AudioClassifier, self).__init__()
        self.encoder = AudioEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio_features):
        vector = self.encoder(audio_features)
        return self.classifier(vector)

class TextClassifier(nn.Module):
    """Wrapper to train and test ONLY the text data."""
    def __init__(self, num_classes=7):
        super(TextClassifier, self).__init__()
        self.encoder = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        vector = self.encoder(input_ids, attention_mask)
        return self.classifier(vector)