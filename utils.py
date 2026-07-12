import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# PATHS — apna path yahan set karo
# ─────────────────────────────────────────────
BASE_PATH   = r"D:\University\projects\deepfake voice detection system\LA"
TRAIN_AUDIO = os.path.join(BASE_PATH, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO   = os.path.join(BASE_PATH, "ASVspoof2019_LA_dev",   "flac")
EVAL_AUDIO  = os.path.join(BASE_PATH, "ASVspoof2019_LA_eval",  "flac")
PROTO_PATH  = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols")
MODELS_PATH = r"D:\University\projects\deepfake voice detection system\models"

# ─────────────────────────────────────────────
# Load Protocol (Labels)
# ─────────────────────────────────────────────
def load_protocol(filepath):
    """Load ASVspoof2019 protocol file → DataFrame"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            audio_id = parts[1]
            system   = parts[3]
            label    = 1 if parts[4] == "bonafide" else 0  # 1=Real, 0=Fake
            data.append({"audio_id": audio_id, "system": system, "label": label})
    return pd.DataFrame(data)

def load_all_protocols():
    train_df = load_protocol(os.path.join(PROTO_PATH, "ASVspoof2019.LA.cm.train.trn.txt"))
    dev_df   = load_protocol(os.path.join(PROTO_PATH, "ASVspoof2019.LA.cm.dev.trl.txt"))
    eval_df  = load_protocol(os.path.join(PROTO_PATH, "ASVspoof2019.LA.cm.eval.trl.txt"))
    print(f"✅ Train: {len(train_df)} | Dev: {len(dev_df)} | Eval: {len(eval_df)}")
    print(f"   Train Real: {train_df['label'].sum()} | Fake: {(train_df['label']==0).sum()}")
    return train_df, dev_df, eval_df

# ─────────────────────────────────────────────
# Mel Spectrogram Extraction
# ─────────────────────────────────────────────
def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, max_len=128):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=4.0)
        target_len = sr * 4
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] > max_len:
            mel_db = mel_db[:, :max_len]
        else:
            mel_db = np.pad(mel_db, ((0,0),(0, max_len - mel_db.shape[1])))

        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# CNN Dataset
# ─────────────────────────────────────────────
class ASVspoofCNNDataset(Dataset):
    def __init__(self, df, audio_dir, max_samples=None):
        self.df = df.reset_index(drop=True)
        if max_samples:
            real = self.df[self.df["label"]==1].sample(
                min(max_samples//2, len(self.df[self.df["label"]==1])), random_state=42)
            fake = self.df[self.df["label"]==0].sample(
                min(max_samples//2, len(self.df[self.df["label"]==0])), random_state=42)
            self.df = pd.concat([real, fake]).reset_index(drop=True)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row["audio_id"] + ".flac")
        mel  = extract_mel_spectrogram(path)
        if mel is None:
            mel = np.zeros((128, 128))
        mel   = torch.FloatTensor(mel).unsqueeze(0)   # (1, 128, 128)
        label = torch.tensor(row["label"], dtype=torch.long)
        return mel, label

# ─────────────────────────────────────────────
# CNN Model
# ─────────────────────────────────────────────
class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# Wav2Vec2 Dataset
# ─────────────────────────────────────────────
class ASVspoofWav2VecDataset(Dataset):
    def __init__(self, df, audio_dir, processor, max_samples=None):
        self.df = df.reset_index(drop=True)
        if max_samples:
            real = self.df[self.df["label"]==1].sample(
                min(max_samples//2, len(self.df[self.df["label"]==1])), random_state=42)
            fake = self.df[self.df["label"]==0].sample(
                min(max_samples//2, len(self.df[self.df["label"]==0])), random_state=42)
            self.df = pd.concat([real, fake]).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.processor = processor
        self.sr        = 16000

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row["audio_id"] + ".flac")
        try:
            y, _ = librosa.load(path, sr=self.sr, duration=4.0)
            target_len = self.sr * 4
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
        except:
            y = np.zeros(self.sr * 4)

        inputs      = self.processor(y, sampling_rate=self.sr, return_tensors="pt",
                                     padding=True, truncation=True, max_length=64000)
        input_values = inputs.input_values.squeeze(0)
        label        = torch.tensor(row["label"], dtype=torch.long)
        return input_values, label

# ─────────────────────────────────────────────
# Wav2Vec2 Classifier
# ─────────────────────────────────────────────
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec2_model, num_classes=2):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.wav2vec2.feature_extractor._freeze_parameters()
        hidden_size   = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, input_values, attention_mask=None):
        out    = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)
