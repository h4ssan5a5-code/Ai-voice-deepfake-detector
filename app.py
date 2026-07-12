
import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from utils import DeepfakeCNN, extract_mel_spectrogram, MODELS_PATH

# New architecture — must match train_wav2vec2.py exactly
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec2_model, num_classes=2):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.wav2vec2.feature_extractor._freeze_parameters()
        hidden_size = self.wav2vec2.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, input_values, attention_mask=None):
        out    = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Deepfake Detector",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
<style>
body { background-color: #0f0f1a; }
.stApp { background-color: #0f0f1a; color: #ffffff; }
.result-real {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border: 2px solid #52b788;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #d8f3dc;
    margin: 10px 0;
}
.result-fake {
    background: linear-gradient(135deg, #6b1a1a, #9d2235);
    border: 2px solid #e63946;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #ffccd5;
    margin: 10px 0;
}
.info-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    path = os.path.join(MODELS_PATH, "cnn_best_model.pth")
    if not os.path.exists(path):
        return None
    model = DeepfakeCNN(num_classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_wav2vec2():
    path = os.path.join(MODELS_PATH, "wav2vec2_best_model.pth")
    if not os.path.exists(path):
        return None, None
    processor      = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    base           = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model          = Wav2Vec2Classifier(base, num_classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, processor

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox(
        "Model Select",
        ["CNN (Fast)", "Wav2Vec2 (Accurate)", "Both (Compare)"]
    )
    st.markdown("---")
    st.subheader("📁 Model Status")
    cnn_path = os.path.join(MODELS_PATH, "cnn_best_model.pth")
    w2v_path = os.path.join(MODELS_PATH, "wav2vec2_best_model.pth")
    if os.path.exists(cnn_path):
        st.success("✅ CNN Model: Ready")
    else:
        st.error("❌ CNN: Run train_cnn.py first")

    if os.path.exists(w2v_path):
        st.success("✅ Wav2Vec2: Ready")
    else:
        st.warning("⚠️ Wav2Vec2: Run train_wav2vec2.py first")
    st.markdown("---")
    st.caption("AI Voice Deepfake Detection\nBS AI – 6th Semester\nASVspoof 2019 LA Dataset")

# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("🎙️ AI Voice Deepfake Detector")
st.markdown("**Upload an audio clip to detect whether it's real human speech or AI-generated.**")
st.markdown("---")

audio_file = st.file_uploader(
    "🎧 Upload Audio File (.wav / .mp3 / .flac)",
    type=["wav", "mp3", "flac", "ogg"]
)

if audio_file is not None:
    suffix = "." + audio_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(audio_file)

    with st.spinner("Loading audio..."):
        y, sr = librosa.load(tmp_path, sr=16000, duration=4.0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Waveform**")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#52b788")
        ax.set_title("Waveform", color="white", fontsize=10)
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Mel Spectrogram**")
        mel    = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                  y_axis="mel", ax=ax, cmap="magma")
        ax.set_title("Mel Spectrogram", color="white", fontsize=10)
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close()

    duration = len(y) / sr
    st.markdown(f'<div class="info-box">📋 <b>File:</b> {audio_file.name} &nbsp;|&nbsp; <b>Duration:</b> {duration:.2f}s &nbsp;|&nbsp; <b>Sample Rate:</b> {sr} Hz</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Detection")

    if st.button("🚀 Analyze Now", use_container_width=True, type="primary"):
        with st.spinner("Analyzing audio..."):
            results = {}

            if model_choice in ["CNN (Fast)", "Both (Compare)"]:
                cnn_model = load_cnn()
                if cnn_model is not None:
                    mel_feat = extract_mel_spectrogram(tmp_path)
                    if mel_feat is not None:
                        inp = torch.FloatTensor(mel_feat).unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            out   = cnn_model(inp)
                            probs = torch.softmax(out, dim=1).numpy()[0]
                        results["CNN"] = {"real": probs[1], "fake": probs[0]}
                else:
                    st.error("❌ CNN model not found! Run train_cnn.py.")

            if model_choice in ["Wav2Vec2 (Accurate)", "Both (Compare)"]:
                w2v_model, proc = load_wav2vec2()
                if w2v_model is not None:
                    target_len = 16000 * 4
                    y_proc = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
                    inputs = proc(y_proc, sampling_rate=16000, return_tensors="pt")
                    with torch.no_grad():
                        out   = w2v_model(inputs.input_values)
                        probs = torch.softmax(out, dim=1).numpy()[0]
                    results["Wav2Vec2"] = {"real": probs[1], "fake": probs[0]}
                else:
                    st.warning("⚠️ Wav2Vec2 model not found. Run train_wav2vec2.py.")

        for model_name, res in results.items():
            real_pct = res["real"] * 100
            fake_pct = res["fake"] * 100
            st.markdown(f"### {model_name} Result")
            if res["real"] > 0.5:
                st.markdown(f'<div class="result-real">✅ REAL VOICE &nbsp;|&nbsp; Confidence: {real_pct:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-fake">⚠️ AI-GENERATED / FAKE &nbsp;|&nbsp; Confidence: {fake_pct:.1f}%</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("🟢 Real Probability", f"{real_pct:.1f}%")
            c2.metric("🔴 Fake Probability", f"{fake_pct:.1f}%")
            st.markdown(f"**Real** {'█' * int(real_pct//5)}{'░' * (20 - int(real_pct//5))} {real_pct:.1f}%")
            st.markdown(f"**Fake** {'█' * int(fake_pct//5)}{'░' * (20 - int(fake_pct//5))} {fake_pct:.1f}%")
            st.markdown("---")

    os.unlink(tmp_path)

else:
    st.markdown("""
    <div class="info-box">
    <h4>ℹ️ How to Use</h4>
    <ol>
        <li>Run <code>train_cnn.py</code> → CNN model will be trained</li>
        <li>Then run <code>train_wav2vec2.py</code> → Wav2Vec2 model will be trained</li>
        <li>Upload an audio file (.wav / .mp3 / .flac)</li>
        <li>Select a model and click <b>Analyze Now</b></li>
    </ol>
    <p>🎯 <b>Best results:</b> Use 2-10 second clear audio</p>
    </div>
    """, unsafe_allow_html=True)