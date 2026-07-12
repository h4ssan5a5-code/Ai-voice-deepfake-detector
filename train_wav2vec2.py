import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_all_protocols, ASVspoofWav2VecDataset,
    TRAIN_AUDIO, DEV_AUDIO, EVAL_AUDIO, MODELS_PATH
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BATCH_SIZE         = 4
GRAD_ACCUM_STEPS   = 4     # effective batch = 4*4 = 16
EPOCHS             = 10
LR                 = 3e-5  # transformer ke liye chhota LR
TRAIN_SAMPLES      = 2400  # 1200 real + 1200 fake (balanced)
DEV_SAMPLES        = 600   # 300 real + 300 fake
EVAL_SAMPLES       = 800   # 400 real + 400 fake
NUM_WORKERS        = 0
MODEL_NAME         = "facebook/wav2vec2-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
# Load Protocols
# ─────────────────────────────────────────────
print("\n📂 Loading protocols...")
train_df, dev_df, eval_df = load_all_protocols()

# ─────────────────────────────────────────────
# Load Wav2Vec2
# ─────────────────────────────────────────────
print(f"\n🤖 Loading Wav2Vec2: {MODEL_NAME}")
processor     = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec2_base = Wav2Vec2Model.from_pretrained(MODEL_NAME)
print("✅ Wav2Vec2 loaded!")

# ─────────────────────────────────────────────
# Improved Classifier — deeper head
# ─────────────────────────────────────────────
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec2_model, num_classes=2):
        super().__init__()
        self.wav2vec2 = wav2vec2_model

        # Sirf CNN feature extractor freeze karo
        # Transformer encoder layers trainable rahenge
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

model = Wav2Vec2Classifier(wav2vec2_base, num_classes=2).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 Trainable Parameters: {trainable:,}")

# ─────────────────────────────────────────────
# Balanced Dataset using WeightedRandomSampler
# ─────────────────────────────────────────────
train_dataset = ASVspoofWav2VecDataset(train_df, TRAIN_AUDIO, processor, max_samples=TRAIN_SAMPLES)
dev_dataset   = ASVspoofWav2VecDataset(dev_df,   DEV_AUDIO,   processor, max_samples=DEV_SAMPLES)

# WeightedRandomSampler — har batch mein equal real/fake
labels_list = train_dataset.df["label"].tolist()
class_count = [labels_list.count(0), labels_list.count(1)]  # [fake_count, real_count]
weights     = [1.0 / class_count[l] for l in labels_list]
sampler     = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=NUM_WORKERS)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=NUM_WORKERS)

print(f"✅ Train: {len(train_dataset)} samples | Dev: {len(dev_dataset)} samples")
print(f"   Fake: {class_count[0]} | Real: {class_count[1]}")

# ─────────────────────────────────────────────
# Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────
# Balanced cross entropy — equal weight
criterion = nn.CrossEntropyLoss()

# Different LR for transformer vs classifier
optimizer = optim.AdamW([
    {"params": model.wav2vec2.parameters(),  "lr": LR},
    {"params": model.classifier.parameters(), "lr": LR * 10},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─────────────────────────────────────────────
# Train & Eval Functions
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for i, (input_values, target) in enumerate(loader):
        input_values = input_values.to(device)
        target       = target.to(device)

        output = model(input_values)
        loss   = criterion(output, target) / GRAD_ACCUM_STEPS
        loss.backward()

        # Gradient accumulation
        if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total   += target.size(0)

        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            print(f"  [{i+1}/{len(loader)}] Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | Acc: {100.*correct/total:.1f}%", end="\r")

        del input_values, target, output, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    return total_loss / len(loader), 100. * correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for input_values, target in loader:
            input_values = input_values.to(device)
            target       = target.to(device)

            output = model(input_values)
            loss   = criterion(output, target)
            total_loss += loss.item()

            probs = torch.softmax(output, dim=1)[:, 1]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total   += target.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            del input_values, target, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_probs


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
print("\n🚀 Starting Wav2Vec2 Training (FIXED)...")
print("=" * 65)

train_losses, dev_losses = [], []
train_accs,   dev_accs   = [], []
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc                       = train_epoch(model, train_loader, optimizer, criterion)
    dv_loss, dv_acc, preds, labels, probs = eval_epoch(model, dev_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    dev_losses.append(dv_loss)
    train_accs.append(tr_acc)
    dev_accs.append(dv_acc)

    # Per-class accuracy
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    fake_acc = 100. * (preds_arr[labels_arr==0] == 0).sum() / max((labels_arr==0).sum(), 1)
    real_acc = 100. * (preds_arr[labels_arr==1] == 1).sum() / max((labels_arr==1).sum(), 1)

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    print(f"  Train → Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
    print(f"  Val   → Loss: {dv_loss:.4f}  Acc: {dv_acc:.2f}%  AUC: {auc:.4f}")
    print(f"  Val per-class → Fake: {fake_acc:.1f}%  Real: {real_acc:.1f}%")

    if dv_acc > best_acc and fake_acc > 10:  # sirf save karo agar fake bhi detect ho raha ho
        best_acc  = dv_acc
        save_path = os.path.join(MODELS_PATH, "wav2vec2_best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved! Acc: {best_acc:.2f}%")

print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")

# ─────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label="Train Loss", color="blue",   marker="o", markersize=4)
axes[0].plot(dev_losses,   label="Val Loss",   color="orange", marker="o", markersize=4)
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label="Train Acc", color="green", marker="o", markersize=4)
axes[1].plot(dev_accs,   label="Val Acc",   color="red",   marker="o", markersize=4)
axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Wav2Vec2 — Training History (Fixed)", fontsize=14)
plt.tight_layout()
curve_path = os.path.join(MODELS_PATH, "wav2vec2_training_curves.png")
plt.savefig(curve_path, dpi=150)
plt.show()

# ─────────────────────────────────────────────
# Final Evaluation
# ─────────────────────────────────────────────
print("\n📊 Final Evaluation on Test Set...")
best_path = os.path.join(MODELS_PATH, "wav2vec2_best_model.pth")
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, weights_only=True))
else:
    print("⚠️  No saved model found, using last epoch weights")

eval_dataset = ASVspoofWav2VecDataset(eval_df, EVAL_AUDIO, processor, max_samples=EVAL_SAMPLES)
eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

_, test_acc, test_preds, test_labels, test_probs = eval_epoch(model, eval_loader, criterion)

print(f"\n✅ Test Accuracy : {test_acc:.2f}%")
try:
    print(f"✅ ROC-AUC Score : {roc_auc_score(test_labels, test_probs):.4f}")
except:
    pass
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"]))

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Wav2Vec2 — Confusion Matrix (Fixed)")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
cm_path = os.path.join(MODELS_PATH, "wav2vec2_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"📊 Saved → {cm_path}")