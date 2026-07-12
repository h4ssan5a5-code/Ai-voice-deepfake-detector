"""
train_cnn.py — CNN Model Training
Run: python train_cnn.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import (
    load_all_protocols, ASVspoofCNNDataset, DeepfakeCNN,
    TRAIN_AUDIO, DEV_AUDIO, EVAL_AUDIO, MODELS_PATH
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 15
LR            = 0.001
TRAIN_SAMPLES = 6000   # increase karo agar time ho
DEV_SAMPLES   = 1500
EVAL_SAMPLES  = 2000
NUM_WORKERS   = 0      # Windows mein 0 rakhna zaroori hai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
print("\n📂 Loading protocols...")
train_df, dev_df, eval_df = load_all_protocols()

train_dataset = ASVspoofCNNDataset(train_df, TRAIN_AUDIO, max_samples=TRAIN_SAMPLES)
dev_dataset   = ASVspoofCNNDataset(dev_df,   DEV_AUDIO,   max_samples=DEV_SAMPLES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"✅ Train batches: {len(train_loader)} | Dev batches: {len(dev_loader)}")

# ─────────────────────────────────────────────
# Model, Loss, Optimizer
# ─────────────────────────────────────────────
model     = DeepfakeCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 CNN Parameters: {total_params:,}")

# ─────────────────────────────────────────────
# Train & Eval Functions
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total   += target.size(0)

        # Progress
        if (i+1) % 20 == 0 or (i+1) == len(loader):
            print(f"  [{i+1}/{len(loader)}] Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%", end="\r")

    print()
    return total_loss / len(loader), 100. * correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)
            total_loss += loss.item()

            probs     = torch.softmax(output, dim=1)[:, 1]
            _, predicted = output.max(1)
            correct  += predicted.eq(target).sum().item()
            total    += target.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_probs


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
print("\n🚀 Starting CNN Training...")
print("=" * 65)

train_losses, dev_losses = [], []
train_accs,   dev_accs   = [], []
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc                        = train_epoch(model, train_loader, optimizer, criterion)
    dv_loss, dv_acc, preds, labels, probs  = eval_epoch(model, dev_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    dev_losses.append(dv_loss)
    train_accs.append(tr_acc)
    dev_accs.append(dv_acc)

    auc = roc_auc_score(labels, probs)
    print(f"  Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
    print(f"  Val   Loss: {dv_loss:.4f}  Acc: {dv_acc:.2f}%  AUC: {auc:.4f}")

    if dv_acc > best_acc:
        best_acc = dv_acc
        save_path = os.path.join(MODELS_PATH, "cnn_best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved → {save_path}")

print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")

# ─────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss", color="blue",   marker="o", markersize=4)
axes[0].plot(dev_losses,   label="Val Loss",   color="orange", marker="o", markersize=4)
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label="Train Acc", color="green", marker="o", markersize=4)
axes[1].plot(dev_accs,   label="Val Acc",   color="red",   marker="o", markersize=4)
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("CNN Model — Training History", fontsize=14)
plt.tight_layout()
curve_path = os.path.join(MODELS_PATH, "cnn_training_curves.png")
plt.savefig(curve_path, dpi=150)
plt.show()
print(f"📊 Training curves saved → {curve_path}")

# ─────────────────────────────────────────────
# Final Test Evaluation
# ─────────────────────────────────────────────
print("\n📊 Final Evaluation on Test Set...")
model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "cnn_best_model.pth")))

eval_dataset = ASVspoofCNNDataset(eval_df, EVAL_AUDIO, max_samples=EVAL_SAMPLES)
eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

_, test_acc, test_preds, test_labels, test_probs = eval_epoch(model, eval_loader, criterion)

print(f"\n✅ Test Accuracy : {test_acc:.2f}%")
print(f"✅ ROC-AUC Score : {roc_auc_score(test_labels, test_probs):.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"]))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.title("CNN Model — Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
cm_path = os.path.join(MODELS_PATH, "cnn_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"📊 Confusion matrix saved → {cm_path}")
