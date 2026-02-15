import os
import time
import random
import json
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, OUTPUT_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS,
    LR, BACKBONE_LR, WEIGHT_DECAY, LABEL_SMOOTH, DROPOUT,
    WARMUP_EPOCHS, NUM_WORKERS, USE_BBOX, SEED,
    USE_SUBSET, CLASS_MIN, CLASS_MAX, BACKBONE,
)
from src.dataset import NABirdsDataset, train_transform, test_transform
from src.model import NABirdsResNet
from src.evaluation import (
    collect_predictions,
    save_classification_report,
    save_confusion_matrix,
    save_per_class_accuracy,
    save_misclassified_examples,
    save_auroc_curves,
)


# ──────────────────────────────────────────────────────────
#  Training & evaluation loops
# ──────────────────────────────────────────────────────────

# Run one training epoch with mixed precision, gradient clipping, and progress logging
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = correct = total = 0
    num_batches = len(loader)
    t0 = time.time()

    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

        # Progress every 10%
        if (i + 1) % max(1, num_batches // 10) == 0 or (i + 1) == num_batches:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (num_batches - i - 1)
            pct = 100 * (i + 1) / num_batches
            filled = int(25 * (i + 1) / num_batches)
            bar = "\u2588" * filled + "\u2591" * (25 - filled)
            print(f"  Epoch {epoch+1:2d} |{bar}| {pct:5.1f}%  "
                  f"loss={loss.item():.4f}  acc={100*correct/total:.1f}%  "
                  f"ETA {eta:.0f}s", flush=True)

    return running_loss / total, 100.0 * correct / total


# Evaluate the model on a data loader and return loss, top-1 accuracy, and top-5 accuracy
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = top5_correct = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        _, t5 = logits.topk(5, 1)
        top5_correct += t5.eq(labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total, 100.0 * top5_correct / total


if __name__ == "__main__":
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )


    subset_str = (f"classes {CLASS_MIN}-{CLASS_MAX}" if USE_SUBSET
                  else "ALL classes")
    print(f"\n{'='*60}")
    print(f"  NABirds {BACKBONE.upper()} Classifier  (Team 12, STAT 426)")
    print(f"  Device : {device}")
    print(f"  Subset : {subset_str}")
    print(f"  Image  : {IMG_SIZE}x{IMG_SIZE}  |  BS: {BATCH_SIZE}  |  "
          f"Epochs: {EPOCHS}")
    print(f"  LR     : {LR}  |  Backbone LR: {BACKBONE_LR}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load train and test datasets with optional subset filtering and bbox cropping
    print("Loading dataset...")
    train_ds = NABirdsDataset(
        DATA_DIR, train=True, transform=train_transform, use_bbox=USE_BBOX,
        use_subset=USE_SUBSET, class_min=CLASS_MIN, class_max=CLASS_MAX)
    test_ds = NABirdsDataset(
        DATA_DIR, train=False, transform=test_transform, use_bbox=USE_BBOX,
        use_subset=USE_SUBSET, class_min=CLASS_MIN, class_max=CLASS_MAX)

    num_classes = train_ds.num_classes
    class_names = train_ds.class_name_list

    # Build a weighted sampler so underrepresented classes are drawn more often
    label_counts = train_ds.get_label_counts()
    class_weights = [1.0 / max(c, 1) for c in label_counts]
    sample_weights = [class_weights[label] for _, label, _ in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds),
                                    replacement=True)

    # Print class distribution summary
    min_count = min(c for c in label_counts if c > 0)
    max_count = max(label_counts)
    print(f"  Class sizes: min={min_count}, max={max_count}  "
          f"(weighted sampling enabled)")
    print(f"  Classes: {num_classes}\n")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin,
                             persistent_workers=NUM_WORKERS > 0)

    # Initialize the ResNet model with pretrained backbone and custom classifier head
    model = NABirdsResNet(num_classes, backbone=BACKBONE,
                          dropout=DROPOUT).to(device)
    params_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {BACKBONE.upper()} -> {num_classes} classes  "
          f"({params_total:.1f}M params)")

    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": BACKBONE_LR},
        {"params": model.classifier.parameters(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return max(1e-6 / LR, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Main training loop with evaluation after each epoch
    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "test_loss": [],
               "test_acc": [], "test_top5": [], "lr": []}

    print(f"\n{'_'*60}")
    print(f"  Starting training  ({BACKBONE.upper()}, {subset_str})")
    print(f"{'_'*60}\n")

    total_t0 = time.time()

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch)
        scheduler.step()

        test_loss, test_acc, test_top5 = evaluate(
            model, test_loader, criterion, device)

        elapsed = time.time() - t0
        total_elapsed = time.time() - total_t0
        eta_total = total_elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        lr_now = optimizer.param_groups[1]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_top5"].append(test_top5)
        history["lr"].append(lr_now)

        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state,
                         "best_acc": best_acc,
                         "epoch": epoch,
                         "num_classes": num_classes,
                         "backbone": BACKBONE,
                         "use_subset": USE_SUBSET,
                         "class_range": [CLASS_MIN, CLASS_MAX]
                             if USE_SUBSET else None},
                        os.path.join(OUTPUT_DIR, "best.pth"))
            marker = "  ** NEW BEST"

        print(f"\n  Epoch {epoch+1}/{EPOCHS} ({elapsed:.0f}s, "
              f"total {total_elapsed/60:.1f}min, "
              f"~{eta_total/60:.1f}min left)")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Test   loss={test_loss:.4f}  acc={test_acc:.2f}%  "
              f"top5={test_top5:.2f}%  lr={lr_now:.2e}{marker}")
        print(f"  Best so far: {best_acc:.2f}%\n")

    total_time = time.time() - total_t0
    print(f"{'='*60}")
    print(f"  Training complete in {total_time/60:.1f} minutes")
    print(f"  Best Test Top-1 Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

    # Save last checkpoint & history
    torch.save({"model_state_dict": model.state_dict(),
                 "best_acc": best_acc,
                 "epoch": EPOCHS - 1,
                 "num_classes": num_classes,
                 "backbone": BACKBONE,
                 "use_subset": USE_SUBSET,
                 "class_range": [CLASS_MIN, CLASS_MAX] if USE_SUBSET else None},
                os.path.join(OUTPUT_DIR, "last.pth"))

    run_info = {
        "backbone": BACKBONE,
        "use_subset": USE_SUBSET,
        "class_range": [CLASS_MIN, CLASS_MAX] if USE_SUBSET else "all",
        "num_classes": num_classes,
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_acc": round(best_acc, 4),
        "total_time_min": round(total_time / 60, 2),
        "device": str(device),
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # Plot loss, accuracy, and learning rate curves over training epochs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_x = range(1, EPOCHS + 1)

    axes[0].plot(epochs_x, history["train_loss"], label="Train")
    axes[0].plot(epochs_x, history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_x, history["train_acc"], label="Train")
    axes[1].plot(epochs_x, history["test_acc"], label="Test Top-1")
    axes[1].plot(epochs_x, history["test_top5"], label="Test Top-5")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs_x, history["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    fig.suptitle(f"Training Curves - {BACKBONE.upper()} ({subset_str})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Training curves -> {OUTPUT_DIR}/training_curves.png")

    # Reload the best checkpoint and run full evaluation with all report outputs
    print("\nLoading best checkpoint for final evaluation...")
    if best_state is not None:
        model.load_state_dict(best_state)

    labels, probs = collect_predictions(model, test_loader, device)
    preds = probs.argmax(axis=1)

    save_classification_report(labels, preds, class_names, OUTPUT_DIR)
    save_confusion_matrix(labels, preds, class_names, OUTPUT_DIR)
    save_per_class_accuracy(labels, preds, class_names, OUTPUT_DIR)
    save_misclassified_examples(labels, preds, probs, test_ds, class_names,
                                OUTPUT_DIR)
    save_auroc_curves(labels, probs, class_names, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  Pipeline done:  ({BACKBONE.upper()}, {subset_str})")
    print(f"  Checkpoints:         {OUTPUT_DIR}/best.pth, last.pth")
    print(f"  Run info:            {OUTPUT_DIR}/run_info.json")
    print(f"  Training curves:     {OUTPUT_DIR}/training_curves.png")
    print(f"  Classification:      {OUTPUT_DIR}/classification_report.txt")
    print(f"  Confusion matrix:    {OUTPUT_DIR}/confusion_matrix.png")
    print(f"  Per-class acc:       {OUTPUT_DIR}/per_class_accuracy.png")
    print(f"  Misclassified:       {OUTPUT_DIR}/misclassified.png")
    print(f"  AUROC plots:         {OUTPUT_DIR}/auroc_*.png")
    print(f"{'='*60}\n")
