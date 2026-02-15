import os
import time
import random
import json
import math
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)

from nabirds_data_loaders import (
    load_image_paths,
    load_image_labels,
    load_train_test_split,
    load_bounding_box_annotations,
    load_class_names,
)


# CONFIG
DATA_DIR       = "."
OUTPUT_DIR     = "checkpoints"
IMG_SIZE       = 224            # standard ResNet input
BATCH_SIZE     = 32
EPOCHS         = 10             # short run to confirm loss decreases
LR             = 1e-3
BACKBONE_LR    = 1e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
DROPOUT        = 0.3
WARMUP_EPOCHS  = 2
NUM_WORKERS    = 4
USE_BBOX       = True
SEED           = 42

# Subset control
USE_SUBSET     = True           # True = classes 295-400 only (fast)
CLASS_MIN      = 295            # inclusive
CLASS_MAX      = 400            # inclusive
# Set USE_SUBSET = False to use all classes

# Architecture
BACKBONE       = "resnet18"     # "resnet18" for baseline, "resnet50" for later


# NABirds dataset with optional class-range filtering and bbox cropping
class NABirdsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, use_bbox=True,
                 use_subset=False, class_min=295, class_max=400):
        self.transform = transform
        self.use_bbox = use_bbox

        # Load via nabirds.py helpers
        image_paths = load_image_paths(root, path_prefix=os.path.join(root, "images"))
        image_labels = load_image_labels(root)
        train_images, test_images = load_train_test_split(root)
        bboxes = load_bounding_box_annotations(root)
        class_names_dict = load_class_names(root)

        # Determine which class IDs to keep
        all_class_ids = sorted(set(int(v) for v in image_labels.values()))
        if use_subset:
            all_class_ids = [c for c in all_class_ids
                             if class_min <= c <= class_max]

        # Contiguous label mapping (sparse IDs -> 0..C-1)
        self.label_to_idx = {lbl: i for i, lbl in enumerate(all_class_ids)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.num_classes = len(all_class_ids)

        # Class names ordered by contiguous index
        self.class_name_list = [
            class_names_dict.get(str(self.idx_to_label[i]),
                                 str(self.idx_to_label[i]))
            for i in range(self.num_classes)
        ]

        # Filter by split AND class range
        valid_classes = set(all_class_ids)
        split_ids = train_images if train else test_images
        self.samples = []
        for img_id in split_ids:
            if img_id not in image_paths:
                continue
            cls_id = int(image_labels[img_id])
            if cls_id not in valid_classes:
                continue
            label_idx = self.label_to_idx[cls_id]
            bbox = list(bboxes.get(img_id, []))
            self.samples.append((image_paths[img_id], label_idx, bbox))

        tag = "Train" if train else "Test "
        subset_tag = f" (classes {class_min}-{class_max})" if use_subset else " (ALL classes)"
        print(f"  [{tag}] {len(self.samples):,} images, "
              f"{self.num_classes} classes{subset_tag}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, bbox = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.use_bbox and bbox:
            x, y, w, h = bbox
            W, H = img.size
            m = 0.1  # 10% margin
            x1, y1 = max(0, int(x - w * m)), max(0, int(y - h * m))
            x2, y2 = min(W, int(x + w * (1 + m))), min(H, int(y + h * (1 + m)))
            img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)
        return img, label

    # Return per-class sample counts, used to build weights for balanced sampling
    def get_label_counts(self):
        counts = Counter(label for _, label, _ in self.samples)
        return [counts.get(i, 0) for i in range(self.num_classes)]


# TRANSFORMS
train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ResNet backbone with a custom classification head for NABirds species
class NABirdsResNet(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", dropout=0.3):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_name = backbone
        self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        # Init classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


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

        # Progress every ~10%
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


# Collect all ground-truth labels and predicted probabilities from the test set
@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
        all_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# Generate and save a sklearn classification report (precision, recall, f1 per class)
def save_classification_report(labels, preds, class_names, output_dir):
    report = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0,
    )
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved -> {path}")


# Compute and save the confusion matrix as a heatmap image and a CSV file
def save_confusion_matrix(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    n = len(class_names)

    # Plot the confusion matrix as a color-coded heatmap
    fig, ax = plt.subplots(figsize=(max(10, n * 0.18), max(10, n * 0.18)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if n <= 60:
        short_names = [(nm[:20] + "..") if len(nm) > 22 else nm
                       for nm in class_names]
        ax.set_xticks(range(n))
        ax.set_xticklabels(short_names, rotation=90, fontsize=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_names, fontsize=5)
    else:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Save the raw confusion matrix values to a CSV for further analysis
    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"),
               cm, delimiter=",", fmt="%d")
    print(f"  Confusion matrix -> {output_dir}/confusion_matrix.png, .csv")


# Compute per-class accuracy and save as a horizontal bar chart and text file
def save_per_class_accuracy(labels, preds, class_names, output_dir):
    n = len(class_names)
    per_class_correct = np.zeros(n)
    per_class_total = np.zeros(n)
    for true, pred in zip(labels, preds):
        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1

    per_class_acc = np.where(per_class_total > 0,
                             per_class_correct / per_class_total * 100, 0)

    # Sort for plot
    order = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[order]
    sorted_names = [class_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.15)))
    colors = ["#e74c3c" if a < 30 else "#f39c12" if a < 60
              else "#27ae60" for a in sorted_acc]
    ax.barh(range(n), sorted_acc, color=colors, height=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=max(4, min(8, 300 // n)))
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)

    # Text file
    lines = []
    for i in np.argsort(-per_class_acc):
        lines.append(f"{per_class_acc[i]:6.2f}%  "
                     f"({int(per_class_correct[i])}/{int(per_class_total[i])})  "
                     f"{class_names[i]}")
    path = os.path.join(output_dir, "per_class_accuracy.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    mean_acc = per_class_acc[per_class_total > 0].mean()
    print(f"  Mean per-class accuracy: {mean_acc:.2f}%")
    print(f"  Per-class accuracy -> {output_dir}/per_class_accuracy.png, .txt")


# Save a grid image of the top most-confidently-wrong predictions for error analysis
def save_misclassified_examples(labels, preds, probs, dataset, class_names,
                                output_dir, max_examples=25):
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("  No misclassified examples!")
        return

    # Sort by confidence in the wrong prediction (most confidently wrong first)
    wrong_conf = probs[wrong, preds[wrong]]
    top_wrong = wrong[np.argsort(-wrong_conf)[:max_examples]]

    ncols = 5
    nrows = (len(top_wrong) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(top_wrong):
        path, true_label, _ = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        ax = axes[i]
        ax.imshow(img)
        true_name = class_names[labels[idx]]
        pred_name = class_names[preds[idx]]
        conf = probs[idx, preds[idx]] * 100
        true_short = true_name[:22] + ".." if len(true_name) > 24 else true_name
        pred_short = pred_name[:22] + ".." if len(pred_name) > 24 else pred_name
        ax.set_title(f"True: {true_short}\nPred: {pred_short}\n({conf:.0f}%)",
                     fontsize=6, color="red")
        ax.axis("off")

    for j in range(len(top_wrong), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Most Confidently Wrong Predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "misclassified.png"), dpi=150)
    plt.close(fig)
    print(f"  Misclassified examples -> {output_dir}/misclassified.png "
          f"({len(wrong)} total wrong, showing top {len(top_wrong)})")


# Compute per-class AUROC scores and save bar charts, best/worst ROC curves, and micro-average ROC
def save_auroc_curves(labels, probs, class_names, output_dir, top_n=10):
    num_classes = probs.shape[1]
    y_bin = np.zeros_like(probs, dtype=int)
    for i, lbl in enumerate(labels):
        y_bin[i, lbl] = 1

    per_class_auc, fpr_d, tpr_d = {}, {}, {}
    for c in range(num_classes):
        if y_bin[:, c].sum() == 0 or y_bin[:, c].sum() == len(labels):
            continue
        fpr_d[c], tpr_d[c], _ = roc_curve(y_bin[:, c], probs[:, c])
        per_class_auc[c] = auc(fpr_d[c], tpr_d[c])

    if not per_class_auc:
        print("  Not enough data for AUROC -- skipping.")
        return

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    try:
        macro_auc = float(roc_auc_score(y_bin, probs, average="macro",
                                        multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")

    print(f"\n  Macro AUROC (OvR) : {macro_auc:.4f}")
    print(f"  Micro AUROC       : {micro_auc:.4f}")

    # Bar chart
    sorted_items = sorted(per_class_auc.items(), key=lambda x: x[1],
                          reverse=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(sorted_items)), [v for _, v in sorted_items],
           width=1.0, color="steelblue", alpha=0.8)
    ax.axhline(float(macro_auc), color="red", ls="--", lw=1.2,
               label=f"Macro AUROC = {macro_auc:.4f}")
    ax.set_xlabel("Classes (sorted)")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-class AUROC (One-vs-Rest)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_bar_chart.png"), dpi=150)
    plt.close(fig)

    # Best & worst ROC curves
    top_n = min(top_n, len(sorted_items))
    for tag, idxs in [("best", [c for c, _ in sorted_items[:top_n]]),
                      ("worst", [c for c, _ in sorted_items[-top_n:]])]:
        fig, ax = plt.subplots(figsize=(8, 8))
        for c in idxs:
            name = class_names[c] if c < len(class_names) else str(c)
            short = (name[:25] + "...") if len(name) > 27 else name
            ax.plot(fpr_d[c], tpr_d[c], lw=1.3,
                    label=f"{short} ({per_class_auc[c]:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC -- {tag.capitalize()} {top_n} Classes")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"auroc_{tag}{top_n}.png"),
                    dpi=150)
        plt.close(fig)

    # Micro-average ROC
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_micro, tpr_micro, color="darkorange", lw=2,
            label=f"Micro-avg (AUC={micro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Micro-Average ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auroc_micro_avg.png"), dpi=150)
    plt.close(fig)

    summary = {
        "macro_auroc": macro_auc,
        "micro_auroc": micro_auc,
        "per_class": {
            class_names[c] if c < len(class_names) else str(c): round(v, 5)
            for c, v in sorted_items
        },
    }
    with open(os.path.join(output_dir, "auroc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  AUROC plots -> {output_dir}/auroc_*.png, auroc_summary.json")


def main():
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


if __name__ == "__main__":
    main()
