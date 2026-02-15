import os
import json

import numpy as np
from PIL import Image
import torch
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)

from src.config import IMG_SIZE


# ──────────────────────────────────────────────────────────
#  Collect predictions
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return ground-truth labels and predicted probabilities for the full loader."""
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
        all_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ──────────────────────────────────────────────────────────
#  Classification report
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
#  Confusion matrix
# ──────────────────────────────────────────────────────────

def save_confusion_matrix(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    n = len(class_names)

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

    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"),
               cm, delimiter=",", fmt="%d")
    print(f"  Confusion matrix -> {output_dir}/confusion_matrix.png, .csv")


# ──────────────────────────────────────────────────────────
#  Per-class accuracy
# ──────────────────────────────────────────────────────────

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

    # Sorted bar chart
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


# ──────────────────────────────────────────────────────────
#  Misclassified examples grid
# ──────────────────────────────────────────────────────────

def save_misclassified_examples(labels, preds, probs, dataset, class_names,
                                output_dir, max_examples=25):
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("  No misclassified examples!")
        return

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


# ──────────────────────────────────────────────────────────
#  AUROC curves
# ──────────────────────────────────────────────────────────

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
