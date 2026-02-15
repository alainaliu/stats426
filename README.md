# NABirds Fine-Grained Bird Species Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=matplotlib&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-3776AB?logo=python&logoColor=white)

> **STAT 426 – Project 1 (Team 12)**
>
> A CNN pipeline for fine-grained bird species classification on the [NABirds dataset](https://dl.allaboutbirds.org/nabirds) (Cornell Lab of Ornithology). The project explores how robust a classifier can be made against real-world image challenges (lighting, blur, distance, background bias) through data augmentation, transfer learning, and evaluation diagnostics.
>
> *Note:* Claude Opus was used to help with the presentation and layout of project repository, but all modeling logic/construction was done by our project team.

---

## Project Structure

```
stats426/
├── pipeline_script.py          # Entry point — training loop + main()
├── requirements.txt            # Python dependencies
├── README.md
├── src/                        # Source package
│   ├── __init__.py
│   ├── config.py               # All hyperparameters & paths
│   ├── dataset.py              # NABirdsDataset class, data loaders, transforms
│   ├── model.py                # NABirdsResNet model architecture
│   └── evaluation.py           # Metrics, plots, and report generation
├── checkpoints/                # Saved models & evaluation outputs (generated)
│   ├── best.pth
│   ├── last.pth
│   ├── run_info.json
│   ├── training_curves.png
│   ├── classification_report.txt
│   ├── confusion_matrix.png / .csv
│   ├── per_class_accuracy.png / .txt
│   ├── misclassified.png
│   └── auroc_*.png / auroc_summary.json
└── nabirds/                    # Dataset root (downloaded separately)
    ├── images/                 # Bird images organised by species
    ├── parts/                  # Part location annotations
    ├── bounding_boxes.txt
    ├── classes.txt
    ├── hierarchy.txt
    ├── image_class_labels.txt
    ├── images.txt
    ├── sizes.txt
    ├── photographers.txt
    └── train_test_split.txt
```

## Setup

### 1. Clone & enter the repo

```bash
cd stats426
```

### 2. Create a virtual environment & install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the NABirds dataset

Download from <https://dl.allaboutbirds.org/nabirds> and extract so that the `images/`, `parts/`, and `.txt` files are inside the `nabirds/` directory.

### 4. Run the pipeline

```bash
python pipeline_script.py
```

All checkpoints and evaluation artifacts are saved to `checkpoints/`.

---

## Pipeline Overview

### Preprocessing
- **Bounding-box crop** – provided by the dataset; images are cropped with a 10 % margin to isolate the bird.
- **Training augmentations** – random resized crop (224×224), horizontal flip, color jitter, ±10° rotation, random erasing.
- **Test transforms** – resize to 256 using center crop to 224×224.
- **Normalization** – ImageNet mean/std (pretrained backbone).

### Class Subset Mode
The full dataset has 555 species. A configurable subset mode (default: classes 295–400, **84 classes**, 3 588 train / 3 735 test images) keeps iteration fast (~6.5 min for 10 epochs on Apple Silicon). Set `USE_SUBSET = False` in `src/config.py` to train on all classes.

### Class Imbalance
Class sizes range from 13 to 120 images. **Weighted random sampling** (inverse frequency) ensures balanced representation per epoch.

### Model Architecture
| Component | Details |
|---|---|
| Backbone | ResNet-18 pretrained on ImageNet (swappable to ResNet-34/50) |
| Head | `BatchNorm → Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.15) → Linear(256→C)` |
| Init | Kaiming Normal (classifier), ImageNet weights (backbone) |
| Params | ~11.2 M (ResNet-18) |

### Training Configuration
| Setting | Value |
|---|---|
| Optimizer | AdamW (backbone LR 1e-4, head LR 1e-3, weight decay 1e-4) |
| Schedule | 2-epoch linear warmup → cosine annealing |
| Loss | Cross-entropy with label smoothing (0.1) |
| Gradient clipping | Max norm 1.0 |
| Mixed precision | AMP (auto on CUDA) |

### Baseline Results (ResNet-18, classes 295–400, 10 epochs)

| Metric | Value |
|---|---|
| **Top-1 Accuracy** | **71.75 %** |
| **Top-5 Accuracy** | **91.86 %** |
| Training Accuracy | 86.33 % |
| Training Time | ~6.5 min (Apple MPS) |
| Random Chance (84 classes) | ~1.2 % |

> Test loss was still decreasing at epoch 10 — more epochs or a larger backbone should push accuracy higher.

### Evaluation Outputs (auto-generated)
| Artifact | Description |
|---|---|
| `training_curves.png` | Loss, accuracy, and LR curves per epoch |
| `classification_report.txt` | Precision, recall, F1 per class |
| `confusion_matrix.png / .csv` | Color-coded heatmap + raw values |
| `per_class_accuracy.png / .txt` | Horizontal bar chart + ranked list |
| `misclassified.png` | Top-25 most confidently wrong predictions |
| `auroc_*.png / auroc_summary.json` | Per-class, macro, micro AUROC + best/worst ROC curves |

---

## Configuration

All hyper-parameters live in `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `USE_SUBSET` | `True` | Filter to a class range for fast iteration |
| `CLASS_MIN / CLASS_MAX` | `295 / 400` | Inclusive class-ID range when subset is active |
| `BACKBONE` | `"resnet18"` | `"resnet18"`, `"resnet34"`, or `"resnet50"` |
| `IMG_SIZE` | `224` | Input resolution |
| `BATCH_SIZE` | `32` | Batch size |
| `EPOCHS` | `10` | Training epochs |
| `LR / BACKBONE_LR` | `1e-3 / 1e-4` | Head / backbone learning rates |
| `DROPOUT` | `0.3` | Classifier dropout rate |
| `USE_BBOX` | `True` | Crop to bounding box before transforms |

---

## Dataset

The **NABirds** dataset was created in collaboration with the Cornell Lab of Ornithology. It contains ~48 000 images across 555 bird species, with bounding boxes, part annotations, and a taxonomic hierarchy.

---

## Example Run of Pipeline
After running pipeline.py you get:

```bash
============================================================
  NABirds RESNET18 Classifier  (Team 12, STAT 426)
  Device : mps
  Subset : classes 295-400
  Image  : 224x224  |  BS: 32  |  Epochs: 10
  LR     : 0.001  |  Backbone LR: 0.0001
============================================================

Loading dataset...
  [Train] 3,588 images, 84 classes (classes 295-400)
  [Test ] 3,735 images, 84 classes (classes 295-400)
  Class sizes: min=10, max=60  (weighted sampling enabled)
  Classes: 84

Model: RESNET18 -> 84 classes  (11.3M params)

____________________________________________________________
  Starting training  (RESNET18, classes 295-400)
____________________________________________________________

  Epoch  1 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=10.3322  acc=0.3%  ETA 81s
  Epoch  1 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=9.5337  acc=1.3%  ETA 46s
  Epoch  1 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=7.8035  acc=3.0%  ETA 33s
  Epoch  1 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=6.9691  acc=3.7%  ETA 25s
  Epoch  1 |████████████░░░░░░░░░░░░░|  49.1%  loss=7.2544  acc=4.2%  ETA 19s
  Epoch  1 |██████████████░░░░░░░░░░░|  58.9%  loss=5.6495  acc=5.0%  ETA 15s
  Epoch  1 |█████████████████░░░░░░░░|  68.8%  loss=6.0858  acc=6.5%  ETA 11s
  Epoch  1 |███████████████████░░░░░░|  78.6%  loss=4.9828  acc=8.7%  ETA 7s
  Epoch  1 |██████████████████████░░░|  88.4%  loss=4.0735  acc=9.9%  ETA 4s
  Epoch  1 |████████████████████████░|  98.2%  loss=4.1870  acc=11.0%  ETA 1s
  Epoch  1 |█████████████████████████| 100.0%  loss=4.5817  acc=11.2%  ETA 0s

  Epoch 1/10 (46s, total 0.8min, ~6.8min left)
  Train  loss=7.3014  acc=11.22%
  Test   loss=4.1197  acc=28.86%  top5=56.92%  lr=1.00e-03  ** NEW BEST
  Best so far: 28.86%

  Epoch  2 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=4.8635  acc=26.1%  ETA 26s
  Epoch  2 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=4.4832  acc=29.0%  ETA 22s
  Epoch  2 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=4.8272  acc=29.7%  ETA 19s
  Epoch  2 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=3.7008  acc=32.0%  ETA 16s
  Epoch  2 |████████████░░░░░░░░░░░░░|  49.1%  loss=2.9441  acc=34.0%  ETA 13s
  Epoch  2 |██████████████░░░░░░░░░░░|  58.9%  loss=4.0446  acc=35.6%  ETA 11s
  Epoch  2 |█████████████████░░░░░░░░|  68.8%  loss=3.4043  acc=37.0%  ETA 8s
  Epoch  2 |███████████████████░░░░░░|  78.6%  loss=4.0021  acc=38.5%  ETA 6s
  Epoch  2 |██████████████████████░░░|  88.4%  loss=3.3548  acc=39.5%  ETA 3s
  Epoch  2 |████████████████████████░|  98.2%  loss=3.1378  acc=40.5%  ETA 0s
  Epoch  2 |█████████████████████████| 100.0%  loss=2.7792  acc=40.8%  ETA 0s

  Epoch 2/10 (34s, total 1.3min, ~5.3min left)
  Train  loss=3.7919  acc=40.79%
  Test   loss=3.0653  acc=50.58%  top5=78.98%  lr=1.00e-03  ** NEW BEST
  Best so far: 50.58%

  Epoch  3 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=3.2271  acc=52.0%  ETA 26s
  Epoch  3 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=3.0160  acc=52.3%  ETA 22s
  Epoch  3 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=2.7698  acc=53.1%  ETA 19s
  Epoch  3 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=2.3906  acc=54.8%  ETA 16s
  Epoch  3 |████████████░░░░░░░░░░░░░|  49.1%  loss=2.8595  acc=55.5%  ETA 14s
  Epoch  3 |██████████████░░░░░░░░░░░|  58.9%  loss=2.6788  acc=56.3%  ETA 11s
  Epoch  3 |█████████████████░░░░░░░░|  68.8%  loss=2.8835  acc=56.1%  ETA 8s
  Epoch  3 |███████████████████░░░░░░|  78.6%  loss=2.3997  acc=56.1%  ETA 6s
  Epoch  3 |██████████████████████░░░|  88.4%  loss=2.5525  acc=56.2%  ETA 3s
  Epoch  3 |████████████████████████░|  98.2%  loss=3.0607  acc=57.2%  ETA 0s
  Epoch  3 |█████████████████████████| 100.0%  loss=2.4509  acc=57.3%  ETA 0s

  Epoch 3/10 (35s, total 1.9min, ~4.5min left)
  Train  loss=2.8707  acc=57.31%
  Test   loss=2.8478  acc=56.22%  top5=82.92%  lr=9.62e-04  ** NEW BEST
  Best so far: 56.22%

  Epoch  4 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=3.2657  acc=66.2%  ETA 28s
  Epoch  4 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=2.6909  acc=64.5%  ETA 23s
  Epoch  4 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=2.2475  acc=64.8%  ETA 20s
  Epoch  4 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=2.2497  acc=65.1%  ETA 17s
  Epoch  4 |████████████░░░░░░░░░░░░░|  49.1%  loss=2.5605  acc=64.5%  ETA 14s
  Epoch  4 |██████████████░░░░░░░░░░░|  58.9%  loss=2.2600  acc=64.4%  ETA 11s
  Epoch  4 |█████████████████░░░░░░░░|  68.8%  loss=2.3122  acc=64.2%  ETA 9s
  Epoch  4 |███████████████████░░░░░░|  78.6%  loss=2.3888  acc=64.3%  ETA 6s
  Epoch  4 |██████████████████████░░░|  88.4%  loss=1.9531  acc=64.7%  ETA 3s
  Epoch  4 |████████████████████████░|  98.2%  loss=1.7655  acc=65.2%  ETA 0s
  Epoch  4 |█████████████████████████| 100.0%  loss=2.7120  acc=65.0%  ETA 0s

  Epoch 4/10 (37s, total 2.5min, ~3.8min left)
  Train  loss=2.4201  acc=65.04%
  Test   loss=2.4452  acc=62.30%  top5=86.02%  lr=8.54e-04  ** NEW BEST
  Best so far: 62.30%

  Epoch  5 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=2.0533  acc=71.9%  ETA 27s
  Epoch  5 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=2.0139  acc=70.9%  ETA 23s
  Epoch  5 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=2.3150  acc=71.1%  ETA 20s
  Epoch  5 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=2.1596  acc=69.9%  ETA 17s
  Epoch  5 |████████████░░░░░░░░░░░░░|  49.1%  loss=2.0335  acc=70.2%  ETA 14s
  Epoch  5 |██████████████░░░░░░░░░░░|  58.9%  loss=1.7616  acc=70.6%  ETA 11s
  Epoch  5 |█████████████████░░░░░░░░|  68.8%  loss=2.3075  acc=70.4%  ETA 9s
  Epoch  5 |███████████████████░░░░░░|  78.6%  loss=2.1043  acc=70.3%  ETA 6s
  Epoch  5 |██████████████████████░░░|  88.4%  loss=1.8908  acc=70.3%  ETA 3s
  Epoch  5 |████████████████████████░|  98.2%  loss=1.8018  acc=70.7%  ETA 0s
  Epoch  5 |█████████████████████████| 100.0%  loss=2.1422  acc=70.5%  ETA 0s

  Epoch 5/10 (38s, total 3.2min, ~3.2min left)
  Train  loss=2.1232  acc=70.54%
  Test   loss=2.2104  acc=63.96%  top5=87.52%  lr=6.91e-04  ** NEW BEST
  Best so far: 63.96%

  Epoch  6 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=2.0882  acc=75.3%  ETA 28s
  Epoch  6 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=1.9156  acc=74.9%  ETA 24s
  Epoch  6 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=2.3578  acc=74.9%  ETA 21s
  Epoch  6 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=1.8338  acc=75.7%  ETA 18s
  Epoch  6 |████████████░░░░░░░░░░░░░|  49.1%  loss=1.7039  acc=76.2%  ETA 15s
  Epoch  6 |██████████████░░░░░░░░░░░|  58.9%  loss=2.0197  acc=75.5%  ETA 12s
  Epoch  6 |█████████████████░░░░░░░░|  68.8%  loss=1.9439  acc=75.4%  ETA 9s
  Epoch  6 |███████████████████░░░░░░|  78.6%  loss=1.8452  acc=75.7%  ETA 6s
  Epoch  6 |██████████████████████░░░|  88.4%  loss=1.7131  acc=76.3%  ETA 3s
  Epoch  6 |████████████████████████░|  98.2%  loss=2.2091  acc=76.6%  ETA 1s
  Epoch  6 |█████████████████████████| 100.0%  loss=1.4364  acc=76.8%  ETA 0s

  Epoch 6/10 (40s, total 3.8min, ~2.6min left)
  Train  loss=1.8668  acc=76.76%
  Test   loss=2.0848  acc=64.85%  top5=88.70%  lr=5.00e-04  ** NEW BEST
  Best so far: 64.85%

  Epoch  7 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=1.7562  acc=77.6%  ETA 30s
  Epoch  7 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=1.8670  acc=76.8%  ETA 25s
  Epoch  7 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=1.6950  acc=77.7%  ETA 22s
  Epoch  7 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=1.6175  acc=78.6%  ETA 19s
  Epoch  7 |████████████░░░░░░░░░░░░░|  49.1%  loss=1.5841  acc=78.2%  ETA 16s
  Epoch  7 |██████████████░░░░░░░░░░░|  58.9%  loss=1.3367  acc=77.9%  ETA 13s
  Epoch  7 |█████████████████░░░░░░░░|  68.8%  loss=1.6770  acc=78.4%  ETA 10s
  Epoch  7 |███████████████████░░░░░░|  78.6%  loss=1.5953  acc=79.0%  ETA 7s
  Epoch  7 |██████████████████████░░░|  88.4%  loss=1.5715  acc=79.1%  ETA 4s
  Epoch  7 |████████████████████████░|  98.2%  loss=1.5197  acc=79.4%  ETA 1s
  Epoch  7 |█████████████████████████| 100.0%  loss=1.4721  acc=79.5%  ETA 0s

  Epoch 7/10 (40s, total 4.5min, ~1.9min left)
  Train  loss=1.7030  acc=79.55%
  Test   loss=1.9794  acc=68.01%  top5=89.91%  lr=3.09e-04  ** NEW BEST
  Best so far: 68.01%

  Epoch  8 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=1.6203  acc=86.9%  ETA 30s
  Epoch  8 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=1.4265  acc=84.2%  ETA 26s
  Epoch  8 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=1.5312  acc=84.5%  ETA 22s
  Epoch  8 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=1.7785  acc=83.7%  ETA 19s
  Epoch  8 |████████████░░░░░░░░░░░░░|  49.1%  loss=1.5333  acc=83.7%  ETA 16s
  Epoch  8 |██████████████░░░░░░░░░░░|  58.9%  loss=1.5494  acc=83.8%  ETA 13s
  Epoch  8 |█████████████████░░░░░░░░|  68.8%  loss=1.5434  acc=84.0%  ETA 10s
  Epoch  8 |███████████████████░░░░░░|  78.6%  loss=1.8095  acc=84.1%  ETA 7s
  Epoch  8 |██████████████████████░░░|  88.4%  loss=1.4689  acc=84.2%  ETA 4s
  Epoch  8 |████████████████████████░|  98.2%  loss=1.3147  acc=83.9%  ETA 1s
  Epoch  8 |█████████████████████████| 100.0%  loss=1.5579  acc=83.8%  ETA 0s

  Epoch 8/10 (42s, total 5.2min, ~1.3min left)
  Train  loss=1.5547  acc=83.82%
  Test   loss=1.8907  acc=69.99%  top5=90.82%  lr=1.46e-04  ** NEW BEST
  Best so far: 69.99%

  Epoch  9 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=1.2296  acc=84.7%  ETA 31s
  Epoch  9 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=1.5225  acc=84.4%  ETA 26s
  Epoch  9 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=1.4055  acc=83.8%  ETA 23s
  Epoch  9 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=1.5315  acc=83.9%  ETA 19s
  Epoch  9 |████████████░░░░░░░░░░░░░|  49.1%  loss=1.4556  acc=83.8%  ETA 16s
  Epoch  9 |██████████████░░░░░░░░░░░|  58.9%  loss=1.5364  acc=84.1%  ETA 13s
  Epoch  9 |█████████████████░░░░░░░░|  68.8%  loss=1.6149  acc=84.1%  ETA 10s
  Epoch  9 |███████████████████░░░░░░|  78.6%  loss=1.4038  acc=84.4%  ETA 7s
  Epoch  9 |██████████████████████░░░|  88.4%  loss=1.3393  acc=84.2%  ETA 4s
  Epoch  9 |████████████████████████░|  98.2%  loss=1.5436  acc=84.2%  ETA 1s
  Epoch  9 |█████████████████████████| 100.0%  loss=1.7309  acc=84.2%  ETA 0s

  Epoch 9/10 (43s, total 5.9min, ~0.7min left)
  Train  loss=1.5119  acc=84.24%
  Test   loss=1.8231  acc=71.22%  top5=91.70%  lr=3.81e-05  ** NEW BEST
  Best so far: 71.22%

  Epoch 10 |██░░░░░░░░░░░░░░░░░░░░░░░|   9.8%  loss=1.4083  acc=84.4%  ETA 31s
  Epoch 10 |████░░░░░░░░░░░░░░░░░░░░░|  19.6%  loss=1.3204  acc=85.9%  ETA 26s
  Epoch 10 |███████░░░░░░░░░░░░░░░░░░|  29.5%  loss=1.6145  acc=86.0%  ETA 23s
  Epoch 10 |█████████░░░░░░░░░░░░░░░░|  39.3%  loss=1.3730  acc=86.0%  ETA 20s
  Epoch 10 |████████████░░░░░░░░░░░░░|  49.1%  loss=1.3094  acc=86.0%  ETA 16s
  Epoch 10 |██████████████░░░░░░░░░░░|  58.9%  loss=1.4755  acc=85.9%  ETA 13s
  Epoch 10 |█████████████████░░░░░░░░|  68.8%  loss=1.5809  acc=86.3%  ETA 10s
  Epoch 10 |███████████████████░░░░░░|  78.6%  loss=1.5166  acc=86.5%  ETA 7s
  Epoch 10 |██████████████████████░░░|  88.4%  loss=1.3957  acc=86.6%  ETA 4s
  Epoch 10 |████████████████████████░|  98.2%  loss=1.3819  acc=86.4%  ETA 1s
  Epoch 10 |█████████████████████████| 100.0%  loss=1.4099  acc=86.3%  ETA 0s

  Epoch 10/10 (43s, total 6.6min, ~0.0min left)
  Train  loss=1.4578  acc=86.33%
  Test   loss=1.7948  acc=71.75%  top5=91.86%  lr=1.00e-06  ** NEW BEST
  Best so far: 71.75%

============================================================
  Training complete in 6.6 minutes
  Best Test Top-1 Accuracy: 71.75%
============================================================
  Training curves -> checkpoints/training_curves.png

Loading best checkpoint for final evaluation...

============================================================
  CLASSIFICATION REPORT
============================================================
                                        precision    recall  f1-score   support

             Common Eider (Adult male)     0.7241    0.7241    0.7241        29
        Long-tailed Duck (Winter male)     0.6970    0.7667    0.7302        30
            Ruddy Duck (Breeding male)     0.8043    0.9737    0.8810        38
         Swainson's Hawk (Dark morph )     0.3571    0.3333    0.3448        30
   Red-tailed Hawk (Light morph adult)     0.5469    0.5833    0.5645        60
              Snow Goose (White morph)     0.7000    0.5385    0.6087        26
             Wood Duck (Breeding male)     0.9231    0.8000    0.8571        60
               Gadwall (Breeding male)     0.6212    0.7321    0.6721        56
       American Wigeon (Breeding male)     0.6909    0.8444    0.7600        45
               Mallard (Breeding male)     0.7959    0.6500    0.7156        60
               Blue-winged Teal (Male)     0.8864    0.9286    0.9070        42
                  Cinnamon Teal (Male)     0.8704    0.8545    0.8624        55
     Northern Shoveler (Breeding male)     0.8571    0.7000    0.7706        60
      Northern Pintail (Breeding male)     0.6600    0.7857    0.7174        42
              Green-winged Teal (Male)     0.8750    0.6829    0.7671        41
            Canvasback (Breeding male)     0.9189    0.7907    0.8500        43
               Redhead (Breeding male)     0.5714    0.5926    0.5818        27
      Ring-necked Duck (Breeding male)     0.6522    0.6818    0.6667        44
         Greater Scaup (Breeding male)     0.3448    0.2326    0.2778        43
          Lesser Scaup (Breeding male)     0.5000    0.5227    0.5111        44
                 Harlequin Duck (Male)     0.8409    0.8605    0.8506        43
                    Surf Scoter (Male)     0.7692    0.5405    0.6349        37
            White-winged Scoter (Male)     0.7857    0.4074    0.5366        27
                   Black Scoter (Male)     0.5714    0.8421    0.6809        19
            Bufflehead (Breeding male)     0.8197    0.8333    0.8264        60
      Common Goldeneye (Breeding male)     0.6981    0.7255    0.7115        51
    Barrow's Goldeneye (Breeding male)     0.7353    0.7143    0.7246        35
      Hooded Merganser (Breeding male)     0.9583    0.8679    0.9109        53
      Common Merganser (Breeding male)     0.6111    0.6875    0.6471        32
Red-breasted Merganser (Breeding male)     0.7692    0.7317    0.7500        41
               California Quail (Male)     0.9153    0.9643    0.9391        56
                 Gambel's Quail (Male)     1.0000    0.9143    0.9552        35
           Ring-necked Pheasant (Male)     0.8780    0.7660    0.8182        47
          Red-throated Loon (Breeding)     0.7000    0.5833    0.6364        12
               Pacific Loon (Breeding)     0.7000    0.7000    0.7000        20
                Common Loon (Breeding)     0.8571    0.8000    0.8276        60
               Horned Grebe (Breeding)     0.8889    0.6857    0.7742        35
           Red-necked Grebe (Breeding)     0.8846    0.7931    0.8364        29
                Eared Grebe (Breeding)     0.8537    0.8333    0.8434        42
     Northern Gannet (Adult, Subadult)     0.7241    0.7000    0.7119        60
   Double-crested Cormorant (Immature)     0.6119    0.7885    0.6891        52
               Great Cormorant (Adult)     0.6207    0.4500    0.5217        40
             Little Blue Heron (Adult)     0.6301    0.7667    0.6917        60
            Reddish Egret (Dark morph)     0.8036    0.8654    0.8333        52
     Black-crowned Night-Heron (Adult)     0.8033    0.8167    0.8099        60
    Yellow-crowned Night-Heron (Adult)     0.9348    0.7167    0.8113        60
                    White Ibis (Adult)     0.8852    0.9310    0.9076        58
          Bald Eagle (Adult, subadult)     0.7708    0.6167    0.6852        60
         Northern Harrier (Adult male)     0.7241    0.3500    0.4719        60
           Sharp-shinned Hawk (Adult )     0.5208    0.4310    0.4717        58
                 Cooper's Hawk (Adult)     0.4462    0.5088    0.4754        57
          Red-shouldered Hawk (Adult )     0.6829    0.7368    0.7089        38
             Broad-winged Hawk (Adult)     0.3396    0.6923    0.4557        26
        Rough-legged Hawk (Dark morph)     0.3902    0.6667    0.4923        24
                  Golden Eagle (Adult)     0.7097    0.5238    0.6027        42
         American Kestrel (Adult male)     0.8361    0.8500    0.8430        60
              Peregrine Falcon (Adult)     0.5849    0.6739    0.6263        46
              Purple Gallinule (Adult)     0.9000    0.9000    0.9000        40
              Common Gallinule (Adult)     0.8846    0.9388    0.9109        49
       Black-bellied Plover (Breeding)     0.9429    0.9429    0.9429        35
          Spotted Sandpiper (Breeding)     0.9796    0.9412    0.9600        51
                 Sanderling (Breeding)     0.6522    0.7895    0.7143        19
                     Dunlin (Breeding)     0.9167    0.9429    0.9296        35
         Wilson's Phalarope (Breeding)     0.8627    0.8302    0.8462        53
 Broad-billed Hummingbird (Adult Male)     0.9375    0.9184    0.9278        49
Ruby-throated Hummingbird (Adult Male)     0.8444    0.7037    0.7677        54
Black-chinned Hummingbird (Adult Male)     0.8140    0.7000    0.7527        50
       Anna's Hummingbird (Adult Male)     0.6970    0.8519    0.7667        54
      Costa's Hummingbird (Adult Male)     0.6136    0.7297    0.6667        37
     Calliope Hummingbird (Adult Male)     0.8205    0.7805    0.8000        41
 Broad-tailed Hummingbird (Adult Male)     0.7442    0.7273    0.7356        44
       Rufous Hummingbird (Adult Male)     0.4348    0.6000    0.5042        50
      Allen's Hummingbird (Adult Male)     0.5111    0.3833    0.4381        60
         Red-headed Woodpecker (Adult)     0.9455    0.8814    0.9123        59
     Northern Flicker (Yellow-shafted)     0.8485    0.9492    0.8960        59
            Black Guillemot (Breeding)     0.2222    0.3000    0.2553        20
           Pigeon Guillemot (Breeding)     0.6047    0.6190    0.6118        42
        Black-legged Kittiwake (Adult)     0.3750    0.6522    0.4762        23
              Laughing Gull (Breeding)     0.7705    0.7966    0.7833        59
               Heermann's Gull (Adult)     0.8000    0.7179    0.7568        39
              Ring-billed Gull (Adult)     0.4545    0.5833    0.5109        60
                  Western Gull (Adult)     0.8065    0.7576    0.7812        33
               California Gull (Adult)     0.2264    0.2927    0.2553        41
                  Herring Gull (Adult)     0.5217    0.4211    0.4660        57

                              accuracy                         0.7175      3735
                             macro avg     0.7189    0.7108    0.7078      3735
                          weighted avg     0.7326    0.7175    0.7188      3735

  Saved -> checkpoints/classification_report.txt
  Confusion matrix -> checkpoints/confusion_matrix.png, .csv
  Mean per-class accuracy: 71.08%
  Per-class accuracy -> checkpoints/per_class_accuracy.png, .txt
  Misclassified examples -> checkpoints/misclassified.png (1055 total wrong, showing top 25)

  Macro AUROC (OvR) : 0.9883
  Micro AUROC       : 0.9895
  AUROC plots -> checkpoints/auroc_*.png, auroc_summary.json

============================================================
  Pipeline done:  (RESNET18, classes 295-400)
  Checkpoints:         checkpoints/best.pth, last.pth
  Run info:            checkpoints/run_info.json
  Training curves:     checkpoints/training_curves.png
  Classification:      checkpoints/classification_report.txt
  Confusion matrix:    checkpoints/confusion_matrix.png
  Per-class acc:       checkpoints/per_class_accuracy.png
  Misclassified:       checkpoints/misclassified.png
  AUROC plots:         checkpoints/auroc_*.png
============================================================
```