import os
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Sample path to NABirds dataset
DATASET_DIR = Path.home() / "Downloads" / "nabirds"


def read_id_file(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            mapping[parts[0]] = parts[1:]
    return mapping


if __name__ == "__main__":
    print("\n--- VALIDATING NABirds DATASET ---\n")

    # Load data
    images = read_id_file(os.path.join(DATASET_DIR, "images.txt"))
    labels = read_id_file(os.path.join(DATASET_DIR, "image_class_labels.txt"))
    split = read_id_file(os.path.join(DATASET_DIR, "train_test_split.txt"))
    classes = read_id_file(os.path.join(DATASET_DIR, "classes.txt"))

    print(f"Total images listed: {len(images)}")
    print(f"Total labels listed: {len(labels)}")
    print(f"Total split entries: {len(split)}")
    print(f"Total classes: {len(classes)}")

    # Class Distribution Check
    print("\n Checking class distribution...")

    class_counts = Counter()

    for image_id, (class_id,) in labels.items():
        class_counts[int(class_id)] += 1

    print(f"Unique classes in labels: {len(class_counts)}")
    print(f"Largest class count: {max(class_counts.values())}")
    print(f"Smallest class count: {min(class_counts.values())}")

    # Histogram of class distribution
    plt.figure(figsize=(8,5))
    plt.hist(class_counts.values(), bins=50, edgecolor='black')
    plt.title("Distribution of Images per Bird Species (NABirds)")
    plt.xlabel("Number of Images per Class")
    plt.ylabel("Number of Classes")
    plt.grid(True)
    plt.show()

    # Species imbalance curve
    # Sort class counts from largest to smallest
    sorted_counts = sorted(class_counts.values(), reverse=True)

    plt.figure(figsize=(10,5))
    plt.plot(sorted_counts)

    plt.title("Species Imbalance Curve (Long-Tail Distribution)")
    plt.xlabel("Species Rank")
    plt.ylabel("Number of Images")

    plt.grid(True)
    plt.show()
