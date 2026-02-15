import os
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.config import IMG_SIZE


# Load bounding box annotations from nabirds, which has the format:
# <image_id> <x> <y> <width> <height>
def load_bounding_box_annotations(dataset_path=''):
  
  bboxes = {}
  
  with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      bbox = list(map(int, pieces[1:]))
      bboxes[image_id] = bbox
  
  return bboxes

# Load part annotations from nabirds
def load_part_annotations(dataset_path=''):
  
  parts = {}
  
  with open(os.path.join(dataset_path, 'parts/part_locs.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      parts.setdefault(image_id, [0] * 11)
      part_id = int(pieces[1])
      parts[image_id][part_id] = list(map(int, pieces[2:]))

  return parts  

# Load Names of parts from parts/parts.txt, which has the format:
# <part_id> <part_name>
def load_part_names(dataset_path=''):
  
  names = {}

  with open(os.path.join(dataset_path, 'parts/parts.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      part_id = int(pieces[0])
      names[part_id] = ' '.join(pieces[1:])
  
  return names  

# Load Names of classes from classes.txt, which has the format:
# <class_id> <class_name>
def load_class_names(dataset_path=''):
  
  names = {}
  
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  
  return names

# Load class labels for each image from image_class_labels.txt, which has the format:
# <image_id> <class_id>
def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = class_id
  
  return labels

# Load image paths for pipeline.py, which has the format:
# <image_id> <relative_path>
def load_image_paths(dataset_path='', path_prefix=''):
  
  paths = {}
  
  with open(os.path.join(dataset_path, 'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(path_prefix, pieces[1])
      paths[image_id] = path
  
  return paths

# Load image sizes for pipeline.py, which has the format:
# <image_id> <width> <height>
def load_image_sizes(dataset_path=''):
  
  sizes = {}
  
  with open(os.path.join(dataset_path, 'sizes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      width, height = map(int, pieces[1:])
      sizes[image_id] = [width, height]
  
  return sizes

# Represents the hierarchy of classes in the dataset
def load_hierarchy(dataset_path=''):
  
  parents = {}
  
  with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      child_id, parent_id = pieces
      parents[child_id] = parent_id
  
  return parents

# Loads photographs associated with each picture
def load_photographers(dataset_path=''):
  
  photographers = {}
  with open(os.path.join(dataset_path, 'photographers.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      photographers[image_id] = ' '.join(pieces[1:])
  
  return photographers

# Helps pipeline.py determine which images are in the training set and which are in the test set, based on the file train_test_split.txt
def load_train_test_split(dataset_path=''):
  train_images = []
  test_images = []
  
  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train:
        train_images.append(image_id)
      else:
        test_images.append(image_id)
        
  return train_images, test_images 


# NABirds dataset with optional class-range filtering and bounding-box cropping
class NABirdsDataset(Dataset):
    # Initializes the dataset by loading image paths, labels, bounding boxes, and class names.
    def __init__(self, root, train=True, transform=None, use_bbox=True,
                 use_subset=False, class_min=295, class_max=400):
        self.transform = transform
        self.use_bbox = use_bbox

        # Load via helper functions
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
        subset_tag = (f" (classes {class_min}-{class_max})"
                      if use_subset else " (ALL classes)")
        print(f"  [{tag}] {len(self.samples):,} images, "
              f"{self.num_classes} classes{subset_tag}")
        
        
        
    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.samples)
    
    # Returns (image, label) where image is a tensor and label is an integer index
    def __getitem__(self, idx):
        path, label, bbox = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.use_bbox and bbox:
            x, y, w, h = bbox
            W, H = img.size
            m = 0.1  # 10 % margin
            x1, y1 = max(0, int(x - w * m)), max(0, int(y - h * m))
            x2, y2 = min(W, int(x + w * (1 + m))), min(H, int(y + h * (1 + m)))
            img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)
        return img, label

    # Return per-class sample counts (used for weighted sampling)
    def get_label_counts(self):
        counts = Counter(label for _, label, _ in self.samples)
        return [counts.get(i, 0) for i in range(self.num_classes)]



# Transforms for training and testing, including data augmentation for training

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
