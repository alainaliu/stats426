import os

def load_bounding_box_annotations(dataset_path=''):
  
  bboxes = {}
  
  with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      bbox = list(map(int, pieces[1:]))
      bboxes[image_id] = bbox
  
  return bboxes

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
  
def load_part_names(dataset_path=''):
  
  names = {}

  with open(os.path.join(dataset_path, 'parts/parts.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      part_id = int(pieces[0])
      names[part_id] = ' '.join(pieces[1:])
  
  return names  
    
def load_class_names(dataset_path=''):
  
  names = {}
  
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  
  return names

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = class_id
  
  return labels
        
def load_image_paths(dataset_path='', path_prefix=''):
  
  paths = {}
  
  with open(os.path.join(dataset_path, 'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(path_prefix, pieces[1])
      paths[image_id] = path
  
  return paths

def load_image_sizes(dataset_path=''):
  
  sizes = {}
  
  with open(os.path.join(dataset_path, 'sizes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      width, height = map(int, pieces[1:])
      sizes[image_id] = [width, height]
  
  return sizes

def load_hierarchy(dataset_path=''):
  
  parents = {}
  
  with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      child_id, parent_id = pieces
      parents[child_id] = parent_id
  
  return parents

def load_photographers(dataset_path=''):
  
  photographers = {}
  with open(os.path.join(dataset_path, 'photographers.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      photographers[image_id] = ' '.join(pieces[1:])
  
  return photographers

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