from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import random
from tqdm import tqdm

#Initialise directories
SOURCE_DIR = Path('dataset/waste-image-dataset')
DEST_DIR = Path('dataset/waste-image-dataset-splitted')


CLASS_LIST = ['glass', 'organic', 'paper', 'plastic', ] #waste-image-dataset
DATA_GROUP = ['train', 'test']
SPLIT_RATIOS = {"train": 0.80, "test": 0.20}
IMG_COUNT = 1500
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

random.seed(42)

def collect_image_paths_labels(source_dir):
    file_paths = []
    labels = []
    for cls in CLASS_LIST:
        cls_path = source_dir / cls
        if not cls_path.exists():
            print(f"Warning: {cls_path} does not exist. Skipping class.")
            continue
        images = [f for f in cls_path.iterdir() if f.suffix.lower() in IMG_EXTENSIONS and f.is_file()]
        images = images[:IMG_COUNT]
        file_paths.extend(images)
        labels.extend([cls] * len(images))
    return file_paths, labels

def stratified_split_and_copy(source_dir, dest_dir):
    file_paths, labels = collect_image_paths_labels(source_dir)
    # Stratified splitting
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_RATIOS['test'], random_state=42)
    for train_idx, test_idx in splitter.split(file_paths, labels):
        split_data = {
            'train': [(file_paths[i], labels[i]) for i in train_idx],
            'test':  [(file_paths[i], labels[i]) for i in test_idx],
        }

    #Copy files to destination based on split
    for group, items in split_data.items():
        counter = {}
        for img_path, cls in tqdm(items, desc=f"Copying {group} images"):
            counter.setdefault(cls, 1)
            ext = img_path.suffix
            counter_str = str(counter[cls]).zfill(5)
            new_img_name = f"{cls}_{group}_{counter_str}{ext}"
            dest_path = dest_dir / group / cls
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_path / new_img_name)
            counter[cls] += 1

stratified_split_and_copy(SOURCE_DIR, DEST_DIR)