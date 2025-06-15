from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#Set directories
DATASET_DIR = Path("dataset/waste-image-dataset-splitted")
OUTPUT_DIR = Path("dataset/waste-image-dataset-extracted")

#Preprocessing parameters
IMG_SIZE = (256, 256) #64x64, 128x128, 256x256,
COLOR = cv2.COLOR_BGR2GRAY #color gray required for LBP
IS_EQUALIZED = False #To improve contrast

#Set LBP parameters
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

#Set HOG parameters #72 features
HOG_PIXELS_PER_CELL = (64, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 6

#Convert class names to numbers ex: {'cardboard': 0, 'clothes': 1, ...}
class_names = sorted([p.name for p in (DATASET_DIR / "train").iterdir() if p.is_dir()])
class_map = {name: idx for idx, name in enumerate(class_names)}

def pre_process_image(image_path):
    image = cv2.imread(str(image_path)) #Load image
    if image is None: #Check if cv2 is able to read the image
        raise ValueError(f"CV2 could not read image: {image_path}")
    
    #Preprocess image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, IMG_SIZE)
    if IS_EQUALIZED:
        processed_image = cv2.equalizeHist(resized_image)  # Histogram equalization
    else:
        processed_image = resized_image
    
    return processed_image

def extract_lbp_feature(image_path):
    image = pre_process_image(image_path)
    lbp_features = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
    n_bins = int(lbp_features.max() + 1)
    hist, bin_edges = np.histogram(lbp_features.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) #normalize
    return hist

def extract_hog_feature(image_path):
    image = pre_process_image(image_path)
    hog_features = hog(image, orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm='L2-Hys', feature_vector=True)
    return hog_features

def process_folder(folder, is_combined):
    feature_list = []
    class_list = []
    image_list = list(folder.rglob('*.*'))

    for image_path in tqdm(image_list, desc=f"Extracting features in {folder.name} folder"):
        #Set folder parent directory as the class label
        class_name = image_path.parent.name
        class_id = class_map.get(class_name)
        
        if class_id is None: #Check if class is present
            print(f"Unknown class: {class_name} -- Skipping")
            continue

        try:
            if is_combined:
                lbp_features = extract_lbp_feature(image_path)
                hog_features = extract_hog_feature(image_path)
                feature = np.concatenate([lbp_features, hog_features])
            else: #use LBP as default
                feature = extract_lbp_feature(image_path)

            feature_list.append(feature)
            class_list.append(class_id)
        except Exception as e:
            print(f"Error encountered in extracting feature {image_path}: {e}")

    return feature_list, class_list

def extract_feature(dataset_dir, output_dir, is_combined):
    for data_split in ['train', 'test']:
        folder = dataset_dir / data_split
        features, labels = process_folder(folder, is_combined)

        if features:
            features_count = len(features[0])
            output_dir.mkdir(parents=True, exist_ok=True) #create output_dir
            data_frame = pd.DataFrame(features, columns=[f"F{i+1}" for i in range(len(features[0]))])
            data_frame['Class'] = labels
            filename = f"{data_split}_lbp_hog_discrete_{features_count}.csv" if is_combined else f"{data_split}_lbp_{features_count}.csv"
            data_frame.to_csv(output_dir / filename, index=False)
            print(f"Extracted feature saved in {str(output_dir)} with file: {filename}")
        else:
            print(f"No features extracted for {data_split}")

def extract_feature_old(dataset_dir, output_dir):
    for data_split in ['train', 'test']:
        folder = dataset_dir / data_split
        features, labels = process_folder(folder)

        if features:
            features_count = len(features[0])
            output_dir.mkdir(parents=True, exist_ok=True) #create output_dir
            data_frame = pd.DataFrame(features, columns=[f"F{i+1}" for i in range(len(features[0]))])
            data_frame['Class'] = labels
            filename = f"{data_split}_lbp_{features_count}A.csv"
            data_frame.to_csv(output_dir / filename, index=False)
            print(f"Extracted feature saved in {str(output_dir)} with file: {filename}")
        else:
            print(f"No features extracted for {data_split}")

def extract_features_with_pca(dataset_dir, output_dir, is_combined):
    scaler = StandardScaler()
    pca = PCA(n_components=15)
    for data_split in ['train', 'test']:
        folder = dataset_dir / data_split
        features, labels = process_folder(folder, is_combined)

        if features:
            features = np.array(features)
            
            if data_split == 'train':
                scaled_features = scaler.fit_transform(features)
                features = pca.fit_transform(scaled_features)
            else:
                scaled_features = scaler.transform(features)
                features = pca.transform(scaled_features)

            features_count = len(features[0])
            output_dir.mkdir(parents=True, exist_ok=True) #create output_dir
            data_frame = pd.DataFrame(features, columns=[f"F{i+1}" for i in range(features_count)])
            data_frame['Class'] = labels
            #Save to csv with name formatted name ex: train_lbp_72.csv
            filename = f"{data_split}_lbp_hog_pca_{features_count}.csv"
            data_frame.to_csv(output_dir / filename, index=False)
            print(f"Extracted feature saved in {str(output_dir)} with file: {filename}")
        else:
            print(f"No features extracted for {data_split}")

#Run extract features in Train and Test folder
#extract_feature(DATASET_DIR, OUTPUT_DIR, True)
extract_features_with_pca(DATASET_DIR, OUTPUT_DIR, True)

#Save label map for reference
class_df = pd.DataFrame(list(class_map.items()), columns=["class_name", "class_id"])
class_df.to_csv(OUTPUT_DIR / "class_map.csv", index=False)

print("Feature Extraction using LBP done.")