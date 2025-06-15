import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

csv_path = "dataset/waste-image-dataset-extracted/train_lbp_hog_pca_15.csv"
target_column = "Class"

df = pd.read_csv(csv_path)
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
expert_knowledge = (mi_scores / np.max(mi_scores)).tolist()

print(f"expert_knowledge: {expert_knowledge}")
