import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

IMG_SIZE = (224, 224)
PROCESSED_DIR = os.path.join("data", "processed")
LABELS_PATH = os.path.join("data", "raw", "idrid_labels.csv")
CACHE_DIR = os.path.join("data", "cache")

def cache_data():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    print(f"Loading labels from {LABELS_PATH}...")
    df = pd.read_csv(LABELS_PATH)
    
    # Map paths
    valid_exts = ['.jpg', '.jpeg', '.png', '.tif']
    def find_path(row):
        base_path = os.path.join(PROCESSED_DIR, "Imagenes", "Imagenes")
        for ext in valid_exts:
            p = os.path.join(base_path, row['id_code'] + ext)
            if os.path.exists(p): return p
        return None

    df['path'] = df.apply(find_path, axis=1)
    df = df.dropna(subset=['path'])
    
    print(f"Found {len(df)} images. Loading into memory...")
    
    X = []
    y = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['path']
        label = row['diagnosis']
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        X.append(img)
        y.append(label)
        
    X = np.array(X, dtype=np.uint8) # Save as uint8 to save space, normalize in generator
    y = np.array(y, dtype=np.int8)
    
    print(f"Saving to {CACHE_DIR}...")
    np.save(os.path.join(CACHE_DIR, 'X.npy'), X)
    np.save(os.path.join(CACHE_DIR, 'y.npy'), y)
    
    print("Caching complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

if __name__ == "__main__":
    cache_data()
