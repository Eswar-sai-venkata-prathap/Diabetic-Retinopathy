import os
import shutil
import kagglehub
import subprocess

def download_data():
    print("Attempting to download IDRiD dataset...")
    
    # Try kagglehub first
    try:
        print("Using kagglehub...")
        path = kagglehub.dataset_download("mariaherrerot/idrid-dataset")
        print(f"Dataset downloaded to cache: {path}")
        
        # Move/Copy to data/raw
        target_dir = os.path.join("data", "raw")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # Check if path is a file or dir
        if os.path.isdir(path):
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        else:
            shutil.copy2(path, target_dir)
            
        print(f"Dataset moved to {target_dir}")
        return
    except Exception as e:
        print(f"kagglehub failed: {e}")
    
    # Fallback to kaggle CLI
    try:
        print("Using kaggle CLI...")
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mariaherrerot/idrid-dataset', '-p', 'data/raw', '--unzip'], check=True)
        print("Download complete via CLI.")
    except Exception as e:
        print(f"kaggle CLI failed: {e}")
        print("Please ensure you have authenticated with Kaggle (placed kaggle.json in ~/.kaggle/).")

if __name__ == "__main__":
    download_data()
