import sys
import importlib

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"[FAIL] {module_name} import failed: {e}")
        return False

def verify():
    print("Verifying environment setup...")
    modules = [
        'tensorflow',
        'cv2',
        'sklearn',
        'matplotlib',
        'streamlit',
        'pandas',
        'seaborn',
        'imblearn',
        'kaggle'
    ]
    
    all_pass = True
    for m in modules:
        if not check_import(m):
            all_pass = False
            
    if all_pass:
        print("\nEnvironment verification passed!")
    else:
        print("\nEnvironment verification failed. Please check installation.")

if __name__ == "__main__":
    verify()
