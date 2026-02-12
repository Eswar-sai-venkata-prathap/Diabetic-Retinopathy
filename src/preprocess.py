import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

IMG_SIZE = (224, 224)
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# =========================
# Robust Radius Scaling
# =========================
def scale_radius(img, target_radius):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = gray[gray.shape[0] // 2, :]
    r = (x > x.mean() / 10).sum() / 2

    if r == 0:
        return cv2.resize(img, IMG_SIZE)

    scale = target_radius / r
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img


# =========================
# Circular Crop
# =========================
def circular_crop(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    result = cv2.bitwise_and(img, img, mask=mask)
    return result


# =========================
# Ben Graham Preprocessing
# =========================
def load_ben_graham(img):
    # 1️⃣ Radius normalize
    img = scale_radius(img, IMG_SIZE[0] // 2)

    # 2️⃣ Crop center square
    h, w = img.shape[:2]
    min_dim = min(h, w)
    img = img[
        (h - min_dim) // 2:(h + min_dim) // 2,
        (w - min_dim) // 2:(w + min_dim) // 2
    ]

    # 3️⃣ Resize to standard input
    img = cv2.resize(img, IMG_SIZE)

    # 4️⃣ Apply circular mask
    img = circular_crop(img)

    # 5️⃣ Ben Graham contrast enhancement
    img = cv2.addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0, 0), 10),
        -4,
        128
    )

    # 6️⃣ CLAHE for lesion visibility
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


# =========================
# Inference Preprocessing
# =========================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img = load_ben_graham(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    return img


# =========================
# Batch Processing
# =========================
def process_single_image(file_path):
    try:
        rel_path = os.path.relpath(file_path, RAW_DIR)
        save_path = os.path.join(PROCESSED_DIR, rel_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        img = cv2.imread(file_path)
        if img is None:
            return

        img = load_ben_graham(img)
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def run_preprocessing():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Scanning for images...")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
    files = []

    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(RAW_DIR, '**', ext), recursive=True)
        )

    print(f"Found {len(files)} images. Processing...")

    with ThreadPoolExecutor() as executor:
        list(executor.map(process_single_image, files))

    print("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
