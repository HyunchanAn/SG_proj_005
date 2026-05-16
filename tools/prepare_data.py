import os
import shutil
import random
import argparse
import urllib.request
import zipfile
from pathlib import Path
import ssl

# Configuration
DATASET_ROOT = Path("datasets/custom")
TRAIN_GOOD_DIR = DATASET_ROOT / "train" / "good"
TEST_GOOD_DIR = DATASET_ROOT / "test" / "good"
TEST_BAD_DIR = DATASET_ROOT / "test" / "bad"

# KolektorSDD unofficial direct link (Best effort)
# Backup: https://www.vicos.si/resources/kolektorsdd/
SAMPLE_DATA_URL = "http://box.vicos.si/skokec/gostop/KolektorSDD.zip" 

def create_structure():
    """Creates the necessary folder structure."""
    for d in [TRAIN_GOOD_DIR, TEST_GOOD_DIR, TEST_BAD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created directory structure at {DATASET_ROOT}")

def organize_user_data(source_dir):
    """
    Distributes user images from a source source_dir into train/test splits.
    Assumption: source_dir contains only 'good' (normal) images.
    """
    source_path = Path(source_dir)
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.bmp"))
    
    if not images:
        print("[WARN] No images found in source directory.")
        return

    random.shuffle(images)
    
    # 80/20 Split
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    
    print(f"[INFO] Found {len(images)} images. Moving to dataset folders...")
    
    for img in train_imgs:
        target = TRAIN_GOOD_DIR / img.name
        if not target.exists():
            shutil.copy(img, target)
        
    for img in test_imgs:
        target = TEST_GOOD_DIR / img.name
        if not target.exists():
            shutil.copy(img, target)
        
    print("[SUCCESS] Data organized.")

def download_kolektor_sdd():
    """Downloads and extracts KolektorSDD dataset."""
    print(f"[INFO] Attempting to download KolektorSDD from {SAMPLE_DATA_URL}...")
    
    zip_path = Path("KolektorSDD.zip")
    extract_path = Path("datasets/raw_kolektor")
    
    # SSL context to avoid certificate errors on some machines
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        if not zip_path.exists():
            with urllib.request.urlopen(SAMPLE_DATA_URL, context=ctx) as u, open(zip_path, 'wb') as f:
                shutil.copyfileobj(u, f)
            print("[INFO] Download complete.")
        else:
            print("[INFO] Zip file already exists, skipping download.")
            
        print("[INFO] Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print(f"[INFO] Extracted to {extract_path}")
        
        # Organize extracted data
        # KolektorSDD structure usually is: kos01/... kos02/...
        # We need to find 'good' images (usually without label in name or specific folder)
        # For simplicity, we will grab all images from the first few folders as 'normal'
        # WARNING: Kolektor images have defects. We need to be careful.
        # Usually images defined as 'ok' are marked or we just use the background.
        # But for this demo, let's look for files that DON'T have 'label' in name (masks).
        
        all_images = list(extract_path.rglob("*.jpg")) + list(extract_path.rglob("*.png"))
        # Filter out masks (files with '_label' or similar)
        # Kolektor format: {ID}.jpg and {ID}_label.bmp
        valid_images = [img for img in all_images if "_label" not in img.name and "_plabel" not in img.name]
        
        # Heuristic: Kolektor has defects. To get "Normal" data, we might need manual sorting.
        # However, many items are clean.
        # Let's take the first 50 valid images and assume they are normal for the DEMO.
        # Real-world: Must verify.
        
        print(f"[INFO] Found {len(valid_images)} candidate images. Selecting 50 for 'Normal' training...")
        
        raw_dir = Path("datasets/raw_images")
        raw_dir.mkdir(exist_ok=True)
        
        count = 0
        for img in valid_images:
            if count >= 50: break
            shutil.copy(img, raw_dir / f"kolektor_{count}.jpg")
            count += 1
            
        print(f"[SUCCESS] Prepared {count} images in 'datasets/raw_images'. Now organizing...")
        organize_user_data(raw_dir)
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print(f"[TIP] Please manually download from https://www.vicos.si/resources/kolektorsdd/ and place images in datasets/raw_images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Surface Data")
    parser.add_argument("--download", action="store_true", help="Download Sample KolektorSDD dataset")
    args = parser.parse_args()

    create_structure()
    
    if args.download:
        download_kolektor_sdd()
    
    # Check if user has put raw images somewhere to organize
    raw_dir = Path("datasets/raw_images")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print("[INFO] datasets/raw_images found. Checking organization...")
        organize_user_data(raw_dir)
    else:
        # Create raw dir for user convenience
        raw_dir.mkdir(exist_ok=True)
        if not args.download:
            print(f"[TIP] Put your 50+ normal surface photos in '{raw_dir}' and run 'python prepare_data.py' to auto-split them.")
            print(f"[TIP] Or run 'python prepare_data.py --download' to get a sample dataset.")
