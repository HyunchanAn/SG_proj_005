import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder

def train():
    print("[INFO] Starting Surface Anomaly Detection Training Pipeline...")
    
    # 1. Setup Data
    # Anomalib Folder Dataset
    # We point to the root. It expects 'train' and 'test' subfolders.
    datamodule = Folder(
        name="surface",
        root="datasets/custom",
        normal_dir="train/good", 
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=32, # Optimized for RTX 5080 (was 8)
        eval_batch_size=32,  # Optimized for RTX 5080 (was 8)
        image_size=(512, 512) # High resolution for fine defect detection
    )
    
    # 2. Setup Model
    print("[INFO] Initializing PatchCore Model (Backbone: Wide ResNet 101)...")
    model = Patchcore(
        backbone="wide_resnet101_2", # Deeper model for better feature extraction (was wide_resnet50_2)
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9
    )

    # 3. Setup Engine
    print("[INFO] Initializing Engine...")
    engine = Engine(
        default_root_dir="results",
        max_epochs=1, 
        logger=False, # Disable logger to avoid symlink issues on Windows without Admin/Dev mode
    )

    # 4. Train
    print("[INFO] Beginning Training (Fitting)...")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("[SUCCESS] Training complete. Model saved in 'results/'.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")

if __name__ == "__main__":
    train()
