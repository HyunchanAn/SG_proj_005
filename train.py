import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torch

def train():
    print("[INFO] Starting Surface Anomaly Detection Training Pipeline (RTX 4060 Optimized)...")
    
    # 1. Setup Data
    # In Anomalib 2.2.0, image_size is often handled via transforms
    datamodule = Folder(
        name="surface",
        root="datasets/custom",
        normal_dir="train/good", 
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=32,
        eval_batch_size=32,
    )
    
    # 2. Setup Model
    print("[INFO] Initializing PatchCore Model (Backbone: ResNet 18)...")
    model = Patchcore(
        backbone="resnet18",
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=9
    )

    # 3. Setup Engine
    print("[INFO] Initializing Engine with CUDA Acceleration...")
    engine = Engine(
        default_root_dir="results",
        max_epochs=1, 
        accelerator="cuda",
        devices=1,
    )
    
    # 4. Train
    print("[INFO] Beginning Training (Fitting)...")
    try:
        # We use a simple fit and manually save/export
        engine.fit(model=model, datamodule=datamodule)
        print("[SUCCESS] Training complete.")
    except Exception as e:
        if "WinError 1314" in str(e):
             print("[WARNING] Caught symlink error (WinError 1314). Continuing to export...")
        else:
            print(f"[ERROR] Training failed: {e}")
            return
    
    # Export immediately from memory
    print("[INFO] Exporting model to TorchScript (.pt)...")
    try:
        exported_path = engine.export(
            model=model,
            export_type="torch",
            export_root="exported_models",
            input_size=(512, 512)
        )
        print(f"[SUCCESS] Model exported to {exported_path}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")

if __name__ == "__main__":
    train()
