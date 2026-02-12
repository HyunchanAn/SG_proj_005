import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder

def train():
    print("[INFO] Starting Surface Anomaly Detection Training Pipeline (M2 Pro Optimized)...")
    
    # 1. Setup Data
    datamodule = Folder(
        name="surface",
        root="datasets/custom",
        normal_dir="train/good", 
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=16,
        eval_batch_size=16,
        image_size=(256, 256),
        task="classification"
    )
    
    # 2. Setup Model
    print("[INFO] Initializing PatchCore Model (Backbone: Wide ResNet 50)...")
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9
    )

    # 3. Setup Engine
    print("[INFO] Initializing Engine with MPS Acceleration...")
    engine = Engine(
        default_root_dir="results",
        max_epochs=1, 
        accelerator="mps",
        devices=1,
        # logger=False removed to avoid 'Console' object has no attribute '_live' error
    )

    # 4. Train
    print("[INFO] Beginning Training (Fitting)...")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("[SUCCESS] Training complete. Model saved in 'results/'.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
