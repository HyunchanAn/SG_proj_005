import os
import re
from pathlib import Path
import anomalib.utils.path

# Monkey-patch to fix [WinError 1314] Symlink privilege error on Windows
def patched_create_versioned_dir(root_dir: str | Path) -> Path:
    version_pattern = re.compile(r"^v(\d+)$")
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    highest_version = -1
    for version_dir in root_dir.iterdir():
        if version_dir.is_dir():
            match = version_pattern.match(version_dir.name)
            if match:
                highest_version = max(highest_version, int(match.group(1)))
    new_version_dir = root_dir / f"v{highest_version + 1}"
    new_version_dir.mkdir()
    # Skip symlink_to() to avoid privilege issues on Windows
    return new_version_dir

anomalib.utils.path.create_versioned_dir = patched_create_versioned_dir

# Import Engine after patching or patch its reference
from anomalib.engine import Engine
import anomalib.engine.engine
anomalib.engine.engine.create_versioned_dir = patched_create_versioned_dir

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
    print("[INFO] Initializing Engine with Auto Acceleration...")
    engine = Engine(
        default_root_dir="results",
        max_epochs=1, 
        accelerator="auto",
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
