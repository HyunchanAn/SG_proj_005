import re
from pathlib import Path
from typing import Union

import anomalib.utils.path
from loguru import logger


# Monkey-patch to fix [WinError 1314] Symlink privilege error on Windows
def patched_create_versioned_dir(root_dir: Union[str, Path]) -> Path:
    """Monkey-patched versioned directory creator to bypass symlink privileges on Windows.
    
    Args:
        root_dir: Root directory to house versioned subdirectories.
        
    Returns:
        The newly created version Path.
    """
    version_pattern = re.compile(r"^v(\d+)$")
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    highest_version = -1
    
    try:
        for version_dir in root_dir.iterdir():
            if version_dir.is_dir():
                match = version_pattern.match(version_dir.name)
                if match:
                    highest_version = max(highest_version, int(match.group(1)))
    except Exception as e:
        logger.warning(f"Failed to read existing versions in {root_dir}: {e}")
        
    new_version_dir = root_dir / f"v{highest_version + 1}"
    new_version_dir.mkdir()
    # Skip symlink_to() to avoid privilege issues on Windows
    logger.debug(f"Versioned directory created safely without symlink: {new_version_dir}")
    return new_version_dir

anomalib.utils.path.create_versioned_dir = patched_create_versioned_dir

# Import Engine after patching or patch its reference
import anomalib.engine.engine
from anomalib.engine import Engine

anomalib.engine.engine.create_versioned_dir = patched_create_versioned_dir

from anomalib.data import Folder
from anomalib.models import Patchcore


def train() -> None:
    """Runs the high-performance Surface Anomaly Detection Training Pipeline.
    
    This setups custom Folder datamodule, initializes Wide-ResNet50 PatchCore model,
    configures PyTorch Lightning Engine with auto-acceleration and triggers training.
    """
    logger.info("Starting Surface Anomaly Detection Training Pipeline (M2 Pro / RTX 5080 Optimized)...")
    
    # 1. Setup Data
    logger.info("Setting up Folder data module with custom surface datasets...")
    try:
        datamodule = Folder(
            name="surface",
            root="datasets/custom",
            normal_dir="train/good", 
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            train_batch_size=16,
            eval_batch_size=16,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Anomalib Folder DataModule: {e}")
        raise

    # 2. Setup Model
    logger.info("Initializing PatchCore Model (Backbone: Wide ResNet 50)...")
    try:
        model = Patchcore(
            backbone="wide_resnet50_2",
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9
        )
    except Exception as e:
        logger.error(f"Failed to initialize Patchcore model: {e}")
        raise

    # 3. Setup Engine
    logger.info("Initializing Anomalib Engine with Auto Acceleration...")
    try:
        engine = Engine(
            default_root_dir="results",
            max_epochs=1, 
            accelerator="auto",
            devices=1,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Anomalib Engine: {e}")
        raise

    # 4. Train
    logger.info("Beginning Training (Fitting)...")
    try:
        engine.fit(model=model, datamodule=datamodule)
        logger.success("Training completed successfully. Model saved in 'results/'.")
    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    train()
