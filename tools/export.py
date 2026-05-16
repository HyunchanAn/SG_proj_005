import torch
# Monkeypatch torch.load to disable weights_only=True security check for Anomalib compatibility
orig_load = torch.load
def hooked_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = hooked_load

from anomalib.engine import Engine
from anomalib.models import Patchcore
from pathlib import Path

def export():
    # Find the latest ckpt
    results_dir = Path("results/Patchcore/surface")
    if not results_dir.exists():
        print("Results directory not found.")
        return

    # Sort by version number
    version_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not version_dirs:
        print("No model version directory found.")
        return
        
    versions = sorted(version_dirs, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else -1, reverse=True)
    
    ckpt_files = list(versions[0].glob("weights/lightning/*.ckpt"))
    if not ckpt_files:
        print(f"No checkpoint found in {versions[0]}")
        return
        
    ckpt_path = ckpt_files[0]
    print(f"[INFO] Found checkpoint: {ckpt_path}")
    print("[INFO] Starting export to TorchScript (.pt)...")
    
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9
    )
    
    engine = Engine()
    
    # Export
    # This will create exported_models/Patchcore/model.pt
    exported_path = engine.export(
        model=model,
        export_type="torch",
        ckpt_path=str(ckpt_path),
        export_root="exported_models",
        input_size=(256, 256) # Matching the M2 Pro optimized resolution
    )
    
    print(f"[SUCCESS] Exported to {exported_path}")

if __name__ == "__main__":
    export()
