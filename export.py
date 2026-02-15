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
    ckpt_path = Path("results/manual/model.ckpt")
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return
    print(f"[INFO] Found manual checkpoint: {ckpt_path}")
    print(f"[INFO] Found checkpoint: {ckpt_path}")
    print("[INFO] Starting export to TorchScript (.pt)...")
    
    model = Patchcore(
        backbone="resnet18",
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=9
    )
    
    engine = Engine(logger=False)
    
    # Export
    # This will create exported_models/Patchcore/model.pt
    exported_path = engine.export(
        model=model,
        export_type="torch",
        ckpt_path=str(ckpt_path),
        export_root="exported_models",
        input_size=(512, 512) # Matching the RTX 4060 resolution
    )
    
    print(f"[SUCCESS] Exported to {exported_path}")

if __name__ == "__main__":
    export()
