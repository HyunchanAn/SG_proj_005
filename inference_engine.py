import os
# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm
from typing import Tuple, Dict, Any, Optional

# --- Anomalib / Torch ---
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

# --- SAM2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

class IntegratedEngine:
    def __init__(self, anomalib_path: str, sam2_checkpoint: str, sam2_config: Optional[str] = None):
        self.device = self._get_device()
        self.load_error = None
        
        # 1. Anomalib Load
        self.anomalib_engine = TorchInferencer(path=anomalib_path, device=self.device)
        
        # 2. SAM2 Load - Auto-detect config if not provided
        if sam2_config is None:
            sam2_config = self._guess_sam2_config(sam2_checkpoint)
            
        self.sam2_predictor = None
        if not SAM2_AVAILABLE:
            self.load_error = "SAM2 library not installed (ImportError)"
        else:
            try:
                print(f"[DEBUG] Loading SAM2 with config: {sam2_config}, checkpoint: {sam2_checkpoint}")
                sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            except Exception as e:
                import traceback
                self.load_error = f"SAM2 Load Error: {e}\n{traceback.format_exc()}"
                print(self.load_error)

    def _guess_sam2_config(self, checkpoint_path: str) -> str:
        name = os.path.basename(checkpoint_path).lower()
        if "tiny" in name: return "sam2_hiera_t.yaml"
        if "small" in name: return "sam2_hiera_s.yaml"
        if "base_plus" in name: return "sam2_hiera_b+.yaml"
        if "large" in name: return "sam2_hiera_l.yaml"
        return "sam2_hiera_t.yaml" # Default

    def _get_device(self):
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                return "cuda"
            except: pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def analyze_anomalib(self, image: Image.Image) -> Dict[str, Any]:
        """Runs PatchCore inference and returns heatmap and peak point."""
        img_arr = np.array(image.convert("RGB"))
        # Anomalib inference
        # We use a similar logic to app.py to bypass forward issues
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.anomalib_engine.model, "model"):
                prediction = self.anomalib_engine.model.model(img_tensor)
            else:
                prediction = self.anomalib_engine.model(img_tensor)
        
        # Extract results
        if isinstance(prediction, dict):
            heat_map = prediction.get("anomaly_map", None)
            pred_score = prediction.get("pred_score", None)
        else:
            heat_map = getattr(prediction, "anomaly_map", None)
            pred_score = getattr(prediction, "pred_score", None)

        if isinstance(heat_map, torch.Tensor):
            heat_map = heat_map.squeeze().cpu().numpy()
        if isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.item()

        # Find Peak Point (Max intensity in heatmap)
        # Resize heatmap to original image size for accurate coordinate
        h, w = img_arr.shape[:2]
        heatmap_resized = np.array(Image.fromarray(heat_map).resize((w, h), resample=Image.BICUBIC))
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        
        return {
            "heatmap": heatmap_resized,
            "score": pred_score,
            "peak_point": (int(peak_x), int(peak_y))
        }

    def segment_with_sam2(self, image: Image.Image, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Runs SAM2 segmentation based on points."""
        if self.sam2_predictor is None:
            print("[DEBUG] SAM2 Predictor is NONE. Skipping segmentation.")
            return None
        
        print(f"[DEBUG] Running SAM2 on point: {points[0]}")
        img_arr = np.array(image.convert("RGB"))
        self.sam2_predictor.set_image(img_arr)
        
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )
        
        if masks is not None and len(masks) > 0:
            mask_sum = np.sum(masks[0])
            print(f"[DEBUG] SAM2 Mask found! Sum of pixels: {mask_sum}")
            return masks[0]
        else:
            print("[DEBUG] SAM2 generated NO masks.")
            return None

    def create_overlay(self, image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> Image.Image:
        """Overlays a binary mask onto an image."""
        img_arr = np.array(image.convert("RGB"))
        mask_overlay = np.zeros_like(img_arr)
        mask_overlay[mask > 0] = color
        
        combined = (img_arr * (1 - alpha) + mask_overlay * alpha).astype(np.uint8)
        return Image.fromarray(combined)

    def create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray, alpha=0.4) -> Image.Image:
        """Overlays a heatmap onto an image."""
        img_arr = np.array(image.convert("RGB"))
        
        # Normalize
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        colormap = cm.get_cmap('jet')
        heatmap_colored = (colormap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
        
        overlay = (img_arr * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
