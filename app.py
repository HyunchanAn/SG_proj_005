import streamlit as st
import os

# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# PyTorch 2.6+ security fix: Monkeypatch torch.load to allow loading legacy checkpoints
orig_load = torch.load
def hooked_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = hooked_load

# Fix for loading CUDA-trained models in CPU-only environment
# The torchmetrics Metric._apply method tries to create a dummy tensor on its saved _device
# which throws an AssertionError if it was saved as "cuda:0" but PyTorch is CPU-only.
try:
    import torchmetrics
    orig_metric_apply = torchmetrics.Metric._apply
    def patched_metric_apply(self, fn):
        if not torch.cuda.is_available() and getattr(self, "_device", None) is not None:
            if "cuda" in str(self._device):
                self._device = torch.device("cpu")
        return orig_metric_apply(self, fn)
    torchmetrics.Metric._apply = patched_metric_apply
except ImportError:
    pass

import matplotlib.pyplot as plt

# Translations
TRANS = {
    "en": {
        "sidebar_header": "Configuration",
        "language": "Language",
        "model_select": "Select Exported Model (.pt)",
        "threshold": "Anomaly Threshold",
        "title": "Surface Anomaly Detection System",
        "upload_label": "Upload Surface Image",
        "button_analyze": "Analyze Defect",
        "error_no_lib": "Anomalib library is not installed in this environment.",
        "spinner_analyzing": "Analyzing surface texture...",
        "error_select_model": "Please select a valid .pt file from the sidebar.",
        "result_label": "Result",
        "normal": "Normal",
        "abnormal": "Abnormal",
        "col_orig": "Original Image",
        "col_heatmap": "Anomaly Heatmap",
        "col_overlay": "Overlay Result",
        "waiting": "Waiting for image upload...",
        "info_upload": "Upload an image and click 'Analyze Defect' to see the heatmap.",
        "status_ready": "System Status: Ready",
        "system_ready": "System Status: Ready",
        "footer_warning": "⚠️ Anomalib not detected."
    },
    "ko": {
        "sidebar_header": "설정",
        "language": "언어 (Language)",
        "model_select": "모델 선택 (.pt)",
        "threshold": "이상치 임계값 (Threshold)",
        "title": "표면 이상 탐지 시스템",
        "upload_label": "표면 이미지 업로드",
        "button_analyze": "결함 분석 시작",
        "error_no_lib": "Anomalib 라이브러리가 설치되어 있지 않습니다.",
        "spinner_analyzing": "표면 텍스처 분석 중...",
        "error_select_model": "사이드바에서 올바른 .pt 파일을 선택해주세요.",
        "result_label": "판정 결과",
        "normal": "정상 (Normal)",
        "abnormal": "비정상/불량 (Abnormal)",
        "col_orig": "원본 이미지",
        "col_heatmap": "이상 히트맵",
        "col_overlay": "오버레이 결과",
        "waiting": "이미지를 업로드해주세요...",
        "info_upload": "이미지를 업로드하고 '결함 분석 시작'을 눌러주세요.",
        "status_ready": "시스템 상태: 준비됨",
        "system_ready": "시스템 상태: 준비 완료",
         "footer_warning": "⚠️ Anomalib가 감지되지 않았습니다."
    }
}

# NOTE: In a real environment, you would import anomalib inference classes.
# For this 'scaffolding' phase, we will simulate the import to avoid crashing 
# if anomalib isn't fully installed on the planning machine, 
# BUT we write the Real code for the home machine.

# --- Streamlit Cloud / Debian Trixie (3.13) Missing System Libs (libGL.so.1) Bypass ---
import sys
try:
    import cv2
except (ImportError, Exception):
    # Mock cv2 if it fails to import or provides broken binary extensions
    from unittest.mock import MagicMock
    cv2_mock = MagicMock()
    # Mock some common constants/functions to prevent basic crashes during import
    cv2_mock.NORM_MINMAX = 32
    cv2_mock.CV_8U = 0
    cv2_mock.COLORMAP_JET = 2
    cv2_mock.COLOR_BGR2RGB = 4
    sys.modules["cv2"] = cv2_mock
    import cv2 

ANOMALIB_ERROR = None
try:
    # 1. First attempt robust import of TorchInferencer
    from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
    ANOMALIB_AVAILABLE = True
except Exception as e:
    # 2. If it still fails, the error message will be recorded
    ANOMALIB_AVAILABLE = False
    ANOMALIB_ERROR = str(e)

# Custom read_image that uses PIL as a fallback if cv2 is mocked
try:
    from anomalib.data.utils import read_image as original_read_image
except Exception:
    original_read_image = None

def read_image(path):
    if original_read_image and not isinstance(sys.modules.get("cv2"), MagicMock):
        try:
            return original_read_image(path)
        except Exception:
            pass
    # Fallback to PIL
    img = Image.open(path).convert("RGB")
    return np.array(img)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Surface Anomaly Detection", layout="wide")

# Sidebar - Language Selection
st.sidebar.header("Language / 언어")
lang_code = st.sidebar.radio("Select Language", ["Korean (한국어)", "English"], index=0)
lang = "ko" if "Korean" in lang_code else "en"
t = TRANS[lang]

st.title(f"🛡️ {t['title']}")

# Sidebar for Model Selection
st.sidebar.header(t["sidebar_header"])
# Use exported models folder
model_dir = "exported_models"
# Finder for .pt files (TorchScript)
ckpt_files = list(Path(model_dir).rglob("*.pt")) if os.path.exists(model_dir) else []
ckpt_files = [str(p) for p in ckpt_files]

selected_ckpt = st.sidebar.selectbox(t["model_select"], ["Auto-detect"] + ckpt_files)

# Threshold slider (optional manual override)
threshold = st.sidebar.slider(t["threshold"], min_value=0.0, max_value=1.0, value=0.5)

st.divider()

col1, col2 = st.columns(2)

uploaded_file = st.sidebar.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Original
    image = Image.open(uploaded_file)
    with col1:
        st.subheader(t["col_orig"])
        st.image(image, use_container_width=True)

    # Analyze Button
    if st.sidebar.button(t["button_analyze"]):
        if not ANOMALIB_AVAILABLE:
            st.error(t["error_no_lib"])
        else:
            with st.spinner(t["spinner_analyzing"]):
                try:
                    # Load Model
                    # We assume the config is at 'configs/surface_config.yaml'
                    # and the user selected a valid checkpoint.
                    config_path = "configs/surface_config.yaml"
                    
                    if selected_ckpt == "Auto-detect" or not os.path.exists(selected_ckpt):
                         st.error(t["error_select_model"])
                    else:
                        # Device detection with robust CPU fallback
                        # CUDA may be installed but no actual GPU driver present (e.g. Streamlit Cloud)
                        device = "cpu"
                        try:
                            if torch.cuda.is_available():
                                torch.zeros(1).cuda()  # Force init to verify driver exists
                                device = "cuda"
                            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                device = "mps"
                        except Exception:
                            device = "cpu"
                        
                        inferencer = TorchInferencer(path=selected_ckpt, device=device)
                        
                        # Manual Preprocessing: Bypass potentially buggy v2 transforms in the exported model
                        # Standard torchvision transforms (v1) are more stable on MPS/CPU
                        from torchvision import transforms
                        preprocess = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        img_arr = np.array(image.convert("RGB"))
                        img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                        
                        # Inference
                        # We try to bypass the 'forward' of the exported model which often includes 
                        # the problematic transforms. We call the internal model directly.
                        with torch.no_grad():
                            if hasattr(inferencer.model, "model"):
                                prediction = inferencer.model.model(img_tensor)
                            else:
                                prediction = inferencer.model(img_tensor)
                        
                        # Result handling
                        # Anomalib 1.1+ InferenceModel returns a namespace or dict-like object
                        # We need to extract 'anomaly_map' and 'pred_score'
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
                            
                        # Bug Fix: Use user-defined threshold instead of internal model default
                        is_abnormal = pred_score > threshold
                        pred_label_str = t["abnormal"] if is_abnormal else t["normal"]
                        
                        # Display Text Result
                        if is_abnormal:
                            st.error(f"{t['result_label']}: {pred_label_str} (Score: {pred_score:.2f})")
                        else:
                            st.success(f"{t['result_label']}: {pred_label_str} (Score: {pred_score:.2f})")

                        # Visualization (Bypassing OpenCV)
                        # 1. Normalize heatmap to 0-1
                        heatmap_min, heatmap_max = heat_map.min(), heat_map.max()
                        if heatmap_max > heatmap_min:
                            heatmap_norm = (heat_map - heatmap_min) / (heatmap_max - heatmap_min)
                        else:
                            heatmap_norm = np.zeros_like(heat_map)
                        
                        # 2. Apply ColorMap (Jet) using matplotlib
                        # cm.jet() returns RGBA, we take RGB
                        colormap = cm.get_cmap('jet')
                        heatmap_colored = (colormap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
                        
                        # 3. Resize original image to match heatmap dimensions
                        heatmap_h, heatmap_w = heatmap_colored.shape[:2]
                        original_resized = np.array(Image.fromarray(img_arr).resize((heatmap_w, heatmap_h), resample=Image.BICUBIC))
                        
                        # 4. Overlay original with heatmap
                        overlay = (original_resized * 0.6 + heatmap_colored * 0.4).astype(np.uint8)
                        
                        with col2:
                            st.subheader(t["col_overlay"])
                            st.image(overlay, caption="Heatmap Overlay", use_container_width=True)

                except Exception:
                    import traceback
                    st.error(f"Error during inference:\n{traceback.format_exc()}")
    else:
        with col2:
             st.info(t["info_upload"])
else:
    with col1:
        st.info(t["waiting"])

st.sidebar.markdown("---")
st.sidebar.info(t["system_ready"])
if not ANOMALIB_AVAILABLE:
    st.sidebar.warning(t["footer_warning"])
    if ANOMALIB_ERROR:
        st.sidebar.error(f"Import error: {ANOMALIB_ERROR}")
