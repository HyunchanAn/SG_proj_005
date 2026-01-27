import streamlit as st
import os

# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

# Translations
TRANS = {
    "en": {
        "sidebar_header": "Configuration",
        "title": "Surface Inspection",
        "model_select": "Model (.pt)",
        "threshold": "Sensitivity",
        "upload_label": "📸 Upload or Take Photo",
        "error_no_lib": "Anomalib not installed.",
        "spinner": "Analyzing...",
        "error_select_model": "Select a model in sidebar.",
        "normal": "PASS (OK)",
        "abnormal": "FAIL (NG)",
        "col_orig": "Original",
        "col_result": "Analysis Result",
        "msg_upload": "Upload an image to start analysis."
    },
    "ko": {
        "sidebar_header": "환경 설정",
        "title": "표면 결함 검사기",
        "model_select": "모델 선택 (.pt)",
        "threshold": "민감도 (Threshold)",
        "upload_label": "📸 사진 촬영 또는 업로드",
        "error_no_lib": "Anomalib 라이브러리 없음",
        "spinner": "AI 분석 중...",
        "error_select_model": "사이드바에서 모델을 선택하세요.",
        "normal": "정상 (PASS)",
        "abnormal": "불량 (FAIL)",
        "col_orig": "원본 사진",
        "col_result": "분석 결과",
        "msg_upload": "위 버튼을 눌러 사진을 찍거나 업로드하세요."
    }
}

# --- Import Simulation ---
try:
    from anomalib.deploy import TorchInferencer
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False

# --- Page Config ---
# 'wide' 모드를 쓰면 PC에서는 넓게, 모바일에서는 꽉 차게 보입니다.
st.set_page_config(page_title="Surface AI", layout="wide")

# --- 1. Settings (Mobile Friendly) ---
# 사이드바 대신 Expander를 사용하여 모바일에서 설정 접근성을 높입니다.
# Label must be static or bilingual because 't' is not defined yet.
with st.expander("⚙️ Settings / 환경 설정", expanded=False):
    lang_code = st.radio("Language / 언어", ["Korean", "English"], index=0, horizontal=True)
    
    st.markdown("---")
    
    # Define t early inside the loop or after? 
    # Actually, we need 't' for the inputs below (model_select labels etc) if we want them localized immediately.
    # So let's define 't' right here.
    lang = "ko" if "Korean" in lang_code else "en"
    t = TRANS[lang]

    model_dir = "exported_models"
    ckpt_files = list(Path(model_dir).rglob("*.pt")) if os.path.exists(model_dir) else []
    ckpt_files = [str(p) for p in ckpt_files]
    selected_ckpt = st.selectbox(t["model_select"], ["Auto-detect"] + ckpt_files)
    
    threshold = st.slider(t["threshold"], 0.0, 1.0, 0.15)

# --- 2. Main Header ---
# Now 't' is defined
st.title(f"{t['title']}")

# --- 2. Input Area ---
uploaded_file = st.file_uploader(
    t["upload_label"], 
    type=["jpg", "png", "jpeg"], 
    label_visibility="collapsed"
)

# --- 3. Analysis & Layout ---
if uploaded_file is not None:
    if not ANOMALIB_AVAILABLE:
        st.error(t["error_no_lib"])
    else:
        if selected_ckpt == "Auto-detect" and not ckpt_files:
             st.error(t["error_select_model"])
        else:
            model_path = ckpt_files[0] if selected_ckpt == "Auto-detect" else selected_ckpt
            
            with st.spinner(t["spinner"]):
                try:
                    # Inference
                    image = Image.open(uploaded_file)
                    inferencer = TorchInferencer(path=model_path)
                    img_arr = np.array(image)
                    prediction = inferencer.predict(image=img_arr)
                    
                    # Process Results
                    heat_map = prediction.anomaly_map
                    if isinstance(heat_map, torch.Tensor):
                        heat_map = heat_map.squeeze().cpu().numpy()
                    
                    pred_score = prediction.pred_score
                    if isinstance(pred_score, torch.Tensor):
                        pred_score = pred_score.item()
                    
                    is_abnormal = pred_score > threshold
                    
                    # --- Result Banner ---
                    if is_abnormal:
                        st.error(f"🚫 {t['abnormal']} (Score: {pred_score:.2f})", icon="🚫")
                    else:
                        st.success(f"✅ {t['normal']} (Score: {pred_score:.2f})", icon="✅")
                    
                    # Visualization
                    heatmap_norm = cv2.normalize(heat_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    original_resized = cv2.resize(img_arr, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
                    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
                    
                    # --- Responsive Layout ---
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"🖼️ {t['col_orig']}")
                        st.image(image, use_container_width=True)
                        
                    with col2:
                        st.subheader(f"🔥 {t['col_result']}")
                        tab_a, tab_b = st.tabs(["Overlay", "Heatmap Only"])
                        with tab_a:
                            st.image(overlay, use_container_width=True)
                        with tab_b:
                            st.image(heatmap_colored, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info(t["msg_upload"])