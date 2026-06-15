import os

import streamlit as st

# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

from pathlib import Path

import numpy as np
from PIL import Image

from inference_engine import IntegratedEngine

# Translations
TRANS = {
    "en": {
        "sidebar_header": "Configuration",
        "language": "Language",
        "model_select": "Select PatchCore Model (.pt)",
        "sam2_select": "Select SAM2 Model (.pt)",
        "threshold": "Anomaly Threshold",
        "title": "Surface Anomaly Detection System",
        "upload_label": "Upload Surface Image",
        "button_analyze": "Run Full Analysis",
        "spinner_analyzing": "Analyzing surface & segmenting...",
        "result_label": "Result",
        "normal": "Normal",
        "abnormal": "Abnormal",
        "col_orig": "Original Image",
        "col_heatmap": "Anomaly Heatmap",
        "col_sam2": "SAM2 Refined Boundary",
        "waiting": "Waiting for image upload...",
        "info_upload": "Upload an image and click 'Run Full Analysis'.",
        "system_ready": "System Status: Ready",
        "error_no_model": "Models not found. Please check paths.",
    },
    "ko": {
        "sidebar_header": "설정",
        "language": "언어 (Language)",
        "model_select": "PatchCore 모델 선택 (.pt)",
        "sam2_select": "SAM2 모델 선택 (.pt)",
        "threshold": "이상치 임계값 (Threshold)",
        "title": "표면 이상 탐지 시스템",
        "upload_label": "표면 이미지 업로드",
        "button_analyze": "전체 분석 시작",
        "spinner_analyzing": "표면 분석 및 세그멘테이션 중...",
        "result_label": "판정 결과",
        "normal": "정상 (Normal)",
        "abnormal": "비정상/불량 (Abnormal)",
        "col_orig": "원본 이미지",
        "col_heatmap": "이상 히트맵",
        "col_sam2": "SAM2 정밀 경계",
        "waiting": "이미지를 업로드해주세요...",
        "info_upload": "이미지를 업로드하고 '전체 분석 시작'을 눌러주세요.",
        "system_ready": "시스템 상태: 준비 완료",
        "error_no_model": "모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.",
    },
}


@st.cache_resource
def load_engine_v3(anomalib_path, sam2_checkpoint):
    return IntegratedEngine(anomalib_path, sam2_checkpoint)


st.set_page_config(page_title="Surface Anomaly Detection", layout="wide")

# Sidebar - Language Selection
st.sidebar.header("Language / 언어")
lang_code = st.sidebar.radio("Select Language", ["Korean (한국어)", "English"], index=0)
lang = "ko" if "Korean" in lang_code else "en"
t = TRANS[lang]

st.title(f"🛡️ {t['title']}")

# Sidebar for Model Selection
st.sidebar.header(t["sidebar_header"])
model_dir = "exported_models"
ckpt_files = [str(p) for p in Path(model_dir).rglob("*.pt")] if os.path.exists(model_dir) else []

selected_ckpt = st.sidebar.selectbox(t["model_select"], ckpt_files)

sam2_dir = os.path.join("models", "sam2")
sam2_files = [str(p) for p in Path(sam2_dir).glob("*.pt")] if os.path.exists(sam2_dir) else []
selected_sam2 = st.sidebar.selectbox(t["sam2_select"], sam2_files)

threshold = st.sidebar.slider(t["threshold"], min_value=0.0, max_value=1.0, value=0.5)

st.divider()

col1, col2, col3 = st.columns(3)

uploaded_file = st.sidebar.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader(t["col_orig"])
        st.image(image, use_container_width=True)

    if st.sidebar.button(t["button_analyze"]):
        if not selected_ckpt or not selected_sam2:
            st.error(t["error_no_model"])
        else:
            with st.spinner(t["spinner_analyzing"]):
                try:
                    engine = load_engine_v3(selected_ckpt, selected_sam2)

                    # 1. PatchCore Analysis
                    results = engine.analyze_anomalib(image)

                    # 2. SAM2 Automatic Refinement
                    peak_x, peak_y = results["peak_point"]
                    points = np.array([[peak_x, peak_y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)

                    mask = engine.segment_with_sam2(image, points, labels)

                    # Visualization
                    heatmap_overlay = engine.create_heatmap_overlay(image, results["heatmap"])
                    sam2_overlay = engine.create_overlay(image, mask) if mask is not None else image

                    # Status & Result
                    is_abnormal = results["score"] > threshold
                    pred_label_str = t["abnormal"] if is_abnormal else t["normal"]

                    if is_abnormal:
                        st.error(f"{t['result_label']}: {pred_label_str} (Score: {results['score']:.2f})")
                    else:
                        st.success(f"{t['result_label']}: {pred_label_str} (Score: {results['score']:.2f})")

                    with col2:
                        st.subheader(t["col_heatmap"])
                        st.image(heatmap_overlay, use_container_width=True)

                    with col3:
                        st.subheader(t["col_sam2"])
                        st.image(sam2_overlay, use_container_width=True)
                        st.caption(f"Refined from peak point: ({peak_x}, {peak_y})")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.exception(e)
    else:
        with col2:
            st.info(t["info_upload"])
else:
    with col1:
        st.info(t["waiting"])

st.sidebar.markdown("---")
st.sidebar.info(t["system_ready"])
