import gradio as gr
import numpy as np
from PIL import Image
import os

# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

from inference_engine import IntegratedEngine

# --- Configuration ---
ANOMALIB_MODEL = "exported_models/weights/torch/model.pt"
SAM2_CHECKPOINT = "models/sam2/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

# Load Engine
engine = None
def get_engine():
    global engine
    if engine is None:
        if os.path.exists(ANOMALIB_MODEL) and os.path.exists(SAM2_CHECKPOINT):
            engine = IntegratedEngine(ANOMALIB_MODEL, SAM2_CHECKPOINT, SAM2_CONFIG)
        else:
            print("Warning: Models not found. Engine not initialized.")
    return engine

def process_image(input_img):
    """Automatic Mode: PatchCore + SAM2 (Auto-peak)"""
    if input_img is None: return None, None, "No image uploaded."
    
    eng = get_engine()
    if eng is None: return None, None, "Models not loaded. Please check model paths."
    
    # 1. Anomalib Analysis
    pil_img = Image.fromarray(input_img)
    results = eng.analyze_anomalib(pil_img)
    
    heatmap_overlay = eng.create_heatmap_overlay(pil_img, results["heatmap"])
    
    # 2. SAM2 Automatic (using peak point)
    peak_x, peak_y = results["peak_point"]
    points = np.array([[peak_x, peak_y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32) # 1 for foreground
    
    mask = eng.segment_with_sam2(pil_img, points, labels)
    sam2_overlay = eng.create_overlay(pil_img, mask, color=(255, 0, 0)) if mask is not None else pil_img
    
    status = f"Analysis Complete.\nAnomaly Score: {results['score']:.4f}\nPeak Point: ({peak_x}, {peak_y})"
    return heatmap_overlay, sam2_overlay, status

def refine_with_points(input_img, evt: gr.SelectData):
    """Interactive Mode: User click on image."""
    if input_img is None: return None, "Upload an image first."
    
    eng = get_engine()
    if eng is None: return None, "Models not loaded."
    
    # Get click coordinates
    x, y = evt.index
    pil_img = Image.fromarray(input_img)
    
    points = np.array([[x, y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    mask = eng.segment_with_sam2(pil_img, points, labels)
    sam2_overlay = eng.create_overlay(pil_img, mask, color=(0, 255, 0)) # Green for manual
    
    return sam2_overlay, f"Manual Selection at ({x}, {y})"

# --- UI Design ---
with gr.Blocks(title="Surface Anomaly Detection (SAM2 + Gradio)") as demo:
    gr.Markdown("# 🛡️ Surface Anomaly Detection & SAM2 Segmentation")
    gr.Markdown("Anomalib(PatchCore)로 결함을 탐색하고, SAM2로 정밀하게 분할합니다.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Surface Image")
            analyze_btn = gr.Button("🚀 자동 분석 시작 (Auto Analyze)", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Anomaly Heatmap"):
                    heatmap_output = gr.Image(label="PatchCore Result")
                with gr.Tab("SAM2 Segmentation"):
                    sam2_output = gr.Image(label="SAM2 Result (Click to refine)")
                    gr.Markdown("**Tip**: 결과를 클릭하여 관심 부위를 수동으로 다시 지정할 수 있습니다.")

    # Event handlers
    analyze_btn.click(
        process_image,
        inputs=[input_image],
        outputs=[heatmap_output, sam2_output, status_text]
    )
    
    # Click to refine on the SAM2 result tab
    sam2_output.select(
        refine_with_points,
        inputs=[input_image],
        outputs=[sam2_output, status_text]
    )

if __name__ == "__main__":
    demo.launch()
