# 표면 이상 탐지 시스템 요구사항 (Surface Anomaly Detection Requirements)

## 시스템 요구사항 (System Requirements)
- **운영체제 (OS)**: macOS 14+ (Silicon) 권장, Linux, Windows 10/11
- **Hardware**:
    - **Apple**: M1/M2/M3 등 Apple Silicon 칩셋 (MPS 가속 사용)
    - **NVIDIA**: RTX 3060/4060 이상 (CUDA 사용 시)
- **Memory**: 16GB RAM 이상 권장

## Python 의존성 (Python Dependencies)
아래 내용을 `requirements.txt`로 저장하고 설치하세요.

```txt
# Core Engine
anomalib[full]==1.0.0
torch>=2.6.0
torchvision>=0.21.0

# Data Handling & Processing
openvino-dev>=2023.0  # Optional
pandas
numpy>=2.0.0
matplotlib
opencv-python

# UI
streamlit>=1.30.0
watchdog
```

## 설치 가이드 (Setup Instructions - macOS/Linux)
1. Python 3.12 설치 (권장).
2. 가상 환경 생성: `python -m venv venv`.
3. 가상 환경 활성화: `source venv/bin/activate`.
4. 패키지 설치: `pip install -r requirements.txt`.
