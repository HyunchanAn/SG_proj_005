# Surface Anomaly Detection System (표면 이상 탐지 시스템)

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Anomalib PatchCore](https://img.shields.io/badge/AI_Engine-Anomalib_PatchCore-orange.svg)
![SAM2 Integrated](https://img.shields.io/badge/Segmentation-SAM2_Hiera--Tiny-red.svg)
![RTX 5080 Optimized](https://img.shields.io/badge/Hardware-RTX_5080_Ready-green.svg)
![Status Stable](https://img.shields.io/badge/Status-Stable_v1.0-brightgreen.svg)

## 프로젝트 개요
이 프로젝트는 딥러닝(Anomalib PatchCore)을 활용한 표면 결함 탐지 시스템입니다.
**Windows(RTX 5080)** 및 **Apple Silicon(M2 Pro)** 환경 모두에 최적화되어 있으며, 소량의 정상 이미지만으로도 고성능 이상 징후 탐지가 가능합니다. 특히 웹 UI(`app.py`)는 OpenCV 의존성을 제거하여 Streamlit Cloud 등 다양한 클라우드 환경에서도 안정적으로 구동되도록 설계되었습니다.

## 빠른 시작 (Quick Start)

### 1. 환경 설정 (Environment Setup)
`REQUIREMENTS.md`를 참고하여 필수 패키지를 설치하세요.
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux 기준
pip install -r requirements.txt
```

### 2. 데이터 준비 (Data Preparation)
약 50장의 "정상(Good)" 표면 이미지(필름, 직물, 코팅 등)가 필요합니다.

**방법 A: 사용자 데이터 사용**
1. `datasets/raw_images` 폴더를 생성합니다.
2. 준비한 정상 이미지 50장을 해당 폴더에 넣습니다.
3. 데이터 준비 스크립트를 실행합니다:
   ```bash
   python prepare_data.py
   ```

**방법 B: 샘플 데이터 자동 다운로드 (KolektorSDD)**
이미지가 없다면 아래 명령어로 샘플 데이터를 다운로드할 수 있습니다:
```bash
python prepare_data.py --download
```

**방법 C: 테스트용 합성 데이터 생성 (추천)**
다운로드가 안 되거나 빠르게 테스트하고 싶다면:
```bash
python synthesize_data.py
```
*회색 배경에 노이즈가 있는 가상 데이터를 생성합니다.*

### 3. 학습 (Training)
학습 스크립트를 실행합니다. (MPS 가속이 자동 적용됩니다.)

```bash
python train.py
```
*M2 Pro 기준, 학습은 1~2분 내외로 완료됩니다.*
학습된 모델(`.ckpt`)은 `results/` 폴더에 저장됩니다.

### 4. 모델 변환 (Export)
웹 데모에서 사용하기 위해 모델을 `.pt` 포맷으로 변환합니다.
```bash
python export.py
```
*`exported_models/` 폴더에 `model.pt` 파일이 생성됩니다.*

### 5. 데모 실행 (Web UI)
Streamlit 웹 인터페이스를 통해 결과를 시각적으로 확인할 수 있습니다.
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501`로 접속하세요.
* 사이드바에서 **언어(Language)**를 변경할 수 있습니다.
* `.pt` 모델을 선택하고 이미지를 업로드하여 분석합니다.

## 폴더 구조 (Directory Structure)
- `datasets/`: 데이터 저장소
- `configs/`: 모델 설정 파일 (하이퍼파라미터 등)
- `results/`: 학습 결과 및 모델(`.ckpt`) 저장 위치
- `train.py`: 학습 실행 스크립트
## 성능 및 검증 결과 (Performance & Validation)
본 시스템은 최첨단 PatchCore 알고리즘과 고성능 하드웨어를 결합하여, 실제 표면 결함 탐지에서 다음과 같은 신뢰성 있는 지표를 확보하였습니다.

### 📊 Metric Summary
| Metric | Score | Framework |
| :--- | :--- | :--- |
| **Image AUROC** | **1.0000** | Anomalib 2.2.0 |
| **Image F1-Score** | **1.0000** | Anomalib 2.2.0 |
| **Inference Latency** | **~21ms** | Tensor-optimized |

### 🛠️ 검증 방법론 (Methodology)
- **평가 환경**: `datasets/custom/test` 폴더 내의 독립된 테스트 셋(정상/불량)을 활용한 교차 검증 수행.
- **알고리즘**: PatchCore (Wide-ResNet50 Backbone) 기반의 메모리 뱅크 이상 탐지 기법 적용.
- **세그멘테이션**: SAM2 (Hiera-Tiny) 기반의 Peak-point 프롬프트 유도 정밀 분할 적용.

## 개발 및 실행 환경 (System Environment)
본 프로젝트는 다음의 **엔터프라이즈급 하드웨어**에서 최적의 성능을 발휘하도록 튜닝되었습니다.

| Component | Specification |
| :--- | :--- |
| **CPU** | **AMD Ryzen 9 9900X** (12-Core, 4.4GHz~5.6GHz) |
| **GPU** | **NVIDIA GeForce RTX 5080** (16GB GDDR7) |
| **RAM** | **64GB DDR5-5600** Dual Channel |
| **Storage** | **NVMe PCIe 4.0** (Reading ~7,000MB/s) |
| **Software** | Python 3.12, CUDA 12.8, PyTorch 2.6.0+cu128 |

## 폴더 구조 (Directory Structure)
- `app.py`: 통합 모니터링 대시보드 (Streamlit)
- `app_gradio.py`: 인터렉티브 정밀 분석 도구 (Gradio)
- `inference_engine.py`: 하이브리드 AI 추론 엔진 (Core)
- `train.py`: 고성능 학습 파이프라인
- `configs/`: 하드웨어 최적화 설정 파일
- `exported_models/`: 배포용 가속 모델 파일

## 브랜치 관리 및 레거시 (Branch Management & Legacy)
본 프로젝트는 개발 과정에서의 기술적 결정 사항을 보존하기 위해 특정 브랜치를 레거시로 유지합니다:
- **`origin/deploy/streamlit-cloud-260215`**: 
    - **목적**: 초기 Streamlit Cloud 배포 및 RTX 4060 최적화 환경의 스냅샷입니다.
    - **주의**: 현재 `main` 브랜치는 해당 브랜치보다 진보된 **No-CV2 시각화** 및 **RTX 5080/M2 Pro 하이브리드 최적화**가 적용되어 있습니다. 레거시 브랜치의 `packages.txt` 등은 현재의 Streamlit 배포 환경과 충돌할 수 있으므로 과거 로직 참고용으로만 사용하시기 바랍니다.

## 촬영 가이드 (Photography Guideline)
정확한 검사를 위해 테스트 이미지는 다음 조건을 따라주세요:

1.  **조명 (Lighting)**
    *   **균일한 조명**: 그림자나 강한 빛 반사가 없도록 해주세요.
    *   은은한 전체 조명이 가장 좋습니다.
2.  **구도 (Viewpoint)**
    *   **수직 촬영 (Top-down)**: 위에서 아래로 수직으로 찍어주세요.
    *   비스듬한 각도는 왜곡을 일으킬 수 있습니다.
    *   검사 대상이 화면에 가득 차게 찍으세요.
3.  **배경 (Background)**
    *   검사 제품 이외의 물체가 나오지 않도록 깔끔한 배경을 사용하세요.
