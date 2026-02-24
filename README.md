# Surface Anomaly Detection System (표면 이상 탐지 시스템)

## 프로젝트 개요
이 프로젝트는 딥러닝(Anomalib PatchCore)을 활용한 표면 결함 탐지 시스템입니다.
**Apple Silicon (M2 Pro)** 환경에서 최적화되어 있으며(Metal/MPS 가속), 소량의 정상 이미지만으로도 이상 징후를 학습하고 탐지할 수 있습니다.

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
- `app.py`: Streamlit 웹 데모 앱

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
