# Surface Anomaly Detection 구현 계획 (Implementation Plan)

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 **Anomalib**의 **PatchCore** 모델을 활용하여 표면(Surface)의 미세 결함을 탐지하는 시스템을 구축하고, **Streamlit**을 통해 사용자 친화적인 데모 UI를 제공하는 것을 목표로 합니다.
기존 복잡한 터미널 기반 실행 방식을 탈피하고, 실제 산업 현장의 "코팅/접착 불량 탐지" 시나리오를 시뮬레이션합니다.

## 2. 권장 폴더 구조 (Recommended Directory Structure)
Anomalib의 `Folder` 데이터셋 포맷을 따르는 구조로 설계합니다.

```
surface-anomaly-detection/
├── datasets/                   # 데이터 저장소
│   └── surface_data/           # (예: 스마트폰 촬영본 또는 KolektorSDD)
│       ├── normal/             # 정상 이미지 (학습용 + 테스트용)
│       └── abnormal/           # 불량 이미지 (테스트용)
├── configs/                    # Anomalib 설정 파일
│   └── surface_config.yaml     # PatchCore 커스텀 설정
├── models/                     # 학습된 모델 저장 (.ckpt)
├── src/                        # 헬퍼 모듈 (필요 시)
├── train.py                    # 모델 학습 실행 스크립트
├── app.py                      # Streamlit 웹 애플리케이션 (메인 UI)
└── README.md                   # 실행 가이드
```

## 3. 주요 구현 단계 (Implementation Steps)

### 단계 1: 데이터셋 준비 및 환경 설정
- **가상환경**: Python 3.8+ (Anomalib 호환)
- **데이터 구조화**: 사용자가 찍은 사진을 `normal` 폴더에 넣기만 하면 바로 학습 가능하도록 경로 설정.
- **샘플 데이터**: 사용자가 당장 데이터가 없을 경우를 대비해 `KolektorSDD` 데이터셋 활용 가이드 제공.

### 단계 2: Anomalib 설정 (config.yaml)
- **Model**: `patchcore` (빠른 학습, 적은 데이터로 높은 성능)
- **Dataset**: `folder` format 사용.
    - `path`: ./datasets/surface_data
    - `category`: surface
    - `image_size`: 224x224 (또는 256x256)
    - `train_batch_size`: 8~32 (GPU 메모리에 맞게 조정)

### 단계 3: 학습 스크립트 (train.py)
- CLI 명령어가 아닌 파이썬 API (`anomalib.engine.Engine`)를 사용하여 유연성 확보.
- 학습 후 결과를 `results/` 또는 `models/`에 저장.

### 단계 4: Streamlit UI (app.py)
- **기능**:
    1. **사이드바**: 모델 가중치 파일(.ckpt) 업로드/선택.
    2. **메인 화면**: 검사할 이미지 업로드.
    3. **분석**: "분석 시작" 버튼 클릭 시 추론 실행.
    4. **시각화**: 원본 이미지와 Heatmap/Segmentation Mask를 나란히 표시.
    5. **판정**: 정상/불량 여부 텍스트 출력.

## 4. 사용자 검토 필요 사항 (User Review Required)
> [!IMPORTANT]
> **GPU 가용성**: PatchCore 학습은 GPU가 권장됩니다. 현재 환경에 GPU가 있는지 확인이 필요합니다. (없으면 CPU 모드로 설정해야 하며 속도가 느릴 수 있습니다.)

> [!NOTE]
> **데이터 수집**: "정상" 이미지 50장 촬영이 가능한지 확인해주세요. 당장 어렵다면 공개 데이터셋으로 먼저 세팅해드리겠습니다.

## 5. 검증 계획 (Verification Plan)
- **Hello World**: 기본 설정으로 학습 코드가 에러 없이 돌아가는지 확인.
- **Inference Check**: 학습된 모델로 테스트 이미지를 넣었을 때 Heatmap이 생성되는지 확인.
- **UI Test**: Streamlit에서 이미지 업로드 및 결과 표시가 정상 작동하는지 확인.
