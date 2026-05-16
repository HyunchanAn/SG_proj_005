# Surface Anomaly Detection Project Tasks

## 프로젝트 초기화 및 계획 (Project Initialization & Planning) [completed]
- [x] 프로젝트 기획서 및 폴더 구조 설계 (`implementation_plan.md`) <!-- id: 0 -->
- [x] 개발 환경 설정 가이드 작성 (`REQUIREMENTS.md` or similar) <!-- id: 1 -->

## 데이터셋 준비 (Data Preparation) [completed]
- [x] Anomalib 호환 데이터 폴더 구조 생성 (`datasets/custom/`) <!-- id: 2 -->
    - [x] `normal` (정상 이미지) 폴더 생성 <!-- id: 3 -->
    - [x] `abnormal` (불량 이미지) 폴더 생성 <!-- id: 4 -->
- [x] (Optional) 샘플 데이터(KolektorSDD 등) 다운로드 스크립트 또는 가이드 제공 (`prepare_data.py --download`) <!-- id: 5 -->

## Anomalib 설정 및 엔진 구축 (Engine Setup) [completed]
- [x] PatchCore 모델 설정 파일 작성 (`configs/surface_anomaly.yaml`) <!-- id: 6 -->
- [x] 학습 실행 스크립트 작성 (`train.py`) <!-- id: 7 -->

## UI 개발 (Streamlit App) [completed]
- [x] Streamlit 기본 레이아웃 구성 (`app.py`) <!-- id: 9 -->
- [x] 이미지 업로드 및 모델 로드 기능 구현 (Multi-device Support) <!-- id: 10 -->
- [x] 결과 시각화 구현 (No-CV2 Heatmap Overlay) <!-- id: 11 -->
- [x] Streamlit Cloud 배포 안정화 (Mocking & Patching)

## 모델 고도화 및 환경 정비 [in-progress]
- [x] 문서 현행화 (`README.md`, `REQUIREMENTS.md` 등)
- [x] 환경 정비 (불필요 파일 정리 및 `.gitignore` 점검)
- [ ] SAM2 (Segment Anything Model 2) 통합 시도
- [ ] 실제 산업용 데이터셋 기반 임계값(Threshold) 최적화

## 문서화 및 최종 시연 [completed]
- [x] 사용 가이드 (README.md) 작성 <!-- id: 12 -->
- [x] 프로젝트 파일 정리 및 로그 아카이브 생성 (`docs/logs_archive/`)
- [x] 최종 데모 시연 및 사용자 피드백 반영 [completed] <!-- id: 13 -->
