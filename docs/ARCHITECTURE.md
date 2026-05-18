# Surface Anomaly Detection System Architecture 명세서

이 문서는 표면 이상 탐지 시스템(SG_proj_005_sad)의 전체 아키텍처 구조, 서브모듈 컴포넌트 간 관계 및 데이터 파이프라인의 설계 명세를 정의합니다.

## 1. 하이레벨 시스템 아키텍처

본 시스템은 소량의 데이터만으로 동작하는 메모리 뱅크 기반 이상 탐지기(PatchCore)와, 검출된 결함 영역의 픽셀 단위를 정밀 분할해낼 수 있는 Segment Anything Model 2(SAM2)로 구성된 하이브리드 추론 아키텍처를 취합니다.

컴포넌트 관계도:

1. Web Frontend UI (app.py / app_gradio.py)
   - 사용자 파일 업로드 수용 및 PIL 이미지 전환
   - IntegratedEngine으로 이미지 전달
   - 최종 시각화 아웃풋(Heatmap / Mask Overlay) 렌더링

2. Integrated Engine Core (inference_engine.py)
   - 최적의 하드웨어 디바이스(CUDA/MPS/CPU)를 자동 감지 및 장치 바인딩
   - PatchCore 추론을 구동하여 1차 이상 픽셀 맵 획득 및 아노말리 스코어 도출
   - 아노말리 픽셀 맵에서 최고 위험 좌표(Peak-point)를 수학적으로 색출
   - Peak-point를 프롬프트 좌표로 인가하여 SAM2 세그멘터 구동
   - 정밀한 바운더리 마스크를 생성하여 최종 시각화 유틸리티(create_overlay 등) 가공

3. Offline Training Core (train.py)
   - Windows/M2 Pro 장치 가속 몽키 패치 적용
   - Anomalib Folder API 기반 정상 데이터 로딩 및 피팅 실행
   - results/ 폴더에 최종 체크포인트 빌드

---

## 2. 세부 추론 데이터 파이프라인

데이터가 입력되어 결과 시각화로 이어지는 구체적인 단계별 흐름은 다음과 같습니다.

[Step 1: 데이터 전처리]
- PIL Image 형태로 수신된 데이터는 torchvision.transforms 파이프라인에 의해 256x256 해상도로 리사이즈됩니다.
- ImageNet 정규화 파라미터(Mean, Std)를 적용하여 1x3x256x256 규격의 Float Tensor로 패킹되어 가속 디바이스로 이동합니다.

[Step 2: PatchCore Wide-ResNet50 특징 추출 및 메모리 뱅크 비교]
- PyTorch 가속 백엔드로 로드된 TorchInferencer가 동작합니다.
- Wide-ResNet50의 미들 레이어 피처 맵을 결합하여 coreset 메모리 뱅크와 거리를 비교 계산합니다.
- 각 픽셀별 이상 농도 세기를 표현하는 anomaly_map 및 최종 이상 점수 pred_score가 출력됩니다.

[Step 3: 결함 Peak-point 산출]
- 256x256 크기의 anomaly_map은 원본 입력 이미지 크기로 Bicubic 보간 리사이즈를 수행합니다.
- 리사이즈된 행렬에서 최댓값을 갖는 인덱스를 unravel_index 함수를 이용해 2차원 좌표 (x, y)로 최종 변환합니다. 이 좌표가 결함이 의심되는 핵심 근거이자 SAM2의 긍정 프롬프트가 됩니다.

[Step 4: SAM2 프롬프트 유도형 정밀 세그멘테이션]
- SAM2ImagePredictor가 대상 이미지를 임베딩 텐서로 변환합니다.
- 전달된 Peak-point 좌표를 긍정 프롬프트(label = 1)로 인가하여 결함의 기하학적 형태를 추종하는 픽셀 마스크를 계산합니다.
- 마스크가 확보되면 오버레이(create_overlay)를 통해 최종적인 불량 영역을 붉은색 반투명 레이어로 시각화합니다.

---

## 3. 핵심 아키텍처 제약 및 복구 설계

- 하드웨어 가용성 대처:
  본 아키텍처는 CUDA 가속 환경을 최선으로 하여 21ms 급의 지연시간을 확보하지만, MPS(macOS)나 CPU 단독 환경에서도 정밀 타입 변환을 통해 오류 없이 무중단으로 Fallback 구동되도록 견고하게 설계되었습니다.
- SAM2 로딩 실패 시 복구:
  만약 segment-anything-2 라이브러리가 존재하지 않거나 가중치 파일 로딩에 실패한 경우, IntegratedEngine은 내부 load_error 플래그에 사유를 기록하고 SAM2를 graceful하게 비활성화합니다. 이후 추론 요청이 왔을 때 PatchCore 결과만을 안정적으로 출력하여, 핵심적인 이상 탐지 및 웹 대시보드 뷰가 완전히 다운되는 사고를 미연에 원천 차단합니다.
