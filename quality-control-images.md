# AMMT 프로젝트 전체 이미지 에셋 아카이브 및 상세 설명

이 문서는 AMMT 프로젝트 내에 존재하는 **모든 이미지(PNG, JPG)의 경로와 각 이미지가 의미하는 바**를 상세하게 정리한 종합 아카이브입니다. 
데이터 전처리(Quality Control), 기하학적 정렬(Calibration), 모델 학습 및 시각적 검증, 그리고 최종 라이브 인퍼런스 결과물까지 프로젝트의 모든 시각적 증거 자료를 포함합니다.

---

## 1. 초기 데이터 및 전처리 품질 관리 (Data Preprocessing & QC)

원본 데이터의 물리적 한계(센서 포화 등)를 파악하고, 최적의 분석 환경을 세팅하기 위한 품질 관리(QC) 이미지들입니다.

* **`processed/audit_after_spreading/qc_contact_sheet.png`**
  * **의미:** A 모달리티(파우더 도포 직후)의 초기 레이어 적층 상태와 센서 포화(Saturation) 현상을 확인합니다. 후반 레이어에서 센서 한계(65535)에 도달하는 영역을 파악하여, 이를 결함이 아닌 Validity Mask(0)로 별도 처리해야 함을 증명하는 초기 점검 이미지입니다.
* **`processed/audit_ab_pairs/ab_pair_contact_sheet.png`**
  * **의미:** A(파우더 도포)와 B(레이저 조사) 이미지 페어링 정렬 상태 및 시각적 차이점을 점검합니다. 두 모달리티 간의 변화량이 뚜렷한지 1차적으로 확인합니다.
* **`processed/roi_candidates/roi_candidate_qc.png`**
  * **의미:** 전체 2000x2000 센서 영역 중 불필요한 장비 배경을 자르고 빛 번짐 비율을 최소화하기 위한 '관심 영역(ROI)' 후보군을 선별하고 분석하는 그림입니다.
* **`processed/roi_audit/roi_candidate_qc.png`**
  * **의미:** 최종 확정된 Working ROI(1500x1500) 내에서 조명별 포화(Saturation) 상태를 상세히 확인하여 모델 입력의 안전성을 보장합니다.
* **`processed/normalization_v1/normalization_qc.png`**
  * **의미:** 각 조명(LED 1~3)별, 층(Layer)별 밝기 분포(p01, p50, p99) 추이를 나타냅니다. 단일 글로벌 스케일링이 아닌, 층/조명별 정규화(Robust Scaling)를 도입하게 된 정량적 근거 자료입니다.

---

## 2. 3D-2D 정렬 및 정답지 생성 (Calibration & Target Generation)

사후 3D XCT 데이터를 공정 중 촬영된 2D 카메라 좌표계에 정밀하게 매핑하고, 학습용 정답지(Weak Target)로 변환하는 과정을 보여줍니다.

* **`processed/xct_target_audit/xct_target_qc.png`**
  * **의미:** XCT 스캔 데이터의 연속적인(Continuous) 밀도 분포 및 결함 점수 분포를 확인하는 이미지입니다.
* **`processed/projected_xct_support/projected_support_qc.png`**
  * **의미:** 3D XCT 데이터를 카메라의 2D 평면으로 투영(Projection)했을 때, 빈 허공이 아닌 실제 부품이 위치한 영역(Support)과 정확히 일치하는지 기하학적으로 검증하는 화면입니다.
* **`processed/weak_target_audit/weak_target_rasterization_qc.png`**
  * **의미:** 2D-3D 매핑 간의 1픽셀 오차를 완화하기 위해, XCT 결함 포인트 주변에 가우시안 릴렉세이션(sigma=2)을 적용하여 부드러운 히트맵 형태의 정답지를 생성한 결과물입니다.
* **`processed/target_semantics_v1/target_semantics_patches.png`**
  * **의미:** XCT 점수 상위 15%의 치명적 결함(Defect) 패치와 정상(Normal) 패치의 시각적 패턴 차이를 대비하여, 모델이 학습할 의미론적(Semantic) 타당성을 검증합니다.
* **`processed/calibration/...` 내부의 수많은 오버레이 이미지들**
  * **의미:** Dot Grid, Checkerboard 등을 이용한 카메라-머신 왜곡 보정, 회전각 검증, 레이어별 픽셀 대 기계 좌표 미세 정렬(Local Refinement) 등 광학적 캘리브레이션의 단계별 기하학적 증명 자료들입니다.

---

## 3. 모델 아키텍처 및 훈련 최적화 검증 (Architecture & Training Analysis)

Gated CBAM Fusion 모델이 데이터를 어떻게 이해하고 학습하는지를 시각적으로 보여줍니다.

* **`outputs/cbam_attention_layer242.png`**
  * **의미:** Gated CBAM 퓨전 모델의 시각적 설명력(XAI)을 보여주는 히트맵입니다. 레이저 코어 주변의 무의미한 스패터 노이즈를 모델 스스로 억제(Gate)하고, 실제 결함이 의심되는 징후에만 강하게 어텐션(Attention)을 쏟고 있음을 증명합니다.
* **`outputs/hyperparameter_tuning_loss.png`**
  * **의미:** 에폭(Epoch) 진행에 따른 훈련 손실(Train Loss)과 검증 손실(Validation Loss) 변화 추이 그래프입니다. 모델이 과적합(Overfitting)되지 않고 적절한 스윗스팟에서 학습이 완료되었음을 보여줍니다.

---

## 4. 모델 성능 평가 및 Ablation Study (Performance Evaluation)

통제된 실험 변인 속에서 우리 모델이 얼마나 우수한 성능을 달성했는지 증명하는 성적표입니다.

* **`outputs/cbam_fusion_comparison.png`**
  * **의미:** B-only 단일 모델이 발생시킨 수백 개의 오탐(False Alarms)을 A+B Fusion 모델이 '시계열적 교차 검증'을 통해 깨끗하게 걸러내는 모습을 시각적으로 대조한 결과물입니다.
* **`outputs/ablation_bar_chart.png`**
  * **의미:** A-only, B-only, A+B Fusion 모델 간의 정밀도(Precision), 재현율(Recall), F1-Score를 막대 그래프로 비교하여, 모달리티 융합의 압도적 성능 향상폭(약 2배)을 입증합니다.
* **`outputs/pixel_vs_blob_comparison.png`**
  * **의미:** 비현실적인 '픽셀 단위 칼채점'과 산업 현장을 반영한 '2mm 객체(Blob) 단위 평가' 간의 결함 탐지 재현율(Recall) 상승폭을 보여주어, 우리 모델이 이미 결함을 잘 찾고 있었음을 증명하는 그래프입니다.
* **`outputs/roc_prc_curves.png`**
  * **의미:** ROC 곡선과 정밀도-재현율(PRC) 곡선입니다. 결함이 1% 미만인 극불균형 데이터 환경에서도 퓨전 모델이 위양성(오탐)을 최소화하며 굳건한 신뢰성을 가짐을 보여줍니다.
* **`outputs/multi_seed_boxplot.png` (및 `processed/evaluation/multi_seed_boxplot.png`)**
  * **의미:** 초기 가중치(Random Seed)를 다르게 설정하여 여러 번 재학습해도 퓨전 모델의 성능(F1-Score)이 흔들림 없이 높게 유지됨을 보여주는 통계적 견고성(Robustness) 검증 박스플롯입니다.

---

## 5. 실시간 조기 경보 파이프라인 시뮬레이터 (Live Inference & Monitoring)

공정 엔지니어링 현장에 즉시 투입 가능한 실시간 모니터링 대시보드의 시각화 결과물입니다.

* **`outputs/live_inference_results_v7_demo_3d.png` (및 `live_stream_results_3d.png`, `defect_distribution_3d.png`)**
  * **의미:** 1층부터 현재 진행 중인 층까지 누적 탐지된 결함 의심 구역들을 기계의 절대 물리 좌표(X, Y, Z mm) 기준 3D 입체 산점도(Scatter plot)로 렌더링한 이미지입니다. 결함의 층간 누적 분포를 한눈에 파악할 수 있습니다.
* **`outputs/live_inference_results_v7_demo_2d.png` (및 `live_stream_results_2d.png`, `defect_distribution_2d.png`)**
  * **의미:** 특정 단면(Layer)에서 새롭게 검출된 이상 징후의 위치와 결함 확률 활성화 점수를 2D 히트맵 평면도로 출력하여 즉각적인 조치가 가능하도록 돕는 화면입니다.

---

## 6. A-only Causal Baseline 및 Temporal Difference 내부 검증 (Audit)

과거 프레임과 현재 프레임을 비교하는 시계열 모델 내부 로직의 안정성을 짚고 넘어가는 품질 검증 이미지들입니다.

* **`outputs/a_only_temporal_difference_candidate_margin_v1/...` 시리즈**
* **`outputs/a_only_candidate_stability_v1/...` 시리즈**
* **`outputs/a_only_temporal_path_mechanism_v1/...` 시리즈**
* **`outputs/a_only_temporal_difference_stability_v1/...` 시리즈**
  * **의미:** 시계열 기반 A-only 예측 시 과거 프레임을 변경(Replacement)하거나 동일하게 반복(Endpoint-repeat)했을 때, 모델이 계산하는 최상위 의심 픽셀(Top-1 Candidate)의 점수 차이(Margin)와 좌표 랭킹(Rank)이 튀지 않고 안정적으로 유지되는지 평가하는 디버깅/내부 검증용 이미지들입니다.
