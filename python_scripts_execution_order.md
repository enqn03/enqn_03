# AMMT 프로젝트 Python 스크립트 실행 순서 및 해설

본 문서는 `ammt_project` 내에 존재하는 모든 파이썬(`*.py`) 코드들의 실행 순서와 역할을 7개의 주요 단계(Phase)로 나누어 설명합니다. 위에서 아래로 갈수록 프로젝트가 발전해 온 시간적 순서(Chronological Order)와 일치합니다.

---

## Phase 1: 초기 설정 및 유틸리티 (Setup & Utilities)
프로젝트 초기에 문서 포맷팅을 맞추거나 환경을 점검하기 위해 1회성으로 사용되었던 스크립트 모음입니다.

- `fix_markdown.py`, `fix_markdown_v2.py`
  - **실행 목적:** 리포지토리 내 마크다운(`*.md`) 파일들의 들여쓰기나 깨진 포맷 등을 일괄적으로 수정하기 위해 실행.
- `patch.py`, `patch_apply.py`
  - **실행 목적:** 코드 및 설정 파일에 대한 부분적인 패치(diff)를 안전하게 적용하기 위한 유틸리티.
- `test_dataloader.py`, `test_dataloader_2.py`, `test_dataloader_3.py`
  - **실행 목적:** 초창기 PyTorch DataLoader와 Memmap(대용량 TIFF 읽기)이 정상적으로 메모리 누수 없이 작동하는지 확인하기 위해 작성된 단위 테스트 코드.
- `ammt_tiff_pytorch_dataset_memmap.py`, `ammt_shared_ab_model_skeleton.py`
  - **실행 목적:** 프로젝트 초창기에 Dataset 구성 및 모델의 공통 뼈대 구조를 잡기 위해 임시로 작성되었던 코어 모듈들. 이후 `src/` 하위 모듈로 발전됨.

---

## Phase 2: 원시 데이터 탐색 및 기초 검증 (Raw Data EDA & Audits)
딥러닝 모델에 들어가기 전, 센서(카메라)에서 수집된 A(가루)와 B(쇳물) 이미지 원본 자체의 결함 여부나 구조를 점검하는 코드들입니다.

- `audit_layer_camera.py`
  - **실행 목적:** 원본 LayerCameraAfterSpreading.tif(A 이미지)의 구조(해상도, 레이어 수)와 포화(Saturation) 상태를 1차 점검하기 위해 실행.
- `audit_ab_pairs.py`
  - **실행 목적:** A 이미지와 B 이미지의 프레임 대응성이 맞는지 파악하고 시계열 차분(Difference)이 유효한지 확인하기 위해 실행.
- `src/audit_roi_saturation.py`, `src/audit_roi_candidates.py`
  - **실행 목적:** 원본 이미지 중 관심 영역(ROI) 안팎의 포화도(Saturation)를 검사하여 센서 렌즈 주변의 빛 번짐 현상을 파악하기 위해 실행.

---

## Phase 3: 데이터 파이프라인 및 타겟(XCT) 결합 (Manifest & Target Definition)
딥러닝이 소화할 수 있도록 데이터를 인과(Causal) 시퀀스로 자르고 정답지(XCT Target)와 정규화(Normalization) 정보를 매핑하는 단계입니다.

- `src/build_causal_manifest.py`
  - **실행 목적:** 레이어들을 K=4 단위의 시계열 시퀀스로 쪼개어, 미래 정보를 참조하는 오류(Data Leakage)를 막기 위해 실행.
- `src/estimate_train_normalization.py`
  - **실행 목적:** Train 셋에서만 P01/P99 통계치를 추출해 Robust 정규화를 수행하여 outlier를 억제하기 위해 실행.
- `src/ammt_causal_dataset.py`
  - **실행 목적:** 모델에 들어갈 6-채널 (3 intensity + 3 validity mask) 데이터 파이프라인이 정상 구축되는지 검증하기 위해 실행.
- `src/audit_registered_xct_targets.py`
  - **실행 목적:** 결함 정답지(Registered XCT)의 희소성(Sparse)과 물리적 분포 밀도를 파악하기 위해 실행.
- `src/ammt_weak_target_dataset.py`
  - **실행 목적:** Causal Input 데이터와 약한 정답지(Weak Continuous Response) 및 Support Mask를 메모리 상에서 실시간 결합(on-the-fly)하기 위해 실행.
- `src/audit_weak_target_rasterization.py`, `src/audit_weak_target_semantics_v1.py`, `src/audit_weak_target_support_density.py`
  - **실행 목적:** Gaussian Sigma 값 조절에 따라 타겟 밀도가 어떻게 변하는지 등 Weak Target 매핑 규칙의 안전성을 검토하기 위해 실행.
- `src/ammt_masked_regression_loss.py`, `src/verify_masked_regression_loss.py`
  - **실행 목적:** 타겟이 존재하는(Support=1) 픽셀에서만 Loss가 흐르고 모르는 픽셀(Unknown)에서는 Gradient가 차단되도록 Smooth L1 Loss를 설계하고 검증하기 위해 실행.

---

## Phase 4: 물리 좌표 및 캘리브레이션 (Physical Calibration & Metrology)
모델이 예측한 좌표(Camera Pixel)를 실제 장비 물리 좌표계(Machine XY)로 변환하는 과정을 설계하고 검증하는 코드들입니다.

- `src/audit_machine_camera_calibration.py`, `src/select_machine_camera_control_points.py`
  - **실행 목적:** 카메라 픽셀과 장비 좌표계를 매핑하기 위한 기준점을 설정하기 위해 실행.
- `src/audit_visible_dotgrid_extent_controls.py`, `src/audit_visible_dotgrid_outer_boundary_diagnostic.py`, `src/select_visible_dotgrid_extent_controls.py`, `src/select_visible_dotgrid_extent_controls_v2.py`
  - **실행 목적:** 닷그리드 패턴의 경계면을 인식하고 프로젝션(투영) 시 발생할 수 있는 왜곡이나 절단면을 방어하기 위해 실행.
- `src/audit_layer125_orientation_overlay.py` 및 최상단 `audit_layer125_orientation_overlay.py`
  - **실행 목적:** 레이어 회전 및 뒤집힘(Mirror/Rotate) 등 캘리브레이션 기본 오리엔테이션이 맞는지 오버레이로 검증.
- `src/audit_independent_metrology_fiducials.py`, `src/audit_independent_metrology_fiducials_refined.py`, `src/audit_independent_metrology_metadata.py`
  - **실행 목적:** 캘리브레이션을 독립된 기준(Fiducial)으로 2차 검증(Metrology)하기 위해 실행.
- `src/audit_independent_method2_*` (시리즈)
  - **실행 목적:** Lattice 맵핑 및 커버리지 검증 등 Method 2 캘리브레이션 정밀화 연구에 사용.
- `src/audit_calibration_design_review_v1.py`, `src/audit_calibration_local_refinement.py`, `src/audit_calibration_photometric_consistency.py`
  - **실행 목적:** 카메라-장비 캘리브레이션의 국소적 오차를 줄이고 밝기(Photometric) 일관성을 확인하기 위해 실행.
- `src/audit_candidate_calibration_rank_sensitivity.py`, `src/audit_projected_xct_target_support.py`
  - **실행 목적:** 추출된 모델 후보군이 실제 물리 좌표계로 변환될 때 생기는 민감도와 Support Mask의 일치율을 검증하기 위해 실행.

---

## Phase 5: A-only 베이스라인 훈련 및 진단 (Baseline Training & Diagnostics)
최초로 가루(A) 이미지만 사용하여 8-Epoch 기준 학습을 수행하고 문제점(Temporal Collapse)을 진단한 단계입니다.

- `src/train_a_only_baseline.py`
  - **실행 목적:** 6-채널 가루(A) 입력만으로 XCT 타겟을 학습하는 최초의 베이스라인 모델 성능(Test Loss)을 확보하기 위해 실행.
- `src/diagnose_a_only_spatial_predictions.py`
  - **실행 목적:** 첫 학습 모델의 결과 맵 전체가 하나의 점수로 고정되어 버리는(Map Invariance) 기이한 평탄화 현상을 검사하기 위해 실행.
- `src/diagnose_a_only_input_sensitivity.py`
  - **실행 목적:** 평탄화 현상이 모델의 어느 레이어(Encoder, Temporal, Decoder)에서 정보를 잃어버려서(Temporal Collapse) 발생하는지 수학적으로 추적하기 위해 실행.
- `src/evaluate_a_only_checkpoint.py`
  - **실행 목적:** E24(24-Epoch) 등 특정 실험 체크포인트의 가중치를 Read-Only로 불러와 평가 지표만을 단독으로 추출하기 위해 실행.
- `src/audit_a_only_candidate_coordinates.py`, `src/audit_a_only_candidate_stability_v1.py`, `src/audit_a_only_ordinary_causal_margin_sweep_v1.py`
  - **실행 목적:** A-only 모델이 예측한 최종 후보 좌표들이 장비 경계선(Part Rectangle) 밖으로 나가버리는 기하학적 안전성 문제 등을 진단하고 개선하기 위해 실행.

---

## Phase 6: 시계열 차분(Temporal Difference) 아키텍처 도입 (Architecture Evolution)
A-only 모델에서 확인된 Temporal Collapse 문제를 해결하기 위해, 현재 레이어와 이전 레이어들의 '차이(Difference)'를 활용하는 구조로 업그레이드한 단계입니다.

- `src/train_a_only_temporal_difference_v1.py`
  - **실행 목적:** 평탄화 현상을 막고 변화량에만 집중하게 하기 위해 Temporal Difference 인코더 구조를 처음으로 A 이미지 모델에 도입.
- `src/train_b_only_temporal_difference_v1.py`
  - **실행 목적:** 가장 강력한 결함 신호(레이저 융융 직후)를 담고 있는 B 이미지(쇳물) 단독 모델의 베이스라인 성능(Test Loss 0.0546)을 측정하기 위해 실행.
- `src/audit_a_only_temporal_difference_candidate_margin_v1.py`, `src/audit_a_only_temporal_difference_stability_v1.py`
  - **실행 목적:** 시계열 차분 구조 변경 후에도 출력 예측치가 미세 노이즈에 안정적인지, Margin Threshold(1%) 적용 시 얼마나 방어가 잘 되는지 감사하기 위해 실행.
- `src/audit_a_only_temporal_path_mechanism_v1.py`, `src/audit_multi_seed_temporal_comparison_v1.py`, `src/audit_margin_based_withholding_distribution.py`
  - **실행 목적:** 다양한 시드 및 마진 환경에서도 Temporal Difference 메커니즘이 이전의 Temporal Collapse를 확실히 억제하고 견고한지 확정 짓기 위해 실행.

---

## Phase 7: A+B 융합 아키텍처 고도화 (Fusion Models)
A(가루)와 B(쇳물) 정보를 동시에 활용하여 최고의 정확도를 확보하기 위해 시도한 최종 실험 단계입니다.

- `src/train_a_b_fusion_temporal_difference_v1.py`
  - **실행 목적:** A와 B의 Feature를 각각 추출해 단순히 이어붙이는(Concat) 융합 모델을 테스트했으나, A의 노이즈가 B의 신호를 방해하여 B-only 모델보다 성능이 떨어지는 것을 확인하기 위해 실행.
- `src/train_a_b_gated_fusion_v2.py`
  - **실행 목적:** Concat의 노이즈 간섭을 해결하기 위해, 모델이 스스로 정보의 중요도를 채널 및 공간별로 판단하여(Gated CNN Fusion) 취합하도록 개선한 최종 아키텍처 학습용으로 실행 (현재 진행 중).
