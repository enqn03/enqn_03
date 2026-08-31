import re

with open('AMMT_프로젝트를_처음부터_이해하기.md', 'r') as f:
    content = f.read()

# 1. Restore the corrupted Chapters 13 to 20
corrupted_start = "과거 3개 레이어의 정보는 시간축으로 압축(Temporal update)하되, 가장 중요한 '현재 층(Endpoint)'의 특징(Feature)은 3D Conv를 거치지 않고 디코더 직전으로 우회(Bypass)시켜 더해주는 'A-only Temporal Difference (Residual)' 구조를 제안했다. 기존의 모든 통제 변인(학습 조건 등)은 고정한## 27. 새 모델(Temporal Difference)의 구조적 타당성 사전 검증(Dry Run)"
corrupted_end = """

**결과 및 의의 (Result & Significance):**
캘리브레이션 변환식 교체 금지, 부품 번호나 기계 좌표 출력 금지, 잔차 모델 재학습 금지, XCT 정답 마스크 디코더 투입 금지 등의 절대 규칙(Immutable Rule)을 확립하여 프로젝트가 통제 불능의 오버피팅(Overfitting)으로 빠지지 않도록 방호벽을 쳤다."""

correct_13_20 = """과거 3개 레이어의 정보는 시간축으로 압축(Temporal update)하되, 가장 중요한 '현재 층(Endpoint)'의 특징(Feature)은 3D Conv를 거치지 않고 디코더 직전으로 우회(Bypass)시켜 더해주는 'A-only Temporal Difference (Residual)' 구조를 제안했다. 기존의 모든 통제 변인(학습 조건 등)은 고정한 채, 오직 잔차 연결(`use_endpoint_feature_residual=true`)의 유무만으로 성능을 비교했다.

**결과 및 의의 (Result & Significance):**
미래의 정보를 훔쳐보지 않는 인과적(Causal) 제약을 유지하면서도, 현재 레이어의 선명한 시각적 특징을 디코더까지 안전하게 전달하는 돌파구를 마련했다.

---

## 14. 잔차 모델(Residual Model) 공식 베이스라인 채택

**연구 목표 (Objective):** 
새로 도입된 Residual Bypass 구조가 실제로 이전 C32 모델의 한계를 극복했는지 검증하여, 공식 베이스라인 채택 여부를 결정하고자 했다.

**검증 과정 (Validation Process):**
두 모델 간의 Test Loss, 예측 분산(Prediction std), 그리고 Top-score 픽셀의 중복률(Tie fraction)을 정량적으로 비교했다.

**결과 및 의의 (Result & Significance):**
잔차 모델은 Test Loss를 5.04% 감소시켰을 뿐만 아니라, 96%에 달하던 Top-score 중복률을 단 1픽셀(0.0015%) 수준으로 획기적으로 낮춰 Map 고착화(Plateau) 문제를 완전히 해소했다. 이에 따라 Residual 모델을 현재의 공식 'A-only Causal Baseline'으로 최종 채택했다.

---

# Part IV. 출력의 안전성과 물리 좌표 매핑 (Safety & Calibration)

## 15. 디코더 안전장치 (Safety Decoder Filters)

**연구 목표 (Objective):** 
모델이 뱉어낸 점수(Score)가 우연한 노이즈이거나 신뢰할 수 없는 평탄한 맵(Plateau)일 경우, 이를 결함 후보 좌표로 함부로 출력하지 않도록 막는 다중 안전망(Safety Gate)을 구축하고자 했다.

**검증 과정 (Validation Process):**
학습용 정답지(XCT support mask)의 도움 없이, 모델이 스스로 만든 맵의 품질만을 평가하는 4단계 필터를 구현했다: (1) 맵에 굴곡이 있는지(Flatness), (2) 최상위 점수가 너무 많이 겹치지 않는지(Tie fraction < 0.1%), (3) 과거와 맵이 똑같지 않은지(Temporal invariance), (4) 7x7 지역 내 확실한 정점(Local maxima)인지 엄격히 검사했다.

**결과 및 의의 (Result & Significance):**
점수가 아무리 높아도 맵 자체가 불확실하면 출력을 보류(Withhold)하는 강력한 Fail-closed 파이프라인을 완성하여, 향후 실시간 도입(Deployment) 시 발생할 수 있는 오작동을 선제적으로 차단했다.

---

## 16. 모델 출력 좌표의 현재 의미 (Raw Camera Primary)

**연구 목표 (Objective):** 
모델이 출력하는 점(Candidate)이 실제 기계의 물리적 좌표(Machine XY)나 부품(Part) 번호를 완전히 대변할 수 있는지 그 책임 한계를 명확히 규정하고자 했다.

**검증 과정 (Validation Process):**
임시 캘리브레이션 룰(Rank 1 vs Rank 2)을 교차 적용하여 동일한 모델 출력 픽셀이 기계 좌표계로 변환될 때 어떤 변화가 생기는지 시뮬레이션했다.

**결과 및 의의 (Result & Significance):**
캘리브레이션 가정 하나만 바뀌어도 결함 부품 번호가 완전히 뒤바뀌는(Part agreement 0/240) 현상을 확인했다. 따라서 현재 모델 출력의 1차적 진실(Primary Truth)은 오직 '카메라 픽셀 좌표(x_pixel, y_pixel)' 뿐이며, 물리 좌표 변환은 반드시 별도의 독립된 캘리브레이션 모듈의 검증을 거쳐야만 유효함을 확정했다.

---

# Part V. 캘리브레이션 모호성과 독립 검증 체계

## 17. 임시 캘리브레이션 (Provisional Calibration)의 한계

**연구 목표 (Objective):** 
카메라 이미지(픽셀)를 실제 프린터 물리 공간(mm)으로 변환하는 변환 행렬(Homography)을 구하고자 했다.

**검증 과정 (Validation Process):**
부품의 화면 모서리를 잡고 가능한 192가지의 거울상/회전(Mirror/Rotation) 조합을 비교했다. 그 결과 90도 회전(Rank1)과 270도 회전(Rank2) 가설이 수학적 오차율(RMSE) 면에서 동점(Tie)을 이루는 모호성(Ambiguity)을 발견했다.

**결과 및 의의 (Result & Significance):**
카메라 내부 조명이나 시각적 단서(Local photometric evidence)를 근거로 임시로 Rank2(mirror_rotate_270)를 작업용(Working) 표준으로 두었으나, 이것이 독립적이고 절대적인 물리적 진실은 아님을 분명히 했다. (이 모호성은 이후 제 36장에서 XCT 오버레이를 통해 Rank2로 최종 타결된다.)

---

## 18. NIST 메타데이터를 통한 독립 검증 시도

**연구 목표 (Objective):** 
스크린 픽셀이 아닌 공식 계측용(Metrology) 레퍼런스 데이터를 활용하여 캘리브레이션 변환을 독립적이고 객관적으로 증명하고자 했다.

**검증 과정 (Validation Process):**
NIST에서 제공하는 DotGrid 이미지, 보조 카메라(SecondaryCamera)의 붉은 레이저 기준점(Red marker), 그리고 체커보드(Checkerboard) 데이터를 검증 테이블에 올렸다.

**결과 및 의의 (Result & Significance):**
DotGrid의 패턴은 선명히 인식되었으나, 붉은 레이저 점 1개만으로는 카메라 방향축(Orientation sign)을 수학적으로 결정할 수 없었다. 체커보드 역시 노이즈(False positive)가 많아, 공식 메타데이터를 활용한 직접적 변환은 아직 신뢰할 수 없는 보류(Hold) 상태로 판정했다.

---

## 19. DotGrid Method #2 격자 추적 알고리즘 보류

**연구 목표 (Objective):** 
NIST Method #2 (DotGrid 기반 변환)를 활용하기 위해 카메라 속 점(Dot)들과 실제 물리적 50x50 격자판을 1:1로 매칭(Correspondence)하는 알고리즘을 개발하고자 했다.

**검증 과정 (Validation Process):**
V1 알고리즘(PCA + 1D 클러스터링)을 통해 1,518개의 점을 찾아냈으나, 미리 약속해둔 5x5 홀드아웃(Held-out) 블록을 통한 잔차(Residual) 테스트를 수행했다.

**결과 및 의의 (Result & Significance):**
점은 찾았으나 그 점이 '몇 번째 줄(Row/Col)'에 있는지 인덱싱하는 과정에서 원본 닷 피치(Dot pitch) 대비 0.41배 이상의 과도한 오차(RMSE)가 발생하여 검증 게이트를 통과하지 못했다. 이로 인해 자동 캘리브레이션 적용을 안전하게 보류(Hold)하고 정교한 2D 그래프(Graph) 기반 추적 알고리즘 고도화(Refinement)로 방향을 선회했다.

---

## 20. 현재 설정(Config)의 엄격한 변경 금지 사항

**연구 목표 (Objective):** 
캘리브레이션 모호성이 완전히 해결되기 전까지 무분별한 데이터나 모델 변경으로 프로젝트의 기반이 흔들리는 것을 막고자 했다.

**검증 과정 (Validation Process):**
위의 모든 실패와 보류(Hold) 상태를 종합하여, 어떤 것들을 '절대 건드리면 안 되는지' 정책으로 명문화했다.

**결과 및 의의 (Result & Significance):**
캘리브레이션 변환식 교체 금지, 부품 번호나 기계 좌표 출력 금지, 잔차 모델 재학습 금지, XCT 정답 마스크 디코더 투입 금지 등의 절대 규칙(Immutable Rule)을 확립하여 프로젝트가 통제 불능의 오버피팅(Overfitting)으로 빠지지 않도록 방호벽을 쳤다."""

start_idx = content.find(corrupted_start)
end_idx = content.find(corrupted_end) + len(corrupted_end)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + correct_13_20 + content[end_idx:]
else:
    print("Failed to find corrupted part 1")

# 2. Replace Old chapters 27-31 with the new formatted ones.
old_27_31_start = "## 27. 새 model을 바로 오래 학습시키지 않고 dry run부터 하는 이유"
old_27_31_end = "이 결과가 있어야 특정 margin safety policy가 과도하게 많은 `XCT-derived continuous quality candidate`를 막지 않는지 설계할 수 있다."

new_27_31 = """## 27. 새 모델(Temporal Difference)의 구조적 타당성 사전 검증(Dry Run)

**연구 목표 (Objective):** 
새로 설계된 A-only Temporal Difference 구조가 실제 학습 과정에 돌입하기 전, 텐서(Tensor) 흐름과 메모리 할당, 손실 함수(Loss function) 연산에 구조적 결함이 없는지 조기에 점검하고자 했다.

**검증 과정 (Validation Process):**
단 1개의 학습 샘플(z=128)만을 주입하여 입력 텐서(`[1,4,6,256,256]`)가 출력 텐서(`[1,1,256,256]`)로 정상 변환되는지, XCT 지원 영역(Supported pixel) 내에서 손실값(Loss)이 무한대(NaN/Inf)로 발산하지 않고 유한한 숫자(`0.1218`)로 계산되는지 확인하는 드라이런(Dry run)을 수행했다.

**결과 및 의의 (Result & Significance):**
모델의 순전파(Forward pass)와 오차 역전파(Backward) 파이프라인이 정상적으로 맞물려 작동함을 증명(배선 검사 통과)했다. 무작위 초기화(Random initialization) 상태에서의 무의미한 손실값을 성능으로 오판하지 않도록 방어 기제를 설정하고, 본격적인 8 Epoch 훈련으로 넘어가는 안전한 징검다리를 놓았다.

---

## 28. 검증(Validation)과 평가(Test) 성능의 불일치 진단

**연구 목표 (Objective):** 
새로운 Temporal Difference 모델이 학습 데이터가 아닌 미지의 데이터(Held-out Test set)에서도 실질적인 결함 예측 성능 향상을 이뤄냈는지 엄밀하게 검증하고자 했다.

**검증 과정 (Validation Process):**
단일 난수 시드(Seed)로 8 Epoch을 학습한 뒤, 검증 데이터(Validation)에서 오차가 가장 낮았던 Epoch 7의 체크포인트를 추출했다. 이를 미지의 Test 데이터에 적용하여 기존 베이스라인(C32 Residual)의 Test Loss와 비교했다.

**결과 및 의의 (Result & Significance):**
검증 오차는 4.1% 낮아졌으나, 최종 Test 오차는 오히려 0.41% 상승한 불일치(Discrepancy) 현상을 발견했다. 이 단일 시드 결과를 보고 섣불리 구조를 변경(Data leakage)하는 대신, "모델이 과거 데이터를 실제로 보고 있는지(Response)"를 먼저 입증하고 다중 시드(Multi-seed) 평균 테스트로 넘어가야 한다는 과학적 검증 순서를 확립했다.

---

## 29. 인과적 시계열 반응성(Causal Sensitivity)의 확증

**연구 목표 (Objective):** 
Test 성능의 미세한 하락이 모델 구조의 근본적 실패인지, 아니면 과거의 정보를 학습하긴 했으나 위치(Localization)만 불안정한 상태인지 원인을 분리해내고자 했다.

**검증 과정 (Validation Process):**
이전 실패작(Residual)과 동일하게 과거 이미지 3장을 현재 이미지로 강제 교체하거나 단일 프레임만 조작하는 반사실적 교란(Counterfactual perturbation)을 가한 뒤 출력 맵의 변화량을 관측했다.

**결과 및 의의 (Result & Significance):**
기존 모델이 0.0의 반응성을 보였던 반면, 새 모델은 과거 이미지 변경 시 출력 맵의 오차가 최대 0.105까지 크게 변동하며 기준치(0.0001)를 아득히 돌파했다. 즉, 모델이 '과거의 변화'를 결함 판단에 확실하게 이용(Utilize)하는 시계열 감수성을 회복했음을 성공적으로 증명했다.

---

## 30. 예측 좌표 변동성의 두 가지 가설(Rank Switch vs Peak Relocation) 분리

**연구 목표 (Objective):** 
과거 이미지가 변할 때 예측 결함 좌표(Candidate coordinate)가 크게 흔들리는 현상의 원인이 단순한 '점수 경합'인지, 아니면 '근본적인 예측 실패'인지 규명하고자 했다.

**검증 과정 (Validation Process):**
1위 후보 좌표가 바뀔 때, 기존 1위와 새로운 1위 간의 점수 차이(Score margin)를 측정하는 진단 프로토콜을 도입했다. 점수 차이가 5% 이내로 초박빙(Near-tie)일 때만 순위가 바뀌는지, 아니면 압도적인 차이(High-margin 20% 이상)로 1위가 완전히 날아가 버리는지 추적했다.

**결과 및 의의 (Result & Significance):**
결함 좌표가 불안정하다는 표면적 결과 뒤에 숨겨진 구조적 메커니즘을 해부할 수 있는 마진(Margin) 진단 프레임워크를 구축하여, 향후 실시간 도입 시 애매한 결함을 기각(Withhold)할 수 있는 안전 마진 정책의 기반을 마련했다.

---

## 31. 점수 경합(Near-tie) 현상 증명과 안전 마진 도입의 당위성

**연구 목표 (Objective):** 
12가지 극한 교란 조건(Stress test) 하에서 좌표가 바뀌는 실제 양상을 확인하여, 향후 시스템에 도입될 안전장치(Safety margin)의 방향성을 확정하고자 했다.

**검증 과정 (Validation Process):**
12개의 반사실적 맵(Counterfactual map)을 전수 조사한 결과, 1위 좌표가 완전히 소실(High-margin relocation)되는 경우는 단 1건도 없었으며, 순위가 바뀌는 3건 모두 기존 1위가 2위나 3위로 여전히 살아남아 초박빙 점수 경합(Near-tie rank switch)을 벌이는 상태임을 확인했다.

**결과 및 의의 (Result & Significance):**
모델의 예측이 근본적으로 흔들리는 것이 아니라, 비슷한 확률의 결함 후보들 사이에서 미세한 점수 차이로 순위만 엎치락뒤치락한다는 물리적 진실을 밝혀냈다. 이에 따라 모델을 재학습하는 대신, 향후 **실제 48개 테스트 레이어에 안전 마진 커트라인(예: 1%, 2%, 5%)을 적용하는 차기 시뮬레이션 연구(Margin-based Withholding Audit)로 나아가야 한다는 강력한 증거**를 확보했다."""

start_idx_2 = content.find(old_27_31_start)
end_idx_2 = content.find(old_27_31_end) + len(old_27_31_end)

if start_idx_2 != -1 and end_idx_2 != -1:
    content = content[:start_idx_2] + new_27_31 + content[end_idx_2:]
else:
    print("Failed to find old chapters 27-31")

with open('AMMT_프로젝트를_처음부터_이해하기.md', 'w') as f:
    f.write(content)

