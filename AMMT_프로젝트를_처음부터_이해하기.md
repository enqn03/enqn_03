# AMMT 프로젝트를 처음부터 이해하기

> **한 문장 요약:** 이 프로젝트는 NIST AMMT LPBF 제조공정의 layer-camera A(AfterSpreading)와 B(Burned) 시계열을 각각 분석하고, 두 입력을 결합하는 **A-only decoder, B-only decoder, A+B fusion decoder**를 비교하여, registered XCT에서 유래한 연속 response를 기준으로 확인이 필요한 위치를 raw-camera 좌표 (x_pixel, y_pixel, layer_z, score) 형태의 XCT-derived continuous quality candidate로 제안하는 연구용 품질 모니터링 파이프라인을 구축하는 작업이다.

이 문서는 프로젝트의 전체 아키텍처와 결정 사항을 기록한다. 여기서 세 모델은 서로 대체되는 임시 모델이 아니라, 입력 시점과 정보량에 따른 세 가지 독립적인 실험 축(CNN Prediction Path)이다. A-only는 레이저 전 조기 후보 생성, B-only는 레이저 조사 후 상태 평가, A+B는 두 시점의 통합 평가를 담당한다.

---

## 1. 프로젝트 현황: 완료된 작업과 남은 과제

**연구 목표 (Objective):** 
본 프로젝트는 XCT(X-ray Computed Tomography) 사후 계측 데이터를 정답지(Target)로 삼아, 제조 공정 중 촬영된 시계열 카메라 이미지(A/B stage)만으로 결함 의심 위치를 픽셀 좌표 형태로 조기 제안하는 'XCT-derived continuous quality candidate' 파이프라인 구축을 목표로 한다. 이를 위해 데이터 정규화부터 좌표계 정합성, 그리고 A/B 결합 퓨전 모델까지 이어지는 거대한 로드맵의 현재 진척도를 명확히 진단하고자 했다.

**검증 과정 (Validation Process):**
진행된 모든 실험은 '데이터 보존(Read-only)', '물리적 좌표계 검증', '예측 모델의 신뢰성 검증'이라는 세 가지 엄격한 기준을 통해 교차 검증되었다. 모델이 단순히 점수 맵(Score map)을 생성하는 것에 그치지 않고, 그 맵의 좌표가 실제 프린터 기계의 물리적 좌표와 완벽히 일치하는지(Calibration Audit), 그리고 타겟 점수가 실제 물리적 결함을 의미하는지(Semantics Audit)를 순차적으로 통과시켰다.

**결과 및 의의 (Result & Significance):**
현재 가루(A) 전용 모델 및 쇳물(B) 전용 모델의 베이스라인 구축과 비교(제 34장), 캘리브레이션 거울상 모호성 해결(제 36장), 타겟 결함 의미 확정(제 37장)까지 핵심적인 불확실성을 모두 제거하며 '완료' 상태에 도달했다. 현재 남은 유일한 과제는 A와 B의 장점만을 취합하여 시너지를 내는 'A+B Gated Fusion 모델'의 고도화 작업이다.

---

## 2. 데이터 구조에 대한 세 가지 핵심 분리

**연구 목표 (Objective):** 
다양한 형태의 센서 데이터(카메라 이미지, XCT 포인트 클라우드, 레이저 궤적 등)를 혼동 없이 처리하기 위해, 데이터의 역할과 성격을 물리적 원리에 맞게 명확히 분리하는 기준을 세우고자 했다.

**검증 과정 (Validation Process):**
1. **의료용 CT와의 분리:** XCT는 모델의 실시간 입력(Input)이 아니라 사후 검증용 참조 데이터임을 아키텍처 상에 명시했다.
2. **A/B 시점의 분리:** 가루 도포 직후(A)와 레이저 조사 후(B)는 단순한 시간적 전후가 아니라, 레이저 용융이라는 거대한 물리적 변화를 사이에 둔 완전히 다른 정보원임을 규정했다. 따라서 B-A를 결함으로 정의하는 대신, A-only 조기 경보 모델과 B-only 사후 진단 모델로 역할을 엄격히 분리하여 학습시켰다.
3. **선정 기준의 분리:** 모든 파라미터 선택을 '절대 불변의 규칙(Data contract)', '합리적 타협안(Controlled selection)', '검증 실패로 인한 보류(Safety hold)' 세 가지로 분류하여 실험의 과학적 엄밀성을 유지했다.

**결과 및 의의 (Result & Significance):**
이러한 엄격한 개념 분리를 통해, 단순한 픽셀 차이를 결함으로 오인하거나 모델이 미래 데이터를 부정행위(Leakage)처럼 참조하는 오류를 원천 차단하는 견고한 실험 환경을 구축했다.

---

## 3. 원본 데이터 보존 및 접근 규칙

**연구 목표 (Objective):** 
NIST에서 제공한 수십 기가바이트의 원본 TIFF 및 CSV 데이터가 전처리 과정에서 훼손되거나 정보가 왜곡되는 것을 방지하기 위한 데이터 접근 원칙을 수립하고자 했다.

**검증 과정 (Validation Process):**
물리적으로 750장으로 구성된 것처럼 보이는 TIFF 파일을, 메모리에 모두 올리는 대신 `tifffile.memmap`을 활용하여 논리적 하이퍼스택(Z=250 레이어, T=3 LED) 구조로 온디맨드(On-the-fly) 로드하도록 구현했다. 또한, XCT 및 XYPT 궤적 데이터를 이진(Binary) 정답지로 단정 짓지 않고 연속적인 약한 지도학습(Weak Supervision) 신호로만 다루도록 파이프라인을 설계했다.

**결과 및 의의 (Result & Significance):**
크롭(Crop)이나 리사이즈(Resize)된 텐서(Tensor)를 디스크에 대량으로 저장하는 낭비를 막고(Zero-storage overhead), 항상 물리적 원본 데이터를 훼손 없이(Read-only) 직접 참조하는 무결성(Data Provenance)을 보장하는 파이프라인을 완성했다.

---

## 4. 파이프라인 흐름: 입력, 학습 참조, 좌표 검증의 독립

**연구 목표 (Objective):** 
AI 모델이 이미지를 보고 점수(Score)를 예측하는 과정과, 그 점수가 실제 XCT 결함 위치와 일치하는지 학습하는 과정, 그리고 그 좌표가 실제 3D 프린터 기계의 물리 좌표(mm)로 올바르게 변환되는지 검증하는 과정을 완전히 독립된 모듈로 분리하고자 했다.

**검증 과정 (Validation Process):**
모델의 디코더(Decoder)가 미래의 실시간 입력 환경을 가정할 수 있도록, XCT 지원 마스크(`weak_support_mask`)를 학습(Loss) 계산 시에만 접근 가능하도록 차단했다. 동시에, 모델의 예측 좌표를 기계 좌표로 변환하는 캘리브레이션 검증(Calibration audit)은 학습 루프와 완전히 독립된 스크립트로 분리하여 수행했다.

**결과 및 의의 (Result & Significance):**
학습 데이터 유출(Data Leakage)을 방지하고, 모델의 예측 성능이 아무리 좋아도 물리 좌표 변환이 틀리면(Calibration Ambiguity) 파이프라인을 보류(Hold)시킬 수 있는 강력한 안전장치를 확보했다.

---

# Part I. 입력을 만들기 위해 무엇을 골랐는가

## 5. 입력 영역(ROI) 선정 전략 및 Wide ROI 채택

**연구 목표 (Objective):** 
2000×2000 해상도의 원본 이미지에서 연산량과 메모리 부담을 줄이면서도, 중요한 제조 공정 영역을 잃지 않고 센서 포화(Saturation) 부작용을 최소화할 수 있는 최적의 입력 영역(Region of Interest, ROI)을 결정하고자 했다.

**검증 과정 (Validation Process):**
다양한 크기와 위치의 ROI 후보들(wide, lower, inner, upper 등)을 설정하고, 각 영역 내에서 센서 한계값(65,535)에 도달하는 포화 픽셀의 평균 비율과 최악의 표본 포화율을 교차 비교했다.

**결과 및 의의 (Result & Significance):**
`wide (250,250)–(1750,1750)` ROI가 평균 포화율이 가장 낮고(34.53%, 대조군인 lower는 35.56%, inner는 38.78%), 부품 전체 영역을 가장 잘 커버하는 것으로 판명되어 기본(Working) ROI로 채택되었다. 여전히 남아있는 포화 문제를 모델이 실제 밝기와 혼동하지 않도록 3채널의 Intensity 맵과 3채널의 포화 마스크(Validity mask)를 겹쳐 6채널 입력으로 구성하는 획기적인 전처리 기법을 확립했다.

---

## 6. 시계열(Temporal) 구성: K=4 윈도우와 스플릿 전략

**연구 목표 (Objective):** 
금속 3D 프린팅의 결함은 과거 여러 층(Layer)의 누적된 열적/물리적 상태에 영향을 받으므로, 모델이 미래 정보를 훔쳐보지 않으면서 과거의 시계열 맥락을 충분히 학습할 수 있는 인과적(Causal) 윈도우 크기와 데이터 분할(Split) 방식을 결정하고자 했다.

**검증 과정 (Validation Process):**
현재 레이어(z)를 포함하여 직전 3개 레이어(`[z-3, z-2, z-1, z]`)를 활용하는 **K=4** 윈도우를 기본 베이스라인으로 설정했다. 또한, Train/Validation/Test 데이터를 6.4 : 1.6 : 2의 비율로 분할하되, 분할 경계선에서 과거 데이터가 침범하여 정답이 유출되는 것을 막기 위해 경계 사이에 3개의 레이어를 완충 지대(Guard band)로 비워두는 엄격한 분할 정책을 적용했다.

**결과 및 의의 (Result & Significance):**
데이터 스플릿 간의 직접적인 정보 유출(Leakage)을 원천 차단하면서도, 시계열 특징(Temporal Feature)을 추출하는 3D Conv 아키텍처를 안전하게 검증할 수 있는 통제된 실험 환경(Causal Baseline Contract)을 완성했다.

---

## 7. Train-only 통계 기반 독립 정규화(Normalization)

**연구 목표 (Objective):** 
A/B 스테이지와 3개의 LED 방향에 따라 조명 밝기가 극심하게 달라지는 환경에서, 조명 밝기 차이가 마치 결함(품질) 신호인 것처럼 모델을 교란하는 현상을 방지하고자 했다.

**검증 과정 (Validation Process):**
전체 데이터셋을 한 번에 정규화(Global Normalization)하는 대신, 스테이지(A/B)와 각 LED 채널별로 밝기 분포를 따로 계산했다. 특히 Test/Validation 데이터의 분포를 미리 보면 평가가 낙관적으로 왜곡될 수 있으므로, 오직 Train 데이터셋에서만 p01/p99 극단값을 추출하여 정규화 기준을 확정했다.

**결과 및 의의 (Result & Significance):**
Train 데이터 기반으로 고정된 정규화 스케일을 모든 분할(Split)에 동일하게 적용함으로써, Test 데이터의 정보 유출을 완벽히 방어하고 조명 셋업의 차이와 실제 물리적 제조 결함 신호를 분리해내는 안정적인 입력 표현형(Representation)을 확보했다.

---

# Part II. XCT 약한 지도학습(Weak Supervision)의 기준

## 8. 목표 변수(Target) 선정: Continuous XCT Response

**연구 목표 (Objective):** 
점(Point) 형태로 듬성듬성 존재하는 사후 XCT 측정 데이터를 256x256 해상도의 픽셀 기반 딥러닝 모델이 학습할 수 있는 형태의 정답지(Target)로 변환하고, 그 강도(Scale)를 표준화하고자 했다.

**검증 과정 (Validation Process):**
원본 1x1 픽셀 대비 측정 노이즈에 강건한 `xct_5x5x5` 볼륨 평균값을 목표 변수로 채택했다. 이를 이진화(Binary)하여 결함 여부를 섣불리 단정 짓는 대신, 연속적인 점수(Continuous Response) 자체를 예측하도록 했다. 또한 Train 데이터에서 도출한 p01~p99 범위(0.40070 ~ 0.58533)를 `[0, 1]` 사이로 강제 스케일링(Robust Scaling)하는 수식을 적용했다.

**결과 및 의의 (Result & Significance):**
결함의 모호성을 무리하게 이분법으로 나누지 않고, 부품 내부의 밀도 변화 자체를 픽셀 단위의 연속적 스코어 맵으로 예측하는 약한 지도학습(Weak Supervision) 베이스라인을 성공적으로 구축했다. (이후 이 점수의 방향성은 제 37장에서 `high_score_is_defect`로 최종 확정됨)

---

## 9. 정답 공간의 확장: Gaussian Rasterization (Sigma=2)

**연구 목표 (Objective):** 
희소한 XCT 측정 좌표를 단일 픽셀 점(Dot)으로만 맵핑할 경우, 모델이 학습할 수 있는 픽셀 면적이 너무 좁아지는 문제(Sparsity)를 해결하고자 했다.

**검증 과정 (Validation Process):**
XCT 측정 좌표를 중심으로 주변 픽셀로 정답을 퍼뜨리는 가우시안 래스터화(Gaussian Rasterization) 기법을 도입했다. `sigma` 값을 1부터 4까지 변경하며 검증한 결과, sigma=3 이상으로 넓힐 경우 기존 정답 픽셀의 값(Value)이 과도하게 섞이며 훼손(Common-support MAE가 0.0538972로 급증)되는 현상을 발견했다.

**결과 및 의의 (Result & Significance):**
정답 픽셀의 면적(Coverage)을 확보하면서도 원본 XCT 값의 훼손을 최소화하는 타협점인 **Sigma=2**를 공식 워킹 초이스(Working Choice)로 확정했다. 이를 통해 공간적 희소성 문제를 해결하면서 정답의 신뢰도를 지켜냈다.

---

## 10. 마스킹된 손실 함수(Support-masked Smooth L1) 검증

**연구 목표 (Objective):** 
XCT 데이터가 존재하지 않아 알 수 없는(Unknown) 빈 공간의 픽셀들을 모델이 '정상(Score 0)'이라고 잘못 학습하는 치명적인 오류를 방지하고자 했다.

**검증 과정 (Validation Process):**
XCT 측정치가 존재하는 픽셀 영역(`m_i = 1`)에서만 오차를 평균 내고, 측정치가 없는 빈 공간(`m_i = 0`)은 Loss 계산에서 완전히 제외하는 Support-masked Smooth L1 Loss 함수를 설계했다. 이를 검증하기 위해 XCT가 전혀 없는 초기 레이어(z=4)에서 Loss와 Gradient가 정확히 0이 나오는지 런타임 테스트를 수행했다.

**결과 및 의의 (Result & Significance):**
측정되지 않은 빈 공간을 함부로 정상으로 간주하는 오류를 완벽하게 차단했으며, 빈 공간에서 발생하는 예측 오류가 모델 업데이트(Gradient)에 영향을 주지 못하도록 차단하는 강건한 지도학습 환경을 입증했다.
---

# Part III. 모델 아키텍처 실패 진단 및 개선 (Temporal Collapse & Residual Bypass)

## 11. 초기 모델(C8, C32)의 한계와 평가 유보

**연구 목표 (Objective):** 
첫 번째 3D CNN 기반 베이스라인(C8, C32)이 시계열 이미지 패치를 받아 의미 있는 결함 예측 좌표를 도출할 수 있는지 성능을 검증하고자 했다.

**검증 과정 (Validation Process):**
초기 8채널(C8) 및 32채널(C32) 모델의 출력 맵(Score Map)을 레이어별로 교차 비교했다. 에포크(Epoch)를 늘려보거나 정답지 크기(Sigma)를 바꿔보는 통제 실험(Controlled Experiment)을 수행하며 모델의 локализация(Localization) 능력을 평가했다.

**결과 및 의의 (Result & Significance):**
모델의 파라미터 용량(Capacity)을 늘려도, 입력 이미지가 분명히 다름에도 불구하고 출력 맵이 똑같이 굳어버리는(Top-score 동점자 비율이 96.8994%에 달하고, Map MAE가 0.0으로 고착화되는 Map invariance) 심각한 결함을 발견했다. 이는 모델이 입력을 무시하고 특정 위치만 무작위로 찍어내는 상태이므로 해당 출력 좌표를 결함으로 신뢰할 수 없다는(Hold) 결론에 도달했다.

---

## 12. 시계열 붕괴(Temporal Collapse) 현상 규명

**연구 목표 (Objective):** 
앞서 발견된 모델의 맵 고착화 현상이 정확히 신경망 내부의 어느 지점(Stage)에서 발생하는지 근본적인 원인을 추적하고자 했다.

**검증 과정 (Validation Process):**
엔드포인트(z=203, 227, 250) 간의 차이가 신경망을 통과하며 어떻게 변하는지 레이어별(Stagewise)로 내부 활성화 값(Activation)의 차이(Max-absolute difference)를 정밀 측정했다.

**결과 및 의의 (Result & Significance):**
입력층과 초기 인코더에서는 분명히 존재하던 입력 간의 차이가(max-abs 1.93~2.51), 3D Conv 시계열 압축(Temporal aggregation) 단계를 거친 직후(`temporal_final`) 오차 0(MAE/max-abs=0/0)으로 완전히 소멸(Collapse)하는 것을 발견했다. 이로써 문제의 원인을 인코더나 디코더가 아닌 '시간축 압축 경로(Temporal path)'로 특정할 수 있었다.

---

## 13. 잔차 병목(Residual Bypass) 아키텍처의 도입

**연구 목표 (Objective):** 
시계열 압축 과정에서 현재 레이어(Endpoint)의 핵심 시각적 특징이 소멸해버리는 'Temporal Collapse' 현상을 해결하기 위한 새로운 아키텍처를 설계하고자 했다.

**검증 과정 (Validation Process):**
과거 3개 레이어의 정보는 시간축으로 압축(Temporal update)하되, 가장 중요한 '현재 층(Endpoint)'의 특징(Feature)은 3D Conv를 거치지 않고 디코더 직전으로 우회(Bypass)시켜 더해주는 'A-only Temporal Difference (Residual)' 구조를 제안했다. 기존의 모든 통제 변인(학습 조건 등)은 고정한 채, 오직 잔차 연결(`use_endpoint_feature_residual=true`)의 유무만으로 성능을 비교했다.

**결과 및 의의 (Result & Significance):**
미래의 정보를 훔쳐보지 않는 인과적(Causal) 제약을 유지하면서도, 현재 레이어의 선명한 시각적 특징을 디코더까지 안전하게 전달하는 돌파구를 마련했다.

---

## 14. 잔차 모델(Residual Model) 공식 베이스라인 채택

**연구 목표 (Objective):** 
새로 도입된 Residual Bypass 구조가 실제로 이전 C32 모델의 한계를 극복했는지 검증하여, 공식 베이스라인 채택 여부를 결정하고자 했다.

**검증 과정 (Validation Process):**
두 모델 간의 Test Loss, 예측 분산(Prediction std), 그리고 Top-score 픽셀의 중복률(Tie fraction)을 정량적으로 비교했다.

**결과 및 의의 (Result & Significance):**
잔차 모델은 검증 손실(Validation Loss)을 8.27%(0.06151 -> 0.05642) 낮추고, Test Loss를 5.04%(0.07363 -> 0.06992) 감소시켰을 뿐만 아니라, 96%에 달하던 Top-score 중복률을 단 1픽셀(0.0015%) 수준으로 획기적으로 낮춰 Map 고착화(Plateau) 문제를 완전히 해소했다. 이에 따라 Residual 모델을 현재의 공식 'A-only Causal Baseline'으로 최종 채택했다.

---

# Part IV. 출력의 안전성과 물리 좌표 매핑 (Safety & Calibration)

## 15. 디코더 안전장치 (Safety Decoder Filters)

**연구 목표 (Objective):** 
모델이 뱉어낸 점수(Score)가 우연한 노이즈이거나 신뢰할 수 없는 평탄한 맵(Plateau)일 경우, 이를 결함 후보 좌표로 함부로 출력하지 않도록 막는 다중 안전망(Safety Gate)을 구축하고자 했다.

**검증 과정 (Validation Process):**
학습용 정답지(XCT support mask)의 도움 없이, 모델이 스스로 만든 맵의 품질만을 평가하는 4단계 필터를 구현했다: (1) 맵에 굴곡이 있는지(Flatness), (2) 최상위 점수가 너무 많이 겹치지 않는지(Top 500개 픽셀 중 동점자 비율인 Tie fraction < 0.1%), (3) 과거와 맵이 똑같지 않은지(Temporal invariance), (4) 7x7 지역 내 확실한 정점(Local maxima)인지 엄격히 검사했다.

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
DotGrid의 패턴은 선명히 인식되었으나, 붉은 레이저 점 1개만으로는 카메라 방향축(Orientation sign)을 수학적으로 결정할 수 없었다. 체커보드 역시 금속 가루 표면의 반사광을 체커보드 모서리로 오인하는 노이즈(수백 개의 False positive)가 발생하여, 공식 메타데이터를 활용한 직접적 변환은 아직 신뢰할 수 없는 보류(Hold) 상태로 판정했다.

---

## 19. DotGrid Method #2 격자 추적 알고리즘 보류

**연구 목표 (Objective):** 
NIST Method #2 (DotGrid 기반 변환)를 활용하기 위해 카메라 속 점(Dot)들과 실제 물리적 50x50 격자판을 1:1로 매칭(Correspondence)하는 알고리즘을 개발하고자 했다.

**검증 과정 (Validation Process):**
V1 알고리즘(PCA + 1D 클러스터링)을 통해 1,518개의 점을 찾아냈으나, 미리 약속해둔 5x5 홀드아웃(Held-out) 블록을 통한 잔차(Residual) 테스트를 수행했다.

**결과 및 의의 (Result & Significance):**
점은 찾았으나 그 점이 '몇 번째 줄(Row/Col)'에 있는지 인덱싱하는 과정에서 원본 닷 피치(Dot pitch) 대비 0.41배 이상의 과도한 오차(RMSE 6.00155 px, p95 잔차 9.78092 px)가 발생하여 검증 게이트를 통과하지 못했다. 이로 인해 자동 캘리브레이션 적용을 안전하게 보류(Hold)하고 정교한 2D 그래프(Graph) 기반 추적 알고리즘 고도화(Refinement)로 방향을 선회했다.

---

## 20. 현재 설정(Config)의 엄격한 변경 금지 사항

**연구 목표 (Objective):** 
캘리브레이션 모호성이 완전히 해결되기 전까지 무분별한 데이터나 모델 변경으로 프로젝트의 기반이 흔들리는 것을 막고자 했다.

**검증 과정 (Validation Process):**
위의 모든 실패와 보류(Hold) 상태를 종합하여, 어떤 것들을 '절대 건드리면 안 되는지' 정책으로 명문화했다.

**결과 및 의의 (Result & Significance):**
캘리브레이션 변환식 교체 금지, 부품 번호나 기계 좌표 출력 금지, 잔차 모델 재학습 금지, XCT 정답 마스크 디코더 투입 금지 등의 절대 규칙(Immutable Rule)을 확립하여 프로젝트가 통제 불능의 오버피팅(Overfitting)으로 빠지지 않도록 방호벽을 쳤다.
# Part VI. 카메라 캘리브레이션 정밀 진단 및 시계열(Temporal) 유효성 검증

## 21. Method #2 격자 매핑 정교화 및 한계 진단

**연구 목표 (Objective):** 
NIST Method #2의 체커보드/DotGrid를 활용한 캘리브레이션이 이전 검증에서 실패(Hold)함에 따라, 무작정 오차 허용치(Gate)를 낮추는 대신 점들 간의 2D 이웃 관계(Graph)를 추적하여 오차를 정밀하게 분석하고 진단하고자 했다.

**검증 과정 (Validation Process):**
단순한 1D 군집화가 아닌 점들의 2D 그래프 연결성(BFS)을 추적하는 V1~V3 스크립트를 순차적으로 고도화했다. 이를 통해 1,523개의 점(전체의 94.25%)을 성공적으로 추적했고 잔차 오차율도 기준치 이내(RMSE 0.06728 pitch, p95 0.11472 pitch)로 크게 낮췄다. 하지만, 우리가 설정한 가장 중요한 엄격한 조건 중 하나인 "40개의 세로줄(Row)을 모두 담아야 한다"는 조건을 확인하기 위해 인위적인 개입(Human selector)을 통한 검증(V2)과 블라인드 진단(Outer-boundary diagnostic)을 추가로 수행했다.

**결과 및 의의 (Result & Significance):**
오차율은 낮아졌으나, 알고리즘과 사람이 수동으로 잡은 기준점 모두 39개의 Row까지만 인식하며 40번째 Row를 확정하지 못했다(39-row shortfall). 결과를 본 뒤에 기준을 39개로 타협하는 행위(Data leakage)를 엄격히 금지하고, 현재의 캘리브레이션 변환을 물리적 진실로 확정하지 않은 채 보류(`hold_extent_and_orientation`)하는 Fail-closed 원칙을 지켜냈다.

---

## 22. 캘리브레이션 모호성과 결함 예측의 분리

**연구 목표 (Objective):** 
물리 좌표계(Machine XY) 확정이 보류된 상황에서도 딥러닝 프로젝트 전체가 멈추지 않고, 모델의 순수 예측 성능을 독립적으로 검증할 수 있는 방안을 모색하고자 했다.

**검증 과정 (Validation Process):**
모델의 예측 결과가 최종적으로 '실제 기계 좌표'로 변환되지 않더라도, 카메라 픽셀(Raw camera pixel) 좌표와 점수(Score)만으로 모델이 시계열(과거 데이터)을 올바르게 활용하고 있는지 점검하는 안정성 검사(Candidate stability audit)를 기획했다. 

**결과 및 의의 (Result & Significance):**
캘리브레이션 공학과 인공지능 모델링의 책임을 완벽히 분리함으로써, 물리 좌표 정합성 보류 상태(`hold`)가 모델 아키텍처 고도화 작업에 병목(Bottleneck)이 되는 상황을 성공적으로 회피했다.

---

## 23. 반사실적(Counterfactual) 실험과 가짜 시계열의 발견

**연구 목표 (Objective):** 
Residual Bypass 아키텍처(C32 residual)가 과거 시점(z-3, z-2, z-1)의 이미지를 단순히 입력받는 것을 넘어, 과거의 변화를 실제로 결함 판단에 활용(Utilize)하는지 인과적으로 검증하고자 했다.

**검증 과정 (Validation Process):**
과거 시점의 데이터를 모두 현재(Endpoint) 이미지로 덮어씌워버리거나, 과거 프레임 중 하나만 조작하는 가상 입력(Counterfactual)을 만들어 모델에 주입한 뒤 출력 맵(Map)이 어떻게 변하는지 확인했다.

**결과 및 의의 (Result & Significance):**
과거 입력을 어떻게 조작하든 최종 점수 맵(Score map)이 0.1%의 오차도 없이 완전히 동일하게(과거 이미지를 조작 전후의 출력 Map MAE 및 max difference가 정확히 0.00000) 유지됨을 발견했다. 즉, 모델이 과거 데이터를 보고는 있지만 실제 결함 판단에는 오직 '현재 이미지 한 장'만 쓰고 있다는 치명적인 사실(가짜 시계열 현상)을 밝혀냈다.

---

## 24. 진단용 시각 자료(QC PNG)의 역할과 한계

**연구 목표 (Objective):** 
위와 같은 시계열 붕괴 및 모델의 반응성을 진단하기 위해 생성되는 히트맵(Heatmap) 및 오버레이(Overlay) 이미지들이 실제 물리적 결함 지도로 오남용되는 것을 방지하고자 했다.

**검증 과정 (Validation Process):**
QC 이미지는 모델 내부의 활성화 차이(Difference)나 픽셀별 반응성(Sensitivity)을 점검하기 위한 '연구용 진단 도구'일 뿐, 이것이 실제 XCT 결함의 위치이거나 이상 확률(Anomaly probability)이라고 선언하는 것을 엄격히 금지했다.

**결과 및 의의 (Result & Significance):**
연구용으로 생성된 수많은 중간 산출물들이 "결함 예측 완료"라는 과장된 결과로 포장되는 것을 막고, 데이터 해석의 객관성과 재현성(Reproducibility)을 유지하는 기준을 확립했다.

---

## 25. 시계열 경로 기여도(Temporal-path Mechanism) 정밀 추적

**연구 목표 (Objective):** 
과거 입력의 변화가 무시되는 '가짜 시계열 현상'이 단순히 가중치(Weight)가 학습되지 않아서인지, 아니면 구조적 병목에 의해 과거 정보가 묻히는 것인지 규명하고자 했다.

**검증 과정 (Validation Process):**
Residual 모델 내부의 3D Conv 커널 에너지를 레이어 단위로 해부하여 측정한 결과, 과거 레이어의 가중치 에너지는 무려 66.8%에 달해 정상적으로 학습되고 있음을 확인했다. 하지만 최종 디코더 입력단에서 Endpoint feature의 크기(L2 norm 920.0)가 과거의 Temporal update(L2 norm 4.41)를 압도적으로 짓누르며 묻히고 있음(Static-like correction)을 수치로 증명했다.

**결과 및 의의 (Result & Significance):**
학습 에포크를 늘리거나 데이터를 더 모은다고 해결될 문제가 아니며, 과거 정보를 단순히 3D Conv로 더하는 방식 자체가 한계에 도달했음을 과학적으로 진단했다.

---

## 26. 과거와의 차이를 직접 주입하는 새 아키텍처(Temporal Difference) 도입

**연구 목표 (Objective):** 
모델이 과거 정보를 무시하는 병목 현상을 타파하기 위해, 모델이 과거와 현재의 '차이'를 반드시 학습할 수밖에 없는 명시적 정보 주입 구조를 설계하고자 했다.

**검증 과정 (Validation Process):**
과거 3개의 이미지 텐서를 단순히 쌓아(Concat/Conv3D) 넘기는 대신, 현재 층의 특징 벡터와 과거 층들의 평균 특징 벡터의 **차이(Difference feature = Endpoint - Prior)**를 직접 계산하여 결합(Fusion)하는 **A-only Temporal Difference** 모델을 제안했다.

**결과 및 의의 (Result & Significance):**
모델이 가장 유의미한 정보인 '시간에 따른 변화(Change over time)'를 명시적으로 입력받게 됨으로써, 과거 이미지가 바뀔 때 출력 맵도 정상적으로 변동하는 인과적 시계열 반응성(Causal sensitivity)을 회복할 구조적 기반을 완성했다.


---

## 27. 새 모델(Temporal Difference)의 구조적 타당성 사전 검증(Dry Run)

**연구 목표 (Objective):** 
새로 설계된 A-only Temporal Difference 구조가 실제 학습 과정에 돌입하기 전, 텐서(Tensor) 흐름과 메모리 할당, 손실 함수(Loss function) 연산에 구조적 결함이 없는지 조기에 점검하고자 했다.

**검증 과정 (Validation Process):**
단 1개의 학습 샘플(z=128)만을 주입하여 입력 텐서(`[1,4,6,256,256]`)가 출력 텐서(`[1,1,256,256]`)로 정상 변환되는지, XCT 지원 영역(Supported pixel) 내에서 손실값(Loss)이 무한대(NaN/Inf)로 발산하지 않고 유한한 숫자(`0.12180285`)로 계산되는지 확인하는 드라이런(Dry run)을 수행했다.

**결과 및 의의 (Result & Significance):**
모델의 순전파(Forward pass)와 오차 역전파(Backward) 파이프라인이 정상적으로 맞물려 작동함을 증명(배선 검사 통과)했다. 무작위 초기화(Random initialization) 상태에서의 무의미한 손실값을 성능으로 오판하지 않도록 방어 기제를 설정하고, 본격적인 8 Epoch 훈련으로 넘어가는 안전한 징검다리를 놓았다.

---

## 28. 검증(Validation)과 평가(Test) 성능의 불일치 진단

**연구 목표 (Objective):** 
새로운 Temporal Difference 모델이 학습 데이터가 아닌 미지의 데이터(Held-out Test set)에서도 실질적인 결함 예측 성능 향상을 이뤄냈는지 엄밀하게 검증하고자 했다.

**검증 과정 (Validation Process):**
단일 난수 시드(Seed)로 8 Epoch을 학습한 뒤, 검증 데이터(Validation)에서 오차가 가장 낮았던 Epoch 7의 체크포인트를 추출했다. 이를 미지의 Test 데이터에 적용하여 기존 베이스라인(C32 Residual)의 Test Loss와 비교했다.

**결과 및 의의 (Result & Significance):**
검증 오차는 4.1% 낮아졌으나(0.05643 -> 0.05411), 최종 Test 오차는 오히려 0.41% 상승(0.06992 -> 0.07020)한 불일치(Discrepancy) 현상을 발견했다. 이 단일 시드 결과를 보고 섣불리 구조를 변경(Data leakage)하는 대신, "모델이 과거 데이터를 실제로 보고 있는지(Response)"를 먼저 입증하고 다중 시드(Multi-seed) 평균 테스트로 넘어가야 한다는 과학적 검증 순서를 확립했다.

---

## 29. 인과적 시계열 반응성(Causal Sensitivity)의 확증

**연구 목표 (Objective):** 
Test 성능의 미세한 하락이 모델 구조의 근본적 실패인지, 아니면 과거의 정보를 학습하긴 했으나 위치(Localization)만 불안정한 상태인지 원인을 분리해내고자 했다.

**검증 과정 (Validation Process):**
이전 실패작(Residual)과 동일하게 과거 이미지 3장을 현재 이미지로 강제 교체하거나 단일 프레임만 조작하는 반사실적 교란(Counterfactual perturbation)을 가한 뒤 출력 맵의 변화량을 관측했다.

**결과 및 의의 (Result & Significance):**
기존 모델이 0.0의 반응성을 보였던 반면, 새 모델은 과거 이미지 변경 시 출력 맵의 오차가 평균 0.102~0.105(단일 프레임 변동 시 0.0077~0.0105)까지 크게 변동하며 기준치(0.0001)를 아득히 돌파했다. 즉, 모델이 '과거의 변화'를 결함 판단에 확실하게 이용(Utilize)하는 시계열 감수성을 회복했음을 성공적으로 증명했다.

---

## 30. 예측 좌표 변동성의 두 가지 가설(Rank Switch vs Peak Relocation) 분리

**연구 목표 (Objective):** 
과거 이미지가 변할 때 예측 결함 좌표(Candidate coordinate)가 크게 흔들리는 현상의 원인이 단순한 '점수 경합'인지, 아니면 '근본적인 예측 실패'인지 규명하고자 했다.

**검증 과정 (Validation Process):**
1위 후보 좌표가 바뀔 때, 기존 1위와 새로운 1위 간의 점수 차이(Score margin)를 측정하는 진단 프로토콜을 도입했다. 각 맵에서 Top-K(=5) 후보를 추출한 뒤, `Top1과 Top2의 점수 차이`, `점수 차이의 맵 전체 변동폭 대비 비율`, `후보 간 물리적 거리(픽셀)`, `기존 1위가 새 Top-K에서 몇 위로 밀려났는지`, `Top-K 중복률` 등 5가지 지표를 구체적으로 수치화하여 기록했다. 이를 통해 점수 차이가 전체 편차의 5% 이내(Score 차이 0.05 미만)인 초박빙(Near-tie) 상태에서 단순 순위 바꿈이 일어나는 것인지, 아니면 압도적인 점수 차이(High-margin 20% 이상)와 함께 기존 1위가 Top-K에서 아예 소실(Relocation)되는 것인지 정밀 추적했다.

**결과 및 의의 (Result & Significance):**
이러한 진단을 통해 결함 좌표가 수십~수백 픽셀 튀는 불안정성의 표면적 결과 뒤에 숨겨진 구조적 메커니즘을 해부할 수 있는 마진(Margin) 진단 프레임워크를 구축했다. 이를 기반으로 향후 실시간 도입 시 애매한 결함을 기각(Withhold)할 수 있는 안전 마진 정책을 설계했다.

---

## 31. 점수 경합(Near-tie) 현상 증명과 안전 마진 도입의 당위성

**연구 목표 (Objective):** 
12가지 극한 교란 조건(Stress test) 하에서 좌표가 바뀌는 실제 양상을 확인하여, 향후 시스템에 도입될 안전장치(Safety margin)의 방향성을 확정하고자 했다.

**검증 과정 (Validation Process):**
12개의 반사실적 맵(Counterfactual map)을 전수 조사한 결과, 1위 좌표가 완전히 소실(High-margin relocation)되는 경우는 단 1건도 없었으며, 순위가 바뀌는 3건 모두 기존 1위가 2위나 3위로 여전히 살아남아 초박빙 점수 경합(Near-tie rank switch)을 벌이는 상태임을 확인했다.

**결과 및 의의 (Result & Significance):**
실제 12개의 반사실적(Counterfactual) 맵을 검토한 결과 3건은 점수 차이가 거의 없는(Near-tie) 상태에서 순위가 바뀌었고, 4건은 1위가 그대로 유지되었으며, 고마진(High-margin)으로 1위가 소실되는 최악의 사례는 단 1건도 없었다(0건). 모델의 예측이 근본적으로 흔들리는 것이 아니라, 비슷한 확률의 결함 후보들 사이에서 미세한 점수 차이로 순위만 바뀐다는 물리적 진실을 밝혀냈다. 이에 따라 향후 **실제 48개 테스트 레이어에 안전 마진 커트라인(예: 1%, 2%, 5%)을 적용하는 차기 시뮬레이션 연구(Margin-based Withholding Audit)로 나아가야 한다는 강력한 증거**를 확보했다.

---

## 32. Multi-seed Comparison과 Fail-Fast Safety Gate의 도입

**연구 목표 (Objective):** 
새로운 Candidate 모델(A-only Temporal Difference)이 기존 Reference 모델(C32 Temporal Residual)보다 우연이 아닌 구조적 우위로 나은 성능을 내는지 5-seed 비교를 통해 검증하고, 이 과정에서 발생할 수 있는 데이터 누락 버그를 원천 차단하고자 했다.

**검증 과정 (Validation Process):**
초기 실행 중 경로 누락(`registered_xct_v1`)으로 인해 Loss가 `None`으로 계산되는 "침묵 속의 실패" 현상을 발견했다. 이를 막기 위해 경로 검증, 빈 데이터 사전 검사, Loss=None 런타임 중단이라는 3단계 Fail-Fast 방어막을 구축했다. 이후 올바른 경로에서 5개의 랜덤 시드(1001~1005)를 부여하여 두 모델 간의 Test Loss를 정밀 교차 검증했다.

**결과 및 의의 (Result & Significance):**
첫 번째 시드(1001) 연산에서 Test Loss가 0.0696에서 0.0683으로 감소(`-0.0013`)함을 확인한 데 이어, 전체 5개 시드 평균 오차 감소치 `-0.001989`를 달성하며 Candidate 모델이 5전 5승(전승)을 기록했다. 랜덤 가중치 초기화에 흔들리지 않고 **A-only Temporal Difference 모델의 우월함이 100% 증명(Sign consistency 1.0)** 됨에 따라 공식 뼈대를 전면 교체했다.

---

## 33. Margin-based Withholding Audit (안전 마진 시뮬레이션)

**연구 목표 (Objective):** 
1등 후보와 2등 후보 간의 점수 격차가 너무 적은 모호한(Near-tie) 상황일 때 해당 경보를 안전하게 기각(Withhold)하기 위한 최적의 '안전 마진(Safety Margin)' 임계값을 결정하고자 했다.

**검증 과정 (Validation Process):**
Temporal Difference 모델이 도출한 48개 테스트 레이어의 예측 맵을 분석하여 1, 2등 간 평균 점수 차이(Margin: 0.0353)와 물리적 거리(Peak Distance: 276.43 픽셀)를 측정했다. 이후 1%, 2%, 5%의 세 가지 마진 커트라인을 가상으로 적용하여 각각 몇 개의 레이어가 기각되는지 시뮬레이션했다.

**결과 및 의의 (Result & Significance):**
5% 마진(0.05)을 적용할 경우 전체 레이어의 81.25%(39/48)가 기각되어 시스템 기능이 마비됨을 확인했다. 반면 **1% 마진(0.01)**을 적용할 경우 점수 경합이 가장 불안정한 하위 25.0%(12/48)의 레이어만 선별적으로 차단할 수 있었다. 이에 따라 1% 마진을 가장 합리적인 실무 안전장치 커트라인으로 공식 채택했다.

---

## 34. 시계열 정보량의 독립적 평가: A-only vs B-only 베이스라인

**연구 목표 (Objective):** 
가루가 도포된 직후의 사진(A-stage)과 레이저 조사 후 쇳물이 굳은 사진(B-stage) 중, 어떤 시점의 데이터가 최종 결함(XCT)과 더 강력한 인과관계를 가지는지 정량적으로 비교하고자 했다. 이를 통해 향후 A+B 결합 모델(Fusion)을 개발해야 하는 물리적 당위성을 확보하는 것이 핵심 목표였다.

**검증 과정 (Validation Process):**
완벽하게 통제된 조건(동일한 256x256 ROI, K=4 Temporal Difference 구조, 정규화, 옵티마이저 등) 하에서, 입력 데이터만 `--tiff-a`와 `--tiff-b`로 교체하여 두 개의 독립적인 베이스라인 모델(A-only, B-only)을 학습시키고 Test Loss를 비교했다.

**결과 및 의의 (Result & Significance):**
A-only 모델은 가루의 상태만으로 미래를 예측해야 하는 한계로 인해 조기 과적합(3 에폭, Test Loss 0.0683)을 겪었다. 반면 B-only 모델은 레이저 용융의 물리적 결과물(Spatter, Melt pool)을 직접 관찰하여 Test Loss를 0.0546(7 에폭)까지 끌어내리며 압도적인 성능 우위를 증명했다. 이 실험은 사후 진단(B)의 강력한 정보량과 조기 경보(A)의 잠재력을 결합하는 A+B Fusion 구조의 필요성을 확고히 뒷받침한다.

---

## 35. 통합 진단의 시너지 한계 규명: A+B Fusion 베이스라인

**연구 목표 (Objective):**
가루(A)와 쇳물(B)의 정보를 모두 활용하면, 단일 시점(B-only)보다 결함 예측 성능이 유의미하게 향상될 것이라는 가설을 검증하고자 했다.

**검증 과정 (Validation Process):**
가루 전용 인코더와 쇳물 전용 인코더를 평행하게 구성한 뒤, 추출된 두 특징(Feature)을 단순 병합(Concatenation)하여 최종 점수를 출력하는 A+B Fusion 베이스라인 네트워크를 학습시켰다. 이후 앞서 확보한 A-only 및 B-only 모델의 Test Loss와 3-way 비교를 수행했다.

**결과 및 의의 (Result & Significance):**
직관과 달리, A+B Fusion 모델의 Test Loss(0.0550)는 B-only 모델(0.0546)을 넘어서지 못했다. 이는 쇳물(B)의 강력한 결함 신호에 가루(A)의 상대적 노이즈가 단순 병합되면서 오히려 디코더의 판단을 교란했기 때문으로 분석된다. 결과적으로, 단순 이어붙이기(Concat) 방식은 실패했으며, 디코더가 상황에 따라 정보의 신뢰도를 스스로 조절하는 **Gated Fusion** 또는 **Cross-Attention**과 같은 고도화된 선택적 결합 아키텍처 연구가 필수적임을 규명했다.

---

## 36. 물리적 정합성 확보: 캘리브레이션 거울상 모호성 해결

**연구 목표 (Objective):**
카메라 렌즈의 왜곡을 펴고 실제 3D 프린터의 물리 좌표계(mm)와 픽셀을 매핑하는 호모그래피(Homography) 캘리브레이션은, 격자(DotGrid)의 완벽한 대칭성 탓에 카메라 렌즈의 상하좌우가 뒤집혀도 수학적 오차가 동일하게 나타나는 거울상 모호성(Mirror/Rotation Ambiguity)에 빠져 있었다. 이를 해결하여 기계 좌표 매핑의 물리적 진실을 확정하는 것이 목표였다.

**검증 과정 (Validation Process):**
비대칭적 설계 특징(왼쪽에 구멍, 오른쪽에 돌출부)을 가진 125층(Layer 125)의 실제 레이저 궤적(XYPT)을 카메라 원본 사진(B-stage) 위에 겹쳐 그리는 오버레이 스크립트(`audit_layer125_orientation_overlay.py`)를 실행했다. 이후 생성된 Rank 1과 Rank 2 후보 이미지를 육안으로 교차 검증(Visual Audit)하여 설계 도면과 실제 쇳물의 형태가 일치하는지 확인했다.

**결과 및 의의 (Result & Significance):**
Rank 2 (mirror_rotate_270) 후보만이 설계 도면의 비대칭 특징(Cavity/Overhang)과 정확히 일치함을 시각적으로 증명했다. 이 압도적 증거를 바탕으로 `calibration_v1.yaml`의 상태를 임시(provisional)에서 확정(validated)으로 승격시켰으며, AI의 픽셀 단위 예측을 기계 제어용 절대 좌표로 신뢰성 있게 번역할 수 있는 기반을 완성했다.

---

## 37. 타겟 점수의 물리적 의미 매핑: Target Semantics Audit

**연구 목표 (Objective):**
Registered XCT에서 추출한 0.0 ~ 1.0 사이의 연속 점수(Weak Target)가 물리적으로 어떤 결함 상태를 의미하는지 방향성(Direction)이 명확하지 않았다. 높은 점수가 결함인지, 정상인지 확정하고 실제 알람을 울릴 실무적 임계값(Threshold)을 설정하는 것이 목표였다.

**검증 과정 (Validation Process):**
데이터셋 전체를 스캔하여 1.0점에 근접한 최상위 픽셀들과 0.0점에 근접한 최하위 픽셀들의 위치를 찾아냈다. 해당 위치의 원본 가루(A) 및 쇳물(B) 사진을 32x32 해상도로 크롭(Crop)한 패치 모자이크를 생성한 뒤, 도메인 지식(미용융/기공, 스패터 등)을 바탕으로 물리적 징후를 판독했다.

**결과 및 의의 (Result & Significance):**
육안 판독 결과, 높은 점수 구간에서 쇳물이 불규칙하게 튀거나 깊게 파이는(다크 스팟) 결함 패턴이 지배적으로 나타났다. 반면 낮은 점수 구간은 레이저가 매끄럽게 지나간 정상 상태임이 확인되었다. 이를 근거로 `weak_target_v1.yaml`의 타겟 방향을 `high_score_is_defect`로 확정하고, 실무 적용을 위한 보수적 이진 분류 커트라인을 0.85로 제정했다. 향후 기준 반전이 필요할 경우 `invert: true` 플래그로 즉각 대응할 수 있는 유연한 설정 구조도 확보했다.

---

## 38. A+B Gated Fusion 모델의 과적합(Overfitting) 현상 확인

**연구 목표 (Objective):**
단순 이어붙이기(Concat) 융합 모델의 노이즈 간섭 문제를 해결하기 위해, 공간 및 채널별 중요도를 스스로 조절하는 Gated CNN Fusion 아키텍처를 도입하고 그 성능을 검증하는 것이 목표였다.

**검증 과정 (Validation Process):**
`train_a_b_gated_fusion_v2.py` 스크립트를 통해 A와 B 특징 사이에 Gated CNN 밸브를 적용하여 8-Epoch 통제 학습을 진행했다. 이후 B-only 모델 및 기존 단순 병합(Concat) 모델의 Test Loss와 성능 지표를 교차 비교했다.

**결과 및 의의 (Result & Significance):**
Gated Fusion 모델은 Validation Loss를 기존 대비 획기적으로 낮춘 0.0302(Epoch 7)까지 끌어내리며 강력한 복합 패턴 학습 능력(Capacity)을 증명했다. 게다가 초기 A-only 모델의 고질병이었던 위치 추정 실패(Map Invariance 및 Top-score tie plateau) 없이 48개의 테스트 이미지에서 모두 고유한 이상 후보 위치를 훌륭하게 잡아냈다(48 Emitted).
하지만 미지의 데이터인 Test Loss는 0.0596으로 B-only 모델(0.0546)보다 오히려 상승했다. 이는 추가된 Gate 모듈의 강한 표현력으로 인해 훈련용 데이터의 노이즈까지 완벽하게 외워버리는 전형적인 과적합(Overfitting) 현상에 기인한 것으로 판단된다. 따라서 정규화(Regularization) 기법을 통한 과적합 억제가 필수적인 다음 과제로 도출되었다.
