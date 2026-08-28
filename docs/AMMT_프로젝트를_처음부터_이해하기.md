# AMMT 프로젝트를 처음부터 이해하기

> **한 문장 요약:** 이 프로젝트는 LPBF 제조 중 레이저 조사 **전** 촬영한 layer-camera A 시계열만으로, 향후 확인이 필요한 화면 위치를 `(x_pixel, y_pixel, layer_z, score)` 형태의 **XCT-derived continuous quality candidate**로 제안하는 인과적 baseline을 만드는 작업이다.

이 문서는 코드 목록이 아니라, **무엇을 골랐고 왜 골랐는지**, 그리고 왜 어떤 결과를 통과 또는 보류했는지를 설명한다. 여기서 `score`는 확정 defect, anomaly probability, 원인 진단, 자동 공정 제어 신호가 아니다. 사후 XCT에서 유래한 continuous response와 약하게 대응하도록 학습한 후보 점수이며, response 방향도 아직 확정하지 않았다.

---

## 1. 현재 위치: 완료한 것과 아직 하지 않는 것

| 영역 | 상태 | 현재 결론 |
|---|---|---|
| 원본 보존·TIFF axis 확인 | 완료 | A/B TIFF·XCT CSV·metadata는 원본 그대로 read-only로 다룬다. |
| A-only causal input | 완료 | 현재 endpoint와 과거 3개 layer만 본다. 미래 layer와 B stage는 입력에 넣지 않는다. |
| ROI·normalization·saturation policy | 완료 | wide ROI를 **working ROI**로 고정하고, LED별 intensity와 saturation validity를 분리한다. |
| XCT weak supervision | 완료 | `xct_5x5x5` continuous response를 on-the-fly rasterize하며, 관측되지 않은 곳은 `unknown`이다. |
| C32 temporal-residual baseline | 완료 | temporal collapse를 해소하고 C32보다 held-out masked loss를 5.04% 낮췄다. |
| Candidate output safety | 완료·제한 있음 | flat/tie/temporal-invariance/local-maxima 검사를 통과한 candidate만 raw pixel로 출력한다. |
| Screen-control calibration | 완료·provisional | rank1/rank2 ambiguity가 남아 physical part identity를 주장하지 않는다. |
| Independent metadata feature audit | 완료 | DotGrid과 central red cluster는 보였으나 Checkerboard direct route는 보류다. |
| Independent method-#2 candidate transform | 완료·hold | panel coverage는 통과했지만 held-out dot residual이 기준을 넘었다. 모든 candidate transform을 보류한다. |
| 확정 defect 판정·공정 control | 미완료 | 현 output은 candidate이며 자동 제어 또는 pass/fail 판단에 쓰지 않는다. |

현재 가장 중요한 원칙은 다음과 같다.

> **모델이 공간적으로 변하는 score map을 만들 수 있게 된 것과, 그 pixel이 특정 machine XY 또는 특정 part의 물리 위치라는 것은 다른 주장이다.** 따라서 현재 primary location은 raw camera `(x_pixel, y_pixel)`이다.

---

## 2. 먼저 이해할 세 가지 분리

### 2.1 이것은 의료 CT 프로젝트가 아니다

이 프로젝트의 input은 의료 CT volume이 아니라, 금속 적층 제조 중 촬영된 layer-camera image다. XCT는 제조가 끝난 뒤의 사후 계측 참조이며, model이 실시간으로 보게 되는 input은 아니다. NIST AMMT Overhang Part X4는 이와 같은 layer camera·process·registered XCT 자료를 제공한다.[1] [2]

### 2.2 A와 B는 같은 사진의 전후 버전이 아니다

```text
Layer z 시작
  └─ A: AfterSpreading, 분말 도포 직후·레이저 전  ← 현재 real-time input
       └─ 레이저 노광
            └─ B: Burned, 레이저 후                ← calibration/QC와 미래 확장용
```

A와 B는 같은 `(layer_z, LED)` 기준으로 대응하지만, `B - A`를 defect label로 쓰면 안 된다. 그 차이에는 정상 레이저 용융, 반사광, scan path, 장비 구조도 들어 있기 때문이다. 현재 A-only baseline은 “레이저 조사 전에 무엇을 확인할지”라는 조기 candidate 목적에 맞춘 선택이다.

### 2.3 선정 기준에는 세 종류가 있다

| 기준의 종류 | 의미 | 예시 |
|---|---|---|
| **Data contract** | 원본 구조에서 반드시 지켜야 하는 규칙 | TIFF logical axis, read-only memmap, `65535` saturation 처리 |
| **Controlled selection** | 후보를 비교해 하나를 working choice로 정한 규칙 | wide ROI, train-only normalization, residual bypass |
| **Safety hold** | 근거가 부족해 다음 단계로 진행하지 않는 규칙 | sigma=3 training hold, rank/part identity hold, method-#2 transform hold |

이 구분이 중요하다. “현재 config에 들어 있다”는 말은 항상 “물리적으로 참이다”라는 뜻이 아니다. 어떤 값은 단지 공정하고 재현 가능한 baseline을 위해 고정한 working choice이고, 어떤 값은 검증 실패 때문에 유지되는 hold 상태다.

---

## 3. 원본 데이터와 접근 규칙

| 데이터 | 실제 구조·역할 | 왜 필요한가 | 하지 않는 일 |
|---|---|---|---|
| `LayerCameraAfterSpreading.tif` | ImageJ logical `TZYX=[3,250,2000,2000]`, `uint16` | A-only K=4 input | crop/resize tensor를 대량 저장하거나 원본을 수정하지 않음 |
| `LayerCameraBurned.tif` | A와 동일 axis | B-stage orientation/calibration QC, 후속 fusion 후보 | A-only current input에 사용하지 않음 |
| registered XCT CSV | command XY와 continuous response를 가진 sparse point table | weak supervision | binary defect mask로 단정하지 않음 |
| XYPT command CSV | commanded scan/geometry sequence | projection/calibration audit | exact physical truth로 단정하지 않음 |
| DotGrid / Checkerboard / SecondaryCamera TIFF | independent metrology metadata | camera-to-machine relation 검증 | raw file 수정·자동 config 교체 |

A/B TIFF는 물리 page가 750장처럼 보일 수 있어도, project code는 `tifffile.memmap(..., series=0, mode='r')`로 logical hyperstack을 읽는다. `T=3`은 LED, `Z=250`은 제조 layer이고, model의 시간축은 Z다. 이 접근은 6 GB 원본을 RAM에 전부 올리지 않고 필요한 history frame만 읽게 한다.[1]

---

## 4. 전체 흐름: 입력·학습 참조·좌표 검증은 서로 다르다

```text
Immutable raw data
  ├─ A TIFF ──► causal K=4 Dataset ──► [K, 6, 256, 256] input
  │                 │                    ├─ LED intensity 3 channels
  │                 │                    └─ saturation-validity 3 channels
  │                 ▼
  │            residual causal network ──► [1, 256, 256] score map
  │                                            │
  │                                            └─ safety decoder
  │                                               └─ (x_pixel, y_pixel, layer_z, score)
  │
  ├─ registered XCT CSV ──► on-the-fly continuous response + support
  │                              └─ training/evaluation loss에서만 사용
  │
  └─ metadata TIFF + documented NIST geometry
       └─ independent calibration audits
          └─ machine-coordinate interpretation의 근거를 검증
```

| 항목 | 사용 위치 | 왜 분리하는가 |
|---|---|---|
| `weak_support_mask` | 학습/evaluation loss | XCT가 측정하지 않은 pixel을 잘못된 normal target으로 만들지 않기 위해서 |
| Score-map decoder | 실제 candidate decoding | 미래 real-time input에는 XCT support가 없으므로 decoder가 support를 보면 안 됨 |
| Provisional geometry gate | local maxima 뒤 optional filter | configured rectangle 안 여부만 보는 보조 안전장치이며 physical truth가 아님 |
| Calibration audit | model과 분리된 metadata/geometry 검증 | score가 좋아도 coordinate mapping이 맞는지를 별도로 확인해야 함 |

---

# Part I. 입력을 만들기 위해 무엇을 골랐는가

## 5. ROI는 어떻게 고르고, 왜 wide ROI가 되었는가

### 5.1 ROI를 crop하는 목적

원본 frame은 2000×2000 pixel이다. 모델에 전체 frame을 그대로 넣으면 메모리·연산량이 크고, 실제 build region 밖의 chamber/fixture가 많이 섞인다. 반대로 너무 작은 crop은 유효한 공정 영역을 잃고, 포화가 줄어드는 것처럼 보여도 model이 봐야 할 구조를 잘라낼 수 있다.

그래서 ROI는 “보기 좋은 image”를 고르는 것이 아니라 다음 세 조건의 균형으로 골랐다.

1. **공정 영역 coverage:** 4개 part와 주변 powder field를 충분히 포함해야 한다.
2. **포화 부담:** `raw==65535` pixel의 평균 비율이 가능한 낮아야 한다.
3. **worst-case stability:** 일부 layer에서 saturation이 극단적으로 커지는 ROI를 피해야 한다.

### 5.2 실제 비교 후보와 결과

초기 audit은 아래 ROI 후보에서 A/B·LED·표본 layer의 saturation을 비교했다. `full-scale saturation`은 `uint16` sensor upper limit인 65,535와 정확히 같은 pixel 비율이다.

| ROI 후보 | Raw pixel 경계 `(x0,y0)–(x1,y1)` | 평균 full-scale saturation | 최악 표본 saturation | 선택에 미친 영향 |
|---|---|---:|---:|---|
| **wide** | `(250,250)–(1750,1750)` | **34.53%** | 92.42% | 평균 포화가 비교 후보 중 가장 낮고 build coverage가 넓음 |
| lower | `(350,450)–(1650,1750)` | 35.56% | **91.28%** | 최악값은 약간 낮지만, 평균은 wide보다 1.03%p 높고 coverage가 더 좁음 |
| inner | `(350,350)–(1650,1650)` | 38.78% | 96.42% | 중심 crop으로 saturation이 줄지 않음 |
| upper | `(350,250)–(1650,1550)` | 40.21% | 97.19% | upper region의 saturation 부담이 큼 |
| smaller inner | `(450,450)–(1550,1550)` | 40.52% | 99.10% | 가장 작은 중심 crop도 해결책이 아님 |

wide ROI는 평균 saturation이 가장 낮고 1500×1500 raw pixel의 넓은 공정 context를 유지한다. lower ROI는 worst-case saturation이 wide보다 1.14%p 낮았지만, 그 이점이 더 좁은 spatial coverage와 평균 saturation 손실을 정당화할 만큼 크지 않았다. 따라서 **wide는 현재의 working ROI**로 채택했다.

> **중요:** wide ROI는 “포화가 사라졌다”거나 “최종 물리 ROI다”라는 결론이 아니다. 비교 후보 중 가장 안전한 baseline 선택일 뿐이며, saturation은 별도 channel로 남겨야 한다.

### 5.3 ROI 선택 뒤에도 포화가 남아 왜 mask를 추가했는가

wide ROI에서도 stage/LED별 평균 saturation은 A LED1=35.62%, A LED2=51.19%, A LED3=11.90%, B LED1=45.44%, B LED2=49.94%, B LED3=13.11%였다. 즉 crop만으로 sensor ceiling 문제를 해결할 수 없고, 특히 LED1/2의 `65535`를 단순히 매우 밝은 physical signal로 해석하면 안 된다.

그래서 각 history layer에 대해 다음 두 종류의 channel을 함께 만든다.

| Channel | 계산 | 선정 이유 |
|---|---|---|
| Intensity 3ch | train-only normalization 뒤 `[0,1]` | LED별 raw brightness scale을 공통 model input range로 옮김 |
| Validity 3ch | `raw < 65535`이면 1, 아니면 0 | saturation과 실제 높은 intensity를 구분하게 함 |

모델용 grid는 256×256이다. wide ROI 1500 pixel을 256으로 줄이므로 model pixel 하나는 약 `1500/256 = 5.859` raw pixel 폭을 대표한다. crop/resize/mask는 Dataset이 메모리에서만 만들며, dense model frame이나 saturation mask file은 저장하지 않는다.

---

## 6. 시간 sequence K=4와 split은 어떻게 고정했는가

### 6.1 K=4를 현재 baseline으로 둔 이유

현재 sequence는 endpoint z를 포함한 `[z-3,z-2,z-1,z]` 네 layer다. K=4는 충분한 과거 context를 주면서도 첫 usable endpoint가 z=4가 되게 하는 **initial causal baseline contract**다. K=4가 전 제조 현상에 대해 최적인지를 이미 증명한 것은 아니다. 현재 우선순위는 더 긴 sequence sweep이 아니라, 미래 information leakage 없이 A-only data path·weak supervision·candidate safety가 정확하게 동작하는지 확인하는 것이었다.

| K=4가 주는 것 | K=4가 보장하지 않는 것 |
|---|---|
| endpoint의 현재 A frame과 직전 3 layer context | physical defect mechanism에 최적인 memory length |
| z=4부터 sample 구성 가능 | long-term layer drift를 모두 포착한다는 보장 |
| Conv3D temporal path를 진단할 수 있는 최소한의 sequence | 더 큰 K보다 성능이 좋다는 hyperparameter 결론 |

### 6.2 6.4 : 1.6 : 2 split과 guard band

사용자가 지정한 endpoint 비율 64%/16%/20%를 250 layer와 K=4 제약에 맞춰 정수 endpoint로 구성했다.

| 구분 | endpoint z | endpoint 수 | history 예시 | 선정 기준 |
|---|---:|---:|---|---|
| Train | 4–157 | 154 | z=4 → 1;2;3;4 | model fitting 및 normalization 통계 추정 |
| Guard 1 | 158–160 | 0 | validation의 pre-context | K−1=3 layer buffer |
| Validation | 161–199 | 39 | z=161 → 158;159;160;161 | checkpoint/experiment 선택 |
| Guard 2 | 200–202 | 0 | test의 pre-context | K−1=3 layer buffer |
| Test | 203–250 | 48 | z=203 → 200;201;202;203 | 한 번의 held-out generalization 확인 |

guard 길이를 3으로 둔 이유는 K=4 history가 split boundary를 넘을 때 필요한 최대 과거 layer 수가 3이기 때문이다. 예를 들어 validation z=161은 train endpoint z=157을 보지 않고 guard 158–160만 context로 쓴다. 이 방식은 train/validation/test의 시간적 인접성을 완전히 없애는 것이 아니라, **다른 split의 endpoint image가 history로 흘러가는 직접 leakage**를 막는다.

### 6.3 Split 뒤 train-only rule을 다시 적용한 이유

normalization percentile, response scale, 어떤 threshold든 test/validation 데이터를 보고 계산하면 held-out evaluation이 낙관적으로 변할 수 있다. 그래서 밝기 normalization과 XCT response p01/p99는 train endpoint/history에서만 추정하고 validation/test에는 고정된 값을 적용한다.

---

## 7. Normalization은 왜 LED별·train-only인가

세 LED와 A/B stage는 illumination과 saturation 비율이 다르다. 모든 channel을 하나의 global min/max로 normalize하면 LED exposure 차이가 model에게 fake quality signal처럼 보일 수 있다. 그래서 stage·LED별 robust percentile을 train data에서만 계산하고, 이후 모든 split에 그대로 사용한다.

| 비교 대상 | 선택하지 않은 방법 | 채택한 방법 | 이유 |
|---|---|---|---|
| LED brightness scale | 모든 LED의 global normalization | stage·LED별 train p01/p99 normalization | LED illumination scale 차이와 manufacturing pattern을 분리 |
| Sensor ceiling | intensity=1만 제공 | intensity + validity mask | 65,535 saturation을 real brightness와 혼동하지 않음 |
| Validation/test statistics | split마다 재추정 | train-derived config 고정 | test distribution leakage 방지 |
| Dense preprocessing storage | resized tensor file 저장 | Dataset on-the-fly | raw provenance 유지·storage 폭증 방지 |

이 normalization은 defect label 만들기가 아니라 **input representation을 일관되게 만드는 선택**이다.

---

# Part II. XCT weak supervision은 어떻게 정했는가

## 8. 왜 `xct_5x5x5`를 continuous response로 골랐는가

registered XCT CSV에는 original, `3x3x3`, `5x5x5` voxel response가 함께 있으며, train finite count는 각각 2,329,476개로 같았다. 첫 baseline은 `xct_5x5x5`를 **연속 weak response**로 사용한다. 5×5×5 aggregation은 point-level voxel fluctuation/registration sensitivity를 완화하는 reference 후보이지만, 이를 defect label로 바꾸거나 방향을 뒤집지 않는다.

| 선택 항목 | 현재 choice | 이유 | 아직 보류한 것 |
|---|---|---|---|
| XY columns | command XY, CSV 3–4열 | registered point의 location contract | actual physical coordinate의 완전한 truth 선언 |
| Response column | `xct_5x5x5`, 40열 | continuous weak response baseline | defect/normal binary threshold |
| Response scale | train p01/p99 = 0.40070 / 0.58533 | extreme value에 덜 민감한 fixed `[0,1]` scale | response direction inversion |
| Missing/unsupported pixel | `unknown`, support=0 | no observation ≠ normal | dense negative labels |

Scaling은 다음과 같이 고정한다.

\[
y = \mathrm{clip}\left(\frac{\mathrm{xct}-0.40070}{0.58533-0.40070},0,1\right)
\]

이 식은 response를 model-friendly range로 옮기는 규칙일 뿐이다. `y=1`이 반드시 더 나쁜 품질이라는 해석은 아직 없다.

## 9. Gaussian sigma=2는 어떻게 유지되었는가

XCT point는 sparse이고 model grid는 256×256이므로, 한 point를 단 하나의 pixel에만 찍으면 supervision이 지나치게 희소해진다. 따라서 projected point 주변에 Gaussian weight를 만들고 support가 있는 local region에서만 loss를 계산한다.

초기 rasterization audit에서 z=125 기준 support fraction은 sigma=1/2/3/4에 대해 2.3010% / **3.6179%** / 4.8889% / 6.2622%였다. baseline은 coverage와 excessive smoothing의 중간값인 **sigma=2 model pixel**을 fixed working choice로 두었다. 이후 “더 넓히면 학습이 좋아질 것인가”를 별도 read-only audit으로 검증했다.

| Sigma=2→3 비교 gate | 기준 | 실제 결과 | 결정 |
|---|---:|---:|---|
| Median support gain | ≥25% | 29.3990% | coverage만 보면 통과 |
| Base support retention | 100% | 100% | 기존 supervised pixel 보존 |
| Common-support response MAE | ≤0.05000 | **0.0538972** | 기존 known pixel value가 과도하게 바뀜 |
| Component count ratio | ≥0.5 | 1.0 | 이 데이터에서는 binary component가 모두 하나라 판별력 제한 |
| Largest component share growth | ≤1.5 | 1.0 | giant component 증가는 없음 |

sigma=3은 layer당 1,041 support pixel을 추가했지만, 이미 support였던 pixel의 response도 Gaussian mixture 때문에 달라졌다. 따라서 `weak_target_v1.yaml`은 **sigma=2 유지**, sigma=3 training은 **hold**다. 이 hold는 sigma=3이 물리적으로 틀렸다는 뜻이 아니라, 현재 target contract를 바꿀 만큼 안정적이지 않았다는 뜻이다.

## 10. Support-masked Smooth L1을 어떻게 검증했는가

loss는 support=1인 pixel만 평균한다.

\[
L = \frac{\sum_i m_i\,\mathrm{SmoothL1}(\hat y_i,y_i)}{\sum_i m_i+\varepsilon}
\]

여기서 \(m_i=1\)은 XCT-derived response가 있는 pixel, \(m_i=0\)은 unknown이다. `beta=0.1`은 current baseline config의 fixed robust-regression parameter다. 이 값이 global optimum이라는 의미는 아니며, prior audit의 목적은 beta sweep이 아니라 unknown masking contract 검증이었다.

| Runtime test | 실제 확인 | 막는 오류 |
|---|---|---|
| Early z=4 sample | support=0, loss=0, gradient=0 | XCT가 없는 early layer를 normal zero label로 학습하는 오류 |
| Available z=128 sample | supervised pixel=3,439, finite loss | on-the-fly target과 loss wiring 오류 |
| Unknown prediction을 1000으로 변경 | loss difference=0 | support 밖 prediction이 loss에 새는 오류 |
| Unsupported gradient | sum=0 | unknown pixel이 training update를 만드는 오류 |

---

# Part III. 모델 선택은 어떻게 실패를 진단하고 바뀌었는가

## 11. 처음에는 왜 output을 믿지 않았는가

처음 C8 baseline은 test map이 전역적으로 완전히 상수는 아니어도, support region에서는 동일한 plateau를 만들었다. z=203/227/250의 top-score tie가 63,504 pixel, 즉 96.8994%였고 endpoint 간 map MAE/max-absolute도 0/0이었다. 따라서 top-k 좌표를 강제로 꺼내면 모든 layer에 같은 임의 pixel을 반복할 위험이 있었다.

후속 C32는 top-score tie를 0.3845%까지 낮췄지만, endpoint 간 map이 여전히 완전히 같고 support prediction std=0이었다. 즉 channel capacity만 높이면 static spatial geometry는 바뀌어도 input-dependent localization은 회복되지 않았다.

| Controlled comparison | 바꾼 것 | 유지한 것 | 결과 | 다음 판단 |
|---|---|---|---|---|
| Run 1 → E24 | epoch 8→24 | data/loss/model family | test loss 0.35% 악화, plateau 유지 | under-training 가설 hold |
| Run 1 → C32 | base channels 8→32 | epoch/data/loss | test loss 0.24% 악화, map invariant 유지 | capacity만으로 해결 안 됨 |
| sigma=2→3 audit | target width만 변경 | input/model/scale | response stability gate 실패 | sigma change hold |

이 controlled experiments의 의도는 “좋아 보이는 hyperparameter를 찾기”가 아니라, 다음 변경을 어디에 한정할지 결정하는 것이었다.

## 12. Temporal collapse는 어떻게 발견했는가

C32 checkpoint에서 z=203/227/250의 pairwise difference를 model 내부 stage별로 측정했다.

| Stage | 관측 | 해석 |
|---|---|---|
| `input_history` | max-abs=1.0 | 세 causal input history는 실제로 다름 |
| `encoded_final_history_frame` | max-abs≈1.93–2.30 | frame encoder가 endpoint image 차이를 보존 |
| `encoded_history` | max-abs≈2.15–2.51 | 4-step encoder sequence도 구별됨 |
| `temporal_final` | MAE/max-abs=0/0 | variation이 여기서 처음 완전히 사라짐 |
| `logits`, `score` | MAE/max-abs=0/0 | collapse가 decoder와 sigmoid까지 전달됨 |

또한 stagewise reconstruction score와 ordinary `model.forward()` score의 difference가 0이어서 diagnostic implementation 자체가 다른 forward path를 본 것이 아님을 확인했다. 따라서 다음 변경은 frame encoder나 sigmoid가 아니라 `Conv3D → GroupNorm → SiLU` temporal aggregation path로 한정할 수 있었다.

## 13. Residual bypass를 왜 선택했고, 무엇을 고정했는가

선정한 단일 가설은 endpoint image feature가 decoder까지 사라지지 않도록 다음 연결을 넣는 것이었다.

```text
encoded_endpoint = frame_encoder(current endpoint A frame)
temporal_update  = SiLU(GroupNorm(past-only Conv3D(K=4 history)))
temporal_final   = encoded_endpoint + temporal_update
score             = sigmoid(decoder(temporal_final))
```

이 선택은 현재 A endpoint feature와 past-only temporal update만 사용하므로 future-layer leakage를 만들지 않는다. 그리고 `use_endpoint_feature_residual=False`를 default로 두어 기존 C8/E24/C32 config와 checkpoint behavior를 보존했다.

| Residual experiment에서 고정한 것 | 바꾼 것 |
|---|---|
| A stage, K=4, 6 channels, train-only normalization, response scaling, sigma=2, unknown policy, masked Smooth L1, optimizer, seed, batch size, epoch=8, decoder safety thresholds | `use_endpoint_feature_residual=true`와 별도 output path |

이 고정이 있어야 loss 변화가 input/target/split 변경 때문이 아니라 residual path 때문이라고 해석할 수 있다.

## 14. Residual model을 current baseline으로 둔 선정 기준

| 검증 | C32 temporal-only | C32 temporal-residual | 선정 판단 |
|---|---:|---:|---|
| Best validation loss | 0.06151949 | **0.05642983** | 8.27% 감소 |
| Held-out test loss | 0.07363038 | **0.06992126** | 5.04% 감소 |
| Test support/sample | 172,834 / 48 | 동일 | 공정 비교 가능 |
| z203/227/250 support prediction std | 0 | **0.043991 / 0.041168 / 0.045247** | zero plateau 해소 |
| Map MAE across selected endpoints | 0 | **0.009254–0.011778** | input에 따라 map 변화 |
| Top-score tie | 0.3845% | **0.001526% (1 pixel)** | 0.1% safety gate 통과 |
| Stagewise earliest collapse | `temporal_final` | `null` | selected history에서 collapse 해소 |

따라서 residual model은 **현재의 controlled A-only weak-supervision baseline**으로 선택됐다. 이는 held-out sparse regression과 numerical map sensitivity에서의 개선이다. XCT response direction, physical defect truth, unsampled pixel의 dense truth를 검증한 것은 아니다.

---

# Part IV. Candidate는 어떤 기준을 통과해야 출력되는가

## 15. Decoder의 순서와 기준

score가 가장 높은 pixel이라도 map 자체가 불신할 만하면 좌표를 내보내지 않는다. safety decoder는 다음 순서로 동작한다.

| 순서 | 검사 | 현재 기준·의도 | 실패 시 |
|---:|---|---|---|
| 1 | Spatial flatness | 전체 map variation이 사실상 없는가 | candidate hold |
| 2 | Top-score plateau | tie fraction이 0.001(=0.1%)보다 큰가 | `withheld_top_score_plateau` |
| 3 | Temporal invariance | selected endpoint map MAE와 max-abs가 모두 `≤1e-6`인가 | `withheld_temporally_invariant_map` |
| 4 | Local maxima | 7×7 local peak가 존재하고 top-k가 중복되지 않는가 | peak만 ranking |
| 5 | Optional provisional geometry | local maxima가 configured part rectangle에 들어가는가 | `withheld_outside_provisional_part_geometry` 또는 filtered candidate |

이 검사들은 **score map과 past endpoint maps만** 사용한다. XCT support mask는 deployment decoder가 보지 않는다. 이 분리는 real-time layer에 XCT가 없다는 data availability 조건 때문에 필수다.

## 16. Candidate coordinate의 현재 의미

| 필드 | 현재 의미 | 선정/보류 기준 |
|---|---|---|
| `x_pixel`, `y_pixel` | raw layer-camera pixel | **primary location**; model-grid center와 ROI mapping round-trip이 맞는지 확인 |
| `layer_z` | causal history의 endpoint manufacturing layer | future layer를 쓰지 않음 |
| `score` | sigmoid-scaled XCT-derived continuous quality candidate | confirmed defect 또는 probability로 부르지 않음 |
| `x_model_pixel`, `y_model_pixel` | 256×256 grid provenance | same-endpoint duplicate/edge margin 검증 |
| `provisional_machine_xy_rank2` | optional inverse-projected metadata | rank sensitivity caveat 없이는 사용 금지 |
| `provisional_part_rank2` | optional geometry metadata | physical part ID가 아닌 configured convention label |

geometry-gated evaluation에서는 240/240 candidates가 configured rank2 rectangle 안에 들어가고 raw/model round-trip도 통과했다. 하지만 rank1과 rank2를 비교하면 같은 part agreement가 0/240이고 machine XY shift median이 14.463이다. 그래서 **internal geometry pass는 absolute metrology accuracy pass가 아니다.**

---

# Part V. Calibration과 metrology는 왜 별도 프로젝트처럼 다루는가

## 17. Screen controls로 만든 provisional calibration

초기 camera calibration은 B LED3 화면에서 part screen corners를 잡고, machine part rectangle/order와 image orientation 후보를 비교하는 방식이었다. 192 hypothesis를 residual/leave-one-out RMSE로 비교했으며 residual-only rank1 `mirror_rotate_90`과 selected rank2 `mirror_rotate_270`가 displayed precision에서 동률이었다.

| 검증 결과 | 의미 |
|---|---|
| rank1/rank2 둘 다 240/240 geometry-gated candidate를 어떤 rectangle 안에 둠 | containment만으로 orientation 선택 불가 |
| same containing part=0/240 | part identity가 rank 선택에 따라 완전히 뒤바뀜 |
| machine XY shift min/median/p95/max=4.168/14.463/33.857/38.053 | raw camera pixel을 physical coordinate로 단정하면 안 됨 |
| rank2 local photometric evidence가 더 높음 | current provisional convention 유지 근거일 뿐 independent validation은 아님 |

따라서 rank2 `mirror_rotate_270`과 raw offset `(0,-6)`은 **working provisional mapping**으로만 남아 있다. model target projection과 optional geometry gate가 같은 convention을 써 internal arithmetic을 재현할 수는 있지만, user-facing candidate location은 camera-primary다.

## 18. NIST metadata는 왜 확인했고, 무엇이 통과했는가

NIST는 layer-camera DotGrid, layer-camera Checkerboard, red laser marker가 있는 SecondaryCamera DotGrid를 metadata로 제공하며 두 calibration route를 문서화한다.[1]

| Artifact / route | Pre-audit·V1 결과 | V2 refinement 결과 | 판단 |
|---|---|---|---|
| DotGrid layer-camera | target pattern visible | 1,616 ROI candidates, NN CV=0.3262 | local lattice evidence pass |
| Secondary red reference | global red centroid가 diffuse | 34 components cluster, `(2582.34,2029.18)`, spread=40.27 px | visual-reference candidate pass; origin assertion 금지 |
| Checkerboard direct route | false positives/background mixture | 410 candidates, NN CV=0.5040 > gate 0.45 | hold |
| DotGrid + Secondary method #2 | feature audit 준비 | candidate D→C transform audit 실행 | hold at residual validation |

NIST method #2는 50×50, 1.00 mm-pitch dot grid, D origin=lower-left dot, `A(0,0)=D(28.25,24.25) mm`, D/A relative orientation 2.5°를 보고한다.[1] 하지만 red marker 하나만으로 image-axis orientation sign이 코드 수준에서 결정되지는 않는다. 따라서 project code는 8 image-lattice axis assignment와 `±2.5°` sign alternatives를 모두 보존했다.

## 19. Independent method-#2 candidate audit은 왜 hold인가

V1 audit은 DotGrid panel을 자동 ROI로 제한하고 local weighted subpixel centers를 만들었다. 1,518 unique indexed cell, PCA columns 50, PCA rows 48로 coverage gate는 통과했다. 즉 **board가 보이고 넓게 sample됐다는 것**은 확인됐다.

그러나 board visibility와 exact row/column correspondence는 다르다. 현재 PCA + independent 1D 50-cluster assignment는 perspective를 가진 2D lattice를 충분히 안정적으로 index하지 못했다.

| Held-out measurement | Predeclared criterion | Actual V1 | 판정 |
|---|---:|---:|---|
| Indexed coverage | cells≥1,200, rows/cols≥40 | 1,518 / 50 / 48 | 통과 |
| Robust fit inlier | diagnostic | 1,422/1,518 (93.68%) | 참고용; deployment proof 아님 |
| Detected camera dot pitch | normalizing scale | 14.59340 px | pixel-unit reference |
| Held-out 5×5-block count | spatial validation | 298 | test set은 충분 |
| Held-out RMSE | ≤0.25 dot pitch | **6.00155 px = 0.41125 pitch** | 실패 |
| Held-out p95 residual | ≤0.50 pitch=7.29670 px | **9.78092 px** | 실패 |
| All candidate gates | true | **false** | 16 transforms 전체 hold |

QC overlay에서도 board ROI 위치는 맞지만, grid right side에 robust rejection이 많고 upper/central/lower held-out block 여러 곳에서 predicted dot이 actual dot과 체계적으로 어긋난다. 이 결과는 새 transform이 옳다는 증거도, 기존 rank2가 틀렸다는 증거도 아니다. **correspondence algorithm을 더 정제해야 한다**는 evidence다.

---

# Part VI. 현재 변경 금지 사항과 다음 단계

## 20. 현재 config와 model에 대해 하지 않는 일

| 금지/hold 항목 | 이유 |
|---|---|
| `calibration_v1.yaml` 교체 | independent method-#2 held-out gate 실패 |
| rank1/rank2/method-#2 alternative 자동 선택 | orientation/part ambiguity가 해소되지 않음 |
| machine XY 또는 part ID를 physical fact로 출력 | candidate rank sensitivity가 큼 |
| weak target 재투영 또는 residual model 재학습 | calibration transform이 승인·검증된 변경이 아님 |
| response inversion 또는 binary defect threshold | XCT response direction과 physical interpretation unresolved |
| decoder에 XCT support mask 투입 | real-time deployment input에 존재하지 않음 |

## 21. 다음 코드 작업의 올바른 범위

다음 기술 작업은 “새 homography를 반복해서 fit”하거나 “gate를 느슨하게 조정”하는 일이 아니다. 먼저 DotGrid에서 perspective-aware 2D local-neighbor consistency를 써 row/column correspondence를 정제한 다음, **동일한 5×5-block held-out rule**로 residual을 다시 검사해야 한다.

| 다음 refinement에서 개선할 것 | 그대로 유지할 것 |
|---|---|
| 2D row/column assignment, local grid consistency, off-grid rejection | immutable raw TIFF와 memmap read-only access |
| indexed/inlier/held-out residual overlay | K=4 causal A-only residual baseline |
| same held-out residual gate | XCT response scale, sigma=2, unknown policy |
| candidate transform을 human-review-only output으로 기록 | camera-primary reporting, no automatic config update |

승인된 다음 구현은 `src/audit_independent_method2_lattice_correspondence_refinement.py`다. 이 코드는 DotGrid TIFF 하나만 read-only로 열고, V1의 독립적인 1D clustering 대신 다음 순서를 사용한다.

1. 자동 DotGrid ROI 안에서 dark-dot response/NMS 후보와 response-weighted subpixel center를 만든다.
2. 각 후보의 가까운 여섯 neighbor를 조사한다. 거리 `0.45–1.75×` local dot pitch, PCA axis alignment `≥0.92`를 동시에 만족하는 edge만 남긴다.
3. 이 edge를 따라 BFS로 provisional 2D image-lattice row/column을 전파하고, cycle conflict와 graph에 연결되지 않은 후보를 수치로 기록한다.
4. 가장 dense한 provisional 50×50 image-lattice window를 고른다. 이것은 machine D origin이 아니라 image-plane correspondence의 후보 범위다.
5. 그 window의 image-lattice homography를 in-memory로 사용해 후보를 가장 가까운 2D cell에 다시 배정한다. `0.45×` pitch 밖의 candidate는 off-grid로 보류하며, cell 안 duplicate는 residual이 더 작은 점 하나만 남긴다.
6. 마지막으로 V1과 **완전히 같은 5×5-block held-out rule**, RMSE `≤0.25×` train-inlier pitch, p95 `≤0.50×` 같은 pitch gate를 적용한다.

| 이 코드가 읽는 것 | 이 코드가 쓰는 것 | 이 코드가 하지 않는 것 |
|---|---|---|
| `DotGrid_2000x2000.tif` 한 파일 | compact feature CSV, neighbor-edge CSV, summary JSON, QC overlay 3장 | config/control JSON·A/B·XCT·model/checkpoint 접근, transform/rank 선택, `calibration_v1.yaml` 변경 |

실행은 다음과 같다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_independent_method2_lattice_correspondence_refinement.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --output-dir processed/calibration/independent_method2_lattice_correspondence_refinement_v1
```

첫 실행은 `Dense 50x50 provisional lattice window contains too few graph labels`에서 의도적으로 중단됐다. 이 오류는 raw TIFF나 calibration/model data가 바뀌었다는 뜻이 아니며, CSV·JSON·overlay·held-out residual도 만들기 전 단계에서 fail-closed한 것이다. 원인은 graph BFS가 가장 강한 dark-response 점 하나에서만 시작되어, 큰 DotGrid component 대신 작은 disconnected component를 seed로 선택할 수 있었기 때문이다. 즉 DotGrid pattern이 사라진 것이 아니라 **graph seed policy가 panel-wide correspondence에 충분히 안정적이지 않았다**는 뜻이다.

| 지금의 처리 | 이유 |
|---|---|
| 현재 output directory를 `--overwrite`로 덮어쓰지 않음 | 실패한 실행 기록을 숨기지 않고, 다음 code patch와 결과를 분리하기 위해서 |
| RMSE/p95 값을 새로 해석하지 않음 | held-out validation 전에 멈췄으므로 새 residual evidence가 없음 |
| V1 method-#2 transform hold 유지 | 새 correspondence가 아직 검증되지 않았음 |
| 다음 patch를 별도 승인으로 분리 | graph component 선택 방식도 correspondence algorithm 변경이기 때문 |

가장 작은 보완은 모든 edge-connected component의 size를 세고, **candidate 수가 가장 큰 component**를 BFS seed로 고르는 것이다. size가 같을 때만 aggregate response와 deterministic spatial order로 tie-break한다. detector threshold, fixed 5×5 held-out rule, RMSE/p95 gate, calibration config, model/XCT data는 그대로 유지한다.

이 보완은 새 파일 `src/audit_independent_method2_lattice_correspondence_refinement_v2.py`로 구현됐다. V2는 먼저 `method2_refined_2d_graph_components.csv`를 작성해 component별 candidate 수, edge 수, aggregate response, BFS seed, raw image bounding box를 남긴다. 그래서 다시 일찍 멈추더라도 단순 traceback만 남지 않고, **어느 component가 왜 선택됐는지**를 확인할 수 있다.

| V2가 바꾸는 것 | V2가 바꾸지 않는 것 |
|---|---|
| BFS가 시작하는 graph component 선택 규칙과 component diagnostics | DotGrid detector·ROI·subpixel center rule |
| failure 시 compact JSON/CSV/graph overlay를 먼저 남기는 순서 | `0.45–1.75×` neighbor distance, `≥0.92` axis alignment |
| 새 `_v2` output directory | same 5×5 holdout, coverage/RMSE/p95 gate |
| largest component 안의 high-response seed | calibration config/rank, A/B/XCT/target/model data, camera-primary reporting |

실행할 때는 V1 failure directory를 지우거나 `--overwrite`하지 않고 새 V2 directory를 쓴다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_independent_method2_lattice_correspondence_refinement_v2.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --output-dir processed/calibration/independent_method2_lattice_correspondence_refinement_v2
```

V2의 첫 실행은 graph fragmentation을 넘어서 component CSV·neighbor-edge CSV·refined lattice-feature CSV와 graph/correspondence overlay까지 만들었다. 그러나 마지막 held-out residual overlay를 저장하는 줄에서 **함수 인자를 하나 더 넘긴 code mismatch**로 멈췄다. 따라서 V2 directory는 partial debug evidence로 보존하고, final summary JSON과 held-out residual overlay가 없으므로 pass/fail 수치를 채택하지 않는다.

| 이미 만들어진 것 | 아직 없는 것 | 현재 처리 |
|---|---|---|
| component/edge/feature compact CSV와 graph/correspondence overlay | held-out residual overlay와 final summary JSON | V2를 덮어쓰지 않고 V3의 새 output directory를 사용 |
| raw/config/model/target을 바꾸지 않은 computation | accepted V2 validation result | transform/rank/config hold 유지 |

다음 V3는 same calculation의 plotting call에서 obsolete `residual` argument 하나만 제거한다. static check로 definition/call arity도 맞춘다. detector, largest-component seed, fixed 5×5 holdout, coverage/RMSE/p95 gates는 모두 고정한다.

V3는 `src/audit_independent_method2_lattice_correspondence_refinement_v3.py`로 구현됐다. 쉽게 말해 “grid를 찾는 수학”이나 “합격 기준”을 다시 바꾼 것이 아니라, 계산이 끝난 뒤 held-out error 화살표를 그림으로 저장할 때 주던 불필요한 입력 하나를 제거한 것이다. 따라서 V2와 V3의 결과가 다르게 나와도 그 차이는 new algorithm이 아니라 V2가 summary/last overlay까지 도달하지 못했던 실행 경계 때문이다.

| V3에서 확인하는 것 | V3에서 확인하지 않는 것 |
|---|---|
| function definition과 call의 5-argument 일치 | DotGrid detector threshold가 더 좋아졌는지 |
| same V2 correspondence가 final JSON·held-out overlay까지 완주하는지 | calibration transform/rank가 더 정확한지 |
| fixed gate를 계산해 review 가능한 evidence가 남는지 | machine origin, physical part, confirmed defect 위치 |

V3 command는 다음과 같다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_independent_method2_lattice_correspondence_refinement_v3.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --output-dir processed/calibration/independent_method2_lattice_correspondence_refinement_v3
```

V3는 정상 완료했다. 15개 graph component 중 가장 큰 component가 candidate 1,523개(전체 1,616개 중 94.25%)와 edge 2,812개(전체 2,860개 중 98.32%)를 담고 있어, V1의 작은 component seed 문제가 실제로 해소됐음을 확인했다. BFS label conflict도 0개였다.

가장 중요한 변화는 **미리 고정한 held-out test**다. V1에서 RMSE는 dot pitch의 0.41125, p95는 0.67037이었지만, V3에서 313 held-out cell을 사용한 RMSE는 0.06728 pitch, p95는 0.11472 pitch가 됐다. 각각 기준 `0.25`, `0.50`보다 작으므로 residual gate는 통과했다. 마지막 overlay의 yellow held-out dot과 magenta residual arrow도 panel 여러 위치에 분포하고 화살표가 작아, 수치가 특정 한 구역만 잘 맞춘 결과가 아니라는 점을 visual cross-check했다.

하지만 모든 gate가 통과한 것은 아니다.

| Gate | V3 수치 | 왜 중요한가 | 현재 판정 |
|---|---:|---|---|
| Unique cells | 1,554 ≥ 1,200 | 충분한 수의 cell에서 검증했는지 | 통과 |
| Image-lattice columns | 50 ≥ 40 | panel 가로 방향이 넓게 포함됐는지 | 통과 |
| Image-lattice rows | **39 < 40** | 세로 방향도 사전에 정한 최소 coverage를 충족하는지 | 보류 |
| Held-out RMSE | 0.06728 pitch ≤ 0.25 | average correspondence generalization | 통과 |
| Held-out p95 | 0.11472 pitch ≤ 0.50 | 큰 local mismatch가 제한되는지 | 통과 |

39 row는 기준보다 단 한 row 부족하지만, 결과를 본 뒤 “39도 충분하다”고 기준을 낮추면 validation의 신뢰성이 사라진다. 더구나 NIST source note에는 50×50 dot grid가 문서화되어 있으므로, **visible field-of-view인지, target-count convention인지, detector/reassignment boundary인지**를 먼저 따로 확인해야 한다. 현재 low residual은 2D correspondence algorithm이 좋아졌다는 strong evidence지만, published D coordinate 또는 machine calibration config를 적용할 충분한 evidence는 아니다.

따라서 다음 작업은 gate를 낮추는 것이 아니라 separate read-only coverage-definition audit이다. 이 audit은 구현된 `src/audit_independent_method2_dotgrid_coverage_definition.py`로 수행한다. V3 feature CSV와 DotGrid image를 같이 읽어 50×50 nominal image-lattice cell 각각에 아래 네 질문을 묻는다.

1. 이 nominal cell은 V3 final correspondence에서 실제 assigned됐는가?
2. V3 feature만으로 예측한 cell 위치가 2000×2000 camera sensor 안에 있는가?
3. sensor 안이라면 새 DotGrid detector candidate가 기존 V3 assignment bound(`0.45×` pitch) 안에 있는가?
4. 그 위치의 local raw-image darkness는 어떠한가?

| Missing cell의 관찰 pattern | 이 audit이 허용하는 해석 | 이 audit이 하지 않는 해석 |
|---|---|---|
| 80% 이상이 sensor 밖 | field-of-view clipping이 plausible | row gate를 자동으로 낮춤 |
| sensor 안이고 가까운 fresh detector candidate가 다수 | indexing/reassignment boundary가 plausible | grid size·machine coordinate를 자동 변경 |
| sensor 안이지만 가까운 detector candidate가 다수 없음 | visible target extent 또는 detector boundary가 plausible | physical target specification 오류를 단정 |
| 어느 한 pattern도 다수 아님 | mixed evidence | 임의의 하나를 원인으로 선택 |

Script는 row/column occupancy와 contiguous run, nominal 2,500 cell coverage table, JSON summary, QC plot 두 장만 새 output directory에 쓴다. temporary image-lattice mapping은 dot이 sensor 안에 있을지를 점검하는 데만 메모리에서 쓰며, calibration H나 config file로 저장하지 않는다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_independent_method2_dotgrid_coverage_definition.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --output-dir processed/calibration/independent_method2_dotgrid_coverage_definition_v1
```

Coverage-definition audit은 정상 완료했다. 가장 중요한 수치는 `946/946`이다. V3에서 nominal 50×50 cell 중 할당되지 않은 946개는 모두 2000×2000 camera sensor 안에 예측됐고, 새 dot detector candidate가 V3의 기존 assignment bound(`6.55369 px`) 안에 있는 경우는 `0/946`이었다.

| 먼저 예상했던 원인 | 결과 | 왜 제외/보류됐는가 |
|---|---|---|
| 센서가 grid를 잘라 냈다 | 지지되지 않음 | missing 946개 중 sensor 밖 예측이 0개 |
| dot은 찾았지만 V3 assignment만 놓쳤다 | 지지되지 않음 | missing 946개 중 fresh detector candidate가 가까운 경우가 0개 |
| visible target extent, detector footprint, 또는 provisional image-lattice window/index convention | 여전히 plausible | inside sensor이지만 fresh detector support가 없는 structured missing region |

QC overlay는 이 해석을 보완한다. Cyan assigned cell은 printed dot 위에 모이지만, red missing cell은 image left의 넓은 영역과 dot panel right/central boundary에 structured하게 모인다. Row occupancy는 0–37까지 연속이지만 초반에는 10개에서 점차 늘고, 39번째 row는 2개뿐이다. 반면 column 0–49는 모두 등장하지만 각 column은 약 25–37개 dot만 가진다. 즉 **“39개의 완전한 row가 보이고 한 row만 잘렸다”**라고 간단히 말할 수 없다.

따라서 40-row gate는 계속 fail이다. 39가 40보다 하나 작다고 해서 결과를 본 뒤 기준을 낮추면, 원래 정한 validation이 project에 주는 안전장치가 사라진다. 이 audit은 gate를 바꾸기 위한 증거가 아니라, future human review가 무엇을 확인해야 하는지 좁혀 준 evidence다.

다음 작업은 자동 transform fit이 아니라 human-reviewed visible DotGrid extent control audit이다. `select_visible_dotgrid_extent_controls.py`가 full DotGrid image를 screen에 맞춰 보여 주면, 사람이 **실제로 보이는 가장 바깥 dot의 중심** 네 개를 아래 순서로 click한다.

| Click 순서 | 선택할 것 | 선택하면 안 되는 것 |
|---:|---|---|
| 1 | top-left outer visible dot centre | white paper corner, screw, text |
| 2 | top-right outer visible dot centre | 한 column 안쪽의 dot |
| 3 | bottom-right outer visible dot centre | black panel shadow나 paper edge |
| 4 | bottom-left outer visible dot centre | 한 row 안쪽의 dot |

Preview image가 작아도 문제가 없다. script가 preview click을 raw camera pixel coordinate로 되돌려 compact JSON에 저장한다. Click 하나를 잘못했으면 right-click으로 마지막 point만 지우고 다시 click한다. 네 point가 보인 후 middle mouse button으로 끝낸다.

그 다음 `audit_visible_dotgrid_extent_controls.py`가 click을 calibration point가 아니라 **visible panel evidence**로만 검증한다.

1. each click가 fresh dot candidate의 `0.60×` camera pitch 이내인지 검사한다.
2. 네 click이 서로 다른 candidate로 snap되는지 검사한다.
3. TL→TR→BR→BL 순서가 self-crossing 없는 convex quadrilateral인지 검사한다.
4. 네 edge마다 `0.55×` pitch band 안에 fresh dot candidate가 최소 3개 있는지 검사한다.
5. human quad 안에 V3 assigned cell, fresh candidate, nominal V3 prediction이 각각 얼마나 들어가는지 센다.

| 이 workflow가 답하는 질문 | 답하지 않는 질문 |
|---|---|
| 현재 V3 nominal 50×50 window가 사람이 확인한 visible dot panel보다 어느 방향에서 넓거나 좁은가 | physical 50×50 target의 D origin / physical cell index가 무엇인가 |
| 39-row shortfall이 clicked outer extent 밖에서 생기는가 | 40-row gate를 바꿔도 되는가 |
| V3 assignment가 visibly physical dot panel과 얼마나 겹치는가 | machine calibration transform, rank, orientation, part ID |

Human selector V1은 click 창을 열지 못했다. Terminal에 `FigureCanvasAgg is non-interactive, and thus cannot be shown`이라는 warning이 나타났다. 이는 DotGrid가 깨졌거나 사용자가 click을 못한 문제가 아니다. 이전 batch-QC script는 PNG를 재현 가능하게 만들기 위해 Matplotlib의 `Agg` backend를 사용한다. `Agg`는 화면 창을 열지 않는 backend이므로, 사람의 mouse click을 받는 `ginput`과 함께 사용할 수 없다.

| Batch QC script | Human click selector |
|---|---|
| PNG를 background에서 안정적으로 생성해야 함 | 화면 창과 mouse event를 받아야 함 |
| noninteractive `Agg` backend가 적합 | macOS GUI backend가 필요 |
| imported helper가 `Agg`를 미리 고정해도 됨 | `pyplot` import 전에 GUI backend를 명시해야 함 |

V1은 control JSON을 쓰기 전에 click을 기다렸으므로, 네 click control evidence는 생성되지 않았다. raw DotGrid TIFF와 config/grid/gate/model/target도 바뀌지 않았다.

### 21.1 V2 selector: 배치 PNG 작업과 사람 click 작업을 분리한 수리

승인 후 `src/select_visible_dotgrid_extent_controls_v2.py`를 새로 만들었다. 이 script는 V1처럼 batch-QC helper를 import하지 않는다. 대신 자신 안에서 DotGrid TIFF가 grayscale `YX` image인지 확인하고 `tifffile.memmap(..., mode='r')`로 read-only 접근한다. 그 뒤 **`pyplot`를 import하기 전에** macOS GUI용 `MacOSX` backend를 먼저 선택한다. `MacOSX`가 이 Python 환경에서 불가능할 때만 `TkAgg`를 대안으로 시도한다. 어떤 GUI backend도 쓸 수 없거나 결과가 `Agg`이면 script는 error로 멈추고 JSON을 전혀 만들지 않는다.

| V2가 하는 일 | V2가 하지 않는 일 |
|---|---|
| 축소 preview에서 four outer-dot centre click을 받음 | raw TIFF를 변경·복사·재저장하지 않음 |
| preview click에 recorded stride를 곱해 raw camera `(x,y)`로 저장 | calibration point, machine origin, physical DotGrid index를 정하지 않음 |
| 네 번째 left-click 후 자동 완료 | 네 점 이전의 partial result를 JSON으로 저장하지 않음 |
| right-click으로 마지막 click 하나만 취소 | `GRID_SIZE=50`, 40-row gate, V3 result를 바꾸지 않음 |

사용자는 새 Terminal에서 아래처럼 V2만 실행한다. `visible_dotgrid_extent_controls_v2.json`이 이미 있으면 먼저 검토하고 자동 overwrite하지 않는다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/select_visible_dotgrid_extent_controls_v2.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --output-json processed/calibration/visible_dotgrid_extent_controls_v2.json
```

V2 JSON이 생긴 **후에만** existing validator를 새 output folder에서 실행한다. validator는 click가 실제 dot에 가까운지, 네 점이 TL→TR→BR→BL 순서의 convex panel인지, 그 visible panel 안에 V3/fresh/nominal footprint가 얼마나 있는지를 확인한다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_visible_dotgrid_extent_controls.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --controls-json processed/calibration/visible_dotgrid_extent_controls_v2.json \
  --output-dir processed/calibration/visible_dotgrid_extent_validation_v2
```

V2 source는 syntax·backend-before-pyplot·read-only memmap contract를 정적으로 점검했지만, 실제 GUI click과 validator 실행은 사용자가 수행한다. validator가 pass해도 그것은 사람이 볼 수 있는 DotGrid panel의 범위 evidence일 뿐이다. fixed 40-row rule, transform/rank/orientation, machine origin, calibration config, target projection, model 또는 raw-camera-primary `XCT-derived continuous quality candidate` 표현을 자동으로 바꾸지 않는다.

If `FigureCanvasAgg is non-interactive` appears again, the user should verify that the executed filename ends with `_v2.py` and use a fresh Terminal session. If V2 says both `MacOSX` and `TkAgg` are unavailable, the user should retain the no-JSON state and provide the full error rather than forcing `Agg` or hand-writing four controls.

Human click을 넣더라도 `GRID_SIZE=50`과 40-row gate는 바뀌지 않는다. Passed validation은 future human design review를 위한 evidence일 뿐이며, `calibration_v1.yaml`, target re-projection, retraining, machine/part candidate metadata를 수정하는 trigger가 아니다. `status=completed`여도 same fixed held-out gate를 통과한 뒤 human review만 가능하다. refinement가 통과해도 transform selection, config revision, target re-projection, retraining은 각각 분리된 다음 결정이다. 반대로 gate가 실패하면 candidate 수치를 좋게 보이도록 gate를 느슨하게 하거나 H만 다시 맞추지 않고, correspondence/outlier handling을 다시 검토한다.

---

## 22. 프로젝트에서 기억할 핵심 용어

| 용어 | 쉬운 뜻 |
|---|---|
| Causality | z layer 판단 시 z보다 미래의 image를 보지 않는 규칙 |
| Endpoint | K=4 history에서 현재 판단을 내리는 마지막 layer z |
| Working ROI | 비교 결과로 baseline에 채택했지만 physical final truth로 단정하지 않은 crop |
| Saturation validity | sensor ceiling 65,535인지 여부를 intensity와 별도로 알려주는 channel |
| Train-only normalization | validation/test를 보지 않고 train data에서만 scale을 정하는 규칙 |
| Weak supervision | 완전한 defect mask 대신 sparse/indirect XCT response를 쓰는 학습 방식 |
| Unknown | 관측이 없어 normal/abnormal이라고 말할 수 없는 pixel; zero label이 아님 |
| Temporal collapse | input은 다른데 temporal stage 이후 output이 똑같아지는 failure |
| Residual bypass | endpoint feature를 temporal update와 더해 decoder까지 직접 전달하는 path |
| Provisional calibration | internal convention으로는 계산 가능하지만 independent metrology 전에는 physical fact가 아닌 transform |
| Method #2 | red-reference/secondary-camera와 DotGrid D coordinate를 이용해 layer-camera C와 machine A 관계를 검증하는 NIST route |

---

## 23. 포트폴리오에서 강조할 수 있는 실제 역량

| 역량 | 이 프로젝트의 증거 |
|---|---|
| 대용량 scientific IO | 6 GB TIFF를 memmap으로 필요한 frame만 read-only 접근 |
| Leakage-aware time-series design | K=4 causal manifest, K−1 guard band, train-only statistics |
| Sensor artifact handling | saturation을 intensity와 validity channel로 분리 |
| Imperfect-label learning | continuous XCT weak response, support-masked Smooth L1, unknown≠normal |
| Scientific experiment design | epoch/capacity/sigma를 한 축씩 비교하고 negative result도 hold로 기록 |
| Neural failure diagnosis | stagewise sensitivity로 temporal collapse의 최초 stage를 직접 측정 |
| Safe inference design | plateau/invariance/local-maxima gate, deployment decoder의 XCT support exclusion |
| Metrology awareness | rank sensitivity, independent metadata, held-out dot residual, non-selection policy |
| Reproducibility | raw/processed/output 분리, code/config/docs Git tracking, compact regenerable QC |

---

## 24. 마지막 안전 문장

> 지금의 candidate는 **“A-only causal model이 XCT-derived continuous response와 관련 있어 보인다고 제안한 raw layer-camera pixel”**이다. 이것은 “defect가 확정됐다”, “특정 part의 특정 machine XY에서 이상이 생겼다”, “공정 조건을 즉시 바꿔야 한다”를 뜻하지 않는다.

---

## References

[1] [Lane, B. and Yeung, H. (2020). *Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): Overhang Part X4*. Journal of Research of NIST, 125, 125027.](https://doi.org/10.6028/jres.125.027)

[2] [NIST PDR: AMMT Overhang Part X4 dataset record.](https://data.nist.gov/od/id/mds2-2233)


---

## 21.2 V2 validator 결과: 사각형 모양은 맞아도 dot 중심 근접성이 통과하지 못할 수 있다

V2 selector는 정상적으로 창을 열었고 네 점을 저장했다. 이어 validator도 정상 종료했다. 여기서 중요한 점은 “script가 끝났다”와 “사람이 찍은 네 점이 실제 outer dot의 중심이라는 evidence가 통과했다”가 서로 다르다는 것이다. 이번 결과에서 네 점은 sensor 안에 있고 TL→TR→BR→BL 사각형도 뒤집히거나 교차하지 않는 convex shape였다. 하지만 validator는 각 click가 자동 detector가 새로 찾은 실제 dot 중심에 충분히 가까운지를 **별도로** 검사한다.

| Click | 실제 dot 후보까지 거리 | 허용 상한 | 판정 |
|---|---:|---:|---|
| TL | 3.48 px | 8.74 px | 통과 |
| TR | 170.42 px | 8.74 px | 실패 |
| BR | 87.56 px | 8.74 px | 실패 |
| BL | 16.03 px | 8.74 px | 실패 |

즉 사용자가 흰 판의 모서리가 아닌 중앙 dot panel을 선택했다는 큰 방향은 맞지만, 오른쪽과 아래쪽 outer click가 detector가 인식한 fresh dot 중심과 일치하지 않았다. 특히 오른쪽 edge는 그 line 근처 fresh dot가 5개뿐이고, top/bottom/left의 38/43/51개보다 현저히 적다. 이 때문에 “사각형은 그럴듯하다”는 조건만으로 outer extent를 확정하지 않는다. 이 엄격함은 사람이 panel border·shadow·희미한 dot·white plate edge를 클릭했을 때 coverage rule을 그럴듯하게 바꿔버리는 오류를 막는다.

두 overlay는 후속 검토를 위한 제한적 관찰도 제공한다. V3가 assignment한 1,554 cells 중 1,539개(99.03%)가 human quad 안에 있고, fresh dot 후보도 1,616개 중 1,554개(96.16%)가 안에 있다. 반면 V3 nominal 50×50 prediction은 2,500개 중 1,900개만 안에 있고 600개가 밖에 있다. 그림에서는 nominal prediction의 오른쪽 일부가 human quad 밖으로 나가며, assignment된 점 두 개는 quad 왼쪽에 보인다. 그러나 click-to-dot validation이 실패했기 때문에 이 수치는 physical DotGrid의 실제 행/열 수, D origin, machine direction 또는 coverage gate의 변경 근거가 될 수 없다.

> 결론은 **`hold_extent_interpretation`**이다. 자료가 삭제되거나 실패한 것이 아니라, “현재 네 click만으로는 visible outer boundary를 충분히 정확하게 증명하지 못했다”는 안전한 결과다. `GRID_SIZE=50`, 40-row rule, V3 39-row hold, rank/orientation, machine origin, `calibration_v1.yaml`, target projection, model 또는 raw-camera-primary `XCT-derived continuous quality candidate` 표현은 그대로 유지한다.

따라서 지금 V2 selector/validator를 `--overwrite`로 다시 실행하거나 JSON 숫자를 손으로 고치지 않는다. 기존 control JSON과 validation outputs는 evidence로 보존한다. 이후 click placement 또는 fresh-detector outer-boundary 방법을 바꿀 필요가 있다면, 왜 현재 strict snap check가 충분하지 않은지부터 별도 설계로 검토하고 승인받아야 한다.


---

## 21.3 왜 바로 다시 찍지 않고 outer-boundary diagnostic을 먼저 하는가

V2에서 TR/BR/BL click가 fresh detector candidate와 멀었다고 해서 곧바로 “사용자가 잘못 찍었다”고 결론 내릴 수는 없다. 화면에는 점처럼 보이지만 current automatic detector의 ROI 또는 response threshold가 바깥 열·행을 놓쳤을 수도 있기 때문이다. 반대로 detector만 탓하고 human click를 그대로 인정하면, plate border·shadow·희미한 background texture를 physical outer dot로 잘못 사용할 수 있다. 그래서 같은 자료를 overwrite하지 않고 두 설명을 분리하는 새 read-only diagnostic이 필요하다.

`src/audit_visible_dotgrid_outer_boundary_diagnostic.py`는 각 human click 주변을 camera-dot pitch의 네 배 반경으로 잘라 **메모리에서만** 살펴본다. 그 local patch에 기존과 같은 dark-dot response를 다시 계산하되, local q=0.990 threshold와 8 px NMS로 후보를 찾는다. Click 근처에 어두운 점 하나가 있다는 것만으로는 충분하지 않다. 그 후보 주변에 정상 pitch 범위의 이웃이 두 개 이상 있고 서로 대략 직각인 이웃 방향이 있어야 “DotGrid lattice의 일부처럼 보이는 local evidence”로 인정한다.

| 결과 이름 | 쉬운 뜻 | 다음에 바로 할 수 없는 것 |
|---|---|---|
| `current_detector_supported` | 기존 detector도 이미 그 click 근처 dot를 찾았다 | physical index/origin 확정 |
| `printed_dot_visible_but_current_detector_missed` | local lattice evidence는 있지만 기존 frozen detector가 놓쳤다 | detector threshold·ROI 자동 변경 |
| `click_outside_printed_dot` | local/frozen/V3/nominal reference가 모두 충분히 멀다 | 기존 JSON을 자동 수정하거나 재클릭 |
| `ambiguous` | 서로 다른 reference가 섞여 원인을 정할 수 없다 | 가장 편한 설명을 임의 선택 |

이 diagnostic은 “다음에 무엇을 검토해야 할지”를 결정하는 도구이지, 39 rows를 40 rows로 바꾸는 도구가 아니다. 실행 전의 V2 controls와 validation hold를 그대로 보존하고, 결과는 per-control CSV·summary JSON·네 local patch PNG·한 full-panel PNG만 생성한다. Raw TIFF, detector threshold/ROI, nominal 50×50 grid, rows≥40 gate, homography/rank/orientation, machine origin, calibration config, target/model/checkpoint/decoder에는 손대지 않는다.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_visible_dotgrid_outer_boundary_diagnostic.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --controls-json processed/calibration/visible_dotgrid_extent_controls_v2.json \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --output-dir processed/calibration/visible_dotgrid_outer_boundary_diagnostic_v1
```

이 source는 syntax와 fixed constants, read-only/compact-output contract를 정적으로 점검했다. 실제 실행과 생성 PNG 검토는 사용자가 수행한다. 어떤 evidence class가 나와도 다음 correction은 별도 설계·승인이 필요하며, raw-camera-primary **XCT-derived continuous quality candidate** 표현과 provisional calibration 상태는 유지한다.
