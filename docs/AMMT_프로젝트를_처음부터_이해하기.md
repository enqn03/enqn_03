# AMMT 프로젝트를 처음부터 이해하기

> **한 문장 요약:** 이 프로젝트는 LPBF 제조 중 레이저 조사 **전** 촬영한 layer-camera A 시계열만으로, 향후 각 layer에서 확인이 필요한 위치를 `(x_pixel, y_pixel, layer_z, score)` 형태의 **XCT-derived continuous quality candidate**로 출력하는 인과적(미래 layer 미사용) baseline을 만드는 작업이다.

이 문서는 “현재 무엇이 되었고, 무엇이 아직 안 되었는가”를 먼저 보여 준 뒤, 데이터·모델·검증이 왜 필요한지 순서대로 설명한다. 여기서 `score`는 **확정 결함, 이상 확률, 또는 원인 진단**이 아니다. XCT에서 유래한 연속 response와의 약한 대응을 학습한 후보 점수이며, response가 클수록 무엇을 뜻하는지도 아직 확정하지 않았다.

---

## 1. 지금 프로젝트는 어디까지 왔는가

| 영역 | 현재 상태 | 핵심 결론 |
|---|---|---|
| 원본 보존·입력 구조 | 완료 | A/B TIFF와 XCT/metadata는 원본 그대로 두고 read-only로 접근한다. |
| 인과적 A-only 입력 | 완료 | 현재 endpoint와 과거 3개 layer만 사용한다. 미래 layer는 보지 않는다. |
| XCT weak supervision | 완료 | sparse XCT response를 on-the-fly rasterize하되, support 밖은 정상 0이 아니라 `unknown`으로 둔다. |
| A-only residual baseline | 완료 | temporal collapse를 residual bypass로 해소했고 held-out test loss가 기존 C32보다 5.04% 낮아졌다. |
| Candidate safety | 완료·제한 있음 | pixel coordinate, tie/temporal-invariance/local-maxima safety는 작동한다. 물리 좌표 해석은 아직 provisional이다. |
| Independent metrology feature extraction | 완료 | DotGrid local lattice와 central red reference cluster는 가능성을 보였다. Checkerboard direct route는 보류다. |
| Independent method-#2 candidate transform | 구현·실행 대기 | dot-grid index, held-out residual, rank1/rank2 비교만 수행하며 config는 바꾸지 않는다. |
| 확정 defect 판정·실시간 control | 미완료 | 이 프로젝트의 현재 output은 quality candidate이며 공정 제어 신호가 아니다. |

현재 가장 중요한 원칙은 다음과 같다.

> **모델은 공간적으로 변하는 score map을 만들 수 있게 되었지만, 그 pixel이 실제 machine 좌표의 어느 part인지 독립 metrology로 확정되지 않았다.** 따라서 raw camera pixel을 primary location으로 보고한다.

---

## 2. 문제를 쉽게 비유하면

LPBF 장비는 금속 분말을 한 layer씩 깔고, 레이저로 녹여 형상을 만든다. 우리는 매 layer마다 레이저를 쏘기 **전** 분말 표면을 세 가지 LED 조명으로 촬영한 A image를 본다. 그리고 제작이 끝난 뒤 XCT로 측정된 연속 response를 학습의 약한 참조로 쓴다.

이는 “사진 한 장을 보고 결함 유무를 맞히는 분류 문제”와 다르다. 올바른 질문은 다음과 같다.

1. 현재 layer까지의 과거 A image가 주어졌을 때,
2. 화면의 어느 pixel이 XCT-derived response와 더 관련 있어 보이는가?
3. 하지만 score map이 모든 layer에서 똑같거나 최고점이 너무 넓게 동점이라면,
4. 그 좌표는 그럴듯해 보여도 출력하지 않고 `withheld`해야 한다.

---

## 3. 데이터는 무엇이며, 왜 세 종류가 필요한가

| 데이터 | 프로젝트에서 하는 일 | 절대 하지 않는 일 |
|---|---|---|
| `LayerCameraAfterSpreading.tif` | **A-only model의 입력**. 레이저 전 powder surface 시계열 | 수정·재저장·dense preprocessed image 생성 |
| `LayerCameraBurned.tif` | B-stage visual/geometry calibration QC용 | A-only real-time model의 현재 입력으로 사용 |
| registered XCT CSV | training/evaluation에서 continuous weak response의 참조 | binary defect label로 단정 |
| XYPT command CSV | 각 layer의 commanded machine geometry 및 calibration audit | actual physical position이라고 단정 |
| DotGrid / Checkerboard / SecondaryCamera metadata | 독립 camera-to-machine metrology 검증 | 원본 수정 또는 자동 config 교체 |

### 3.1 A와 B의 시간 차이

```text
Layer z 시작
  └─ A: AfterSpreading, 레이저 전 촬영  ← 현재 baseline의 입력
       └─ 레이저 노광
            └─ B: Burned, 레이저 후 촬영 ← calibration/QC와 미래 확장용
```

A-only baseline은 실시간 가능성을 보려는 의도다. B를 current input으로 쓰면 레이저 조사 이후의 정보를 보게 되므로, “사전에 확인할 candidate”라는 현재 목표와 다르다.

### 3.2 TIFF가 실제로 저장하는 구조

A와 B는 ImageJ hyperstack이며 logical axis는 `T Z Y X = 3 × 250 × 2000 × 2000`이다. `T`는 세 LED, `Z`는 제조 layer다. `tifffile.memmap(..., series=0, mode='r')`로 원본 파일을 전부 메모리에 올리지 않고 필요한 frame만 읽는다. NIST 원문도 layer camera hyperstack의 `xyczt` ordering과 A/B 의미를 설명한다.[1]

---

## 4. 전체 데이터 흐름

```text
Immutable raw data
  ├─ A TIFF ──► K=4 causal Dataset ──► [K, 6, 256, 256] input
  │                 │                    ├─ 3 normalized LED intensity
  │                 │                    └─ 3 saturation-validity masks
  │                 ▼
  │             residual causal network ──► [1, 256, 256] score map
  │                                            │
  │                                            └─ safety decoder
  │                                               └─ (x_pixel, y_pixel, layer_z, score)
  │
  ├─ registered XCT CSV ──► on-the-fly Gaussian weak response + support mask
  │                              └─ training/evaluation loss only
  │
  └─ metadata TIFF + NIST documented geometry
       └─ independent calibration candidate audits
          └─ camera-to-machine relationship 검증 (아직 provisional)
```

### 중요한 분리

| 구분 | 사용 위치 | 이유 |
|---|---|---|
| `weak_support_mask` | training/evaluation loss | XCT가 관측하지 않은 곳을 잘못된 정상 label로 만들지 않기 위해서 |
| score-map decoder | 실제 candidate decoding | 미래 real-time input에는 XCT support가 없으므로 support mask를 보면 안 된다 |
| provisional geometry gate | optional post-local-maxima safety gate | configured rectangle 안 여부만 보는 보조 안전장치이며 physical truth가 아니다 |

---

## 5. 입력을 만들 때 수행한 전처리

### 5.1 인과적 K=4 sequence

모든 sample의 history는 `[z-3, z-2, z-1, z]`이다. endpoint는 train 4–157, validation 161–199, test 203–250으로 나뉘고 split 사이에는 3-layer guard를 둔다. 검증/test sample은 다른 split의 endpoint layer를 history로 볼 수 없다.

| Split | endpoint count | 용도 |
|---|---:|---|
| Train | 154 | normalization 추정과 model parameter 학습 |
| Validation | 39 | epoch/checkpoint 선택 |
| Test | 48 | 마지막 한 번의 held-out 평가 |

이 구조는 “미래 image나 test 분포를 학습에 섞어서 성능이 과대평가되는 leakage”를 막는다.

### 5.2 ROI·normalization·saturation validity

전체 2000×2000 raw image 중 working ROI는 `(250,250)–(1750,1750)`이고, model grid는 256×256이다. normalization 수치는 train split에서만 추정한다. `raw == 65535`은 sensor saturation으로 판단해 intensity 값만 넣지 않고 별도의 validity channel에 0을 넣는다.

따라서 한 시점의 input은 `3 intensity + 3 validity mask = 6 channels`이고, K=4 history 전체 shape은 `[4,6,256,256]`이다. 이 방식은 saturation을 “아주 밝다”는 physical signal로 오해하지 않게 한다.

---

## 6. XCT를 target처럼 쓰되, defect label로 쓰지 않는 이유

registered XCT CSV에서 command XY와 `xct_5x5x5` continuous response를 읽는다. train finite response 2,329,476개에서 추정한 p01/p99 `(0.40070, 0.58533)`으로 response를 0–1에 clip한다. response direction은 아직 unresolved이므로 inverse하지 않고, 임계값으로 binary label도 만들지 않는다.

각 XCT point는 provisional camera mapping을 거쳐 model grid로 옮겨지고, sigma=2 model pixel Gaussian으로 on-the-fly rasterize된다. output은 파일로 저장하지 않는다.

```text
XCT finite point ─► provisional projection ─► Gaussian response
                                            └► local support=1
XCT가 없거나 support 밖 ─────────────────────► support=0 = unknown
```

### 6.1 Loss가 하는 일

loss는 support pixel만 평균한 Smooth L1(`beta=0.1`)이다.

\[
L = \frac{\sum_i m_i\,\mathrm{SmoothL1}(\hat y_i,y_i)}{\sum_i m_i + \varepsilon}
\]

여기서 \(m_i=1\)은 weak supervision이 있는 pixel, \(m_i=0\)은 unknown pixel이다. runtime 검증에서 unknown prediction을 1000으로 바꿔도 loss가 변하지 않고, unknown 영역 gradient가 0임을 확인했다. 즉 “모르는 곳”을 model이 정상이라고 배우지 않는다.

---

## 7. 모델은 왜 residual temporal network가 되었는가

처음의 C8, E24, C32 temporal-only network는 input frame과 frame encoder feature가 달라도 `Conv3D → GroupNorm → SiLU` 뒤의 temporal feature가 sample 사이에서 완전히 같아지는 **temporal collapse**가 발생했다. 그 결과 모든 endpoint score map이 같거나 최고점이 넓게 묶여 candidate를 안전하게 출력할 수 없었다.

이를 해결하기 위해 현재 architecture는 endpoint frame feature를 decoder에 직접 보존한다.

```text
Encoded current A frame ───────────────────────────┐
                                                     ├─► temporal_final ─► decoder ─► score map
Past-only K=4 Conv3D temporal update ───────────────┘
                         encoded_endpoint + temporal_update
```

이 residual bypass는 미래 layer를 추가하지 않으므로 causality를 깨지 않는다. 그리고 default는 `false`여서 기존 checkpoint/forward contract도 보존된다.

| 평가 | 기존 C32 | C32 temporal residual | 해석 |
|---|---:|---:|---|
| Best validation loss | 0.06151950 | **0.05642983** | 8.27% 감소 |
| Held-out test loss | 0.07363038 | **0.06992126** | 5.04% 감소 |
| z=203/227/250 support prediction std | 0 | **0.043991 / 0.041168 / 0.045247** | map이 spatially constant가 아님 |
| selected map MAE | 0 | **non-zero** | endpoint마다 다른 map이 생성됨 |

이것은 **controlled baseline recovery**다. “실제 physical defect를 찾아냈다”는 뜻은 아니다.

---

## 8. Score map에서 candidate를 꺼낼 때의 안전 순서

score map만 높다고 좌표를 내보내지 않는다. decoder는 다음 순서를 지킨다.

1. map이 spatially flat한지 검사한다.
2. 최고 score 동점 영역이 지나치게 넓은 plateau인지 검사한다.
3. 서로 다른 endpoint map이 사실상 동일한지 검사한다.
4. 통과한 map에서만 local maxima를 뽑는다.
5. optional provisional geometry gate가 켜져 있으면, 마지막에 configured part rectangle 밖 maxima만 보류한다.

이 순서에서 XCT support mask는 decoder에 쓰지 않는다. 실제 new layer가 들어올 때 XCT support는 존재하지 않기 때문이다.

### 현재 candidate output의 의미

| 필드 | 의미 | 주의 |
|---|---|---|
| `x_pixel`, `y_pixel` | raw layer-camera pixel | **현재 primary location** |
| `layer_z` | 제조 layer 번호 | past-only input의 endpoint |
| `score` | XCT-derived continuous quality candidate score | defect probability 아님 |
| `provisional_machine_xy_rank2` | optional metadata | rank sensitivity가 크므로 control action 금지 |
| `provisional_part_rank2` | optional metadata | physical part identity 확정 아님 |

---

## 9. 좌표 calibration이 왜 아직 provisional인가

초기 screen-corner controls로 만든 rank1과 rank2 transform은 residual이 비슷했지만, geometry-gated candidates 240개에 대해 같은 part에 들어간 결과가 **0/240**이었다. rank1→rank2 inverse machine coordinate shift의 median은 14.463 px 수준이었다.

즉, pixel↔machine mapping을 하나 고르는 것과 “정확히 이 part의 이 physical position이다”라고 말하는 것은 다르다. 그래서 raw camera pixel을 primary로 둔다.

### 9.1 독립 NIST metadata가 하는 역할

NIST는 layer-camera DotGrid, layer-camera Checkerboard, red indicator가 있는 secondary-camera DotGrid를 제공하고 두 calibration route를 문서화한다.[1]

| Route | 현재 상태 | 이유 |
|---|---|---|
| Checkerboard direct route | hold | V2 detector가 board lower-left subset만 잡았고 spacing CV=0.5040으로 gate 0.45를 넘었다 |
| DotGrid + SecondaryCamera method #2 | candidate audit 준비 | DotGrid candidate 1,616개와 central red cluster가 local gate를 통과했다 |

NIST method #2의 published facts는 dot grid 50×50, 1.00 mm pitch, D origin=lower-left dot, `A(0,0)=D(28.25,24.25) mm`, D/A relative orientation 2.5°다.[1] 다만 source는 red origin 하나만으로 orientation이 결정되지 않음을 설명하므로, 프로젝트는 image-axis orientation과 ±2.5° sign을 multiple alternatives로 유지한다.

### 9.2 지금 새로 구현된 independent method-#2 audit

`src/audit_independent_method2_calibration_candidate.py`는 다음만 한다.

1. layer-camera DotGrid에서 subpixel dot center를 찾는다.
2. image-space PCA와 50-cluster quantization으로 provisional D-lattice cell index를 만든다.
3. 5×5 lattice block을 통째로 held out하고 나머지 cell로 `D→C` homography **candidate**를 fit한다.
4. held-out dot residual을 detected camera dot pitch 대비 비율로 평가한다.
5. 8개 image-lattice axis alternative와 ±2.5° sign을 조합한 transform candidates를 모두 기록한다.
6. current rank1/rank2 transform과 canonical machine anchors에서 raw pixel displacement를 **비교만** 한다.

이 단계에서 config를 자동 변경하거나 candidate를 선택하지 않는다. 통과의 뜻은 “사람이 transform/overlay/residual을 검토할 정보가 준비됐다”는 것뿐이다.

### 9.3 실행 결과: panel detection은 성공했지만 correspondence는 보류

V1 audit은 1,518개의 provisional lattice cell을 만들었고 PCA column 50개·row 48개를 표현해 **coverage 검사**는 통과했다. 이것은 dot board가 잘 보인다는 뜻이다. 그러나 "board가 보인다"와 "각 dot의 50×50 grid 번호를 정확히 안다"는 별개의 문제다.

| 검사 | 결과 | 쉬운 해석 |
|---|---:|---|
| held-out dot 수 | 298 | 학습에 넣지 않은 dot blocks도 충분히 검사함 |
| camera dot pitch | 14.59340 px | 한 dot 간격은 약 14.6 camera pixel |
| held-out RMSE | 6.00155 px = **0.41125 pitch** | 허용 기준 0.25 pitch보다 큼 |
| held-out p95 | **9.78092 px** | 허용 기준 7.29670 px보다 큼 |
| 최종 gate | **fail** | 16개 candidate transform 전체 hold |

Overlay에서도 board 위치 자체는 맞지만, 여러 held-out block에서 yellow actual dot과 magenta prediction이 계속 어긋난다. 특히 PCA와 서로 독립적인 1D 50-cluster가 perspective가 있는 2D grid를 완전하게 row/column으로 index하지 못한 흔적이 보인다. 따라서 이번 실패는 “장비 calibration이 틀렸다”는 결론도, “새 transform이 맞다”는 결론도 아니다. **현재 correspondence algorithm이 아직 충분하지 않다**는 결론이다.

---

## 10. 지금은 무엇을 보류하는가

| 지금 하지 않는 일 | 이유 |
|---|---|
| `calibration_v1.yaml` 교체 | held-out error가 통과 기준을 넘지 못함 |
| rank1/rank2/method-#2 중 하나 선택 | 16 alternatives가 모두 보류 상태 |
| candidate의 machine XY/part 확정 | coordinate ambiguity가 여전히 큼 |
| XCT target 재투영이나 retraining | calibration이 바뀐 것이 아니므로 model data contract도 바꿀 이유가 없음 |

이 시점의 가장 안전한 현재 상태는 변하지 않는다.

> **모델 output은 raw camera `(x_pixel, y_pixel, layer_z, score)`로 보고하며, physical machine coordinate/part interpretation은 hold한다.**

---

## 11. 다음에 필요한 일

다음 기술 작업은 새 homography를 반복해서 fit하는 일이 아니다. 먼저 perspective-aware 2D grid correspondence를 정제하는 **read-only audit**이 필요하다. 목표는 row와 column을 함께 보면서 “이 dot은 정확히 몇 번째 row·column인가”를 더 안정적으로 정한 뒤, 같은 5×5-block held-out test를 다시 수행하는 것이다.

| 다음 audit에서 개선할 것 | 그대로 유지할 것 |
|---|---|
| 2D row/column correspondence와 outlier rejection | immutable raw TIFF와 read-only memmap |
| DotGrid indexing overlay·held-out vector QC | K=4 causal A-only model 및 residual checkpoint |
| same block-held-out residual test | weak target sigma/direction/unknown policy |
| candidate transform을 human-review output으로 기록 | camera-primary reporting, no automatic config update |

새 refinement code의 추가는 별도 승인을 받은 뒤에만 시작한다. 기준을 느슨하게 하거나, 실패한 candidate 중 수치가 작아 보이는 것 하나를 고르는 방식은 사용하지 않는다.

---

## 12. 프로젝트 포트폴리오에서 강조할 수 있는 점

이 프로젝트의 강점은 단순히 CNN을 실행했다는 데 있지 않다. 제조 data의 시간·좌표·weak supervision 문제를 안전하게 분리한 데 있다.

| 포트폴리오 역량 | 이 프로젝트에서 한 증거 |
|---|---|
| 대용량 scientific image IO | 6 GB TIFF를 read-only memmap으로 필요한 frame만 접근 |
| time-series leakage 방지 | causal K=4 + split guard + train-only normalization |
| imperfect label 학습 | support-masked continuous Smooth L1, unknown≠normal |
| failure diagnosis | temporal collapse를 stagewise input-sensitivity diagnostic으로 위치 특정 |
| controlled experiment | endpoint residual bypass 하나만 바꿔 held-out regression 개선 확인 |
| safe candidate decoding | flat/tie/invariance/local-maxima safety gate, decoder의 XCT support exclusion |
| metrology literacy | provisional coordinate policy, rank sensitivity audit, independent NIST metadata route |
| reproducibility | code/config/docs tracked; raw/processed outputs immutable or ignored |

---

## 13. 꼭 기억할 용어

| 용어 | 쉬운 뜻 |
|---|---|
| Causality | z layer를 예측할 때 z보다 미래의 image를 절대 보지 않는 규칙 |
| Endpoint | K=4 sequence에서 현재 판단을 내리는 마지막 layer z |
| Weak supervision | 완전한 defect label 대신 sparse/indirect XCT response를 쓰는 학습 방식 |
| Support | XCT response가 신뢰 가능한 supervision을 제공하는 model pixel 영역 |
| Unknown | 관측이 없어 정상/이상을 말할 수 없는 영역. 0 label이 아님 |
| Temporal collapse | input은 달라도 temporal module 이후 prediction이 완전히 같아지는 failure |
| Residual bypass | current endpoint feature를 temporal update에 더해 정보가 decoder까지 남게 하는 연결 |
| Provisional calibration | 현재 쓸 수는 있으나 independent validation 전에는 physical truth로 주장할 수 없는 좌표 변환 |
| Method #2 | secondary red reference와 dot-grid D coordinate를 이용해 layer-camera C와 machine A 관계를 검증하는 NIST 문서의 route |

---

## 14. 마지막 안전 문장

> 지금 나온 candidate는 **“A-only causal model이 XCT-derived continuous response와 관련 있어 보인다고 제안한 raw camera pixel”**이다. 이 문장은 “defect가 확정됐다”, “이 part의 이 machine coordinate에 이상이 있다”, “공정 조건을 즉시 바꿔야 한다”를 뜻하지 않는다.

---

## References

[1] [Lane, B. and Yeung, H. (2020). *Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): Overhang Part X4*. Journal of Research of NIST, 125, 125027.](https://doi.org/10.6028/jres.125.027)

[2] [NIST PDR: AMMT Overhang Part X4 dataset record.](https://data.nist.gov/od/id/mds2-2233)
