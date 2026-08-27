# AMMT 프로젝트를 처음부터 이해하기

## 1. 이 프로젝트를 한 문장으로 말하면

이 프로젝트는 **금속 3D 프린팅(LPBF)이 진행되는 동안 layer-camera 영상에서 품질 이상이 의심되는 위치를 가능한 이른 시점에 찾고, 사후 XCT 측정값으로 그 후보를 약하게 검증하는 시스템**을 만드는 일이다.

최종적으로 모델은 단순히 “이상 있음/없음”만 말하는 것이 아니라 다음과 같은 후보를 반환하는 것을 목표로 한다.

```
(x_pixel, y_pixel, layer_z, score)
예: (105, 200, 15, 0.82)
```

이는 “제조 15번째 layer에서 카메라 좌표 `(105, 200)` 부근이 품질상 의심된다”는 뜻이다. 여기서 `score`는 현 단계에서는 **확정 결함 확률이 아니라 추가 검토가 필요한 후보 점수**다.

> 이 프로젝트는 의료 CT 분류 프로젝트가 아니다. 제조 중 찍히는 layer-camera 영상과 제조 후 등록된 XCT 측정값을 연결하는 **제조 AI·컴퓨터 비전·공정 데이터** 프로젝트다.

## 2. 왜 이 문제를 풀고 싶은가

금속 3D 프린팅은 한 번에 물체 전체를 만드는 것이 아니라, 매우 얇은 금속 가루 층을 반복해서 깔고 레이저로 녹여 붙이는 방식이다. 만약 한 layer에서 이상이 발생해도 제조가 끝난 뒤 XCT를 찍을 때까지 알 수 없다면 시간과 재료가 낭비될 수 있다.

이 프로젝트의 장기 목표는 다음과 같다.

| 시점 | 사람이 보는 데이터 | 시스템의 역할 |
| --- | --- | --- |
| 레이저 전 | A: AfterSpreading 영상 | 지금 막 도포된 분말 상태에서 조기 이상 후보를 찾는다. |
| 레이저 후 | B: Burned 영상 | 레이저 스캔 뒤 나타난 변화로 후보를 재평가한다. |
| 제조 종료 후 | registered XCT | 실제 내부 품질과 관련된 sparse 측정값으로 공정 중 후보를 약하게 검증한다. |

즉, “XCT로 결함을 찾는 모델”이 아니라, **XCT를 사후 참고 정보로 사용해 layer-camera의 실시간 판단을 학습시키는 모델**이다.

## 3. 우리가 가진 데이터는 무엇인가

### 3.1 A와 B는 한 쌍의 다른 시점 영상이다

다운로드한 대용량 TIFF 두 개는 각각 6 GB이며, 사진 한 장이 아니라 LED와 제조 layer가 쌓인 논리적 hyperstack이다.

| 파일 | 실제 촬영 시점 | 프로젝트에서의 역할 | 주의할 점 |
| --- | --- | --- | --- |
| `LayerCameraAfterSpreading.tif` | 가루를 새로 깐 직후, 레이저 전 | **A head**의 조기 경보 입력 | 아직 레이저를 쏘기 전이므로 실시간 조기 판단에 적합하다. |
| `LayerCameraBurned.tif` | 레이저 스캔 후 | **B head**의 사후 재평가 입력 | 레이저의 정상 반사·용융 변화도 함께 보인다. |

각 파일은 `LED 3개 × 제조 layer 250개 × 높이 2000 × 너비 2000`의 `uint16` 신호다. 따라서 “750장의 독립 사진”처럼만 읽으면 안 되고, **LED는 채널**, **layer는 시간**으로 해석해야 한다.

### 3.2 A와 B를 빼면 결함이 되는가

아니다. 처음에는 B−A 차이가 크면 이상일 것처럼 보일 수 있다. 하지만 B는 레이저를 쏜 뒤 영상이므로, 정상적으로 잘 녹은 구역도 A와 크게 달라진다. 반사, 조명, 용융 흔적, 장비 구조도 차이를 만든다.

따라서 지금 프로젝트는 아래와 같이 해석한다.

| 가능해 보이는 단순 접근 | 왜 위험한가 | 실제 정책 |
| --- | --- | --- |
| `B-A`를 결함 정답으로 사용 | 정상 레이저 공정 변화까지 결함으로 학습할 수 있다. | B−A는 QC·향후 fusion feature 후보일 뿐 label이 아니다. |
| A를 B처럼 만들도록 denoising | B가 clean image가 아니다. | A와 B는 서로 다른 stage의 입력이다. |
| A와 B를 무조건 합침 | 레이저 전 조기 판단의 인과성이 깨진다. | A-only, B-only, fusion을 순서대로 비교한다. |

### 3.3 XCT 데이터는 왜 필요한가

XCT는 제조가 끝난 후 물체 내부 품질을 측정하는 방식이다. 하지만 우리 XCT CSV는 전체 카메라 화면의 결함 그림이 아니다. 각 row는 특정 machine XY 위치에서 얻은 sparse 측정점이며, 그 안에 `xct_5x5x5`라는 연속 값이 있다.

쉽게 비유하면, 카메라는 운동장 전체를 찍지만 XCT는 운동장 여기저기를 표본 조사한 점 목록이다. 따라서 XCT가 없는 픽셀을 “정상”이라고 말할 수 없다. 단지 **그 위치는 모른다(unknown)**고 해야 한다.

## 4. 프로젝트의 가장 중요한 어려움 네 가지

이 프로젝트가 단순 이미지 분류보다 어려운 이유는 데이터가 불완전하기 때문이다.

| 어려움 | 실제 문제 | 잘못 처리하면 생기는 일 |
| --- | --- | --- |
| Sensor saturation | 많은 pixel이 65535로 포화되어 실제 밝기 차이를 잃는다. | 포화된 밝은 영역을 이상 또는 정상 texture로 잘못 학습한다. |
| 시간 누수 | 미래 layer를 입력에 넣으면 실제 실시간 상황보다 성능이 과대평가된다. | 테스트 성능은 좋아 보이지만 현장에서는 재현되지 않는다. |
| 좌표계 불일치 | XCT는 machine mm 좌표, 카메라는 pixel 좌표다. | 맞지 않는 위치에 품질 target을 붙인다. |
| Sparse target | XCT가 측정된 곳만 값이 있다. | 측정되지 않은 곳을 normal=0으로 간주해 잘못된 label을 만든다. |

지금까지의 전처리와 검증은 사실상 이 네 가지 문제를 하나씩 막는 과정이었다.

## 5. 처음부터 지금까지 실제로 한 일

### 단계 1. 원본을 안전하게 읽을 방법을 확인했다

6 GB TIFF를 일반 이미지처럼 전부 메모리에 올리거나 재저장하면 저장 공간과 메모리를 많이 사용하고, 실수로 원본을 훼손할 위험도 있다. 그래서 `memmap` 방식으로 **필요한 layer와 LED만 읽는 Dataset**을 만들었다.

결과적으로 원본 파일은 바꾸지 않고, 예를 들어 “125번째 layer의 LED 1만 읽기”가 가능해졌다. 이것이 이후 모든 단계의 안전한 출발점이다.

### 단계 2. A와 B의 역할을 분리했다

A와 B가 같은 layer와 LED에서 대응되는지 확인했다. 대응은 가능했지만 B−A가 결함 label이라는 근거는 없었다. 이 검증 덕분에 프로젝트의 모델 구조가 다음처럼 정리됐다.

```
A-only: 레이저 전 조기 후보 탐지
B-only: 레이저 후 후보 재평가
Fusion: A와 B를 모두 볼 수 있을 때 위치 안정화
```

현재는 가장 어려운 문제를 작게 시작하기 위해 **A-only baseline**을 먼저 만들 예정이다.

### 단계 3. ROI와 포화를 확인했다

카메라 전체 2000×2000에는 관심 없는 장비 가장자리와 반사 영역이 섞여 있다. 그래서 여러 crop 후보를 비교하고 현재는 raw camera 기준 `(250,250)–(1750,1750)`을 working ROI로 사용한다.

하지만 crop만으로 포화는 해결되지 않았다. LED1·LED2에는 65535로 꽉 찬 pixel이 넓게 존재했다. 이때 “65535는 가장 밝다”라고 그대로 모델에 주면 잘못된 해석을 하게 된다.

그래서 각 LED는 두 가지 정보로 나뉜다.

| 입력 정보 | 의미 |
| --- | --- |
| normalized intensity | 포화가 아닌 valid pixel의 밝기를 stage·LED별 p01/p99로 `[0,1]` 범위에 맞춘 값 |
| validity mask | `raw < 65535`이면 1, 포화면 0. 이 pixel을 믿어도 되는지 알려 주는 신호 |

LED가 3개이므로 한 layer의 최종 입력은 **intensity 3채널 + validity mask 3채널 = 6채널**이다.

### 단계 4. 실시간 규칙을 지키는 시간 분할을 만들었다

실제 현장에서는 15번째 layer를 판단할 때 16번째 layer 영상을 볼 수 없다. 이를 **causality**라고 한다. 현재 모델은 endpoint `z`를 판단할 때 `[z-3, z-2, z-1, z]`의 최근 4개 layer만 본다.

또한 train, validation, test가 서로 미래 정보를 공유하지 않도록 다음처럼 나눴다.

| 구분 | endpoint layer | 역할 |
| --- | --- | --- |
| Train | z=4–157 | 모델 파라미터와 normalization 통계를 학습한다. |
| Guard | z=158–160 | split 경계를 보호하는 buffer다. |
| Validation | z=161–199 | 모델 선택과 설정 비교에 쓴다. |
| Guard | z=200–202 | test 경계를 보호한다. |
| Test | z=203–250 | 마지막 일반화 성능 확인에 쓴다. |

이 작업은 화려해 보이지 않지만, 포트폴리오에서 “실시간이라는 조건을 실제 data split에 반영했다”는 강한 근거가 된다.

### 단계 5. 카메라 pixel과 XCT 위치를 맞췄다

XCT CSV의 위치는 mm 단위 machine XY이고, 카메라는 2000×2000 pixel이다. 둘을 바로 같은 숫자로 보면 안 된다. 그래서 4개 part의 screen corner를 사용해 **homography**라는 투시 변환을 만들었다.

간단히 말하면 homography는 “기울어져 찍힌 카메라 화면의 어떤 pixel이 실제 제조 장비의 어떤 XY 위치에 해당하는가”를 변환하는 규칙이다.

처음에는 좌우/상하가 뒤집힌 mirror 후보 두 개가 수학적으로 거의 동점이었다. 그래서 전역 밝기 상관을 확인했지만 신호가 약했다. 그 뒤 작은 5×5 patch를 사용한 local photometric refinement로 candidate 2와 raw offset `(0,-6)` px를 provisional하게 선택했다.

> 이 변환은 현 단계에서 충분히 검증된 작업용 calibration이지만, 독립 DotGrid/Checkerboard 등의 metrology 검증 전까지 **provisional**이다.

### 단계 6. Sparse XCT를 연속 weak target으로 바꿨다

XCT 측정점은 점으로 흩어져 있다. 점 하나만 supervision으로 쓰면 model grid에서 너무 듬성듬성하다. 그래서 각 점 주변에 작은 Gaussian 모양의 영향 범위를 주어 model 256×256 grid에 옮긴다.

이때 중요한 정책은 다음과 같다.

```
XCT point가 있는 주변: weak supervision support = 1
XCT point가 없는 곳: unknown, loss 계산에서 제외
```

sigma=2 model pixel을 선택했다. 너무 작으면 점이 너무 끊기고, 너무 크면 실제로 모르는 영역까지 target이 퍼진다. sigma=2는 이 두 위험 사이의 절충안이다.

****

## 6. 현재 상태를 한 장으로 보면

```
[제조 중 촬영]
A TIFF (레이저 전) ─┐
                    ├─> causal 6-channel input ─> A-only model (다음 단계)
B TIFF (레이저 후) ─┘                                 │
                                                      ▼
                                      후보 heatmap → (x_pixel, y_pixel, layer_z, score)

[제조 후 참조]
registered XCT CSV → command XY→camera calibration → sparse continuous weak response
                                                       + support mask
                                                       └─> 학습 중 masked loss에만 사용
```

현재까지는 화살표의 왼쪽부터 weak response와 support mask까지 연결했다. 아직 **model을 학습하지는 않았다.**

## 7. 다음에 무엇을 하는가

다음 단계는 support-mask weighted continuous regression loss다. 말은 길지만 원리는 간단하다.

```
support mask가 1인 pixel에서만
모델 예측값과 weak response의 차이를 계산한다.

support mask가 0인 pixel은
XCT가 없어서 정답을 모르므로 loss 계산에 넣지 않는다.
```

그 다음 A-only baseline을 학습한다. A-only 모델은 레이저 전 A 영상 4개 layer를 보고 그 layer의 XCT-derived continuous quality candidate map을 예측한다.

| 학습 단계 | 입력 | target | 출력 해석 |
| --- | --- | --- | --- |
| A-only baseline | 최근 A layer 4개, 6채널 | masked continuous weak response | 조기 품질 후보 지도 |
| B-only 확장 | 최근 B layer 4개, 6채널 | 같은 endpoint weak response | 레이저 후 재평가 지도 |
| A/B fusion | A feature + B feature | 같은 endpoint weak response | 더 안정적인 사후 후보 지도 |
| 방향 검증 | heatmap과 XCT segmentation/manual review | 물리적 의미 | high/low score를 anomaly로 부를 수 있는지 결정 |

## 8. 지금 이 프로젝트가 아직 하지 않은 것

신뢰도 있는 포트폴리오는 “한 것”뿐 아니라 “아직 하지 않은 것”도 명확히 구분한다.

| 아직 하지 않은 일 | 이유 |
| --- | --- |
| YOLO bounding box/class 분류 | 목표가 객체 class가 아니라 pixel heatmap과 위치 후보이기 때문이다. |
| 확정 defect classifier | XCT response direction과 binary threshold가 검증되지 않았다. |
| A/B fusion 성능 주장 | 먼저 A-only와 B-only baseline을 공정하게 비교해야 한다. |
| calibration 확정 선언 | 현재 transform은 local refinement로 선택한 provisional calibration이다. |
| unsupported pixel을 normal로 처리 | XCT가 없어서 모르는 곳이지 정상이라고 증명된 곳이 아니다. |

이 제한 사항은 약점이 아니다. 제조 AI에서는 특히, 불확실성을 숨기고 결함을 단정하는 모델보다 **무엇을 알고 무엇을 모르는지 관리하는 모델**이 더 신뢰할 수 있다.

## 9. 포트폴리오에서 어떻게 설명하면 좋은가

### 30초 요약

> “LPBF 금속 3D 프린팅의 layer-camera 시계열로 실시간 품질 이상 후보 위치를 찾는 프로젝트입니다. 레이저 전 A 영상과 레이저 후 B 영상의 역할을 분리했고, saturated sensor pixel을 validity mask로 처리했습니다. 사후 XCT는 dense defect mask가 아니라 sparse machine-coordinate reference이기 때문에 homography로 camera 좌표에 정합하고, support-aware continuous weak target으로 on-the-fly 생성했습니다. 현재 A-only masked baseline을 준비 중이며, 최종 출력은 `(x,y,layer,score)` 형식의 검토 후보입니다.”

### 면접에서 강조할 수 있는 문제 해결

| 질문을 받을 수 있는 지점 | 답변의 핵심 |
| --- | --- |
| 왜 B−A를 label로 쓰지 않았나 | 레이저 후 정상 변화가 커서 clean–noisy pair가 아니기 때문이다. |
| 왜 mask channel이 필요한가 | 65535 포화 pixel은 실제 intensity를 잃은 관측 실패이므로 모델에 명시해야 한다. |
| 왜 test 데이터를 normalization에 안 썼나 | 미래/평가 분포를 미리 보는 leakage를 막기 위해서다. |
| 왜 target 없는 곳을 0으로 안 뒀나 | XCT가 sparse measurement라 미측정은 정상 증거가 아니라 unknown이기 때문이다. |
| 왜 calibration을 provisional이라고 하나 | machine mm와 camera pixel의 위치 정합은 외부 metrology로 독립 확인해야 하기 때문이다. |

## 10. 가장 중요한 세 문장

> **첫째, A와 B는 clean/noisy 영상 쌍이 아니라 제조 시점이 다른 두 관측이다.**

> **둘째, XCT는 결함 그림이 아니라 sparse reference point이므로, 없는 곳을 정상이라고 학습시키면 안 된다.**

> **셋째, 현재 목표는 ‘결함 확정’이 아니라 실시간으로 검토할 ****`(x,y,layer,score)`**** 후보를 안전하게 만드는 것이다.**

## References

[1]: https://data.nist.gov/od/id/mds2-2233 "NIST, Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT ): Overhang Part X4"



---

## 11. 새로 완료한 단계: 모르는 곳을 억지로 학습하지 않는 loss

이제 **support-mask weighted continuous regression loss**를 만들고 실제로 검사했다. 이름이 길지만, 뜻은 다음 한 문장이다.

> **XCT가 실제로 측정된 위치에서만 모델을 채점하고, XCT가 없는 위치에는 정답을 아는 척하지 않는다.**

### 11.1 왜 이 단계가 필요한가

앞 단계에서 XCT 점을 camera/model grid에 옮겨 `weak_response`와 `support mask`를 만들었다. 하지만 이것만으로 안전하지 않다. 모델을 학습할 때 response map 전체의 오차를 계산하면, support가 0인 픽셀도 loss에 들어간다. support=0은 “정상”이 아니라 “모름”이므로, 이 pixel을 모델 학습에 넣으면 잘못된 신호가 생긴다.

예를 들어 선생님이 답을 채점할 수 있는 문제 10개와 답을 모르는 문제 90개를 섞어 두었다고 생각하면 된다. 답을 모르는 90개를 틀렸다고 처리하면 학생은 부당하게 학습된다. 이 프로젝트의 mask loss는 답을 아는 10개만 채점하는 방식이다.

### 11.2 loss는 실제로 어떻게 계산하는가

모델이 만든 예측을 `prediction`, XCT에서 온 연속 reference를 `weak_response`, XCT support 여부를 `mask`라고 하면 다음처럼 계산한다.

```text
mask = 1인 pixel만 prediction과 weak_response의 차이를 계산
mask = 0인 pixel은 계산에서 완전히 제외
```

구체적으로는 Smooth L1이라는 오차 함수를 사용했다. 매우 큰 오차 하나가 전체 학습을 지나치게 흔들지 않도록 해 주는 회귀용 loss다. 여기서 `beta=0.1`은 `[0,1]` 범위의 continuous response에 맞춘 전환 폭이다.

| 경우 | mask | loss에 포함되는가 | 이유 |
|---|---:|---|---|
| XCT가 있는 점 주변 | 1 | 포함 | reference가 있으므로 모델 예측을 비교할 수 있다. |
| XCT가 없는 camera 영역 | 0 | 제외 | 정상인지 이상인지 알 수 없다. |
| 초기 z=4 layer | 전체 0 | loss=0 | XCT support가 아직 없어 target 학습을 하지 않는다. |
| z=128 layer | 일부 1 | 포함 | 3,439개 model pixel에서 weak response를 비교한다. |

### 11.3 우리가 실제로 확인한 결과

이번 검증은 모델을 학습한 것이 아니라, **loss 규칙이 우리가 정한 unknown 정책을 제대로 지키는지** 확인한 것이다.

| 확인 질문 | 실제 결과 | 의미 |
|---|---|---|
| z=4처럼 XCT가 없는 layer는 어떻게 되는가 | loss=0.0, gradient=0.0 | 없는 XCT를 normal=0으로 만들어 학습하지 않는다. |
| z=128처럼 XCT support가 있는 layer는 학습 신호가 생기는가 | 3,439 pixel이 loss에 참여했고 supported gradient는 0보다 컸다. | support가 있는 위치에서만 model parameter가 update될 수 있다. |
| XCT가 없는 곳의 prediction을 1000으로 바꾸면 loss가 변하는가 | loss difference=0.0 | unknown pixel은 loss에 전혀 새지 않는다. |

따라서 이제 “XCT가 있는 곳만 안전하게 참고하는” 모델 학습 규칙까지 준비됐다. 다음 단계는 이 loss를 실제 **A-only baseline model**에 연결하는 일이다.

### 11.4 지금 위치를 다시 정리하면

```text
완료: 원본 이해 → A/B 역할 분리 → saturation mask → causal split
      → XCT 좌표 정합 → weak response/support → unknown-safe loss 검증

다음: A-only model이 A 영상 4개 layer를 보고
      [0,1] candidate response map을 만들도록 학습

나중: B-only와 A/B fusion을 비교하고,
      XCT response의 high/low가 실제 anomaly를 뜻하는지 별도 검증
```

주의할 점은 이번 loss 검증이 “모델이 결함을 찾는다”는 성공이 아니라, **결함 학습을 시작하기 전에 target가 없는 곳을 잘못 가르치지 않도록 안전장치를 검증한 성공**이라는 점이다.


---

## 12. 다음 단계: A-only baseline 모델을 실제로 학습할 준비

이제 안전장치가 준비됐으므로, 첫 번째 실제 모델인 **A-only baseline**을 만들었다. 여기서 A-only는 “레이저를 쏘기 전의 A(AfterSpreading) 영상만 보고 판단한다”는 뜻이다.

### 12.1 왜 A-only부터 시작하는가

레이저 전 A 영상은 제조 중 가장 이른 시점에 얻을 수 있다. B 영상은 레이저를 쏜 뒤에야 생기므로, A-only 모델이 먼저 성립해야 진짜 조기 경보가 가능하다. 또한 A, B, fusion을 처음부터 모두 학습하면 성능이 좋아져도 어느 정보가 도움이 됐는지 알기 어렵다. 그래서 먼저 A-only를 기준점으로 만들고, 나중에 B-only와 A/B fusion이 얼마나 좋아지는지 공정하게 비교한다.

### 12.2 A-only 모델이 받는 것과 내놓는 것

```text
입력: 최근 A 영상 4개 layer
      각 layer = LED 밝기 3장 + 포화 신뢰도 mask 3장
      전체 shape = [batch, 4, 6, 256, 256]

출력: 현재 endpoint layer의 candidate response map 1장
      전체 shape = [batch, 1, 256, 256]
      값 범위 = 0~1
```

모델은 먼저 각 layer에서 “어디가 밝고, 어디가 포화되어 믿기 어려운가”를 보고, 그 다음 최근 4개 layer가 어떻게 변했는지를 본다. 시간 방향 convolution은 과거 쪽에만 padding하므로 앞으로의 layer를 몰래 보지 않는다.

| 모델 구성 | 쉬운 설명 | 필요한 이유 |
|---|---|---|
| 6채널 frame encoder | 한 layer의 LED 밝기와 신뢰도 mask를 함께 읽는다. | 65535 포화 영역을 실제 밝기 정보처럼 오해하지 않는다. |
| 과거만 보는 temporal convolution | 최근 4개 layer의 변화를 순서대로 합친다. | real-time 상황에서 미래 layer를 쓰지 않는다. |
| sigmoid output | model의 출력값을 0~1 범위로 제한한다. | 0~1로 scaling된 continuous XCT response와 직접 비교한다. |
| support-mask loss | XCT가 있는 위치에서만 오차를 계산한다. | 모르는 pixel을 정상이라고 가르치지 않는다. |

### 12.3 모델이 학습할 때 일어나는 일

학습 초반에는 model output이 거의 무작위다. 예를 들어 z=128에서 support mask가 1인 3,439개 pixel만 골라 model prediction과 XCT-derived weak response의 차이를 계산한다. 그 차이를 줄이는 방향으로 model의 내부 숫자가 조금씩 바뀐다.

반대로 z=4처럼 XCT support가 전혀 없는 layer는 input으로 읽힐 수 있지만 target loss는 0이다. 즉, model은 그 layer를 “정상이다”라고 배우지 않고, **그 layer에는 XCT 선생님이 아직 없으니 채점하지 않는다**고 처리한다.

### 12.4 실행은 두 단계다

| 순서 | 실행 | 파일 변화 | 목적 |
|---:|---|---|---|
| 1 | dry-run | 없음 | 무작위 model이 input→prediction→masked loss까지 shape과 range에 맞게 연결되는지 확인 |
| 2 | 실제 training | `outputs/a_only_baseline_v1/`에만 checkpoint·history·test metric·candidate JSON 생성 | validation으로 best model을 고르고 held-out test에서 candidate를 확인 |

실제 training이 끝나면 model은 test layer별로 top-5 local maximum을 찾아 다음처럼 compact하게 저장한다.

```text
(x_pixel, y_pixel, layer_z, score)
```

여기서도 `score`는 아직 “확정 결함 확률”이 아니다. **XCT-derived continuous quality candidate score**다. 이 score의 high/low가 실제 물리적 이상과 어떤 관계인지 확인하려면, 다음 단계에서 XCT segmentation 또는 사람이 확인한 결함 영역과 비교해야 한다.

### 12.5 현재 우리가 확인하려는 첫 성공 기준

첫 baseline에서 바로 높은 정확도를 기대하거나 주장하지 않는다. 첫 성공은 아래 순서다.

1. dry-run이 input/output/loss shape과 범위 오류 없이 통과한다.
2. training 중 validation supported-pixel loss가 기록되고 best checkpoint가 저장된다.
3. held-out test metric과 `(x,y,layer,score)` 후보 JSON이 생성된다.
4. candidate 위치가 support·XCT 검토와 공간적으로 말이 되는지 확인한다.

이 단계가 끝나면 비로소 “전처리와 target 규칙을 갖춘 A-only 모델이 실제로 학습되고 평가됐다”라고 포트폴리오에 쓸 수 있다.


---

## 13. A-only 모델이 실제로 연결되는지 먼저 확인했다

실제 training을 하기 전에 **dry-run**을 실행했다. dry-run은 model을 한 번 통과시켜 보되, model의 내부 값을 바꾸지 않고 파일도 저장하지 않는 사전 점검이다. 자동차를 출발시키기 전에 시동, 브레이크, 계기판이 연결됐는지 보는 것과 비슷하다.

이번 dry-run의 결과는 다음과 같다.

| 확인한 것 | 결과 | 쉬운 해석 |
|---|---|---|
| 실행 장치 | Apple `mps` | 현재 Mac의 GPU 가속 환경에서 model 계산이 가능했다. |
| model 입력 | `[1,4,6,256,256]` | 한 번에 sample 1개, 최근 layer 4개, layer마다 정보 6장, 각 이미지 256×256으로 들어갔다. |
| model 출력 | `[1,1,256,256]` | 한 endpoint layer에 대한 candidate response map 한 장이 나왔다. |
| output 범위 | 약 0.518 | sigmoid를 썼으므로 0~1 범위를 벗어나지 않았다. 아직 학습 전이라 화면 전체가 거의 같은 값인 것은 정상이다. |
| reference 연결 | z=128, support pixel=3,439 | XCT reference가 존재하는 위치가 model loss 계산까지 연결됐다. |
| masked loss | 0.11099906 | 숫자가 정상적으로 계산됐다. 이 숫자는 아직 model 성능이 아니라 “연결이 된다”는 확인값이다. |

따라서 이제 model, input, XCT weak target, support mask, loss가 한 줄로 이어졌다. 다음 실제 training에서는 model이 여러 train sample을 반복해서 보면서, support가 있는 위치에서만 prediction을 조금씩 바꿔 weak response에 가까워지도록 학습한다.

> **현재 단계:** 전처리와 학습 안전장치는 준비됐고, 첫 A-only training만 남아 있다. training 후에는 validation으로 가장 좋은 checkpoint를 선택하고, 처음 보지 않은 test layer에서 `(x_pixel, y_pixel, layer_z, score)` 후보를 출력한다.


---

## 14. 첫 A-only 학습을 실제로 해 본 결과

이제 A-only model을 8번의 epoch 동안 실제로 학습했다. 한 epoch은 model이 train data 전체를 한 번 훑는 단위라고 생각하면 된다. 매번 model은 XCT support가 있는 곳에서만 자기 prediction을 조금씩 고쳤고, validation data로 얼마나 잘 일반화되는지 확인했다.

### 14.1 잘 된 부분: model과 학습 규칙은 끝까지 연결됐다

| 관찰한 값 | 처음 | 마지막 | 뜻 |
|---|---:|---:|---|
| Train masked loss | 0.11765 | 0.06041 | train support 위치의 오차가 약 48.65% 감소했다. model이 약한 XCT reference에서 학습 신호를 받았다는 뜻이다. |
| Validation masked loss | 0.09674 | 0.06163 | 학습에 직접 쓰지 않은 validation layer에서도 오차가 약 36.28% 감소했다. 가장 낮았던 epoch 8 checkpoint를 선택했다. |
| Held-out test masked loss | - | 0.07345 | 마지막 test layer에서 얻은 값이다. validation보다 약 19.17% 높아졌지만, 미래 layer를 몰래 보지 않고 측정한 초기 generalization gap이다. |

이 결과는 “A 영상만으로도 quality response를 완벽히 맞혔다”는 뜻은 아니다. 하지만 **원본 입력 → sparse XCT target → unknown-safe loss → causal validation → held-out test**라는 전체 학습 체인이 실제로 실행됐다는 중요한 첫 성공이다.

### 14.2 아직 안 된 부분: 위치 후보가 모두 같은 곳을 가리켰다

training 후 model이 test layer마다 top-5 후보를 만들었다. 총 48개 test layer × 5개 = 240개 후보가 나왔다. 그런데 확인해 보니 모든 후보의 score가 `0.67868608` 하나로 같았고, 모든 layer에서 같은 다섯 pixel 위치가 반복됐다.

이것은 결함이 그 위치에 계속 있다는 뜻이 아니다. model output이 화면 전체에서 거의 같은 값인 **flat plateau**가 되었을 때, top-k 함수가 같은 값 중 앞쪽에 놓인 pixel을 임의로 고른 현상이다. 즉, 지금 나온 좌표는 실제 품질 이상 위치가 아니라 computer tensor의 동점 처리 결과다.

| 지금 신뢰할 수 있는 것 | 지금 신뢰하면 안 되는 것 |
|---|---|
| A-only model이 sparse supported pixel에서 학습·validation·test loss를 계산했다는 사실 | `test_coordinate_candidates.json`의 현재 `(x,y,layer,score)` 위치 |
| test loss=0.07345라는 초기 held-out regression 지표 | score=0.67868608을 anomaly probability라고 부르는 것 |
| early unknown layer가 loss에서 제외됐다는 정책 | 반복되는 화면 왼쪽 위 후보를 실제 defect라고 해석하는 것 |

이 문제를 숨기지 않는 것이 중요하다. 포트폴리오에서는 “첫 baseline은 end-to-end regression을 통과했지만 spatial prediction이 평탄해 localization을 보류했고, 다음 diagnostic으로 원인을 분리했다”라고 쓰는 편이 훨씬 신뢰도 높은 설명이다.

### 14.3 이제 무엇을 확인하는가

다음에는 checkpoint가 만든 response map의 최소값·최대값·표준편차를, 입력 영상과 XCT support target의 공간 분포와 비교한다. 이 검사는 model이 정말 화면의 위치 차이를 보고 있는지, 아니면 전체 평균값만 내놓는지를 알려 준다. 결과가 나오기 전까지 현재 top-5 좌표는 사용하지 않는다.


---

## 15. 이제 같은 좌표를 억지로 내지 않도록 안전장치를 넣었다

첫 학습에서 model이 거의 평평한 response map을 만들었는데도 top-5 기능이 화면의 몇 위치를 후보라고 내놓았다. 이것은 top-5가 “가장 높은 값”을 무조건 골라야 하기 때문이다. 화면의 모든 값이 같으면 실제로 높은 곳이 없는데도, 컴퓨터는 앞쪽에 놓인 pixel을 골라야 한다.

그래서 후보를 만드는 규칙을 보완했다. 앞으로 response map의 가장 큰 값과 가장 작은 값의 차이가 `1e-6` 이하이면, model이 위치 차이를 만들지 못했다고 판단하고 좌표를 반환하지 않는다.

```text
기존: flat map → 동점 pixel 중 임의 top-5 좌표 출력
변경: flat map → candidates=[] + withheld_spatial_plateau
```

이 규칙은 defect threshold가 아니다. “score가 이보다 크면 결함” 같은 뜻이 아니라, **map 안에서 위치를 비교할 수 있는 최소한의 차이조차 없는가**를 보는 안전장치다.

### 15.1 다음 diagnostic은 무엇을 볼까

다음에는 첫 training checkpoint를 그대로 읽어서, 실제로 세 test layer에서 map이 얼마나 평평한지 숫자로 확인한다. training도 다시 하지 않고 파일도 새로 저장하지 않는다.

| 볼 값 | 쉬운 의미 | 필요한 이유 |
|---|---|---|
| prediction min/max/std | model이 화면의 장소마다 다른 점수를 냈는지 | 0이면 완전히 평평하고 위치 후보를 낼 수 없다. |
| supported target min/max/std | XCT가 있는 점 주변의 reference가 실제로 얼마나 달랐는지 | target도 거의 같다면 model만의 문제가 아닐 수 있다. |
| support 안 Pearson correlation | model의 위치별 점수 변화가 target 변화와 같이 움직이는지 | loss 하나만으로는 spatial relationship을 알기 어렵다. |
| candidate decoder status | `emitted` 또는 `withheld_spatial_plateau` | 허위 좌표가 막혔는지 확인한다. |

이 결과를 보면 다음에 무엇을 바꿀지 근거가 생긴다. 예를 들어 target은 다양하지만 prediction만 평평하면 model capacity나 learning rule을 살펴보고, target도 거의 평평하면 rasterization 또는 target scaling을 다시 검토한다. 이처럼 한 번에 여러 가지를 바꾸지 않는 것이 실험 결과를 이해하는 방법이다.


### 15.2 코드를 추가할 때도 작은 연결 검사가 필요한 이유

plateau 안전장치를 추가하는 중, model에 input을 넣는 한 줄이 손상되어 Python이 시작 단계에서 멈춘 일이 있었다. 다행히 이 오류는 checkpoint, TIFF, XCT CSV를 읽기 **전**에 발견됐다. 따라서 원본 데이터나 첫 학습 결과는 바뀌지 않았고, 원래 input 연결식을 복구했다.

이것은 왜 diagnostic을 training과 분리하는지 보여 준다. 새 기능을 더할 때 먼저 read-only로 import와 data flow가 연결되는지 확인하면, 실험 결과를 덮어쓰거나 raw data를 건드리지 않고도 문제를 작게 잡을 수 있다. 다음 실행은 복구된 code가 실제 checkpoint를 정상적으로 읽고, flat map에서 좌표를 보류하는지 확인하는 단계다.


---

## 16. 진단 결과: 화면 전체는 변했지만, 중요한 XCT 구역에서는 변하지 않았다

spatial diagnostic을 실제 checkpoint에 실행해 보니 처음 예상보다 더 구체적인 상황이 나왔다. 화면 전체 prediction map만 보면 값이 달라 보였다. 최소값은 거의 0, 최대값은 약 0.679였고, 값 종류도 25개가 있었다. 그래서 단순히 “map 전체가 완전히 평평하다”라고 말할 수는 없었다.

하지만 실제로 XCT reference가 있는 **support 구역 안만 따로 보니** 세 test layer 모두 model prediction이 `0.67868608` 하나로 같았다. 반면 XCT target 값은 장소에 따라 0부터 약 0.82~0.86까지 달랐다. 즉 model은 화면 가장자리나 support 밖에는 다른 값을 냈지만, 정작 채점받는 중요 구역에서는 모든 위치에 같은 답을 쓰고 있었다.

| 질문 | 확인 결과 | 뜻 |
|---|---|---|
| 화면 전체에서 model 점수가 다른가 | 예. global range≈0.679, std≈0.117 | 전체 map만 보면 변화가 있다. |
| XCT support 안에서 model 점수가 다른가 | 아니오. std=0, min=max=0.67868608 | model이 중요한 위치 간 차이를 배우지 못했다. |
| XCT target은 위치마다 다른가 | 예. std≈0.137~0.157 | target이 평평해서 생긴 문제는 아니다. |
| target과 model 점수의 상관은 있는가 | 계산 불가(`null`) | model 값이 모두 같으면 서로 움직이는지 비교할 수 없다. |

따라서 첫 번째 safety gate인 “화면 전체 max-min이 0이면 후보를 숨긴다”는 규칙만으로는 부족했다. 화면 전체는 다르기 때문에 그 규칙을 통과했지만, 실제 중요한 support 구역에서는 여전히 똑같은 값이었다.

이제 decoder는 화면 전체만 보지 않고, 높은 점수가 넓은 plateau를 이루는지, 최댓값 동점이 얼마나 많은지, 다른 layer에서도 map이 똑같이 반복되는지를 추가로 검사해야 한다. 그 기준을 충족하지 못하면 `candidate`를 반환하지 않고 “아직 위치를 구분하지 못함”이라고 명시한다.

> 이 진단의 결론은 “model이 실패했다”가 아니라, **loss 감소와 위치 localization은 다른 검증 항목이며 둘을 따로 통과시켜야 한다**는 것이다. 이제 다음 개선 실험은 이 사실을 기준으로 설계한다.


---

## 17. 실제 실시간 상황에서도 후보를 안전하게 멈추게 하려면

앞의 진단에서는 XCT support 안의 prediction이 모두 같다는 사실을 확인할 수 있었다. 하지만 실제 제조 중에는 XCT 검사 결과가 아직 없으므로, 실시간 model은 “지금 이 pixel이 XCT support에 속하는가”를 알 수 없다. 따라서 실제 candidate decoder가 support mask를 보고 후보를 막으면 안 된다. 그것은 미래 검사 정보를 미리 본 것이기 때문이다.

그래서 decoder는 prediction map 자체만 보고 세 가지 안전 검사를 한다.

| 순서 | decoder가 보는 것 | 멈추는 조건 | 반환 상태 |
|---|---|---|---|
| 1 | 화면 전체의 max-min | 값 차이가 `1e-6` 이하 | `withheld_spatial_plateau` |
| 2 | 최고 score와 사실상 같은 pixel의 면적 | 전체 256×256 pixel의 0.1%를 초과 | `withheld_top_score_plateau` |
| 3 | 이전 endpoint map과 현재 map의 차이 | MAE와 최대 차이가 모두 `1e-6` 이하 | `withheld_temporally_invariant_map` |
| 4 | 위 세 조건을 통과 | finite local maximum 존재 | `emitted` 후 top-k candidate 출력 |

두 번째 규칙은 “가장 높은 점수가 한 점이 아니라 넓은 바닥처럼 퍼져 있으면 어느 좌표가 대표인지 정할 수 없다”는 뜻이다. 세 번째 규칙은 layer가 바뀌어도 model map이 숫자까지 완전히 똑같이 반복되면, 그 map이 현재 input의 변화를 보고 만든 결과인지 의심해야 한다는 뜻이다.

이 규칙들은 **결함을 판정하는 기준이 아니다.** model이 위치를 구분하지 못하는데도 임의 좌표를 내는 일을 막는 품질 안전장치다. 다음 실행에서 이 세 수치를 실제 checkpoint에 적용해 보고, candidate가 올바르게 보류되는지 확인한다.


---

## 18. 안전장치 검증 결과와 다음 한 가지 실험

새 안전장치를 첫 checkpoint에 적용해 보니, z=203·z=227·z=250에서 최고 점수와 사실상 같은 pixel이 각각 63,504개였다. 이는 256×256 화면의 약 96.9%다. 즉 “가장 높은 곳”이 한 점이나 작은 영역이 아니라 화면 대부분을 덮고 있었기 때문에, 좌표를 고르는 것이 의미 없었다.

또한 z=203에서 z=227로, z=227에서 z=250으로 바꿔도 prediction map의 차이는 평균도 최대값도 정확히 0이었다. model은 서로 다른 시간의 A history를 받았지만 같은 그림을 반복하고 있었다.

| 새 규칙 | run 1 결과 | 의미 |
|---|---|---|
| 최고 score 동점 면적 검사 | 96.9%로 허용 0.1%를 크게 초과 | 하나의 대표 위치를 정할 수 없다. |
| endpoint map 변화 검사 | MAE=0, max difference=0 | model이 현재 입력 변화에 반응하지 않았다. |
| candidate 반환 규칙 | 세 layer 모두 `withheld_top_score_plateau`, 후보 0개 | 임의 `(x,y)`가 더 이상 출력되지 않는다. |

따라서 이번 단계는 성공이다. model의 위치 예측이 아직 좋다는 뜻이 아니라, **모델이 위치를 구분하지 못할 때 그 사실을 솔직하고 안전하게 표시하도록 만들었다**는 뜻이다.

### 다음에는 무엇을 하나만 바꿀까?

첫 개선 실험에서는 model 구조, XCT target, Gaussian rasterization, optimizer를 모두 그대로 둔다. 바꾸는 것은 학습 횟수뿐이다. 8 epoch에서 validation loss가 마지막 epoch까지 낮아지고 있었으므로, 먼저 24 epoch까지 충분히 학습시켜 보는 것이 가장 공정한 확인 방법이다.

| 고정하는 것 | 바꾸는 것 | 확인할 질문 |
|---|---|---|
| 데이터 split, 6채널 A input, weak target, masked loss, model 크기, learning rate | epoch `8 → 24` | 8 epoch가 단순히 부족했는가? |

24 epoch 후에도 map이 똑같이 반복되면, 다음 한 가지 실험에서 model capacity를 늘릴 근거가 생긴다. 반대로 spatial variation이 생기면 held-out loss와 candidate safety status를 함께 비교한다. 이렇게 한 번에 하나씩만 바꾸면 무엇이 효과를 냈는지 이해할 수 있다.


---

## 19. 두 번째 학습은 왜 24 epoch만 바꾸는가

첫 학습에서 model map이 똑같이 반복됐다고 해서 바로 model을 크게 바꾸거나 XCT target을 다시 만들면, 무엇이 문제였는지 알 수 없어진다. 아직 8 epoch만 학습했으므로, 단순히 학습 시간이 부족했을 가능성부터 먼저 확인하는 것이 공정하다.

그래서 두 번째 run에서는 첫 번째와 거의 모든 것을 똑같이 둔다. A 영상, 4개 layer history, 6개 input channel, train/validation/test split, XCT weak target, masked loss, model 크기, optimizer, random seed, candidate safety rule은 그대로다. 오직 model이 data를 반복해서 보는 횟수만 8에서 24로 늘린다.

| 그대로 두는 것 | 한 가지만 바꾸는 것 | 실행 후 볼 질문 |
|---|---|---|
| 데이터와 target | 학습 epoch: 8→24 | 더 오래 학습하면 model이 location 차이를 배우는가? |
| model 구조와 optimizer | output directory | 이전 run을 보존한 채 정직하게 비교할 수 있는가? |
| candidate safety gate | 없음 | 변화가 생겨도 허위 좌표는 계속 막히는가? |

두 번째 run의 loss가 더 낮아져도 아직 충분하지 않다. 최고 점수 동점 영역이 run 1의 96.9%보다 줄었는지, 서로 다른 layer의 map이 더 이상 완전히 같지 않은지, 그 뒤 decoder가 candidate를 보류하는지 또는 허용하는지를 모두 함께 본다.

> 이 실험의 목적은 좋은 결과를 빨리 만드는 것이 아니라, **8 epoch가 부족했는지 아닌지를 하나의 변수만 바꿔 확인하는 것**이다. 24 epoch 뒤에도 map이 같다면 다음에는 학습 시간 대신 model capacity 한 가지만 바꾼다.


---

## 20. 학습은 끝났는데 왜 결과 파일이 없고, 다시 학습하지 않는가

24 epoch run은 실제로 24번의 학습을 모두 마쳤다. validation에서 가장 좋았던 checkpoint도 epoch 9에 저장되어 있다. 다만 마지막 단계인 test 평가 중에 code 한 줄에서 “현재 test sample”을 가져오는 부분이 빠져 `NameError`가 발생했다. 이 오류는 이미 끝난 학습을 되돌리거나 TIFF/XCT data를 바꾸지 않는다.

| 구분 | 보존된 것 | 아직 없는 것 |
|---|---|---|
| 학습 | 24 epoch가 끝났고 best checkpoint가 있음 | 없음 |
| validation | best epoch=9, loss=0.06152481 기록 | 없음 |
| held-out test | 원본 test data는 그대로 있음 | test loss와 candidate JSON이 아직 생성되지 않음 |

이때 24 epoch를 다시 돌리는 것은 좋은 방법이 아니다. 이미 학습된 model을 다시 학습시키면 시간만 더 들고, controlled experiment에서 필요했던 checkpoint는 이미 존재하기 때문이다. 그래서 **checkpoint-only evaluator**를 별도로 사용한다.

이 evaluator는 저장된 checkpoint를 읽고 test data를 한 번 통과시킨 뒤, 아직 없던 두 결과 파일만 만든다. optimizer update, backward, 새로운 random initialization, checkpoint overwrite는 전혀 없다. 즉 “학습을 다시 하는 것”이 아니라 “이미 학습한 model의 시험지를 채점하는 것”에 가깝다.

> 이 과정이 끝나면 8 epoch와 24 epoch 중 어느 쪽이 held-out test에서 더 나은지, 그리고 24 epoch가 반복 map 문제를 해결했는지를 공정하게 비교할 수 있다.


---

## 21. 평가 코드의 설정 이름 오류는 왜 학습을 망치지 않았는가

checkpoint-only evaluator를 처음 실행했을 때, loss 설정에서 `loss.smooth_l1_beta`라는 이름을 찾으려 했지만 실제 설정에는 `objective.beta`가 있었다. 즉 evaluator가 설정표의 칸 이름을 잘못 읽은 오류였다.

이 오류는 test data를 읽거나 checkpoint를 수정하기 **전**에 loss 계산기를 준비하는 단계에서 멈췄다. 그래서 이미 저장된 e24 checkpoint와 TIFF/XCT data는 전혀 바뀌지 않았다. e24 output folder에도 checkpoint 하나만 남아 있고, 아직 test result JSON은 없었다.

| 확인한 것 | 결과 | 왜 중요한가 |
|---|---|---|
| 실제 loss 설정 | `objective.beta = 0.1` | training 때 쓴 Smooth L1 설정과 같아야 공정하게 비교 가능 |
| 오류 시점 | evaluation 시작 전 | 원본 data와 checkpoint가 안전함 |
| recovery 방식 | key 이름만 맞추고 checkpoint를 다시 읽음 | 24 epoch 학습을 다시 할 필요가 없음 |
| output 보호 | test JSON이 이미 있으면 evaluator가 쓰기를 거부 | 결과를 실수로 덮어쓰지 않음 |

이제 evaluator는 training과 같은 beta=0.1 loss를 사용해 e24 checkpoint의 held-out test 성능과 candidate safety status만 계산한다. 이것도 “학습”이 아니라 저장된 모델의 시험지를 다시 채점하는 과정이다.


---

## 22. 24 epoch까지 학습했는데도 왜 다음에는 모델 크기를 바꾸는가

E24 evaluator와 spatial diagnostic까지 끝내자, 24 epoch를 허용했어도 가장 좋은 validation checkpoint는 epoch 9였다는 사실이 확인됐다. held-out test loss도 8 epoch run보다 아주 조금 높았다. 더 중요한 것은 map의 모양이었다. 최고 점수 동점 영역은 여전히 96.9%였고, 서로 다른 test layer의 map은 완전히 같았다.

| 비교 항목 | 8 epoch run | 24 epoch run의 best checkpoint | 의미 |
|---|---:|---:|---|
| Best validation loss | 0.06163192 | 0.06152481 | 아주 작게 낮아졌지만 결정적 개선은 아님 |
| Held-out test loss | 0.07344962 | 0.07370871 | 24 epoch 쪽이 0.35% 높음 |
| 최고 점수 동점 면적 | 96.9% | 96.9% | 위치 구분 문제가 그대로임 |
| 서로 다른 layer map 차이 | 0 | 0 | A input 변화에 반응하지 않음 |
| 안전장치 | 보류 필요 | 48개 모두 후보 0개로 보류 | 임의 좌표를 내지 않음 |

따라서 “학습 시간이 부족했다”는 설명은 설득력이 약해졌다. 다음에는 학습 횟수를 다시 8로 고정하고, 모델이 image pattern을 표현할 수 있는 channel 수만 8에서 32로 늘린다.

이것은 모델을 무작정 복잡하게 만드는 것이 아니다. 작은 모델이 충분한 표현 공간을 갖지 못해 평균적인 값을 반복했는지 확인하는 한 가지 실험이다. A input, XCT target, loss, data split, optimizer, random seed는 그대로 두므로, 결과가 달라지면 model capacity가 원인이었을 가능성을 더 강하게 말할 수 있다.


---

## 23. 다음 실험 C32는 무엇을 바꾸고 무엇을 그대로 두는가

C32는 `base_channels`만 8에서 32로 바꾸는 실험이다. Channel은 model이 image에서 동시에 보관하고 조합할 수 있는 feature의 통로 수라고 생각하면 된다. 작은 model이 밝기, texture, saturation validity, 최근 layer 변화처럼 여러 단서를 충분히 나누지 못했다면, 답을 평균값 하나로 단순화할 수 있다.

| 항목 | Run 1 | C32 run | 이유 |
|---|---:|---:|---|
| Model feature channels | 8 | 32 | **유일하게 바꾸는 변수** |
| Training epoch | 8 | 8 | 오래 학습한 효과를 섞지 않음 |
| Input | A 영상 4 layer × 6 channel | 동일 | 새 정보를 추가하지 않음 |
| XCT weak target와 masked loss | 동일 | 동일 | label 의미를 바꾸지 않음 |
| Optimizer, seed, split | 동일 | 동일 | 우연한 학습 조건 차이를 줄임 |
| Safety decoder | 동일 | 동일 | 좌표를 허용하는 기준을 바꾸지 않음 |

C32에서 보는 질문은 “loss가 조금 더 낮아졌는가”만이 아니다. 더 중요한 질문은 **다른 A layer history를 입력했을 때 map 모양이 달라졌는가**, 그리고 **최고 score가 화면 대부분에 동점으로 퍼지는 현상이 줄었는가**다. 두 질문의 답이 모두 아니라면, 다음에는 model 크기보다 target rasterization 또는 A input–XCT target 연결 방식을 조사해야 한다.

> C32가 candidate를 출력해도 그 좌표는 여전히 XCT-derived continuous quality candidate다. XCT response의 high/low 방향이 실제 결함과 어떻게 연결되는지는 별도 검증 전까지 확정하지 않는다.


---

## 24. C32 결과: 모델을 크게 했지만 왜 아직 위치 후보를 내지 않는가

C32는 model의 feature channel만 8에서 32로 늘린 실험이었다. 결과는 한 가지 면에서는 바뀌었고, 가장 중요한 면에서는 바뀌지 않았다.

| 관찰 | Run 1: C8 | C32 | 뜻 |
|---|---:|---:|---|
| 최고 score 동점 비율 | 96.8994% | 0.3845% | C32는 화면 전체의 고정 모양을 조금 더 세밀하게 만들 수 있었음 |
| support 내부 prediction 변화 | 0 | 0 | 학습해야 할 sparse XCT 위치들 사이의 차이는 여전히 못 배움 |
| 서로 다른 test layer map 차이 | 0 | 0 | 서로 다른 A 영상도 model은 동일한 map으로 처리함 |
| held-out test loss | 0.07344962 | 0.07363038 | 더 큰 model이 test 성능을 개선하지 못함 |
| candidate | 보류 필요 | 48개 endpoint 모두 보류 | 안전장치가 임의 좌표를 막음 |

여기서 알 수 있는 것은 “model이 작아서 문제”라는 설명만으로는 충분하지 않다는 점이다. Model은 더 복잡한 **고정된 지도**를 만들었지만, 새로 들어온 A 영상의 차이를 반영한 지도는 만들지 못했다.

다음에는 model을 또 크게 바꾸지 않는다. 대신 sparse XCT 점 주변을 target map으로 바꾸는 Gaussian 폭 `sigma`를 2에서 3으로만 바꾼 audit을 한다. 폭이 커지면 loss를 받는 support pixel 수가 늘어 model이 위치별 학습 신호를 더 많이 받을 수 있다. 하지만 너무 커지면 서로 다른 점의 정보가 퍼져 흐려질 수 있다.

> 먼저 sigma=3 audit으로 support가 얼마나 늘고 얼마나 퍼지는지 확인한다. 원본 CSV, A 입력, calibration, response 방향은 그대로 두며, binary defect label도 만들지 않는다. 충분히 안전하다고 판단된 경우에만 sigma=3으로 학습하는 다음 controlled run을 준비한다.


---

## 25. 다음 검증 코드: sigma=2와 sigma=3은 무엇을 비교하는가

새 파일 `src/audit_weak_target_support_density.py`는 학습을 하지 않는 검증 코드다. A/B TIFF를 열지 않고, registered XCT CSV의 좌표와 값만 사용해 현재 학습 때와 같은 weak target을 RAM에서 다시 만든다. 이때 sigma만 2와 3으로 달리해 두 결과를 비교한다.

| 구분 | 항상 고정하는 것 | 비교하는 것 |
|---|---|---|
| 좌표 | 현재 provisional calibration, `(0,-6)` pixel offset, working ROI, 256×256 grid, pixel rounding | 없음 |
| 값 | train-only p01/p99 response scaling, `[0,1]` clipping, unresolved response direction | 없음 |
| 라벨 의미 | support 밖은 `unknown`, binary defect label 미생성 | 없음 |
| Gaussian support | weighted-average blend 방식 | kernel sigma=2 vs sigma=3 |

이 코드는 layer마다 support pixel 수와 비율, response가 0보다 큰 pixel 수, support가 이어진 덩어리(component)의 개수·최대 비율을 기록한다. 그리고 sigma=3이 sigma=2보다 새 known pixel을 얼마나 늘렸는지, 기존 known pixel이 보존됐는지, 두 response가 공통 support에서 얼마나 달라졌는지, 작은 support island가 지나치게 합쳐졌는지를 수치로 비교한다.

> 기대하는 결과는 “support는 충분히 늘지만, 기존 response의 의미와 위치 구분은 크게 흐려지지 않는다”이다. 이 결과는 sigma=3 training을 검토할 근거일 뿐, defect를 확정하거나 A-only model이 localization을 성공했다는 뜻은 아니다.

실행 뒤에는 `processed/weak_target_support_density_v1/`에 CSV 두 개와 summary JSON 한 개만 생긴다. dense heatmap, model checkpoint, TIFF crop은 만들지 않으며 원본 TIFF/CSV도 그대로다.


---

## 26. sigma=3은 더 많은 학습 지점을 만들었지만, 왜 바로 학습하지 않는가

Audit 결과 sigma=3은 좋은 점과 보류해야 할 점을 동시에 보였다.

| 확인한 항목 | 결과 | 뜻 |
|---|---:|---|
| XCT가 있는 available layer 수 | 8개 | train·validation·test layer를 함께 확인함 |
| z=4의 support | sigma=2/3 모두 0 | 초기 layer의 `unknown`을 정상값으로 바꾸지 않음 |
| median support 증가 | 29.3990% | model이 loss로 볼 수 있는 pixel 수는 늘어남 |
| 기존 support 보존 | 100% | sigma=2에서 알고 있던 위치가 사라지지는 않음 |
| 공통 support response MAE | 0.0538972 | 미리 둔 안정성 기준 0.05000보다 7.79% 큼 |
| binary support component 변화 | 1개→1개 | 이 데이터에서는 component 수만으로 국소적인 섞임을 잘 구분하지 못함 |

쉽게 말하면, Gaussian을 넓히면 각 XCT 점이 주변에 더 넓게 영향을 준다. 그래서 model이 배울 known pixel은 늘어난다. 하지만 가까운 점들이 서로 영향을 더 많이 주면서, **원래 이미 알고 있던 위치의 continuous target 값도 바뀌었다.** 이 변화가 기준보다 컸기 때문에 “support가 늘었으니 sigma=3으로 학습하자”라고 바로 말할 수 없다.

> 결론: `weak_target_v1.yaml`의 production sigma=2를 그대로 유지하고, sigma=3 training은 시작하지 않는다. 이 판단은 sigma=3이 물리적으로 틀렸다는 뜻이 아니라, 현재 evidence만으로는 target 의미를 충분히 보존했다고 말하기 어렵다는 뜻이다.

다음에는 target이나 model 크기를 또 바꾸지 않는다. 이미 저장된 A-only checkpoint를 읽기만 하면서, 서로 다른 A 영상을 넣었을 때 **frame encoder**, **temporal mixer**, **output logit**, **최종 score map** 중 어느 단계부터 결과가 같아지는지 확인한다. 이 검증이 완료되어야 static map collapse가 target 자체의 문제인지, model이 input 정보를 버리는 문제인지 더 정확하게 나눌 수 있다.


---

## 27. 이제는 target이 아니라 model 안에서 A 영상 차이가 어디서 사라지는지 확인한다

지금까지는 “XCT target을 조금 더 넓게 만들면 model이 더 잘 배울까?”를 검사했다. 결과는 보류였다. 이제는 target을 더 건드리지 않고, 이미 학습된 C32 model이 서로 다른 A 영상을 받을 때 내부에서 어떤 일이 생기는지 확인한다.

새 `diagnose_a_only_input_sensitivity.py`는 C32 checkpoint와 A TIFF만 읽는다. XCT CSV, weak target, support mask는 아예 사용하지 않는다. 따라서 이 검증은 “label이 좋았는가”가 아니라 “model이 input을 듣고 있는가”를 묻는다.

| 차이를 확인하는 위치 | 쉽게 말하면 | 결과가 같다면 뜻하는 후보 |
|---|---|---|
| `input_history` | 실제로 넣은 4장의 A 영상과 mask | Dataset 선택 또는 input이 같은지 먼저 확인 |
| `encoded_final_history_frame` | 마지막 A 영상을 encoder가 읽은 특징 | frame encoder가 image 차이를 놓칠 수 있음 |
| `encoded_history` | 4개 history 전체의 encoder 특징 | encoder가 temporal input을 충분히 구분하지 못할 수 있음 |
| `temporal_final` | 과거 4개 layer를 섞은 뒤 특징 | temporal aggregation 또는 normalization이 차이를 줄일 수 있음 |
| `logits` | sigmoid를 통과하기 전 score map | decoder가 feature 차이를 버릴 수 있음 |
| `score` | 최종 0–1 score map | sigmoid saturation 때문에 작은 logit 차이가 사라질 수 있음 |

이 검사는 z=203, 227, 250의 모든 pair에 대해 두 tensor의 MAE·최대 차이·RMSE를 출력한다. `max_abs > 1e-6`이면 적어도 숫자상으로는 차이가 남아 있다고 표시한다. 만약 input과 encoder feature는 다른데 temporal feature부터 같아지면, 다음 개선은 target 확대가 아니라 temporal 부분을 바꾸는 방향이 된다.

> 중요한 점은 “차이가 있다”가 곧 “위치 예측이 성공했다”는 뜻은 아니라는 점이다. 이 검사는 실패의 위치를 좁혀 다음 한 가지 개선 실험을 공정하게 고르기 위한 단계다.


---

## 28. C32 model은 A 영상을 읽었지만, 시간 결합 단계에서 차이를 잃었다

새 diagnostic은 세 test 시점 z=203, 227, 250의 A history를 서로 비교했다. 결과는 매우 명확하다.

| model 내부 위치 | 세 A history 사이의 차이 | 뜻 |
|---|---|---|
| Input history | MAE 약 0.051–0.054, 최대 차이 1.0 | model에 들어간 A 영상은 실제로 다름 |
| Frame encoder feature | MAE 약 0.064–0.068, 최대 차이 약 1.93–2.51 | encoder는 각 A 영상의 차이를 읽어 feature로 보존함 |
| Temporal final feature | MAE=0, 최대 차이=0 | 시간 정보를 섞은 뒤에는 세 결과가 완전히 같아짐 |
| Logit map과 score map | MAE=0, 최대 차이=0 | temporal 단계의 동일한 결과가 최종 출력까지 이어짐 |

즉, 문제는 “A 이미지가 다 똑같다”도 아니고 “encoder가 영상을 못 읽는다”도 아니다. 이 C32 checkpoint에서는 `Conv3D → GroupNorm → SiLU`로 구성된 **temporal aggregation 단계가 input별 feature 차이를 완전히 없애고 있다.**

이 결과가 믿을 만한 이유는 diagnostic이 내부 경로로 다시 계산한 score map과 model의 일반 `forward()` score map이 정확히 같았기 때문이다. 차이는 0이었다. 따라서 내부 측정 경로가 별도의 잘못된 model을 본 것은 아니다.

다음 controlled improvement의 후보는 endpoint frame feature를 decoder로 바로 전달하는 residual bypass다.

```text
기존: temporal update ──> decoder
개선 후보: endpoint encoder feature + temporal update ──> decoder
```

이 bypass는 현재 layer의 A feature를 직접 보존한다. 동시에 temporal update는 과거 3개 layer 정보를 계속 담는다. 현재 layer와 과거 layer만 쓰므로 미래 layer를 보는 leakage는 생기지 않는다.

> 아직 model은 바꾸지 않는다. 먼저 이 변경의 범위, 기존 baseline 보존 방법, separate config/output path를 승인받은 뒤에만 dry-run과 controlled training을 준비한다.


---

## 29. temporal residual bypass는 무엇을 바꾸고, 무엇을 그대로 두는가

이제 새 model option `use_endpoint_feature_residual`을 만들었다. 기본값은 `false`다. 그래서 이전 C8/E24/C32 checkpoint를 읽을 때는 예전 model과 같은 길로 계산되며, 기존 결과는 바뀌지 않는다.

새 residual experiment config에서만 option을 `true`로 켠다. 그때 model 내부는 다음처럼 작동한다.

```text
현재 endpoint A frame ─> frame encoder ────────────────┐
                                                        + ─> decoder ─> score map
최근 4개 A history ─> past-only Conv3D/GroupNorm/SiLU ──┘
```

| 유지하는 것 | 바꾸는 한 가지 |
|---|---|
| A TIFF, K=4 causal history, 6 input channel, normalization, XCT weak target, sigma=2, masked loss, optimizer, seed, epoch=8, base channel=32, safety decoder | endpoint encoder feature를 temporal update에 더하는 residual bypass |

이것은 현재 A frame의 visual feature가 temporal block에서 완전히 사라져도 decoder까지 갈 수 있는 경로를 준다. 동시에 past-only temporal branch는 과거 layer 문맥을 계속 제공한다. 따라서 model이 “현재 A 이미지와 시간 문맥을 모두 이용할 수 있는가”를 검사하는 공정한 실험이 된다.

다음 dry run은 아직 학습이 아니다. model을 RAM에서 한 번 만들고 z=128 sample 하나가 input→target→masked loss까지 error 없이 흐르는지 확인한다. checkpoint, output directory, dense heatmap은 만들지 않는다. dry run이 통과한 뒤에만 8-epoch controlled training을 시작할지 별도로 결정한다.


---

## 30. Residual model의 첫 연결 검사는 통과했다

Residual dry run은 z=128 sample 하나를 사용해 model의 모든 연결을 확인한 검사다. 학습은 하지 않았다.

| 확인 항목 | 실제 결과 | 의미 |
|---|---:|---|
| 실행 device | MPS | 현재 Mac 환경에서 residual model을 실행할 수 있음 |
| Input shape | `[1,4,6,256,256]` | 기존 A-only causal input contract 유지 |
| Prediction shape | `[1,1,256,256]` | 출력이 continuous 2D response map으로 유지 |
| XCT weak target | available=true, support=3,439 | sigma=2와 unknown policy가 그대로 연결됨 |
| Masked loss | 0.18406811, finite | loss가 support 위치에서 정상 계산됨 |
| 저장 파일 | 없음 | checkpoint/heatmap/output folder 없이 RAM 검사만 수행 |

처음 만들어진 random residual model의 score 범위는 약 0.125–0.683이었다. 이 숫자나 loss를 기존 C32의 학습 완료 score/loss와 비교하면 안 된다. 학습을 전혀 하지 않은 model의 한 sample 결과이기 때문이다.

> 지금 확인된 것은 “residual model이 data를 안전하게 받아 학습 준비가 됐다”는 점이다. 아직 “residual model이 더 좋은 localization을 한다”는 것은 아니다. 다음 8-epoch controlled training에서만 그 질문을 확인할 수 있다.


---

## 31. Residual model은 같은 A 영상 조건에서 더 나은 held-out score와 위치별 변화를 보였다

Residual model을 8 epoch 학습한 결과, best validation checkpoint는 epoch 6에서 선택됐다. C32 temporal-only model과 비교하면 같은 test split에서 regression loss가 낮아졌고, 무엇보다 같은 모양을 반복하던 score map 문제가 사라졌다.

| 항목 | C32 temporal-only | C32 temporal-residual | 무엇이 달라졌나 |
|---|---:|---:|---|
| Validation loss | 0.06151949 | 0.05642983 | 8.27% 낮아짐 |
| Held-out test loss | 0.07363038 | 0.06992126 | 5.04% 낮아짐 |
| Support 내부 prediction std | 0 | 약 0.041–0.045 | sampled location별 score 차이가 생김 |
| z203·227·250 map 차이 | 정확히 0 | MAE 약 0.009–0.012 | A history에 따라 score map이 달라짐 |
| Top-score 동점 | 0.3845% | 1 pixel=0.001526% | plateau safety gate를 통과 |
| Candidate decoder | 48/48 withheld | 48/48 emitted, 총 240개 | endpoint당 compact top-5를 만들 수 있음 |

또한 model 내부를 다시 확인했을 때 input → frame encoder → temporal final → logit → score의 모든 단계에서 세 test history의 차이가 남아 있었다. 특히 C32에서 0이었던 temporal final의 차이가 residual model에서는 MAE 약 0.048–0.051, 최대 차이 약 1.58–1.80으로 측정됐다. 이는 residual bypass가 current A frame의 feature를 temporal block 뒤까지 보존했다는 직접적인 근거다.

> 지금까지 통과한 것은 **internal technical validation**이다. 즉 model이 A input에 반응하고, sparse XCT-supported response와의 held-out loss가 개선되며, 안전 gate가 통과한 compact candidate를 내보낼 수 있음을 확인했다.

하지만 아직 candidate를 “확정 결함”이라고 할 수는 없다. 현재 calibration은 provisional이어서 pixel 좌표가 machine/part 좌표로 얼마나 정확히 바뀌는지 점검해야 하고, XCT response가 높을수록 또는 낮을수록 나쁜 quality인지도 정해지지 않았다. 다음에는 score 자체를 다시 바꾸지 않고 emitted candidate의 좌표가 calibration FOV 안에 있는지, camera→machine→camera round-trip에서 얼마나 오차가 나는지 검사한다.


---

## 32. 후보 점수와 좌표가 나왔어도 좌표 변환 검사가 먼저다

Residual model은 `(x_pixel, y_pixel, layer_z, score)` 후보를 내보낼 수 있게 됐다. 이때 `x_pixel`, `y_pixel`은 우선 **layer camera의 raw pixel center**이고, 그 원래 위치는 256×256 model grid의 index다. 제조 장비에서 쓸 machine/part 좌표로 읽으려면 calibration을 역으로 적용해야 한다.

새 `audit_a_only_candidate_coordinates.py`는 이미 만든 compact candidate 240개만 읽어 다음 경로를 계산한다.

```text
256×256 model-grid index
        ↓  ROI center convention
raw layer-camera pixel
        ↓  configured offset을 제거하고 homography 역변환
machine / part coordinate
        ↓  homography 정방향 변환과 offset 재적용
raw layer-camera pixel
        ↓  ROI-grid 역변환
원래 model-grid index
```

| 검사 | 확인하는 내용 | 통과가 뜻하는 것 | 통과해도 뜻하지 않는 것 |
|---|---|---|---|
| Grid·ROI·sensor bounds | 후보가 256 grid, working ROI, 2000×2000 camera 안에 있는지 | 좌표 domain이 맞음 | 실제 defect 여부 |
| Round trip | camera→machine→camera와 grid→camera→grid의 수치 차이 | 현재 코드·offset·homography가 서로 일관됨 | 외부 metrology accuracy |
| Part containment | 역투영 좌표가 configured four-part rectangle에 속하는지 | provisional geometry 안에서 part domain이 맞음 | part edge의 진짜 위치 |
| Edge margin | 후보가 model boundary에서 최소 3 pixel 떨어졌는지 | NMS padding 영향 warning 확인 | candidate score의 진짜 품질 |
| Duplicate | 같은 endpoint에서 같은 model cell을 반복했는지 | top-5의 기본 좌표 다양성 확인 | 시간에 따른 defect persistence |

이 audit은 TIFF, XCT CSV, target, model, checkpoint를 전혀 읽지 않는다. 새로운 dense heatmap도 만들지 않고, 작은 CSV/JSON 숫자만 남긴다.

> 특히 “round trip이 통과했다”는 말은 같은 provisional calibration 식을 정방향·역방향으로 적용했을 때 숫자가 되돌아온다는 뜻이다. 현재 calibration에는 fit RMSE 약 5.2px, LOO RMSE 약 7.0px가 있으므로, 이 검사만으로 절대 machine coordinate 정확도가 확정되지는 않는다.


---

## 33. 좌표 계산은 맞았지만, 현재 part 영역에서는 후보를 보류해야 한다

Candidate coordinate audit의 결과는 두 가지를 분리해서 읽어야 한다.

| 결과 | 값 | 의미 |
|---|---:|---|
| Grid/ROI/sensor domain | 240/240 통과 | 후보 pixel이 camera와 model grid의 유효 범위 안에 있음 |
| Score contract | 240/240 finite, 0–1 | continuous score 형식이 정상 |
| Camera round trip | 최대 약 `4.69e-13` pixel | 현재 homography/offset 식을 정·역으로 적용하면 수치상 되돌아옴 |
| Grid round trip | 최대 0 pixel | model grid center convention이 decoder와 일치 |
| Edge margin | 모두 3 pixel 이상 | model boundary warning 없음 |
| 같은 endpoint 좌표 중복 | 0 | endpoint별 top-5가 같은 cell을 반복하지 않음 |
| Inverse machine coordinate가 known part rectangle 안에 있음 | **0/240** | physical machine/part coordinate로 해석하면 안 됨 |

즉 “계산식은 서로 맞는다”와 “후보가 실제 part 위에 있다”는 완전히 다른 질문이다. 전자는 통과했지만 후자는 현재 provisional calibration과 configured part geometry 아래에서 전부 실패했다.

가능한 설명은 아직 셋 중 하나 이상으로 열어 둔다. model이 part 밖의 visual pattern에 높은 score를 주었을 수 있고, provisional calibration의 절대 위치가 충분히 정확하지 않을 수 있으며, screen-derived part rectangle과 working ROI의 convention에 차이가 있을 수도 있다. 지금 데이터만으로 하나를 원인으로 단정하지 않는다.

> 따라서 현 240개는 machine coordinate로 operational하게 사용하거나 confirmed defect라고 보고하지 않는다. 안전한 다음 단계는 XCT support가 아니라 existing provisional part geometry만 사용해 part 밖 후보를 explicit hold하는 decoder safety gate다. 이 gate는 score map을 바꾸지 않고, “이 candidate는 current geometry convention에서 안전하게 해석할 수 없다”는 사실을 알려 준다.


---

## 34. Geometry-aware safety gate는 model을 다시 학습시키지 않는 후단 안전장치다

All 240 original top-5 candidates가 current provisional part rectangles 밖으로 역투영됐기 때문에, score map을 다시 학습시키거나 XCT support를 deployment에 넣지 않고 candidate decoding 뒤에 geometry safety gate를 추가했다.

```text
score map
  → flat / top-tie / temporal-invariance 기존 safety checks
  → 7×7 local maxima 전체 탐색
  → provisional part rectangle 안의 maxima만 남김
  → 남은 maxima를 top-5로 정렬
  → 하나도 없으면 explicit hold
```

이 gate는 `enabled: false`가 기본이다. 그러므로 이전 C8, E24, C32, residual config에는 영향을 주지 않는다. 오직 새 geometry-gated evaluation config에서만 켜진다.

| gate 결과 | 의미 | 의미하지 않는 것 |
|---|---|---|
| `emitted` | current provisional geometry 안에 local maximum이 있어 compact candidate가 남음 | physical defect confirmation, absolute coordinate accuracy |
| `withheld_outside_provisional_part_geometry` | current configured part geometry 안에 local maximum이 없어 coordinate를 안전하게 보류 | score map이 flat임, model이 실패했음, calibration이 확실히 틀림 |
| 기존 `withheld_top_score_plateau` 등 | geometry 검사보다 먼저 기존 map quality safety가 동작 | XCT support를 deployment decoder가 사용함 |

여기서 중요한 점은 gate가 score map, training loss, target, checkpoint를 바꾸지 않는다는 것이다. 저장된 residual epoch-6 checkpoint를 다시 읽어 test evaluation만 하고, original residual output은 건드리지 않는 별도 output directory에 compact metric와 candidate JSON만 쓴다. 따라서 geometry filter의 효과를 model quality 변화와 혼동하지 않고 비교할 수 있다.
