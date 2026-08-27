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
