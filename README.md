# AMMT Layer-Camera Real-Time Anomaly Candidate Localization

LPBF 적층제조 공정의 layer-camera 시계열을 사용해 **실시간 노이즈 저감**과 **이상 후보 위치 탐지**를 구축하는 연구 프로젝트다. 이 저장소는 의료 영상이 아닌 NIST AMMT Overhang Part X4 제조 공정 데이터를 대상으로 한다.[1]

> 현재 모델 출력은 확정 결함 판정이 아니라 `(x_pixel, y_pixel, layer_z, score)` 형태의 **이상 후보 위치**다.

## 현재 위치

데이터 구조와 A/B 대응, ROI 후보 포화 분석, 인과적 sequence split, train-only stage·LED별 normalization, causal Dataset, registered XCT sparse response audit, provisional machine XY→camera calibration, projected support, weak-target rasterization audit, support-masked loss runtime 검증이 완료됐다. LED 1·2의 넓은 full-scale saturation을 분리하기 위해 A-only baseline input은 3개 normalized intensity channel과 3개 validity-mask channel을 사용한다. `AMMTWeakTargetDataset`은 endpoint layer의 command XY를 provisional calibration으로 투영해 continuous `weak_response`와 `weak_support_mask`를 **on-the-fly** 반환한다. z=4는 `unknown`/loss 제외로, z=128은 3,439 supervised model pixel로 사용자 실행 검증을 통과했다. `AOnlyCausalCandidateNet`과 training/evaluation script는 구현됐고, z=128에서 `[1,4,6,256,256]` input·`[1,1,256,256]` sigmoid output·3,439 supervised pixel·finite masked loss=0.11099906의 `mps` dry-run을 통과했다. run 1 training은 epoch 8까지 완료되어 best validation supported-pixel Smooth L1=0.06163210, held-out test=0.07344962를 기록했다. checkpoint spatial diagnostic은 test z=203/227/250에서 global map range=0.67868543·std=0.11746240이지만 support 내부 prediction std=0.0·Pearson=null인 고정 plateau를 확인했다. 따라서 global `max-min ≤ 1e-6` gate만으로는 candidate를 안전하게 보류할 수 없으며, 현재 `emitted` coordinate는 사용하지 않는다. 확장 diagnostic은 z=203/z=227/z=250에서 top-score tie=63,504 pixel=96.8994%, 선택 endpoint 간 map MAE/max-absolute=0.0을 확인했다. 세 endpoint는 모두 `withheld_top_score_plateau`, candidate count=0으로 보류됐다. 이는 real-time decoder가 XCT support mask 없이 arbitrary coordinate를 막았음을 보여 준다. e24 controlled training의 checkpoint-only held-out evaluation은 완료됐다. Best validation은 epoch 9에서 run 1보다 0.17% 낮았으나, held-out test loss는 0.07370871로 run 1보다 0.35% 높았다. e24도 top-score tie=96.8994%, selected endpoint map MAE/max-absolute=0/0, candidate=0(48/48 withheld)이었다. 따라서 epoch만 늘리는 방법은 fixed map collapse를 해소하지 못했다. C32 controlled capacity run도 완료됐다. C32는 top-score tie를 96.8994%에서 0.3845%로 줄였지만, selected endpoint map MAE/max-absolute=0/0 및 support-region prediction std=0을 유지했다. Held-out test loss=0.07363038은 run 1보다 0.24% 높았다. 따라서 capacity 증대는 static geometry를 바꿨지만 input-dependent localization을 복원하지 못했다. 다음 controlled axis는 target rasterization에서 Gaussian sigma만 2→3으로 바꾸어 sparse support density를 검사하는 read-only audit이다. `src/audit_weak_target_support_density.py`가 구현되어 있으며, 학습 Dataset과 동일한 projection·ROI·rounding·train p01/p99 scaling·Gaussian weighted average로 두 sigma를 memory에서만 비교한다. 출력은 layer별 compact CSV 2개와 summary JSON이고 raw TIFF/CSV·dense target·checkpoint는 변경하지 않는다. response direction은 여전히 unresolved이므로 B−A나 XCT response를 direct defect label로 사용하지 않는다.

| 단계 | 상태 | 핵심 산출물 |
|---|---|---|
| A/B hyperstack 구조 검증 | 완료 | `TZYX=[3,250,2000,2000]`, `uint16` |
| A/B pair 대응 검증 | 완료 | A/B raw 차이를 label로 쓰지 않는 정책 |
| ROI·saturation 분석 | 완료 | 후보 ROI별 포화 비교와 QC |
| 인과적 train/validation/test split | 완료 | `manifests/causal_sequence_manifest.csv` |
| Train-only normalization·validity mask | 완료 | `configs/normalization_v1.yaml` |
| Causal Dataset 연결 | 완료·train/validation/test sample 검증 | 3 intensity + 3 validity-mask channel |
| Registered XCT sparse target audit | 완료·1,000 CSV schema·coverage 검증 | train-only finite response 2,329,476개/target column |
| Machine XY→camera pixel calibration | 완료·provisional | rank2 `mirror_rotate_270`, raw correction `(0,-6)` px; 독립 calibration 전까지 provisional |
| Projected sparse support·rasterization | 완료 | FOV 100%, sigma=2 model px, support 밖=unknown |
| Weak target Dataset 연결 | 완료·available/unknown sample 검증 | `[1,256,256]` response/mask, z=4 loss 제외, z=128 3,439 supervised pixel |
| Support-mask weighted continuous regression loss | 완료·runtime 검증 통과 | z=4 loss/gradient=0, z=128 support-only regression, unknown 영향=0 |
| A-only support-mask weighted baseline | run 1 end-to-end 완료·localization hold | best validation loss=0.06163210, test loss=0.07344962. support-region prediction은 flat하여 current coordinate 무효 |
| Checkpoint spatial diagnostic | 완료 | top-score tie=96.8994%, selected endpoint map MAE/max-absolute=0.0으로 input-invariant map 확인 |
| Support-independent candidate decoder | runtime 통과 | 세 test endpoint에서 `withheld_top_score_plateau`, candidate count=0; XCT support 없이 동작 |
| A-only controlled training-duration run | 완료·negative result | best validation e9=0.06152481, test=0.07370871; duration만 연장해도 plateau 유지 |
| E24 checkpoint-only evaluator | 완료 | saved checkpoint read-only evaluation으로 48/48 withheld·candidate=0을 확인 |
| A-only model-capacity controlled run | 완료·capacity 가설 보류 | tie 0.3845%로 축소됐지만 input-invariant map·support plateau·test +0.24% 유지 |
| Weak target support-density rasterization audit | 구현·사용자 실행 대기 | sigma=2/3의 support coverage·response stability·component merge 지표를 compact CSV/JSON으로 비교 |
| B·fusion heatmap | 확장 단계 | 사후 재평가 및 위치 안정화 |

## 연구 흐름

```text
A/B raw layer-camera hyperstacks
        ↓
structure · correspondence · saturation audits
        ↓
causal K=4 manifest with guarded temporal splits
        ↓
train-only normalization + saturation validity mask
        ↓
causal 6-channel Dataset (3 intensity + 3 mask)
        ↓
A-only candidate heatmap baseline
        ↓
B head and A/B fusion heatmap
        ↓
registered XCT sparse support → screen-corner controls → 192 part/orientation hypothesis residual+overlay audit → weak heatmap → manual review for spatial validation
```

`프로젝트과정.md`의 **“데이터 전처리에 적용한 수학적·기술적 기법”** 절에는 실제 적용된 TIFF 축 변환, causal split, percentile normalization, saturation mask, sparse supervision, DLT homography, residual, orientation hypothesis, photometric correlation의 수식·입력·출력·오류 방지 목적이 정리되어 있다.

## 데이터와 코드

| 경로 | 내용 |
|---|---|
| `raw_original/` | 원본 A/B TIFF, process signal, metadata, registered XCT. Git에서 제외 |
| `processed/` | 재생성 가능한 audit 결과와 QC 이미지. Git에서 제외 |
| `manifests/` | 인과적 sample index와 split policy |
| `src/` | 재현 가능한 분석·전처리 코드 |
| `프로젝트과정.md` | 기술적 의사결정, 현재 상태, 다음 검증 흐름 |
| `docs/quality-control-images.md` | Git으로 보존한 11개 QC PNG의 panel별 의미·판정 한계·모델 영향 |
| `docs/AMMT_프로젝트를_처음부터_이해하기.md` | 초심자 관점의 프로젝트 목적·데이터·검증·학습 흐름 안내서 |

## 핵심 기술 원칙

A는 분말 도포 후·레이저 전 영상이고 B는 레이저 스캔 후 영상이다. A/B는 같은 `(layer_z, LED)`에 대응하지만, B−A에는 정상 용융 변화와 반사 변화도 포함된다. 따라서 A/B 차이를 결함 정답으로 직접 사용하지 않는다.

시계열 입력은 layer 축을 시간으로 사용하며, 각 endpoint는 현재와 과거 layer만 참조한다. 현재 기본 정책은 K=4, train endpoint z=4–157, validation z=161–199, test z=203–250이고 경계에는 3-layer guard band를 둔다.

자세한 데이터 검증 결과, ROI 포화 수치, 모델 설계, split 규칙과 다음 기술 단계는 [프로젝트과정.md](프로젝트과정.md)에 정리되어 있다.

## Reference

[1] [NIST, *Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): Overhang Part X4*](https://data.nist.gov/od/id/mds2-2233)


## 구현 파이프라인 아키텍처와 데이터 흐름

```mermaid
flowchart TD
    A[원본 A/B TIFF\nTZYX: LED×Layer×Y×X] --> B[구조·A/B·ROI·saturation audit]
    B --> C[causal manifest\nK=4 + guard split]
    C --> D[train-only stage·LED normalization\np01/p99]
    D --> E[read-only memmap Dataset]
    E --> F[입력 X\nK×6×H×W\n3 intensity + 3 validity masks]
    X1[Registered XCT sparse CSV\nmachine XY + xct_5x5x5] --> G[geometry + photometric calibration]
    G --> H[sparse point projection\nraw camera → model 256×256]
    H --> I[on-the-fly continuous weak response\n+ unknown support mask]
    F --> J[향후 A-only heatmap baseline]
    I --> J
    J --> K[이상 후보\n(x, y, layer_z, score)]
    K --> L[향후 B head·A/B fusion\n및 수동/XCT 검증]
```

데이터 흐름의 핵심은 **원본 TIFF를 재저장하지 않고** memmap으로 읽어 model input을 구성하며, XCT의 sparse supervision도 dense label file로 저장하지 않고 calibration과 rasterization 규칙으로 필요한 sample 시점에만 생성하는 것이다. 이 방식은 저장 공간을 제한하면서도 raw data provenance와 temporal causality를 보존한다.

## 데이터 전처리에 적용한 수학적·기술적 기법

| 단계 | 적용 기법·수식 | 해결하는 문제 | 현재 결과 |
|---|---|---|---|
| Hyperstack 해석 | ImageJ logical axes `T,Z,Y,X`; `tifffile.memmap` | 물리 TIFF page가 1개인 hyperstack을 일반 multi-page TIFF로 잘못 읽는 오류 | LED=channel, layer=시간축을 정확히 유지 |
| Causal sequence | `H_z=[z-K+1,...,z]`, K=4 | 미래 layer를 보는 시간 누수 | train/val/test causal manifest 생성 |
| Guard split | train `4–157`, guard `158–160`, val `161–199`, guard `200–202`, test `203–250` | validation/test endpoint가 train history에 섞이는 문제 | endpoint와 history의 split 경계 보존 |
| Saturation mask | `m(x)=1[x<65535]` | sensor full-scale을 정상·이상 신호로 오인하는 문제 | LED별 validity mask를 intensity와 함께 제공 |
| Robust normalization | `clip((x-p01)/(p99-p01),0,1)`; p01/p99는 train만 사용 | LED·stage별 밝기 차이와 평가 데이터 누수 | A/B×LED별 p01/p99 config 확정 |
| ROI·resize | working ROI crop 후 intensity=bilinear, mask=nearest resize | 외곽 artifact와 mask interpolation 오류 | model space 256×256 contract 확정 |
| Sparse XCT support | finite `xct_5x5x5` point만 supervision; support 밖은 unknown | 미관측 영역을 normal/negative label로 오인하는 문제 | 2,329,476 train finite point, FOV 100% |
| Homography | `s[u,v,1]^T=H[X,Y,1]^T`, normalized DLT/SVD | machine XY와 raw pixel 좌표를 임의 동일시하는 문제 | geometry RMSE와 orientation 후보를 분리 검증 |
| Calibration validation | leave-one-out RMSE, 192 part/orientation hypothesis, local 5×5 photometric correlation | mirror ambiguity와 control-point overfit | candidate 2 + `(0,-6)` px provisional correction |
| Weak-target rasterization | model-space Gaussian kernel, sigma=2 px; support mask | sparse target이 끊기거나 과도하게 퍼지는 문제 | continuous response+unknown mask policy 확정 |
| Sigma support-density audit | 동일 projection·ROI·response scaling에서 sigma=2/3 in-memory 비교; retention/MAE/component 지표 | loss gradient support가 너무 sparse하거나 support island가 과도하게 병합되는 문제 | 구현·정적 구문/whitespace 검증 완료, 실행 결과 대기 |

## 현재 전처리 진행률

현재 전처리와 첫 baseline 구현 준비는 **약 92% 완료**로 판단한다. 이 수치는 raw input 준비, sparse spatial supervision의 runtime 연결, unknown-safe loss 검증을 함께 포함한 실무적 기준이다. 원본 구조 검증, causal split, normalization, saturation mask, Dataset input, registered XCT audit, provisional calibration, sparse-support projection, rasterization kernel audit, on-the-fly weak target Dataset, available/unknown sample 검증, support-masked regression loss runtime 검증까지 완료됐다.

| 구간 | 상태 | 전처리 비중 |
|---|---|---:|
| 원본 구조·품질 audit | 완료 | 15% |
| 시계열 split·normalization·input Dataset | 완료 | 25% |
| XCT sparse supervision·calibration·support audit | 완료 | 35% |
| weak target을 Dataset output으로 연결·sample 검증 | 완료 | 10% |
| XCT response direction 검증·A-only baseline 연결 | 진행 예정 | 10% |

남은 핵심 과제는 **A-only spatial localization diagnostic·decoder 보완 및 target 의미 검증**이다. `xct_5x5x5` response는 train-only p01/p99로 `[0,1]` robust scaling되지만, 아직 anomaly 방향으로 invert하거나 binary defect label로 변환하지 않는다. `weak_support_mask==1`에서만 Smooth L1 regression을 계산하는 loss는 z=4 unknown·z=128 supported sample에서 runtime 검증을 통과했고, six-channel causal Conv3D A-only baseline에 연결됐다. 다음 gate는 support-region plateau를 잡는 decoder diagnostic과 candidate withholding 검증이다. 이 gate를 통과할 때까지 score/좌표를 물리적 품질 후보로 사용하지 않는다.

## 실행 순서

프로젝트 전체의 코드 실행 순서, 코드별 역할, 필요 입력, 생성 출력, 실행 전후 상태는 [실행가이드.md](실행가이드.md)에 지속적으로 관리한다. 새 코드가 추가될 때마다 이 guide와 `프로젝트과정.md`를 함께 갱신한다.
