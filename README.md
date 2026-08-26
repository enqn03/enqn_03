# AMMT Layer-Camera Real-Time Anomaly Candidate Localization

LPBF 적층제조 공정의 layer-camera 시계열을 사용해 **실시간 노이즈 저감**과 **이상 후보 위치 탐지**를 구축하는 연구 프로젝트다. 이 저장소는 의료 영상이 아닌 NIST AMMT Overhang Part X4 제조 공정 데이터를 대상으로 한다.[1]

> 현재 모델 출력은 확정 결함 판정이 아니라 `(x_pixel, y_pixel, layer_z, score)` 형태의 **이상 후보 위치**다.

## 현재 위치

데이터 구조와 A/B 대응, ROI 후보 포화 분석, 인과적 sequence split, train-only stage·LED별 normalization, causal Dataset sample 검증은 완료됐다. LED 1·2에는 넓은 full-scale saturation이 남아 있으므로, baseline input은 정규화된 3개 LED intensity channel과 3개 validity-mask channel을 함께 사용한다. registered XCT의 sparse machine-coordinate response audit은 완료됐다. 초기 layer에는 XCT supervision이 없는 구간이 있으므로 이를 negative label이 아니라 unknown으로 처리한다. 다음 단계는 machine XY→camera pixel calibration을 검증한 뒤 support-aware weak heatmap으로 확장하는 것이다. B−A는 직접 label로 사용하지 않는다.

| 단계 | 상태 | 핵심 산출물 |
|---|---|---|
| A/B hyperstack 구조 검증 | 완료 | `TZYX=[3,250,2000,2000]`, `uint16` |
| A/B pair 대응 검증 | 완료 | A/B raw 차이를 label로 쓰지 않는 정책 |
| ROI·saturation 분석 | 완료 | 후보 ROI별 포화 비교와 QC |
| 인과적 train/validation/test split | 완료 | `manifests/causal_sequence_manifest.csv` |
| Train-only normalization·validity mask | 완료 | `configs/normalization_v1.yaml` |
| Causal Dataset 연결 | 완료·train/validation/test sample 검증 | 3 intensity + 3 validity-mask channel |
| Registered XCT sparse target audit | 완료·1,000 CSV schema·coverage 검증 | train-only finite response 2,329,476개/target column |
| Machine XY→camera pixel calibration | 도구 구현 완료·control-point selection 대기 | homography residual·overlay가 support-aware weak heatmap의 전제 조건 |
| A-only heatmap baseline | 이후 단계 | `(x,y,z,score)` 출력 |
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
registered XCT sparse support → manual control points → residual/overlay calibration audit → weak heatmap → manual review for spatial validation
```

## 데이터와 코드

| 경로 | 내용 |
|---|---|
| `raw_original/` | 원본 A/B TIFF, process signal, metadata, registered XCT. Git에서 제외 |
| `processed/` | 재생성 가능한 audit 결과와 QC 이미지. Git에서 제외 |
| `manifests/` | 인과적 sample index와 split policy |
| `src/` | 재현 가능한 분석·전처리 코드 |
| `프로젝트과정.md` | 기술적 의사결정, 현재 상태, 다음 검증 흐름 |

## 핵심 기술 원칙

A는 분말 도포 후·레이저 전 영상이고 B는 레이저 스캔 후 영상이다. A/B는 같은 `(layer_z, LED)`에 대응하지만, B−A에는 정상 용융 변화와 반사 변화도 포함된다. 따라서 A/B 차이를 결함 정답으로 직접 사용하지 않는다.

시계열 입력은 layer 축을 시간으로 사용하며, 각 endpoint는 현재와 과거 layer만 참조한다. 현재 기본 정책은 K=4, train endpoint z=4–157, validation z=161–199, test z=203–250이고 경계에는 3-layer guard band를 둔다.

자세한 데이터 검증 결과, ROI 포화 수치, 모델 설계, split 규칙과 다음 기술 단계는 [프로젝트과정.md](프로젝트과정.md)에 정리되어 있다.

## Reference

[1] [NIST, *Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): Overhang Part X4*](https://data.nist.gov/od/id/mds2-2233)
