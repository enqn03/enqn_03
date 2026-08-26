# AMMT 품질관리 PNG 아카이브

이 문서는 AMMT Overhang Part X4 전처리와 sparse XCT weak-target 연결 과정에서 생성된 **11개 품질관리(QC) 이미지**를 설명한다. 원본 TIFF와 registered XCT CSV는 Git에 포함하지 않으며, 이 디렉터리의 PNG는 각 audit의 판단 근거를 장기적으로 검토할 수 있도록 복사한 경량 시각 기록이다. 각 그림은 특정 검증 가설을 점검한 결과이며, 어떤 그림도 단독으로 결함을 확정하거나 binary label을 제공하지 않는다.

> **공통 해석 원칙:** display-only percentile scaling은 화면 표시를 위한 것이며 학습 normalization과 다르다. B−A map, projected XCT response, Gaussian rasterized map은 모두 확정 defect map이 아니다. support 밖과 early XCT 미지원 layer는 normal=0이 아니라 **unknown**이다.

## 1. 입력·ROI·포화 품질관리

| PNG | 내용과 읽는 방법 | 검증 목적 및 모델 영향 |
|---|---|---|
| [01 AfterSpreading 초기 layer QC](qc_images/01_afterspreading_initial_layers_qc.png) | A(AfterSpreading) LED1의 z=1–20 frame을 4×5 contact sheet로 배치하고, 각 tile에 표시용 p1/p99.8을 표기한다. 후반 tile에서 p99.8이 반복적으로 65535인 것은 full-scale saturation이 존재함을 보여 준다. | 원본 hyperstack을 logical layer 순서로 읽는지, 초기 적층 형상과 sensor ceiling이 어떤지 빠르게 확인한다. 포화 영역은 texture나 anomaly가 아니라 validity mask=0으로 별도 제공해야 함을 뒷받침한다. |
| [02 A/B pair QC](qc_images/02_ab_pair_qc.png) | z=1, 10, 20, 125, 230, 250의 LED1마다 A, B(Burned), display-only `|B-A|` candidate map을 3열로 비교한다. | A/B가 같은 layer·관찰영역에서 대응됨을 확인한다. 다만 `|B-A|`에는 정상 레이저 용융·반사 변화가 포함되므로 clean target이나 defect label이 아니라 stage 변화 QC로만 사용한다. |
| [03 ROI 후보 선별 QC](qc_images/03_roi_candidate_screening_qc.png) | 다섯 raw-pixel ROI 후보를 A/B reference와 A/B LED1 saturation frequency map에 겹치고, 하단에서 후보별 stage·LED saturation을 비교한다. | 외관상 그럴듯한 crop을 임의 선택하지 않고, 공정 영역 보존과 saturation 비용을 함께 비교한다. 이 그림은 descriptive screening이며 final geometric ROI 확정 근거는 아니다. |
| [04 Working ROI saturation QC](qc_images/04_working_roi_saturation_qc.png) | z=125 LED1 A/B 원본 및 saturation frequency map에 현재 working ROI `(250,250)–(1750,1750)`를 red rectangle으로 표시한다. | A/B에 동일한 raw camera ROI가 적용되는지 확인한다. 이 ROI는 현재 Dataset contract의 working ROI이지만 외부 calibration 기반 final geometric ROI는 여전히 provisional이다. |
| [05 Train normalization QC](qc_images/05_train_normalization_qc.png) | train history z=1–157에서만 계산한 A/B×LED1–3 valid-pixel p01/p50/p99와 sampled full-scale saturation을 막대그래프로 요약한다. | LED·stage별 intensity scale과 saturation 차이를 수치로 확인한다. 그래서 global normalization 대신 stage·LED별 p01/p99 robust scaling 및 `raw<65535` validity mask 3채널을 입력에 결합한다. validation/test 통계는 사용하지 않아 leakage를 막는다. |

## 2. Registered XCT·calibration 품질관리

| PNG | 내용과 읽는 방법 | 검증 목적 및 모델 영향 |
|---|---|---|
| [06 Registered XCT sparse target QC](qc_images/06_registered_xct_sparse_target_qc.png) | 2×2 panel은 layer 100의 command-XY sparse support와 `xct_5x5x5` value, part별 finite support 시작 layer, train-only XCT response 분포, layer별 finite point 수를 보여 준다. | registered CSV가 camera heatmap이 아닌 sparse machine-coordinate measurement임을 확인한다. `xct_5x5x5`는 continuous response 후보이며, 초기 finite-value 부재는 loss 제외/unknown을 뜻한다. train p01=0.40070, p99=0.58533은 runtime robust scaling에만 사용되며 defect threshold가 아니다. |
| [07 Geometry candidate QC](qc_images/07_calibration_geometry_candidates_qc.png) | top 4 part/orientation homography 후보의 projected part outlines와 screen controls를 B camera frame에 겹친다. rank 1과 rank 2는 LOO RMSE 7.03 px로 mirror tie이고, 3·4위는 훨씬 큰 residual을 보인다. | control point geometry가 plausible transform을 만들었는지와 mirror ambiguity가 남았는지를 확인한다. geometry residual만으로 방향을 확정하지 않고 별도 photometric 검증으로 넘어가야 한다. |
| [08 Global photometric QC](qc_images/08_calibration_global_photometric_qc.png) | rank 1·2의 registered LWI와 camera raw intensity global consistency를 median absolute Spearman correlation으로 비교하고 비교 수 n=432를 표시한다. | 두 score가 낮고 근접해 global photometric tie-break가 충분하지 않았음을 기록한다. 이 hold 결과가 local patch·offset refinement의 필요성을 정당화한다. |
| [09 Local calibration refinement QC](qc_images/09_calibration_local_refinement_qc.png) | candidate 1·2 각각에 대해 raw global offset `(dx,dy)` grid의 5×5 valid-patch median absolute Spearman score를 heatmap으로 표시한다. candidate 2는 내부 `(0,-6)` px 근처에서 0.54282 peak를 보인다. | candidate 1의 경계 peak 0.20692보다 강하고 안정적인 candidate 2 peak로 mirror tie를 provisional하게 해소한다. 따라서 `mirror_rotate_270`, screen A→D=`part04,part03,part02,part01`, `(0,-6)` correction이 sparse support projection에 사용된다. 독립 metrology calibration 전까지 provisional이다. |

## 3. Camera-space sparse support와 weak target 품질관리

| PNG | 내용과 읽는 방법 | 검증 목적 및 모델 영향 |
|---|---|---|
| [10 Projected XCT support QC](qc_images/10_projected_xct_support_qc.png) | layer 125의 finite `xct_5x5x5` command-XY points를 provisional homography와 correction으로 raw 2000×2000 camera plane에 투영하고 part별 색으로 표시한다. | 네 part support 영역이 camera FOV에서 분리되어 나타나는지 검증한다. 전체 train finite point의 FOV 통과율은 100%, rounded-pixel collision은 2.888%였으며, support가 없는 pixel은 normal이 아니라 unknown으로 유지한다. |
| [11 Weak target rasterization QC](qc_images/11_weak_target_rasterization_qc.png) | layer 125의 projected continuous response와 binary support mask를 model 256×256 grid에서 Gaussian sigma 1–4별로 비교한다. 상단은 Gaussian-weighted response, 하단은 support mask다. | support 밀도와 spatial bleeding의 절충을 확인한다. sigma=2의 support 약 3.62%를 기본으로 채택해 runtime에서 on-the-fly rasterize한다. Dataset의 loss는 `weak_support_mask==1` 위치에만 적용하며, 이 response는 direction-unresolved continuous quality target이다. |

## 4. Git에 포함하는 이유와 보존 정책

`processed/`는 재실행 가능한 수치·시각 산출물이라 기본적으로 Git에서 제외한다. 그러나 이 문서의 PNG 사본은 **모델 학습용 데이터나 dense label이 아니라 판단 근거를 보존하는 QC 아카이브**다. 총 11개, 약 11 MB이며 원본 6 GB TIFF나 1,000개 registered XCT CSV를 대체하지 않는다. 새 audit이 기존 의사결정을 변경하면 원본 `processed/` 결과를 재검토하고 해당 QC 사본·설명을 함께 갱신한다.

## References

[1] [NIST, *Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): Overhang Part X4*](https://data.nist.gov/od/id/mds2-2233)
