# AMMT Layer-Camera Real-Time Anomaly Candidate Localization

LPBF 적층제조 공정의 layer-camera 시계열을 사용해 **실시간 노이즈 저감**과 **이상 후보 위치 탐지**를 구축하는 연구 프로젝트다. 이 저장소는 의료 영상이 아닌 NIST AMMT Overhang Part X4 제조 공정 데이터를 대상으로 한다.[1]

> 현재 모델 출력은 확정 결함 판정이 아니라 `(x_pixel, y_pixel, layer_z, score)` 형태의 **이상 후보 위치**다.

## 현재 위치

데이터 구조와 A/B 대응, ROI 후보 포화 분석, 인과적 sequence split, train-only stage·LED별 normalization, causal Dataset, registered XCT sparse response audit, provisional machine XY→camera calibration, projected support, weak-target rasterization audit, support-masked loss runtime 검증이 완료됐다. LED 1·2의 넓은 full-scale saturation을 분리하기 위해 A-only baseline input은 3개 normalized intensity channel과 3개 validity-mask channel을 사용한다. `AMMTWeakTargetDataset`은 endpoint layer의 command XY를 provisional calibration으로 투영해 continuous `weak_response`와 `weak_support_mask`를 **on-the-fly** 반환한다. z=4는 `unknown`/loss 제외로, z=128은 3,439 supervised model pixel로 사용자 실행 검증을 통과했다. `AOnlyCausalCandidateNet`과 training/evaluation script는 구현됐고, z=128에서 `[1,4,6,256,256]` input·`[1,1,256,256]` sigmoid output·3,439 supervised pixel·finite masked loss=0.11099906의 `mps` dry-run을 통과했다. run 1 training은 epoch 8까지 완료되어 best validation supported-pixel Smooth L1=0.06163210, held-out test=0.07344962를 기록했다. checkpoint spatial diagnostic은 test z=203/227/250에서 global map range=0.67868543·std=0.11746240이지만 support 내부 prediction std=0.0·Pearson=null인 고정 plateau를 확인했다. 따라서 global `max-min ≤ 1e-6` gate만으로는 candidate를 안전하게 보류할 수 없으며, 현재 `emitted` coordinate는 사용하지 않는다. 확장 diagnostic은 z=203/z=227/z=250에서 top-score tie=63,504 pixel=96.8994%, 선택 endpoint 간 map MAE/max-absolute=0.0을 확인했다. 세 endpoint는 모두 `withheld_top_score_plateau`, candidate count=0으로 보류됐다. 이는 real-time decoder가 XCT support mask 없이 arbitrary coordinate를 막았음을 보여 준다. e24 controlled training의 checkpoint-only held-out evaluation은 완료됐다. Best validation은 epoch 9에서 run 1보다 0.17% 낮았으나, held-out test loss는 0.07370871로 run 1보다 0.35% 높았다. e24도 top-score tie=96.8994%, selected endpoint map MAE/max-absolute=0/0, candidate=0(48/48 withheld)이었다. 따라서 epoch만 늘리는 방법은 fixed map collapse를 해소하지 못했다. C32 controlled capacity run도 완료됐다. C32는 top-score tie를 96.8994%에서 0.3845%로 줄였지만, selected endpoint map MAE/max-absolute=0/0 및 support-region prediction std=0을 유지했다. Held-out test loss=0.07363038은 run 1보다 0.24% 높았다. 따라서 capacity 증대는 static geometry를 바꿨지만 input-dependent localization을 복원하지 못했다. Sigma=2→3 support-density audit은 완료됐고, sigma=3 training은 보류됐다. Available endpoint 8개에서 median support gain=29.3990% 및 base support retention=100%로 coverage는 늘었지만, median common-support response MAE=0.0538972가 stability gate 0.05000을 7.79% 초과했다. 모든 available layer는 binary support component 1개·largest share 1.0이어서 component metric은 local merge risk를 식별하지 못했다. 따라서 `weak_target_v1.yaml`의 sigma=2, response direction, unknown policy를 유지한다. C32 input-sensitivity diagnostic도 완료됐다. z=203/227/250의 A input history와 frame-encoder feature는 모든 pair에서 distinct했지만, `temporal_final`부터 logit/score까지 MAE=max-abs=0이었다. 따라서 current collapse는 frame encoder·sigmoid가 아니라 `Conv3D→GroupNorm→SiLU` temporal aggregation path에서 처음 발생한다. 다음 단일 가설은 endpoint frame feature가 decoder까지 causal하게 전달되도록 optional residual bypass `encoded_endpoint + temporal_update`를 도입하는 것이다. `use_endpoint_feature_residual=false` default로 C8/E24/C32 state-dict와 forward contract를 보존했고, C32와 residual flag·output path만 다른 `configs/a_only_baseline_c32_temporal_residual_v1.yaml`을 준비했다. training/spatial diagnostic/checkpoint evaluator/input-sensitivity diagnostic의 모든 model 복원 경로가 이 optional flag를 읽는다. Residual dry run도 통과했다. z=128에서 `[1,4,6,256,256]` input, `[1,1,256,256]` prediction, support=3,439, finite masked loss=0.18406811, MPS execution을 확인했고 output directory/checkpoint/dense target은 생성하지 않았다. 이 random-initialization loss는 trained C32 result와 비교하지 않는다. C32와 동일 조건의 8-epoch residual training도 완료됐다. Best validation loss=0.05642983 (epoch 6), held-out test=0.06992126으로 C32 대비 각각 8.27%·5.04% 낮아졌다. z=203/227/250 spatial diagnostic에서 support prediction std=0.043991/0.041168/0.045247, Pearson=0.396817/0.353880/0.195059, top-score tie=1 pixel=0.001526%로 all three endpoint가 `emitted`됐다. Endpoint map MAE/max-abs도 z203→227=0.010572/0.225400, z227→250=0.009254/0.164584로 zero가 아니었다. Stagewise diagnostic은 temporal_final/logit/score까지 모든 pair variation이 남음(earliest collapsed stage=`null`)을 확인했다. 따라서 residual bypass는 selected held-out inputs에서 temporal collapse를 해소하고 input-dependent continuous score map을 만들었다. 다만 calibration은 provisional이고 response direction/physical defect truth는 unresolved이므로 후보는 계속 XCT-derived continuous quality candidate다. 다음 validation인 `src/audit_a_only_candidate_coordinates.py`는 구현·정적 검증을 마쳤다. Residual compact candidate JSON을 model grid→raw camera→inverse machine coordinate→raw/model round-trip으로 검사하며, ROI/sensor/known-part containment, edge margin, same-endpoint duplicate cell을 compact CSV/JSON으로 audit한다. Coordinate audit은 internal arithmetic과 edge safety는 통과했지만 operational geometry는 보류됐다. 240/240 candidate가 grid/ROI/sensor domain, score `[0,1]`, raw/model round trip, same-endpoint dedup, 3px edge margin을 통과했으나, configured rank-2 provisional transform의 inverse machine coordinates는 모두 known part rectangles 밖이었다. 따라서 `coordinate_consistency_pass=true`, `edge_safety_pass=true`, `operational_geometry_pass_under_provisional_calibration=false`이며 current emitted candidates는 machine/part coordinate interpretation에서 hold한다. 다음 단일 candidate-output improvement인 geometry-aware decoder safety gate는 구현·정적 검증을 마쳤다. Gate는 default-disabled이며 existing spatial/tie/temporal safety gate를 먼저 유지한 뒤 configured provisional part rectangles 안의 local maxima만 compact ranking한다. In-geometry maximum이 없으면 `withheld_outside_provisional_part_geometry`를 명시적으로 반환한다. Saved residual checkpoint의 geometry-gated checkpoint-only evaluation과 filtered candidate coordinate audit도 완료됐다. Held-out loss=0.0699212649, optimizer steps=0으로 original residual score map/loss를 재현했고 48/48 endpoint에서 240 candidates가 emitted됐다. Ungated output의 240/240 outside-part result와 달리 geometry gate output은 part01=84, part02=126, part03=27, part04=3으로 240/240 configured known rectangle 안에 역투영됐다. Coordinate consistency·operational geometry·edge safety가 모두 pass했고 raw round-trip max=`6.43e-13`px, model round-trip max=0px다. 이는 provisional transform **내부에서만** candidate coordinate safety pathway가 일관됨을 뜻한다. Rank-1 versus selected-rank-2 transform sensitivity를 read-only로 비교하는 `src/audit_candidate_calibration_rank_sensitivity.py`도 구현·정적 검증을 마쳤다. Geometry-gated raw candidates에 대해 alternative/selected containment, same-part agreement, inverse machine XY shift, endpoint-level agreement를 compact CSV/JSON으로만 기록하며 rank 2를 재선정하거나 calibration config를 수정하지 않는다. Rank 1 versus rank 2 sensitivity audit도 완료됐다. Both transforms place 240/240 geometry-gated candidates inside some known part rectangle, but same-part agreement=0/240 (0%) and rank1→rank2 inverse machine-coordinate shift=4.168/14.463/33.857/38.053 (min/median/p95/max)다. Rank 1/2 fit·LOO RMSE는 displayed precision에서 동일하고 orientation/part mapping이 다르므로, current transform rank selection은 absolute part identity를 uniquely determine하지 않는다. 따라서 candidate reporting은 raw layer-camera `(x_pixel,y_pixel)` 및 `layer_z`,`score`를 primary로 하고, rank-2 machine XY/part는 `provisional` metadata로만 선택적으로 제공한다. NIST authoritative build-layout anchor 조사도 완료됐다. 공식 dataset description의 Fig. 2는 layer 125에서 Part 1–4 numbering/placement를, same-layer shape에서 cavity=`-X` left / overhang=`+X` right 기준을 제공한다.[1] PDR와 논문은 layer-camera registration에 필요한 `DotGrid_2000x2000.tif`, `SecondaryCamera_Laser00.tif`, `Checkerboard_2000x2000.tif`가 `Layer Camera Metadata.zip`에 있음을 열거한다.[1] [2] 이는 rank1/rank2 mirror ambiguity를 visual/metrology로 분리할 authoritative anchor source지만, raw camera pixel→machine homography를 바로 제공하지는 않는다. Current project에는 archive가 없고 sandbox PDR retrieval도 timeout됐으므로 calibration config는 변경하지 않았다. NIST Fig.2 criterion을 raw pixels에서 검토한 `src/audit_layer125_orientation_overlay.py` 실행도 완료됐다. B-stage `[LED=3,z=125]` read-only memmap frame 및 local laser-on XYPT paths로 rank1/rank2 deterministic overlay PNG two files를 만들었고, both hypotheses have 100% projectable/in-sensor command paths and identical 5.212059/7.028278px fit/LOO RMSE. The raw B-stage full-frame view did not provide unambiguous cavity/overhang visual correspondence, so `visual_preference_hypothesis=inconclusive`; rank2, `provisional` status, and camera-primary reporting remain unchanged. The ignored processed QC PNGs are not model heatmaps or labels. The required independent artifacts were then found already extracted under `raw_original/metadata/Layer Camera Metadata/`: `DotGrid_2000x2000.tif` (4,000,384 bytes), `SecondaryCamera_Laser00.tif` (36,869,338 bytes), and `Checkerboard_2000x2000.tif` (4,000,148 bytes). Each has a recorded local SHA-256 inventory and TIFF signature. The ZIP is absent, so archive-level PDR hash cannot be compared, but duplicate download is cancelled. `src/audit_independent_metrology_metadata.py` pre-audit execution is complete: DotGrid and Checkerboard are both actual `YX=[2000,2000]` uint8 grayscale ImageJ TIFFs with visibly resolved patterns; SecondaryCamera is actual `YXS=[3036,4048,3]` uint8 RGB and has a visible red-reference candidate. Its global red-dominance weighted centroid is `(2341.28,2039.94)` secondary pixels but its weighted spread is 613.08 px, so that centroid is not precise origin evidence; later component isolation and visual confirmation are required. This execution neither fits a homography nor selects rank 1/2 or modifies `calibration_v1.yaml`. Independent fit-audit feasibility is supported, but calibration remains provisional and camera-primary reporting remains mandatory. `src/audit_independent_metrology_fiducials.py` execution is complete and safely **holds** calibration fit: Dot candidates reached the 5,000 cap but include plate/text/screw false positives (global NN CV=0.7133>0.65); Checkerboard candidates are partial/background-contaminated (679, CV=1.2555>0.75); the compact top red component is a left reflection `(791.03,2109.00)`, while the visually salient central spot is split across nearby components. No homography, rank selection, origin attribution, or config change is justified. `src/audit_independent_metrology_fiducials_refined.py` execution is complete. DotGrid ROI evidence passes (1,616 candidates; NN CV=0.3262) and its overlay aligns candidates to the printed panel. The central red spot becomes cluster #1 (34 components, `(2582.34,2029.18)`, spread=40.27 px), separate from the left reflection. Checkerboard remains held (410 candidates; CV=0.5040>0.45; only a lower-left board subset). NIST documents a method-#2 route from dot-grid D coordinates to layer-camera C and reports `A(0,0)=D(28.25,24.25) mm` and 2.5° grid orientation; an independent D→C candidate transform with held-out lattice residuals is now a separately approved **design-review** candidate, not an update. No image rectification/homography/rank choice/origin attribution/config edit has been made.[2] 이 arithmetic pass는 calibration fit/LOO error를 대체하지 않으며, calibration status는 계속 provisional이다. B−A나 XCT response를 direct defect label로 사용하지 않는다.

[1] [NIST PDR: Overhang Part X4](https://data.nist.gov/od/id/mds2-2233)
[2] [Lane & Yeung (2020), J. Res. NIST 125:125027](https://doi.org/10.6028/jres.125.027)

### Independent NIST method-#2 candidate calibration audit — implementation ready

`src/audit_independent_method2_calibration_candidate.py` is implemented and statically verified. It reads only immutable `DotGrid_2000x2000.tif` and the existing provisional calibration config. It refines dot centers to local response-weighted subpixel candidates, creates provisional 50×50 PCA lattice indices, fits `D→C` homography **candidates**, and evaluates deterministic held-out 5×5 lattice blocks. It retains all eight image-lattice axis variants and both `±2.5°` D/A angle-sign alternatives, yielding 16 non-selected `A→D→C` candidates.

The gate measures only image-pixel consistency: at least 1,200 unique cells and 40 represented rows/columns, held-out RMSE≤0.25 and p95≤0.50 of detected inlier camera-dot pitch. The script emits compact CSV/JSON and two DotGrid QC overlays. It does not write raw metadata, modify `calibration_v1.yaml`, choose a candidate/rank, assert a red reference as machine origin, access model/XCT/target/checkpoint data, or change camera-primary candidate reporting. Passing the gate means that human transform review can begin, not that any calibration is deployed.[2]

#### Method-#2 candidate audit V1 execution result — hold all transforms

The V1 run indexed 1,518 unique DotGrid cells across 50 PCA columns and 48 PCA rows, so the coverage gate passed. However, 5×5-block held-out validation failed: 298 held-out cells gave RMSE=`6.00155 px` = `0.41125` of the detected inlier camera-dot pitch (`14.59340 px`), and p95=`9.78092 px` exceeded the `0.50`-pitch limit (`7.29670 px`). The in-sample robust fit was similarly too coarse for deployment (`6.14185 px` RMSE, p95=`9.80831 px`). QC overlays show correct board localization but systematic predicted-vs-actual dot offsets in multiple blocks and rejected/off-panel candidates near text/hardware. The failure is therefore attributed to the provisional PCA + separate 1D 50-cluster correspondence/indexing, not to a demonstrated physical calibration result. All 16 method-#2 alternatives remain held; no rank/config/origin change is allowed. The next possible scope is a separately approved read-only perspective-aware 2D lattice-correspondence refinement, followed by a fresh held-out audit—not another unrestricted homography fit.[2]

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
| Weak target support-density rasterization audit | 완료·sigma=3 training 보류 | support gain 29.3990%·retention 100%이나 response MAE 0.0538972가 stability gate 초과 |
| A-only input-sensitivity diagnostic | 완료·temporal collapse 확인 | input/encoder는 distinct, `temporal_final`부터 pairwise MAE/max-abs=0 |
| Endpoint-feature temporal residual controlled run | 완료·selected held-out internal validation 통과 | validation −8.27%, test −5.04%, 48/48 emitted; spatial·stagewise map variation 확인 |
| Calibration-aware candidate validation | 완료·operational geometry hold | internal coordinate/edge consistency pass, 240/240 inverse coordinates outside known part rectangles |
| Geometry-aware candidate safety gate | 완료·provisional internal safety pass | 48/48 emitted; 240/240 candidates inside configured part rectangles; model/score/loss unchanged |
| Calibration rank-sensitivity audit | 완료·machine coordinate ambiguity 확인 | both-rank containment 100%이나 same-part agreement 0%; median rank shift 14.463 |
| Camera-primary coordinate reporting policy | 확정 | raw camera `(x_pixel,y_pixel)` primary; rank-2 machine/part는 provisional metadata |
| NIST authoritative build-layout anchor research | 완료·direct config update 보류 | Fig. 2 machine layout/asymmetric feature 및 official calibration artifacts 확인; direct PDR archive retrieval timeout |
| Layer-125 visual orientation overlay audit | 완료·visual hold | rank1/rank2 paths both in sensor; full-frame B visual asymmetry not decisive, so no rank change |
| Independent metrology metadata pre-audit | 완료·fit feasibility supported | all TIFF schemas verified; dot-grid/checkerboard patterns visible; red candidate is diffuse and requires component isolation; no refit/rank selection |
| Independent fiducial detector feasibility audit | 완료·calibration fit hold | valid feature visibility but dot/corner false positives and red-reflection top rank; needs ROI/component grouping refinement |
| Detector-refinement audit | 완료·mixed result | DotGrid/red cluster local evidence pass; checkerboard direct route held; no transform fitted |
| Independent method-#2 calibration-fit design audit | 별도 승인 대상 | D→C candidate fit with dot-lattice indexing and held-out residuals; compare to rank1/rank2, no config update |
| Checkerboard-route calibration fit audit | hold | V2 does not yet provide fully indexed checkerboard lattice; not used in method-#2 design |
| Control-point perturbation robustness audit | 후속 검증 단계 | independent anchor 결과 후 perturbation 범위 결정 |
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


### Perspective-aware 2D lattice-correspondence refinement — implementation ready

`src/audit_independent_method2_lattice_correspondence_refinement.py` is the separately approved read-only follow-up to the V1 method-#2 candidate audit. It reads only immutable `DotGrid_2000x2000.tif` through the existing `tifffile.memmap(..., mode='r')` path. After automatic planar ROI restriction and subpixel dark-dot centers, it builds a local graph from up to six nearest neighbors. An edge is retained only when its distance is `0.45–1.75×` the estimated local dot pitch and its PCA-axis direction alignment is at least `0.92`. BFS propagation forms provisional 2D image-lattice labels; a maximum-count 50×50 image-lattice window and iterative projective nearest-cell reassignment then refine correspondence and suppress off-grid candidates.

The audit repeats the **same** deterministic 5×5-block held-out scheme and fixed V1 gates: at least 1,200 unique cells, at least 40 represented rows and columns, held-out RMSE≤0.25 and p95≤0.50 of the train-inlier camera-dot pitch. It writes two compact CSV files, one JSON, and exactly three deterministic QC overlays. It does not read or modify `calibration_v1.yaml`, existing controls, A/B manufacturing TIFF, XCT, weak target/support, model/checkpoint, decoder, candidate output, transform rank, or camera-primary reporting. The intermediate image-lattice homography is used only in memory to test correspondence; it is not a deployed machine calibration. Any passing result permits human review of the correspondence evidence only, never automatic transform selection or config replacement.


#### First 2D correspondence runtime attempt — fail-closed graph-fragmentation hold

The first user execution of the perspective-aware refinement stopped at `best_dense_grid_window` before generating features, overlays, summary JSON, an image-lattice transform, or held-out residual measurements. The raised condition was `Dense 50x50 provisional lattice window contains too few graph labels`. Raw metadata, calibration config, existing rank choice, A/B/XCT data, weak target, model, checkpoint, candidate output, and fixed V1 validation gates were not changed.

Static diagnosis shows that the initial BFS used the single maximum dark-response point as its root; a high-response but small disconnected graph component can therefore exclude the panel-wide component and cause fail-closed coverage termination. This is a graph-seed policy limitation, not evidence of a physical calibration outcome or permission to loosen residual gates. The proposed minimum follow-up is a separately approved component-size-first deterministic BFS seed with component diagnostics, followed by the exact same correspondence procedure and V1 held-out gates. Until then all method-#2 transform candidates remain held and raw camera pixel reporting remains primary.


#### Largest-component correspondence refinement V2 — implementation ready

`src/audit_independent_method2_lattice_correspondence_refinement_v2.py` is a new, separate-output repair for the V1 graph-fragmentation failure. V1 started BFS at the global maximum dark-response candidate; a small disconnected component could therefore be selected before the panel-wide DotGrid graph. V2 enumerates all edge-connected components and deterministically chooses the largest by vertex count, breaking ties with aggregate detector response and then minimum raw `(y,x)`. It uses the highest-response candidate only **inside that selected component** as the propagation seed.

The V2 script writes graph-component diagnostics before attempting correspondence. Thus if it fails before held-out validation, it returns a compact JSON with `status=fail_closed_before_heldout_validation` plus graph components/edges CSVs and one overlay; it does not leave an unexplainable empty result. The detector/ROI, local-neighbor threshold, 2D projective reassignment, exact 5×5 held-out scheme, coverage rule, and RMSE/p95 rules are unchanged from V1. No raw TIFF/CSV, `calibration_v1.yaml`, controls, rank, machine transform, A/B/XCT/target/model/checkpoint/decoder data, or camera-primary reporting is changed.


#### V2 partial completion — plotting-call hold

V2 reached the largest-component correspondence computation and wrote its component/edge/feature compact CSVs plus neighbor-graph and correspondence overlays. It stopped only when saving the final held-out residual overlay: `plot_heldout_residual` has a five-argument signature while the main call supplied obsolete sixth `residual` input. No final JSON or held-out residual overlay exists, so no V2 validation metric or transform conclusion is accepted. The V2 output directory is preserved and must not be overwritten. The only proposed corrective scope is a new V3 source/output path that removes the obsolete argument and statically verifies definition/call arity; all detector, component, correspondence, fixed-gate, config/model/data, and camera-primary policies remain unchanged.


#### V3 implementation — overlay-call arity correction only

`src/audit_independent_method2_lattice_correspondence_refinement_v3.py` preserves the V2 detector, largest-component graph seed, 2D reassignment, and every fixed held-out gate. It changes only the final QC call from six inputs to the function's defined five inputs: `plot_heldout_residual(gray, rows, block_test, predicted, residual_overlay)`. `py_compile` passed and static source inspection confirmed the definition and unique call both have five positional arguments. V3 writes only a new ignored output directory and never overwrites the V2 partial artifacts. A V3 completion can make the same fixed-gate result reviewable; it still cannot select a transform/rank, update `calibration_v1.yaml`, assert a machine/part location, or change camera-primary XCT-derived continuous quality candidate reporting.


#### V3 runtime result — correspondence residual pass, fixed coverage hold

V3 completed on the immutable DotGrid TIFF and established a coherent largest graph component: 1,523/1,616 candidates (94.25%) and 2,812/2,860 accepted edges (98.32%) are in component #1, with zero BFS label-cycle conflicts. The same fixed 5×5-block held-out residual gate improved substantially from V1: held-out RMSE=`0.98184 px`=`0.06728` detected camera-dot pitch and p95=`1.67410 px`=`0.11472` pitch, so the residual gate passes. QC overlays show correspondence and sampled held-out blocks on the printed DotGrid panel without a visible multi-region residual pattern.

However, final reassignment has 1,554 cells and 50 image-lattice columns but **39 rows**. The predeclared V1 coverage rule requires both dimensions≥40, so `coverage_pass_same_v1_rule=false` and `all_fixed_gates_pass=false`. This near-miss is not silently promoted to a pass and the threshold is not lowered after observation. The remaining question is whether the 39-row result arises from the visible physical target/field of view, target-count convention, detector/reassignment boundary, or another indexing limitation. All method-#2 transform candidates therefore remain held. The next permissible scope is a separate read-only coverage-definition audit; `calibration_v1.yaml`, rank/orientation, machine/part claims, model/target, and camera-primary XCT-derived continuous quality candidate reporting remain unchanged.


#### Coverage-definition audit — implementation ready after the V3 39-row hold

`src/audit_independent_method2_dotgrid_coverage_definition.py` is a separate, read-only evidence audit. It reads only the immutable DotGrid TIFF and the completed V3 compact feature CSV. A temporary in-memory image-lattice mapping predicts the nominal 50×50 cells; compact per-cell/per-row/per-column tables then separate three conditions: nominal prediction outside the 2000×2000 sensor, in-sensor prediction near a fresh detector candidate but unassigned in V3, and in-sensor prediction without fresh detector support. Its two deterministic QC overlays show nominal coverage evidence and occupancy profiles.

The script cannot and does not change `GRID_SIZE=50`, the fixed `rows/columns≥40` gate, `calibration_v1.yaml`, transform/rank/orientation, machine origin, A/B/XCT/target/model/checkpoint/decoder data, or camera-primary XCT-derived continuous quality candidate reporting. Its evidence class only tells a later human review whether FOV, correspondence assignment, or target/detector visibility remains plausible; it never applies a remedy automatically.


#### Coverage-definition runtime result — no simple FOV explanation for 39 rows

The read-only coverage audit found that all `946/946` nominal-but-unassigned image-lattice cells are predicted inside the 2000×2000 sensor; none is outside. Moreover, none is within the frozen `6.55369 px` (`0.45×` V3 train-inlier pitch) of a fresh ROI-restricted detector candidate. Thus the observed pattern is not simple sensor clipping and is not merely a set of known dot-centre candidates missed by V3 assignment. It is classified narrowly as in-sensor nominal cells without fresh detector support, with visible target extent, detector footprint, and provisional image-lattice window/index convention all remaining plausible.

The two QC plots strengthen this caution: missing cells form structured left and right/central regions rather than one clean clipped border; row occupancy ramps to near-full coverage and then has only a two-cell row, while all columns remain represented only partially. The fixed 40-row gate remains failed and is not relaxed. All method-#2 transform candidates, rank/orientation claims, machine/part metadata, and config revisions remain held. The next possible work is a separately approved human-reviewed DotGrid extent/index-convention design audit, not another automatic transform fit.


#### Human-reviewed visible DotGrid outer-extent workflow — implementation ready

The next coverage evidence is deliberately human-reviewed rather than another threshold sweep. `select_visible_dotgrid_extent_controls.py` records only four visible outer dot centres in fixed visual order `TL → TR → BR → BL`, after reading the immutable DotGrid TIFF. `audit_visible_dotgrid_extent_controls.py` snaps those clicks to distinct fresh dot candidates within `0.60×` V3 pitch, verifies a strictly convex ordered quadrilateral and per-edge candidate support, then reports V3/fresh-detector/nominal-footprint inclusion relative to the human visible panel.

This produces visible-panel extent evidence in raw camera coordinates only. It cannot identify D origin, physical cell indices, machine axes, transform/rank/orientation, or an accepted grid/coverage policy. The compact control JSON and validation CSV/JSON/two overlays remain ignored regenerable outputs. Raw data, V3 artifacts, `GRID_SIZE=50`, the 40-row gate, `calibration_v1.yaml`, model/target/checkpoint/decoder, and camera-primary XCT-derived continuous quality candidate reporting remain unchanged.


#### Human outer-extent selector V1 — GUI backend hold

The initial `select_visible_dotgrid_extent_controls.py` correctly preserved raw/control/config state but could not accept clicks on macOS: it imported a batch-QC module that sets the noninteractive Matplotlib `Agg` backend, so `plt.ginput` emitted `FigureCanvasAgg is non-interactive`. No four-click controls JSON was written and no human extent evidence exists. The next proposed correction is an isolated V2 selector with its own read-only TIFF/grayscale helper and an explicit GUI backend selected before `pyplot` import. It will use a separate control JSON path, preserve the same ordered four visible outer-dot clicks and compact-output policy, and make no grid/gate/config/rank/model/target/candidate change.


#### Human outer-extent selector V2 — GUI backend isolated implementation ready

`src/select_visible_dotgrid_extent_controls_v2.py` is the separately approved repair for the V1 `Agg` backend hold. It intentionally imports no batch-QC/audit module. Instead, it reads only the immutable `DotGrid_2000x2000.tif` through a local `tifffile.TiffFile` metadata check and `tifffile.memmap(..., series=0, mode='r')` YX grayscale reader. Before importing `matplotlib.pyplot`, it requests the macOS `MacOSX` interactive backend; only when that is unavailable does it attempt `TkAgg`. The selector rejects a noninteractive `Agg` result and raises an explanatory error if neither GUI backend can be selected.

The V2 display is downsampled only for click visibility. It saves raw-camera coordinates by multiplying the display clicks by the recorded stride, returns automatically after four left-clicks, supports right-click removal of the latest point, and writes `processed/calibration/visible_dotgrid_extent_controls_v2.json` **only after exactly four clicks** in `TL → TR → BR → BL` order have been obtained. `py_compile`, source-order inspection, read-only memmap contract inspection, and whitespace validation passed; the interactive selector itself was deliberately not executed by the assistant. V1 source/output remain untouched. V2 neither edits raw data nor changes `GRID_SIZE=50`, the 40-row gate, V3 results, `calibration_v1.yaml`, transform/rank/orientation, machine origin, A/B/XCT/weak target/model/checkpoint/decoder data, or camera-primary XCT-derived continuous quality candidate reporting. Its future JSON is human visible-panel extent evidence only and must be inspected by the existing validator before any separate policy discussion.

To create a new V2 control file, the user runs:

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/select_visible_dotgrid_extent_controls_v2.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --output-json processed/calibration/visible_dotgrid_extent_controls_v2.json
```

If the output JSON already exists, it must be listed and reviewed before any deliberate replacement; this guide does not recommend `--overwrite` automatically.


#### Visible-extent validator CLI correction

The existing validator defines its completed-V3 feature input as `--v3-features`, not `--v3-features-csv`. An initial V2 guide invocation stopped at `argparse` before opening any TIFF or control data because the former spelling was supplied; this was a documentation-only command mismatch, not a calibration/validation outcome. The corrected read-only command is:

```bash
/usr/local/bin/python3 src/audit_visible_dotgrid_extent_controls.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --controls-json processed/calibration/visible_dotgrid_extent_controls_v2.json \
  --output-dir processed/calibration/visible_dotgrid_extent_validation_v2
```

This correction changes no source, raw/processed data, V2 controls, calibration/gate/model policy, or candidate reporting. The validator result remains pending.


#### V2 human visible-extent validation result — click/snap hold

The user completed `audit_visible_dotgrid_extent_controls.py` with the V2 controls and the actual V3 feature CSV. The audit executed read-only and created only its three compact CSVs, one JSON summary, and two deterministic overlays. It preserved raw TIFF/CSV, the V2 controls JSON, V3 outputs, `GRID_SIZE=50`, the rows≥40 gate, `calibration_v1.yaml`, transform/rank/orientation, machine-origin status, target/model/checkpoint/decoder, and camera-primary XCT-derived continuous quality candidate policy.

The controls are sensor-contained and form a strictly convex ordered `TL→TR→BR→BL` quadrilateral (all cross-products positive). However, the fixed click-to-fresh-candidate snap test fails: only TL is within the frozen `8.73835 px` bound (`3.47989 px`); TR, BR, and BL are respectively `170.41794 px` (11.701 V3 camera-dot pitches), `87.56457 px` (6.012 pitches), and `16.03069 px` (1.101 pitches) from their nearest fresh detector candidate. The minimum per-edge support count is 5≥3, but the right edge alone has only 5 candidates versus 38/43/51 on the top/bottom/left. Therefore `all_control_validity_checks_pass=false` and the correct result is `hold_extent_interpretation`, not a panel-extent pass.

The overlays nevertheless supply narrow diagnostic evidence: V3 assigned cells are 1,539/1,554 (99.0347%) inside the human quadrilateral, fresh candidates are 1,554/1,616 (96.1634%) inside, and nominal 50×50 predictions are 1,900/2,500 inside with 600 outside. Visually, the selected quad covers the central detector-supported lattice, while a right-side nominal region extends beyond it and two V3 assigned points lie left of the quad. Because three outer clicks failed fresh-dot snap and the right edge has sparse support, these counts cannot resolve whether the shortfall is visible physical extent, detector footprint, click placement, or nominal-window convention. No grid/gate/config/calibration decision follows from this run.


#### Visible DotGrid outer-boundary diagnostic — implementation ready

`src/audit_visible_dotgrid_outer_boundary_diagnostic.py` is the separately approved read-only follow-up to the V2 strict snap hold. It preserves the existing V2 controls and V3 outputs, reconstructs the same frozen refined detector/ROI and V3 nominal 50×50 image-space predictions, and adds a fixed local four-pitch patch around each click. Within each patch it recomputes the same dark-dot response, applies deterministic q=0.990/NMS=8 px candidate extraction, and requires a near-click candidate plus at least two camera-pitch-band neighbors with an approximately orthogonal pair before supporting `printed_dot_visible_but_current_detector_missed`.

Each control is classified only as `current_detector_supported`, `printed_dot_visible_but_current_detector_missed`, `click_outside_printed_dot`, or `ambiguous`. The result is diagnostic evidence, not an automatic reclick, detector replacement, tolerance change, grid/gate decision, homography fit, calibration/rank/origin claim, or model/target change. The script writes one compact per-control CSV, one JSON summary, four local patch QC PNGs, and one full-panel QC PNG under a new ignored output directory; it persists no dense crop, response, mask, rectification, target, or model output. Static `py_compile`, fixed-constant/source-contract inspection, and whitespace validation passed. The assistant did not run the diagnostic.

```bash
cd ~/ammt_project
/usr/local/bin/python3 src/audit_visible_dotgrid_outer_boundary_diagnostic.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --controls-json processed/calibration/visible_dotgrid_extent_controls_v2.json \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --output-dir processed/calibration/visible_dotgrid_outer_boundary_diagnostic_v1
```

If the output directory already exists, it must be listed and reviewed; `--overwrite` is not recommended automatically. The script never changes the frozen V2 snap result, `GRID_SIZE=50`, rows≥40 gate, V3 39-row hold, `calibration_v1.yaml`, transform/rank/orientation, machine origin, A/B/XCT/weak-target/model/checkpoint/decoder, or raw-camera-primary XCT-derived continuous quality candidate reporting.


#### Outer-boundary diagnostic V1 result — one detector-footprint miss, two lower-edge ambiguities

The user completed `audit_visible_dotgrid_outer_boundary_diagnostic.py` with existing V2 controls and V3 compact features. It read only those immutable/compact inputs and wrote one per-control CSV, one summary JSON, four local-patch QC PNGs, and one full-panel QC PNG. V2 controls, V3 outputs, detector parameters, raw TIFF/CSV, `GRID_SIZE=50`, rows≥40 gate, calibration/rank/orientation, model/target/checkpoint/decoder, and camera-primary XCT-derived continuous quality candidate reporting were unchanged.

The four evidence classes are `TL=current_detector_supported`, `TR=printed_dot_visible_but_current_detector_missed`, and `BR/BL=ambiguous`; the predeclared recommendation is therefore `hold_outer_boundary_diagnosis; mixed_or_ambiguous_image_space_evidence`. TR is 13.959 px right of the frozen detector ROI, but its local detector identifies a candidate at `(1704,799)` only 4.984 px from the human click, with three pitch-band neighbors and an orthogonal pair. Its patch visibly contains a regular dark-dot lattice, so the frozen ROI/detector footprint excludes a locally supported outer-right dot. This supports only a future detector-boundary design review, not an automatic ROI change.

BR is 11.405 px outside the frozen ROI and the nearest local dark-dot candidate is `(1701,1586)`, 14.346 px from the click near `(1706,1599)`; the visible dot lattice ends above the clicked point. BL is inside the frozen ROI but the nearest frozen/local candidate `(975,1614)` is 16.031 px above the click near `(973,1630)`, while V3/nominal reference points lie around y≈1595–1615. Both lower controls are therefore conservatively `ambiguous`, not promoted to confirmed click-placement or detector-miss evidence. Overlay review agrees with this classification.

The diagnostic establishes mixed image-space evidence only. The existing V2 JSON and outputs must remain preserved; no reclick, JSON edit, `--overwrite`, threshold/ROI change, gate relaxation, homography refit, calibration/config update, or model/target action follows automatically. A future correction workflow, if proposed, requires separately approved human-reselection and/or detector-boundary design with fixed predeclared acceptance criteria.


#### Calibration design review V1 — implementation ready

`src/audit_calibration_design_review_v1.py` implements the approved read-only review after the mixed outer-boundary evidence hold. It uses only the immutable DotGrid TIFF, V2 validation summary/control CSV, V1 outer-boundary diagnostic per-control CSV, V3 compact feature CSV, and the existing 192-hypothesis ranking CSV. It compares two fixed extent candidates—frozen refined-detector ROI and the held V2 human quad—without constructing a new expanded extent. An evidence-expanded extent is explicitly blocked unless all four controls show local lattice evidence; the current mixed classes do not meet that rule.

For each fixed candidate the audit reports V3 point inclusion, unique assigned rows/columns, missing nominal `0..49` row/column indices, and the unchanged `rows>=40` plus `columns>=50` descriptive occupancy gate. It separately records whether the top residual-ranked orientation is tied within `1e-6 px` LOO RMSE. It marks the current top tie as unresolved because no independent asymmetric cross-camera anchor is supplied. Neither residual rank nor image extent is selected for calibration deployment.

```bash
cd ~/ammt_project
ls -ld processed/calibration/calibration_design_review_v1

/usr/local/bin/python3 src/audit_calibration_design_review_v1.py \
  --dot-grid 'raw_original/metadata/Layer Camera Metadata/DotGrid_2000x2000.tif' \
  --v2-validation-summary processed/calibration/visible_dotgrid_extent_validation_v2/visible_dotgrid_extent_validation_summary.json \
  --v2-controls-csv processed/calibration/visible_dotgrid_extent_validation_v2/visible_dotgrid_extent_control_validation.csv \
  --outer-boundary-csv processed/calibration/visible_dotgrid_outer_boundary_diagnostic_v1/visible_dotgrid_outer_boundary_diagnostic_by_control.csv \
  --v3-features processed/calibration/independent_method2_lattice_correspondence_refinement_v3/method2_refined_2d_lattice_features.csv \
  --orientation-ranking-csv processed/calibration/orientation_audit_v1/calibration_candidate_ranking.csv \
  --output-dir processed/calibration/calibration_design_review_v1
```

The expected compact output is three CSVs, one JSON summary, and two deterministic QC overlays. No raw input, V2/V3 artifact, detector setting, `GRID_SIZE=50`, rows>=40 gate, homography, rank/orientation, `calibration_v1.yaml`, XCT target/model/checkpoint/decoder, or raw-camera-primary XCT-derived continuous quality candidate policy can change. Static `py_compile`, fixed-rule inspection, and whitespace validation passed; the assistant did not run the audit.


#### Calibration design review V1 result — neither fixed extent reaches 40 rows; mirror tie remains

The user completed `audit_calibration_design_review_v1.py` successfully. It read only the immutable DotGrid TIFF plus preserved V2/V3/outer-boundary/ranking compact artifacts and wrote three compact CSVs, one JSON summary, and two deterministic QC overlays. Raw data, V2 controls, outer-boundary results, detector threshold/ROI, `GRID_SIZE=50`, coverage gate, calibration config, model/target/checkpoint/decoder, and raw-camera-primary XCT-derived continuous quality candidate reporting remain unchanged.

Neither pre-existing extent candidate reaches the fixed `rows>=40` rule. The frozen detector ROI contains all 1,554 V3 assigned points with 50 columns but only 39 distinct rows; it contains row indices 0–37 and 39, while 38 and 40–49 are absent. The held V2 human quad excludes 15 V3 points and contains only 38 rows, 0–37, while retaining all 50 columns. Thus the wider frozen detector footprint is the less restrictive descriptive candidate but **still fails** the frozen row gate; the V2 quad is strictly worse for coverage. The output correctly blocks evidence-expanded extent construction because outer-control evidence remains mixed.

The orientation ranking retains an exact two-way residual tie: rank1 `mirror_rotate_90` / `part01;part02;part03;part04` has LOO RMSE `7.028278322386677 px`, rank2 `mirror_rotate_270` / `part04;part03;part02;part01` has `7.028278322386715 px`, a maximum difference of only `3.82e-14 px`. All remaining tested candidates are materially worse, but no independently validated asymmetric cross-camera anchor is available. Therefore the correct decision is `hold_extent_and_orientation; no candidate satisfies all predeclared independent requirements`, not a rank/config update.


#### SecondaryCamera asymmetric-anchor feasibility — current metadata is insufficient

A read-only inventory and NIST method-#2 source check were completed after the calibration design-review hold. The local metadata directory contains only `DotGrid_2000x2000.tif` and a single `SecondaryCamera_Laser00.tif`; no `SecondaryCamera_Laser01...` or multiple red-dot-at-known-grid-position sequence is available. The existing refined audit detects a compact red cluster at secondary-camera `(2582.34,2029.18)` px, but that is a pixel-space visual candidate only. There is no validated SecondaryCamera→LayerCamera bridge and no LayerCamera observation of that same red indicator.

Lane and Yeung describe the red indicator image as locating `A(0,0)` relative to DotGrid, but explicitly state that this origin alone does not provide orientation; their 2.5° DotGrid-to-machine orientation required **additional secondary-camera measurements at various red-dot positions on the grid** [1]. The public PDR currently lists 11 top-level files and exposes only `Layer Camera Metadata.zip` as the metadata archive [2]. Since the local extracted archive contains one `Laser00` image, the additional multi-position evidence required to reproduce the orientation step is not present in the current project inputs.

Consequently a new cross-camera homography/anchor audit is not implemented: any such source would fabricate correspondence rather than validate it. Rank1/rank2 residual tie, `calibration_v1.yaml` provisional rank2 status, fixed extent/coverage holds, and raw-camera-primary XCT-derived continuous quality candidate reporting remain unchanged. The next valid project work is model-side evaluation that does not require machine-coordinate truth; a future orientation-resolution task requires either independently documented cross-camera correspondences, the missing multi-position secondary-camera measurements, or a different asymmetric LayerCamera-visible fiducial.

[1]: https://doi.org/10.6028/jres.125.027 "Lane & Yeung (2020), Process Monitoring Dataset from the AMMT: Overhang Part X4"
[2]: https://data.nist.gov/od/id/mds2-2233 "NIST Public Data Repository: AMMT Overhang Part X4"
