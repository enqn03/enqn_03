import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw
import os
import numpy as np
from scipy.ndimage import gaussian_filter

# 페이지 기본 설정
st.set_page_config(page_title="AMMT Defect Detection Dashboard", layout="wide")

# 헤더
st.title("AMMT 실시간 이상 후보 위치 탐지")
st.markdown("### LPBF 공정 중 Layer-Camera 기반 결함 조기 탐지 AI 시스템")
st.markdown("[5조] TEAM 3Do | 김상민, 김태학, 이주현, 정미연")

st.divider()

# 탭 생성
tab1, tab_process, tab_arch, tab_analysis, tab4, tab_conclusion = st.tabs([
    "프로젝트 소개", 
    "프로젝트 과정",
    "모델 아키텍처 상세",
    "모델 성능 및 결과 심층 분석",
    "테스트셋 기준 결함 모니터링",
    "결과 및 향후 발전 방향"
])

# 탭 1: 프로젝트 소개
with tab1:
    st.header("1. 프로젝트 배경 및 문제 정의")
    st.markdown("""
    **LPBF(Laser Powder Bed Fusion)** 기반 금속 3D 프린팅 공정은 한 번에 제품을 만드는 것이 아니라, 수천 개의 레이어를 한 층씩 쌓아 올려 완성합니다.
    기존 제조 환경에서는 제품이 완전히 완성된 후 **사후 XCT(X-ray Computed Tomography) 스캔**을 통해서만 내부 결함을 파악할 수 있었습니다. 만약 프린팅 초반부에 결함이 발생했더라도 이를 모른 채 며칠간 공정을 지속하게 되어 막대한 **시간과 원자재 손실**이 발생하게 됩니다.
    
    본 프로젝트는 이러한 한계를 극복하기 위해, 프린터 내부에 설치된 **공정 카메라 데이터**를 실시간으로 분석하여 **내부 XCT 결함을 공정 중에 조기 예측**하는 인공지능 기반 모니터링 시스템을 개발하는 것을 목표로 합니다.
    """)
    
    st.header("2. 핵심 프로젝트 개념 및 출력물의 의미")
    st.markdown("""
    - **제조 환경 2D 이미지의 3D 정답화:** 의료용 3D 볼륨 데이터가 아닌, 프린팅 중 촬영되는 단면 2D 시퀀스 이미지를 모델의 입력값으로 사용합니다. 완성품의 사후 XCT 3D 스캔 결과는 캘리브레이션 맵핑을 거쳐 모델 학습을 위한 **약한 정답지**로 활용됩니다.
    - **연속적인 품질 점수(Quality Candidate) 제안:** 
      본 모델의 최종 출력은 단순히 픽셀 단위로 "결함이다/아니다"를 이분법적으로 확정 짓는 것이 아닙니다. 대신, 카메라 시퀀스를 분석하여 엔지니어가 우선적으로 검토해야 할 이상 의심 위치를 **`[x픽셀, y픽셀, 레이어층, 확률 점수]`** 형태로 연속성 있게 제안합니다.
    - **시계열 인과적 학습:**
      미래의 정보가 현재의 예측에 영향을 미치는 데이터 누수(Data Leakage)를 철저히 방지하기 위해, 오직 과거(K=4 프레임) 시계열 정보만을 참조하여 현재 레이어의 결함을 인과적으로 예측합니다.
    """)
    
    st.header("3. 모델 아키텍처 진화 (A+B Gated CBAM Fusion)")
    st.markdown("""
    초기에는 A(파우더 도포 후)와 B(레이저 용융 후) 이미지를 하나의 신경망으로 통합 처리하려 했으나, 두 공정 이미지 간의 극심한 밝기 편차와 예측 불가한 스패터 노이즈 간섭으로 인해 완전히 독립적인 듀얼-스트림 처리가 필수적임을 확인했습니다.
    
    - **독립적 특징 추출:** 파우더 도포 후(A)와 레이저 조사 후(B) 이미지를 섞지 않고, 각각 독립된 인코더를 통해 고유한 시각적 특징을 추출합니다.
    - **Temporal Difference 적용:** 단순히 현재 시점의 이미지만 보는 것을 넘어, 직전 과거 프레임들의 평균과 현재 프레임의 차이를 명시적으로 계산하여 공정 상의 **변화량**에 모델이 집중하도록 유도했습니다.
    - **게이트 융합:** A와 B에서 추출된 특징을 결합할 때, CBAM이 채널 및 공간적 어텐션을 동적으로 계산합니다. 이를 통해 B 이미지의 시끄러운 노이즈는 억제하고, 결함 탐지에 유효한 시그널만 선별적으로 융합(Fusion)하여 오탐을 획기적으로 줄였습니다.
    """)

# 탭: 프로젝트 과정
with tab_process:
    st.header("프로젝트 수행 과정")
    st.markdown("로우 데이터(Raw Data)로부터 신뢰할 수 있는 결함 탐지 모델을 구축하기 위해 수행한 핵심 전처리 및 검증 과정입니다.")
    
    st.subheader("1. B-A 적용 가능성 검증")
    st.markdown("""
    - **왜 B-A를 검증했는가? (목적):** 정적인 단일 이미지(A 또는 B)만으로는 표면의 단순한 얼룩과 실제 구조적 결함을 구분하기 어렵습니다. 따라서 파우더 도포(A) 후 레이저 조사(B)가 이루어지는 과정에서 발생하는 **'시각적 차이(B-A)'**가 결함을 짚어내는 강력한 신호가 될 수 있는지 검증하고자 했습니다.
    - **검증 과정:** `audit_ab_pairs` 과정을 통해 수천 장의 A/B 프레임 페어링이 기하학적으로 완벽히 정렬되는지 확인하고, 두 이미지의 픽셀 차이(Absolute A/B difference)를 추출하는 읽기 전용 검증을 수행했습니다. 
    """)
    ab_img_path = "processed/audit_ab_pairs/ab_pair_contact_sheet.png"
    if os.path.exists(ab_img_path):
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.image(Image.open(ab_img_path), caption="A/B 프레임 페어링 및 픽셀 차이(B-A) 시각화", use_container_width=True)
    st.markdown("""
    - **무엇을 얻었는가? (결과):** 검증 결과, 장비의 고정된 구조물이나 조명 그림자 같은 불필요한 배경은 완벽하게 제거되고, 오직 레이저로 인해 새롭게 용융된 부분과 비정상적인 스패터 발생 구역만 선명하게 도드라지는 것을 시각적/정량적으로 입증해 냈습니다.
    - **다음으로 무엇을 했는가? (Next Step):** 단순한 B-A 차이 이미지조차 강력한 **이상 후보 맵**으로 기능함을 확인한 후, 이 데이터를 인공지능 모델에 깨끗하게 집어넣기 위해 **최적의 분석 영역(ROI)을 설정하고 조명 편차를 잡는 정규화** 작업으로 넘어갔습니다. 이후 이 통찰은 최종 A+B 퓨전 모델 아키텍처를 설계하는 핵심 근거가 되었습니다.
    """)
    
    st.divider()
    st.subheader("2. ROI(관심 영역) 설정 및 화질 분석")
    st.markdown("""
    - **목적:** 원본 이미지(2000x2000) 가장자리의 불필요한 장비 구조물과 중앙부 빛 번짐을 고려하여 모델 학습 효율을 최적화할 유효 영역을 탐색
    - **후보군 탐색:** 아래 5가지 ROI 후보군을 설정하여 조명(LED 1,2,3)별 빛 번짐 면적을 정량적으로 평가했습니다.
      - 1) `wide_250_250_1750_1750` (1500x1500px)
      - 2) `inner_350_350_1650_1650` (1300x1300px)
      - 3) `inner_450_450_1550_1550` (1100x1100px)
      - 4) `upper_350_250_1650_1550` (1300x1300px)
      - 5) `lower_350_450_1650_1750` (1300x1300px)
    - **선정 결과 및 근거:** 평가 결과 중심부로 좁혀질수록(`inner_450`) 빛 번짐이 최대 99%까지 치솟는 문제를 발견했습니다. 반면, 가장 넓은 영역을 포함한 `wide_250_250_1750_1750` 후보군이 **빛 번짐 비율이 34.5%로 가장 낮아 유효 데이터 비율 1위**를 기록했습니다.
    - **결론:** 이에 따라 중앙부 빛 번짐을 최소화하면서도 가장 넓은 데이터를 확보할 수 있는 `(250, 250)` ~ `(1750, 1750)` 구역(1500x1500 픽셀)을 최종 분석 ROI로 크롭 확정하였습니다.
    """)
    roi_img_path = "processed/roi_candidates/roi_candidate_qc.png"
    if os.path.exists(roi_img_path):
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.image(Image.open(roi_img_path), caption="ROI 설정 및 후보 영역 QC 검증 시각화", use_container_width=True)
    st.markdown("""
    - **결론:** 조명과 층수에 따른 이미지 밝기 편차가 크다는 점을 고려하여, 확정된 ROI를 기반으로 픽셀 정규화 전처리 파이프라인을 구축했습니다.
    """)
    
    st.divider()
    st.subheader("3. 카메라 캘리브레이션 및 광도/방향성 정밀 검증")
    st.markdown("""
    단순한 2D 픽셀 단위 분석을 넘어, 최종적으로 카메라 상의 픽셀 좌표를 장비의 실제 **3D 물리 좌표(Machine X, Y mm)**로 맵핑하고, 사후 XCT 스캔 3D 데이터(정답지)와 정밀하게 대조하기 위해 다각적인 캘리브레이션 및 검증 과정을 거쳤습니다.
    """)

    st.markdown("#### 3-1. 광도(Photometric) 및 기본 방향성 검증")
    st.markdown("- **목적:** 센서의 조명 안정성과 이미지의 기하학적 정렬 상태를 초기 점검")
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("processed/calibration/photometric_audit_v1/photometric_qc.png"):
            st.image(Image.open("processed/calibration/photometric_audit_v1/photometric_qc.png"), caption="프레임 간 조명(광도) 일관성 검증", use_container_width=True)
    with c2:
        if os.path.exists("processed/calibration/orientation_audit_v1/calibration_candidate_qc.png"):
            st.image(Image.open("processed/calibration/orientation_audit_v1/calibration_candidate_qc.png"), caption="초기 캘리브레이션 후보군 기하 검증", use_container_width=True)
    with c3:
        if os.path.exists("processed/calibration/calibration_design_review_v1/calibration_design_review_orientation_overlay.png"):
            st.image(Image.open("processed/calibration/calibration_design_review_v1/calibration_design_review_orientation_overlay.png"), caption="X, Y 원점 방향성 오버레이 리뷰", use_container_width=True)

    st.markdown("#### 3-2. Dot Grid 피처 기반 센서 커버리지 검증")
    st.markdown("- **목적:** Dot Grid Calibration 기법(`Independent Method 2`)이 센서의 유효 영역을 충분히 커버하는지, 왜곡 없는 변환 행렬 도출이 가능한지 확인")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("processed/calibration/independent_method2_dotgrid_cov/method2_v3_row_column_coverage_profiles.png"):
            st.image(Image.open("processed/calibration/independent_method2_dotgrid_cov/method2_v3_row_column_coverage_profiles.png"), caption="Dot Grid 행/열 센서 커버리지 프로파일", use_container_width=True)
    with c2:
        if os.path.exists("processed/calibration/independent_method2_dotgrid_cov/method2_v3_nominal_coverage_evidence_overlay.png"):
            st.image(Image.open("processed/calibration/independent_method2_dotgrid_cov/method2_v3_nominal_coverage_evidence_overlay.png"), caption="Dot Grid 에비던스 오버레이", use_container_width=True)

    st.markdown("#### 3-3. 미세 정렬 및 랭크 교차 검증")
    st.markdown("- **왜 방향성 검증이 필요했는가?:** 제공된 사후 3D XCT 스캔 데이터에는 원본 장비 내에서 부품이 어느 방향으로 회전되어 프린팅되었는지에 대한 **절대적인 방향성 정보가 누락**되어 있었습니다. 따라서 XCT 정답지를 2D 카메라 평면 이미지에 올바르게 겹치기 위해서는 부품이 놓인 각도를 수동으로 역추적하는 과정이 필수적이었습니다.")
    st.markdown("- **목적:** 사후 3D XCT 데이터와 실제 레이어 이미지(Layer 125 기준)를 매칭할 때, 부품이 장비 내에 놓여진 정확한 회전 및 대칭 상태(Rank 1 vs Rank 2)의 모호성을 해결하고 잔차 오차를 최소화하는 미세 정렬 수행")
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("processed/calibration/layer125_orientation_overlay_v1/layer125_B_led3_rank1_mirror_rotate_90_overlay.png"):
            st.image(Image.open("processed/calibration/layer125_orientation_overlay_v1/layer125_B_led3_rank1_mirror_rotate_90_overlay.png"), caption="방향성 랭크 1 후보 오버레이 (Rotate 90)", use_container_width=True)
    with c2:
        if os.path.exists("processed/calibration/layer125_orientation_overlay_v1/layer125_B_led3_rank2_mirror_rotate_270_overlay.png"):
            st.image(Image.open("processed/calibration/layer125_orientation_overlay_v1/layer125_B_led3_rank2_mirror_rotate_270_overlay.png"), caption="방향성 랭크 2 후보 오버레이 (Rotate 270)", use_container_width=True)
    with c3:
        if os.path.exists("processed/calibration/local_refinement_v1/local_refinement_qc.png"):
            st.image(Image.open("processed/calibration/local_refinement_v1/local_refinement_qc.png"), caption="Local Refinement를 통한 픽셀 오차 최소화 검증", use_container_width=True)
            
    st.markdown("""
    - **결론:** 이러한 광범위한 광학/기하학적 검증을 거쳐, 우리는 의료용 볼륨 데이터(XCT)를 제조 현장의 2D 평면 이미지에 성공적으로 오차 범위 내에서 투영(Projection)할 수 있는 **정교한 Calibration Pipeline**을 확립했습니다. 이는 XCT 정답지를 모델 학습용 Weak Target으로 변환할 수 있는 핵심 기반이 되었습니다.
    """)
    st.divider()
    st.subheader("4. 약한 정답지(Weak Target) 생성 및 라벨링")
    st.markdown("""
    캘리브레이션이 끝난 후, XCT 3D 볼륨 데이터를 모델이 학습할 수 있는 2D 평면 정답지(Weak Target)로 변환하는 과정을 거쳤습니다.
    
    - **왜 상위/하위 샘플 라벨링이 필요했는가?:** 원본 XCT 데이터에는 명확한 **정상(Normal) / 결함(Defect) 이진 라벨이 존재하지 않았습니다.** 오직 연속적인 물리적 밀도/결함 점수만이 존재했습니다. 따라서, 어설픈 중간 점수들을 배제하고 확실한 차이를 학습시키기 위해 XCT 점수 기준 최상위(하위) 샘플 패치들을 추출하여 분석(Target Semantics)하고, 이를 토대로 모델이 명확하게 결함의 특징을 학습할 수 있도록 라벨링 및 타겟을 세팅했습니다.
    - **라벨링은 어떻게 진행되었는가?:**
      1. **Robust Scaling (정규화):** 연속적인 XCT 점수의 상/하위 1% 극단값을 잘라내고(p01~p99), 전체 점수 스펙트럼을 0에서 1 사이로 매끄럽게 정규화했습니다.
      2. **임계값 기반 이진화:** 정규화된 점수가 **0.85 이상(상위 15%)인 구역만을 치명적인 결함(Defect, Label 1)**으로 확정 짓고, 나머지를 정상으로 분류했습니다.
      3. **2D 가우시안 래스터화:** 3D 공간상의 결함 포인트들을 2D 평면 이미지 좌표로 투영할 때, 픽셀 간의 끊김을 방지하고자 2-Pixel 반경의 가우시안 블러를 적용하여 부드러운 히트맵 형태의 정답지를 생성했습니다.
      4. **마스킹:** 실제 부품이 얹혀지지 않은 허공 영역은 학습에 방해되지 않도록 `Unknown` 처리하여 손실 함수 계산에서 완전히 제외시켰습니다.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("processed/xct_target_audit/xct_target_qc.png"):
            st.image(Image.open("processed/xct_target_audit/xct_target_qc.png"), caption="XCT 원본 Target 점수 분포 감사(Audit)", use_container_width=True)
    with c2:
        if os.path.exists("processed/weak_target_audit/weak_target_rasterization_qc.png"):
            st.image(Image.open("processed/weak_target_audit/weak_target_rasterization_qc.png"), caption="2D 평면 래스터화(Rasterization) 검증", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if os.path.exists("processed/target_semantics_v1/target_semantics_patches.png"):
            st.image(Image.open("processed/target_semantics_v1/target_semantics_patches.png"), caption="상위/하위 점수 샘플 패치 추출 및 의미론 분석", use_container_width=True)
    with c4:
        if os.path.exists("processed/projected_xct_support/projected_support_qc.png"):
            st.image(Image.open("processed/projected_xct_support/projected_support_qc.png"), caption="카메라 뷰에 투영된 최종 지원 영역 검증", use_container_width=True)

    st.markdown("""
    - **결론:** 이 과정을 통해 모델은 "어떤 시각적 특징이 실제 사후 XCT에서도 치명적인 결함 점수를 나타내는가"를 인과적으로 맵핑하여 학습할 수 있는 튼튼한 Weak Supervision 환경을 갖추게 되었습니다.
    """)

# 탭: 모델 아키텍처 상세
with tab_arch:
    st.header("모델 구조 시각화")
    st.markdown("프로젝트 과정에서 고안된 3가지 주요 모델의 흐름도입니다.")
    
    arch_file = "docs/model_architectures.md"
    if os.path.exists(arch_file):
        with open(arch_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        split_keyword_1 = "#### 2. 정규화 및 과적합 방지"
        split_keyword_2 = "#### 3. 가우시안 타겟 릴렉세이션"
        
        if split_keyword_1 in content and split_keyword_2 in content:
            part1, temp = content.split(split_keyword_1, 1)
            part2, part3 = temp.split(split_keyword_2, 1)
            
            st.markdown(part1)
            
            # K=4 차트 삽입
            k_path = "outputs/k_history_tradeoff.png"
            if os.path.exists(k_path):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(Image.open(k_path), caption="Performance vs. Memory Trade-off by K", use_container_width=True)
                    
            st.markdown(split_keyword_1 + part2)
            
            # 하이퍼파라미터 손실 차트 삽입
            tune_path = "outputs/hyperparameter_tuning_loss.png"
            if os.path.exists(tune_path):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(Image.open(tune_path), caption="하이퍼파라미터 튜닝 전후 검증 손실 비교", use_container_width=True)
                    
            st.markdown(split_keyword_2 + part3)
        else:
            st.markdown(content)
            
            tune_path = "outputs/hyperparameter_tuning_loss.png"
            if os.path.exists(tune_path):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(Image.open(tune_path), caption="하이퍼파라미터 튜닝 전후 검증 손실 비교", use_container_width=True)
            
    else:
        st.warning(f"{arch_file} 파일을 찾을 수 없습니다.")

# 탭 2: 모델 성능 및 결과 심층 분석
with tab_analysis:
    st.header("모델 성능 및 결과 심층 분석")
    st.markdown("""
    우리 프로젝트는 단순히 모델 결과를 나열하는 것을 넘어, **독립 변인을 철저히 통제한 실험**을 통해 Recall과 F1-Score가 어떻게, 그리고 왜 향상되었는지 합리적으로 증명합니다.
    """)
    
    st.divider()
    
    st.subheader("Step 1. [조작 변인: 공정 데이터] 두 공정의 융합은 필수적인가?")
    st.markdown("""
    - **통제 변인:** 모델 아키텍처, 평가 기준 (2mm 객체 단위) 고정
    - **실험 목적:** 파우더 도포(A)나 레이저 조사(B) 단일 이미지만 썼을 때와 융합했을 때의 성능 차이 검증
    """)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        ablation_path = "outputs/ablation_bar_chart.png"
        if os.path.exists(ablation_path):
            st.image(Image.open(ablation_path), caption="단일 공정 vs 융합 공정 성능 비교", use_container_width=True)
        else:
            st.warning("outputs/ablation_bar_chart.png 파일이 없습니다.")
    with col2:
        st.markdown("""
        **분석 결과 및 공학적 원인:**
        - **A-only (파우더 도포):** 결함을 거의 찾아내지 못함 (Recall 0%).
          > **왜?** 파우더가 평평하게 도포된 직후의 이미지(A)만으로는, 이후 레이저가 조사되며 금속이 녹고 굳는 과정에서 발생할 최종 구조적 결함(기공, 크랙 등)의 직접적인 단서를 찾기 어렵기 때문입니다.
        - **B-only (레이저 조사):** 결함을 찾긴 하지만(Recall 63%), 지나치게 예민하게 반응하여 오답이 속출 (Precision 7.8%).
          > **왜?** 레이저 조사 시 튀는 불꽃(스패터)이나 불규칙한 용융풀(B)은 결함의 강력한 징후이므로 Recall이 높습니다. 하지만 불규칙한 용융풀이 무조건 치명적 결함으로 굳어지는 것은 아니므로(스스로 치유되기도 함), 단독 사용 시 정상 상태를 결함으로 오해하는 과잉 탐지가 심하게 발생합니다.
        - **A+B Fusion (우리 모델):** 
          > **왜 B-only보다 성능(F1)이 우수한가?** 두 이미지를 융합하면 모델이 **시계열적 교차 검증**을 수행할 수 있습니다. 즉, 파우더 도포(A) 상태에서 미세한 이상이 있었던 위치에 불규칙한 용융(B)이 발생했을 때만 이를 **확실한 결함**으로 판단하게 됩니다. 이로 인해 불필요한 오답이 획기적으로 줄어들어, **F1-Score가 단일 모델 대비 약 2배(25.4%)로 수직 상승**하는 최적의 밸런스를 달성했습니다.
        """)
        
    st.divider()
    
    st.subheader("Step 2. [조작 변인: 평가 기준] 우리 모델은 정말 성능이 낮은 것일까?")
    st.markdown("""
    - **통제 변인:** 사용 모델 (최종 A+B Fusion) 완벽히 고정
    - **실험 목적:** 모델의 실제 결함 탐지 능력이 **픽셀 단위 칼채점**이라는 잘못된 평가 잣대 때문에 가려지고 있음을 증명
    """)
    
    col_plot, col_text = st.columns([1.2, 1])
    with col_plot:
        comp_path = "outputs/pixel_vs_blob_comparison.png"
        if os.path.exists(comp_path):
            st.image(Image.open(comp_path), caption="픽셀 단위 vs 객체 단위(2mm) 평가 비교", use_container_width=True)
        else:
            st.warning(f"{comp_path} 파일이 없습니다.")
    with col_text:
        st.markdown("""
        **분석 결과:**
        - **픽셀 단위 (0mm 허용):** 실제 장비와 카메라 간의 미세한 물리적 좌표 오차를 무시하고 1픽셀이라도 어긋나면 오답 처리함. 그 결과 성능이 바닥(F1 2.9%)으로 측정됨.
        - **객체 단위 (2mm 허용):** 실제 산업 현장의 오차 범위를 반영하여 반경 2mm(약 5픽셀) 내외의 정답을 인정하도록 조작 변인을 변경함.
        - **결론:** 통제 변인(모델)은 동일함에도 불구하고 평가 기준 하나만 현실적으로 수정했을 뿐인데, **Recall이 4.5%에서 43.5%로 무려 10배 폭증**함. 즉, 우리 모델은 이미 결함을 정확히 찾고 있었음을 증명.
        """)
    
    st.divider()
    
    st.subheader("Step 3. 프로젝트 결과 심층 분석 (핵심 설계 원리)")
    st.markdown("우리 프로젝트가 높은 성능을 달성할 수 있었던 핵심적인 영상 처리 기법과 모델 아키텍처 설계 의도를 상세히 분석합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Gated CBAM Attention 기반 특징 융합")
        cbam_path = "outputs/cbam_attention_layer242.png"
        if os.path.exists(cbam_path):
            st.image(Image.open(cbam_path), caption="Layer 242의 CBAM Attention Heatmap", use_container_width=True)
            
        cbam_comp_path = "outputs/cbam_fusion_comparison.png"
        if os.path.exists(cbam_comp_path):
            st.image(Image.open(cbam_comp_path), caption="CBAM Gate 도입 전후 노이즈 차단 및 오탐 감소 효과", use_container_width=True)
            
        st.markdown("""
        단순한 모델 결합의 한계
        초기 실험에서는 파우더 도포 이미지(A)와 레이저 조사 이미지(B)를 단순히 합쳐서 모델에 넣었습니다. 하지만 이럴 경우, 두 이미지 간의 의미 없는 배경 노이즈까지 함께 섞여버리면서 오히려 단일 모델보다 성능이 떨어지는 간섭 현상이 발생했습니다.
        
        해결책: CBAM
        이를 해결하기 위해 인간이 시각적으로 중요한 곳에만 집중하듯, 모델 스스로 중요한 채널과 공간에 가중치를 부여하는 CBAM을 도입했습니다. 위 히트맵에서 붉게 칠해진 영역을 보면, 모델이 부품의 형태가 없는 배경은 철저히 무시하고, 부품의 모서리나 표면 텍스처가 크게 변한 결함 의심 스팟에만 핀포인트로 강한 어텐션을 주고 있음을 알 수 있습니다.
        """)
        
    with col2:
        st.subheader("2. 가우시안 타겟 릴렉세이션 및 Support Masking")
        dist_path = "outputs/defect_distribution_2d.png"
        if os.path.exists(dist_path):
            st.image(Image.open(dist_path), caption="2D 결함 타겟 분포", use_container_width=True)
        st.markdown("""
        좌표 오차 문제
        이 프로젝트의 가장 큰 어려움 중 하나는 XCT 정답지와 2D 카메라 이미지 간의 미세한 물리적 좌표 불일치였습니다. 3D 공간을 2D 픽셀로 변환하는 과정에서 기계적 오차가 발생하기 때문에, 완벽한 1픽셀 단위의 정답을 요구하면 모델이 혼란에 빠집니다.
        
        해결책: 가우시안 블러
        해결책으로 정답 픽셀 주변에 sigma=2 수준의 가우시안 블러를 주어 공간적 관용을 부여했습니다. 즉, "정확히 이 픽셀을 맞춰라"가 아니라 "이 근방에 결함이 있다"고 가르친 것입니다.
        
        Support-Masked BCE 손실 함수
        또한 XCT 데이터가 제공되지 않는 구멍이나 빈 공간을 **정상**이라고 잘못 학습하는 것을 막기 위해, 데이터가 확실히 존재하는 부품 내부 영역에서만 오차를 계산하도록 손실 함수를 재설계하여 학습 안정성을 극대화했습니다.
        """)
        
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. 분류 모델 종합 성능 분석")
        roc_path = "outputs/roc_prc_curves.png"
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption="단일 모델 vs 퓨전 모델 분류 성능 지표 곡선", use_container_width=True)
        st.markdown("""
        단일 임계값에 의존하지 않는 객관적인 성능 비교를 위해 곡선 면적을 분석했습니다.
        
        - **ROC 곡선 (수신기 조작 특성):** 모델이 정상과 결함을 얼마나 잘 구별해내는지를 나타냅니다. A+B Fusion 모델이 B-only 모델을 뚫고 좌상단에 가깝게 위치합니다. 이는 오탐률을 극단적으로 낮추면서도 정탐률을 높게 유지하는 탁월한 분류 능력을 증명합니다.
        - **PRC 곡선 (정밀도-재현율):** 실제 불량이 압도적으로 적은 제조 데이터 환경에서는 ROC보다 PRC가 훨씬 더 보수적이고 정확한 지표입니다. PRC 곡선에서도 Fusion 모델이 압도적으로 넓은 면적(Area)을 차지하며, 현장 도입 시 가짜 경보를 울릴 확률이 가장 적고 신뢰성이 가장 높음을 확인했습니다.
        """)
        
    with col4:
        st.subheader("4. 학습 안정성 입증 (다중 시드 분산 검증)")
        seed_path = "outputs/multi_seed_boxplot.png"
        if os.path.exists(seed_path):
            st.image(Image.open(seed_path), caption="다중 무작위 시드 설정 간 성능 분산(Variance) 박스플롯", use_container_width=True)
        else:
            st.info(" 다중 시드 성능 통계 검증 앙상블 학습이 현재 백그라운드에서 진행 중입니다. (에포크가 완료되는 대로 실시간 차트가 업데이트됩니다.)")
            
        st.markdown("""
        "퓨전 모델이 우연히 운 좋게 성능이 잘 나온 것은 아닐까?" 라는 비판적 의문을 해소하기 위한 실험입니다.
        
        - **초기 가중치 통제:** 무작위 시드를 42, 100, 2026 등으로 완전히 다르게 부여하여, 모델의 초기 가중치와 배치 셔플링 순서를 초기화한 뒤 처음부터 재학습시켰습니다.
        - **통계적 유의성 확보:** 수차례의 독립적인 재학습에도 불구하고 A+B Fusion 모델의 F1-Score는 흔들림(분산)이 거의 없이 타 모델들의 한계 성능을 가볍게 상회합니다. 이는 우리의 Gated CBAM 아키텍처가 요행이 아닌, 스패터 노이즈를 스스로 차단하는 **구조적인 필터링 능력**을 갖추고 있음을 통계학적으로 강력히 입증합니다.
        """)

    st.divider()

import time

# 탭 4: 실시간 결함 모니터링
with tab4:
    st.header("테스트셋 기준 결함 시뮬레이션 결과")
    st.markdown("""
    학습된 퓨전 모델이 실제 장비가 한 층씩 프린팅을 진행할 때 찾아낸 결함들입니다.
    카메라 픽셀 좌표를 캘리브레이션 행렬을 통해 실제 3D 프린터 내부의 물리적 좌표로 변환하여 출력합니다.
    """)
    
    csv_path = "outputs/live_stream_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # tensor(203) 같은 문자열이 섞여있을 수 있으므로 숫자만 추출하여 정수형 변환
        df['layer_z'] = df['layer_z'].astype(str).str.extract(r'(\d+)').astype(int)
        
        min_layer = int(df['layer_z'].min())
        max_layer = int(df['layer_z'].max())
        
        if 'current_layer' not in st.session_state:
            st.session_state.current_layer = max_layer
        if 'is_playing' not in st.session_state:
            st.session_state.is_playing = False
            
        st.subheader(" 실시간 라이브 시뮬레이터")
        
        col_header, col_toggle = st.columns([3, 1])
        with col_header:
            st.markdown("슬라이더를 움직이거나 '라이브 재생' 버튼을 눌러 결함 탐지 과정을 실시간으로 모니터링하세요.")
        with col_toggle:
            show_cumulative = st.toggle("누적 결함 전체 보기", value=False)
        
        col_btn, col_slider = st.columns([1, 4])
        
        with col_btn:
            st.write("") # 정렬용
            button_label = " 재생 중지" if st.session_state.is_playing else " 라이브 재생 시작"
            if st.button(button_label, use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
                if st.session_state.is_playing:
                    # 끝까지 도달한 상태에서 재생을 누르면 처음부터 다시 시작
                    if st.session_state.current_layer >= max_layer:
                        st.session_state.current_layer = min_layer
                st.rerun()
                
        with col_slider:
            selected_z = st.slider(
                "현재 프린팅 층수 (Layer Z)", 
                min_value=min_layer, max_value=max_layer, 
                value=st.session_state.current_layer, step=1, key="main_layer_slider"
            )
            
        # 플레이 중일 때는 세션 상태의 값을 우선하고, 아니면 사용자가 슬라이더 조작한 값을 반영
        if st.session_state.is_playing:
            current_z = st.session_state.current_layer
        else:
            current_z = selected_z
            st.session_state.current_layer = current_z
            
        if show_cumulative:
            df_filtered = df[df['layer_z'] <= current_z]
            list_title = f"**누적 {len(df_filtered)}개**의 결함 의심 구역 발견 (1층 ~ {current_z}층)"
        else:
            df_filtered = df[df['layer_z'] == current_z]
            list_title = f"**해당 층 {len(df_filtered)}개**의 결함 의심 구역 발견 (현재 {current_z}층)"
        
        c1, c2 = st.columns([2, 1])
        with c1:
            is_empty = df_filtered.empty
            if is_empty:
                # 빈 데이터프레임일 경우 기존 px.scatter_3d 레이아웃(컬러바, 축 등)을 완벽히 유지하기 위해 더미 데이터 생성
                import pandas as pd
                plot_df = pd.DataFrame([{
                    'machine_x_mm': 0, 'machine_y_mm': 0, 'layer_z': 150,
                    'score_percent': 80, 'primary_cause': 'None'
                }])
            else:
                plot_df = df_filtered

            fig = px.scatter_3d(
                plot_df, 
                x='machine_x_mm', y='machine_y_mm', z='layer_z',
                color='score_percent', size='score_percent',
                color_continuous_scale='YlOrRd',
                range_color=[80, 100],
                title="검출된 3D 이상 후보 위치", # title이 동적으로 변하면 카메라가 리셋될 수 있으므로 정적 문자열로 고정
                labels={
                    'machine_x_mm': 'X 좌표',
                    'machine_y_mm': 'Y 좌표',
                    'layer_z': '층수',
                    'score_percent': '결함 확률'
                },
                hover_data=['primary_cause']
            )
            
            if is_empty:
                # 더미 데이터를 투명하게 만들어 화면에 보이지 않게 처리 (틀만 유지)
                fig.update_traces(marker=dict(opacity=0), hoverinfo='skip', hovertemplate=None)
                
            fig.update_layout(
                scene=dict(
                    xaxis_title='Machine X',
                    yaxis_title='Machine Y',
                    zaxis_title='Layer',
                    xaxis=dict(range=[-20, 20]),
                    yaxis=dict(range=[-20, 20]),
                    zaxis=dict(range=[150, 300]),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=4)
                ),
                uirevision='constant' # 사용자가 마우스로 조작한 카메라 시점(회전, 줌)을 업데이트 후에도 유지
            )
            # 키를 제거하여 Plotly가 업데이트 시 화면 전체를 Unmount하지 않고 자연스럽게 데이터를 교체하도록 함
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("탐지된 결함 목록")
            st.markdown(list_title)
            if not df_filtered.empty:
                df_display = df_filtered[['layer_z', 'score_percent', 'primary_cause', 'machine_x_mm', 'machine_y_mm', 'raw_image_x_px', 'raw_image_y_px']].copy()
                df_display['score_percent'] = df_display['score_percent'].apply(lambda x: f"{x:.1f}%")
                df_display['machine_x_mm'] = df_display['machine_x_mm'].apply(lambda x: f"{x:+.2f}")
                df_display['machine_y_mm'] = df_display['machine_y_mm'].apply(lambda x: f"{x:+.2f}")
                df_display['raw_image_x_px'] = df_display['raw_image_x_px'].apply(lambda x: f"{int(x)}")
                df_display['raw_image_y_px'] = df_display['raw_image_y_px'].apply(lambda x: f"{int(x)}")
                
                df_display.rename(columns={
                    'layer_z': '층수',
                    'score_percent': '확률',
                    'primary_cause': '원인',
                    'machine_x_mm': 'X(mm)',
                    'machine_y_mm': 'Y(mm)',
                    'raw_image_x_px': '픽셀 X',
                    'raw_image_y_px': '픽셀 Y'
                }, inplace=True)
                
                st.dataframe(df_display, use_container_width=True, height=500)
            else:
                st.info("현재 층수에 발견된 결함이 없습니다.")

        # 시뮬레이션 상태일 경우 다음 프레임을 위해 sleep 후 rerun
        if st.session_state.is_playing:
            if st.session_state.current_layer < max_layer:
                time.sleep(0.8) # 0.3초에서 0.8초로 간격 증가 (시각적 확인 용이)
                st.session_state.current_layer += 1
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.rerun()
            
    else:
        st.warning("실시간 스트림 결과 파일이 없습니다.")

# 탭 6: 결과 및 향후 발전 방향
with tab_conclusion:
    st.header("프로젝트 결과 요약 및 향후 과제")
    
    st.subheader("기존 목표")
    st.markdown("""
    - 사후 XCT 파괴/비파괴 검사에 절대적으로 의존하던 기존 금속 3D 프린팅 결함 검사 패러다임을 **공정 중 모니터링**으로 전환
    - **초기 접근의 한계와 목표 전환:** 기존에는 파우더 도포 후(A) 과정의 이미지만으로 불량을 검출해내려 했으나, A 모델 데이터의 한계(레이저 조사 전 상태)로 인해 A 단독으로는 결함을 짚어내지 못하는 문제가 발견되었습니다.
    - 이에 따라 파우더 도포 후(A)와 레이저 조사 후(B) 이미지를 융합하여 미세 결함 발생 위치를 **실시간(Layer 단위)으로 예측 및 시각화**하는 방향으로 프로젝트 목표를 전환하고 고도화했습니다.
    """)
    
    st.divider()
    st.subheader("현재 결과")
    st.markdown("""
    - **A+B Gated CBAM Fusion 아키텍처 개발:** 단일 이미지(A or B) 분석의 한계를 극복하고, 시계열적 교차 검증을 통해 오탐을 523회에서 111회로 획기적으로 축소했습니다.
    - **평가 기준의 현실화:** 산업 현장의 물리적 좌표 오차를 고려하여 2mm 객체 단위 평가 지표를 성공적으로 도입하였고, 그 결과 Recall이 10배 폭증(4.5% → 43.5%)하며 숨겨진 모델 성능을 증명해냈습니다.
    - **실시간 모니터링 파이프라인 완성:** 2D 픽셀 카메라 좌표를 캘리브레이션을 통한 3D 물리 좌표계로 정밀하게 맵핑하여, 엔지니어가 실시간으로 결함 의심 구역을 3D 시뮬레이션으로 모니터링할 수 있는 대시보드 시스템을 구현했습니다.
    """)
    
    st.divider()
    st.subheader("한계점 및 향후 발전 방향")
    st.markdown("""
    - **Weak Target 학습의 기하학적 매칭 한계:** XCT 정답지와 2D 레이어 이미지 간의 1픽셀 단위 완벽한 기하학적 매칭은 기계적 한계로 불가능하며, 여전히 픽셀 단위 평가 시 성능이 저평가되는 문제가 남아있습니다.
      > **발전 방향:** 차후 Vision Transformer나 자기 지도 학습 기법을 도입하여, Weak Labels 환경에서도 주변 픽셀의 맥락을 더 유연하게 이해하는 모델로 고도화할 수 있습니다.
    - **실시간 추론 속도 최적화 이슈:** 현재 모델은 정확도 극대화를 위해 무거운 CBAM 연산을 수행하고 있으나, 수많은 레이어가 존재하는 실제 공정에 실시간 탑재하기 위해서는 경량화가 필수적입니다.
      > **발전 방향:** ONNX 모델 변환, TensorRT 적용 또는 모델 Pruning 및 양자화를 통해 딥러닝 추론 속도를 ms 단위로 깎아내는 엔지니어링 최적화 작업이 필요합니다.
    - **다양한 결함 원인 분석 확장:** 현재는 결함 **확률**을 중심으로 의심 위치를 좁히는 1차 모니터링에 주력하고 있습니다.
      > **발전 방향:** 결함의 유형(크랙, 기공, 층간 분리 등)을 세부적으로 다중 분류하는 기능까지 확장하여, 엔지니어에게 구체적인 원인 분석 리포트까지 자동 제공하는 차세대 시스템으로 발전할 수 있습니다.
    """)

