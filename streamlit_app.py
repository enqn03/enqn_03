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
tab1, tab_arch, tab2, tab3, tab4, tab5 = st.tabs([
    "프로젝트 소개", 
    "모델 아키텍처 상세",
    "모델 성능 비교",
    "이미지 생성 원리 갤러리",
    "실시간 결함 모니터링", 
    "단일 이미지 결함 테스트"
])

# 탭 1: 프로젝트 소개
with tab1:
    st.header("1. 문제 정의 및 목표")
    st.markdown("""
    LPBF 기반 금속 3D 프린팅 공정은 한 번에 제품을 만드는 것이 아니라, 수백~수천 개의 레이어를 쌓아올립니다.
    기존에는 제품이 완성된 후 XCT로 내부를 비파괴 검사해야만 결함을 알 수 있어 막대한 시간과 원자재 손실이 발생했습니다.
    
    본 프로젝트는 고정된 Layer-Camera 이미지(파우더 도포 후, 레이저 조사 후)만으로 내부 XCT 결함을 공정 중에 사전에 예측하는 AI 모델을 개발했습니다.
    """)
    
    st.header("2. A+B Gated CBAM Fusion 아키텍처")
    st.markdown("""
    - 독립적 특징 추출: 파우더 도포 후(A-stage)와 레이저 조사 후(B-stage) 이미지를 각각 독립된 인코더로 분석
    - Temporal Difference: 단순히 현재 이미지만 보는 것이 아니라, 과거 3프레임 평균과 현재의 차이를 명시적으로 계산
    - CBAM Attention: 채널과 공간 어텐션을 통해 A와 B 중 어떤 특징이 치명적인지 스스로 판단하여 융합
    """)

# 탭: 모델 아키텍처 상세
with tab_arch:
    st.header("모델 구조 시각화 (Architecture Diagrams)")
    st.markdown("프로젝트 과정에서 고안된 3가지 주요 모델의 흐름도입니다.")
    
    arch_file = "docs/model_architectures.md"
    if os.path.exists(arch_file):
        with open(arch_file, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.warning(f"{arch_file} 파일을 찾을 수 없습니다.")

# 탭 2: 모델 성능 비교
with tab2:
    st.header("모델 성능 비교 (A-only vs B-only vs A+B Fusion)")
    st.markdown("""
    단일 공정 이미지만을 사용했을 때와 두 이미지를 융합했을 때의 성능 차이를 정량적으로 비교합니다.
    - A-only 모델: 파우더 도포 이미지만 관찰 (조기 경보에는 유리하지만, 레이저 조사로 인한 최종 결함을 놓침)
    - B-only 모델: 레이저 조사 이미지만 관찰 (최종 상태는 확인 가능하지만 도포 불량을 놓침)
    - A+B Fusion 모델: 두 정보를 융합하여 테스트 손실을 획기적으로 낮추고 안정성을 확보!
    """)
    
    ablation_path = "outputs/ablation_bar_chart.png"
    if os.path.exists(ablation_path):
        st.image(Image.open(ablation_path), caption="단일 모델과 융합 모델의 Test Loss 비교", use_container_width=False, width=800)
    else:
        st.warning("outputs/ablation_bar_chart.png 파일이 없습니다.")
        
    st.info("""
    왜 픽셀 단위 지표는 낮고, 객체 단위 성능은 높을까요?
    우리 정답지는 완벽한 픽셀 마스크가 아닌 XCT에서 추출한 대략적인 덩어리입니다. 
    따라서 픽셀 단위로 칼채점하면 오차가 크게 발생하지만, 결함 덩어리를 맞췄는지로 평가하면 약 30%의 치명적 결함을 성공적으로 잡아냅니다.
    """)
    roc_path = "outputs/roc_prc_curves.png"
    if os.path.exists(roc_path):
        st.image(Image.open(roc_path), caption="픽셀 단위 평가 곡선", use_container_width=False, width=600)

# 탭 3: 이미지 생성 원리 갤러리
with tab3:
    st.header("이미지 생성 원리 갤러리")
    st.markdown("우리 프로젝트에서 생성된 히트맵과 마스크 이미지들이 왜 이렇게 생겼는지, 어떤 의미를 가지는지 해설합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Gated CBAM Attention 히트맵")
        cbam_path = "outputs/cbam_attention_layer242.png"
        if os.path.exists(cbam_path):
            st.image(Image.open(cbam_path), caption="Layer 242의 CBAM Attention Heatmap", use_container_width=True)
        st.markdown("""
        Q: 왜 히트맵이 부품 모서리나 특정 영역에만 진하게 표시될까요?
        A: 모델이 파우더와 레이저 이미지를 단순 합치지 않고, 어텐션 메커니즘을 통해 이상 징후가 강한 영역에만 가중치를 주기 때문입니다. 이는 모델이 공정의 물리적 차이를 제대로 이해하고 있다는 증거입니다.
        """)
        
    with col2:
        st.subheader("2. 가우시안 타겟과 서포트 마스크")
        dist_path = "outputs/defect_distribution_2d.png"
        if os.path.exists(dist_path):
            st.image(Image.open(dist_path), caption="2D 결함 타겟 분포", use_container_width=True)
        st.markdown("""
        Q: 왜 타겟을 점이 아니라 둥근 덩어리로 퍼뜨렸나요?
        A: 카메라 픽셀과 실제 3D 프린터 기계 사이에는 미세한 좌표 오차가 존재합니다. 정확한 픽셀 하나를 맞추라고 강요하면 학습이 무너집니다. 따라서 sigma=2의 가우시안 블러를 주어 공간적 관용을 부여했습니다.
        
        Q: Support-Masked BCE란?
        A: XCT 데이터가 없는 텅 빈 영역을 '정상'이라고 모델에게 거짓말하지 않기 위해, 데이터가 확실히 존재하는 영역에서만 오차를 계산했습니다.
        """)

# 탭 4: 실시간 결함 모니터링
with tab4:
    st.header("실시간 결함 시뮬레이션 결과")
    st.markdown("""
    학습된 퓨전 모델이 실제 장비가 한 층씩 프린팅을 진행할 때 찾아낸 결함들입니다.
    카메라 픽셀 좌표를 캘리브레이션 행렬을 통해 실제 3D 프린터 내부의 물리적 좌표로 변환하여 출력합니다.
    """)
    
    csv_path = "outputs/live_stream_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter_3d(
                df, 
                x='machine_x_mm', y='machine_y_mm', z='layer_z',
                color='score_percent', size='score_percent',
                color_continuous_scale='YlOrRd',
                title="검출된 3D 이상 후보 위치 분포",
                labels={
                    'machine_x_mm': 'X 좌표',
                    'machine_y_mm': 'Y 좌표',
                    'layer_z': '층수',
                    'score_percent': '결함 확률'
                },
                hover_data=['primary_cause']
            )
            fig.update_layout(scene=dict(
                xaxis_title='Machine X',
                yaxis_title='Machine Y',
                zaxis_title='Layer'
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("탐지된 결함 목록")
            st.markdown(f"총 {len(df)}개의 결함 의심 구역 발견")
            df_display = df[['layer_z', 'score_percent', 'primary_cause', 'machine_x_mm', 'machine_y_mm']].copy()
            df_display['score_percent'] = df_display['score_percent'].apply(lambda x: f"{x:.1f}%")
            df_display['machine_x_mm'] = df_display['machine_x_mm'].apply(lambda x: f"{x:+.2f}")
            df_display['machine_y_mm'] = df_display['machine_y_mm'].apply(lambda x: f"{x:+.2f}")
            
            df_display.rename(columns={
                'layer_z': '층수',
                'score_percent': '확률',
                'primary_cause': '원인',
                'machine_x_mm': 'X 좌표',
                'machine_y_mm': 'Y 좌표'
            }, inplace=True)
            
            st.dataframe(df_display, use_container_width=True, height=500)
    else:
        st.warning("실시간 스트림 결과 파일이 없습니다.")

# 탭 5: 단일 이미지 결함 테스트
with tab5:
    st.header("단일 이미지 결함 테스트")
    
    st.error("""
    데모 모드 한계 명시
    실제 우리가 학습시킨 딥러닝 퓨전 모델은 정확한 결함 탐지를 위해 과거 4장의 시계열 이미지와 파우더/레이저 6채널 영상을 동시에 요구합니다.
    본 화면은 임의의 단일 이미지 1장만 업로드했을 때, 이미지 처리 휴리스틱을 통해 부품 내에서 결함일 확률이 가장 높은 곳을 찾아 빨간 점을 찍어주는 웹 시연용 간소화 시뮬레이터입니다.
    """)
    
    uploaded_file = st.file_uploader("AMMT 공정 사진을 업로드하세요", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        
        # 데모 시뮬레이션: 가장 비정상적인 픽셀 찾기
        img_array = np.array(img.convert("L")) # 흑백 변환
        
        # 간단한 블러 처리로 노이즈 제거
        blurred = gaussian_filter(img_array, sigma=3)
        
        # 이미지의 중간 50% 영역 안에서만 찾기
        h, w = blurred.shape
        margin_h, margin_w = h // 4, w // 4
        roi = blurred[margin_h:h-margin_h, margin_w:w-margin_w]
        
        # 가장 어두운 픽셀 찾기
        min_y_roi, min_x_roi = np.unravel_index(np.argmin(roi), roi.shape)
        
        # 원본 이미지 좌표로 복원
        target_y = min_y_roi + margin_h
        target_x = min_x_roi + margin_w
        
        # 빨간 점과 테두리 그리기
        draw = ImageDraw.Draw(img)
        r = 15 # 점의 반지름
        
        # 빨간색 채워진 원
        draw.ellipse((target_x - r, target_y - r, target_x + r, target_y + r), fill='red', outline='white', width=2)
        # 주변을 감싸는 박스
        box_r = 40
        draw.rectangle((target_x - box_r, target_y - box_r, target_x + box_r, target_y + box_r), outline='red', width=3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("결함 후보 탐지 결과")
            st.image(img, caption="빨간 점: 가장 유력한 이상 픽셀", use_container_width=True)
            
        with col2:
            st.subheader("예측 리포트")
            st.success(f"탐지된 결함 좌표: (X: {target_x}, Y: {target_y})")
            st.info("이 위치는 입력된 이미지 내에서 주변 대비 가장 이질적인 밝기 값을 갖는 곳으로 추정됩니다.")
            st.markdown("""
            > 실제 시스템 작동 방식
            > 실제 파이프라인에서는 이러한 2D 픽셀 좌표가 캘리브레이션 행렬을 거쳐 프린터 물리 좌표로 변환된 후 작업자에게 알람으로 전송됩니다.
            """)
