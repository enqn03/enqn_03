import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# 페이지 기본 설정
st.set_page_config(page_title="AMMT Defect Detection Dashboard", layout="wide", page_icon="🏭")

# 헤더
st.title("🏭 AMMT 실시간 이상 후보 위치 탐지")
st.markdown("### LPBF 공정 중 Layer-Camera 기반 결함 조기 탐지 AI 시스템")
st.markdown("**[5조] TEAM 3Do | 김상민, 김태학, 이주현, 정미연**")

st.divider()

# 탭 생성
tab1, tab2, tab3 = st.tabs([
    "📖 프로젝트 소개 (Overview)", 
    "🚨 실시간 결함 모니터링 (Live Stream)", 
    "🧠 핵심 설계 의도 및 성능 (Rationale)"
])

# 탭 1: 프로젝트 소개
with tab1:
    st.header("1. 문제 정의 및 목표")
    st.markdown("""
    **LPBF(Laser Powder Bed Fusion)** 기반 금속 3D 프린팅 공정은 한 번에 제품을 만드는 것이 아니라, 수백~수천 개의 레이어를 쌓아올립니다.
    기존에는 제품이 완성된 후 XCT(X-ray CT)로 내부를 비파괴 검사해야만 결함을 알 수 있어 막대한 시간과 원자재 손실이 발생했습니다.
    
    본 프로젝트는 **고정된 Layer-Camera 이미지(파우더 도포 후, 레이저 조사 후)만으로 내부 XCT 결함을 공정 중에 사전에 예측**하는 AI 모델을 개발했습니다.
    """)
    
    st.header("2. A+B Gated CBAM Fusion 아키텍처")
    st.markdown("""
    - **독립적 특징 추출:** 파우더 도포 후(A-stage)와 레이저 조사 후(B-stage) 이미지를 각각 독립된 인코더로 분석
    - **Temporal Difference:** 단순히 현재 이미지만 보는 것이 아니라, 과거 3프레임 평균과 현재의 **'차이(변화량)'**를 명시적으로 계산
    - **CBAM Attention:** 채널과 공간 어텐션을 통해 A와 B 중 어떤 특징이 치명적인지 스스로 판단하여 융합
    """)

# 탭 2: 실시간 결함 모니터링
with tab2:
    st.header("실시간 결함 시뮬레이션 결과")
    st.markdown("""
    학습된 퓨전 모델이 실제 장비가 한 층씩 프린팅을 진행할 때 찾아낸 결함들입니다.
    카메라 픽셀 좌표를 캘리브레이션 행렬(Homography)을 통해 **실제 3D 프린터 내부의 물리적 좌표(X, Y mm)**로 변환하여 출력합니다.
    """)
    
    csv_path = "outputs/live_stream_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plotly 3D Scatter 차트 생성
            fig = px.scatter_3d(
                df, 
                x='machine_x_mm', y='machine_y_mm', z='layer_z',
                color='score_percent', size='score_percent',
                color_continuous_scale='YlOrRd',
                title="🔍 검출된 3D 이상 후보 위치 분포 (Interactive)",
                labels={
                    'machine_x_mm': 'X 좌표 (mm)',
                    'machine_y_mm': 'Y 좌표 (mm)',
                    'layer_z': '층수 (Layer Z)',
                    'score_percent': '결함 확률 (%)'
                },
                hover_data=['primary_cause']
            )
            fig.update_layout(scene=dict(
                xaxis_title='Machine X (mm)',
                yaxis_title='Machine Y (mm)',
                zaxis_title='Layer (Z)'
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("📋 탐지된 결함 목록")
            st.markdown(f"총 **{len(df)}개**의 결함 의심 구역 발견")
            # 보기 좋게 포맷팅
            df_display = df[['layer_z', 'score_percent', 'primary_cause', 'machine_x_mm', 'machine_y_mm']].copy()
            df_display['score_percent'] = df_display['score_percent'].apply(lambda x: f"{x:.1f}%")
            df_display['machine_x_mm'] = df_display['machine_x_mm'].apply(lambda x: f"{x:+.2f}")
            df_display['machine_y_mm'] = df_display['machine_y_mm'].apply(lambda x: f"{x:+.2f}")
            
            df_display.rename(columns={
                'layer_z': 'Layer',
                'score_percent': '확률',
                'primary_cause': '원인',
                'machine_x_mm': 'X (mm)',
                'machine_y_mm': 'Y (mm)'
            }, inplace=True)
            
            st.dataframe(df_display, use_container_width=True, height=500)
    else:
        st.warning("⚠️ 실시간 스트림 결과 파일(`outputs/live_stream_results.csv`)을 찾을 수 없습니다. 터미널에서 `live_inference_stream.py`를 먼저 실행해주세요.")

# 탭 3: 모델 설계 의도 및 성능
with tab3:
    st.header("엔지니어링 의사결정 (Design Rationale)")
    st.markdown("단순히 모델을 가져다 쓴 것이 아니라, **데이터의 한계를 극복하기 위해 치열하게 고민한 흔적**입니다.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("#### 1. K=4 시계열 선택의 이유\n"
                "과거 프레임을 너무 길게 잡으면 예전의 노이즈가 현재를 덮어버리는 **Temporal Collapse** 현상이 발생했습니다. "
                "모델이 절대적인 이미지를 외우지 않고 직전 3개 층과의 **'변화량'**에만 집중하도록 의도적으로 시야를 좁혔습니다.")
        
        st.success("#### 2. 가우시안 타겟 (Gaussian Target)\n"
                   "카메라와 XCT 사이의 미세한 물리적 좌표 오차를 인정하고, 점(Point)이 아닌 "
                   "`sigma=2`의 넓은 가우시안 분포를 타겟으로 주어 모델에게 공간적 관용(Spatial Tolerance)을 부여했습니다.")
        
    with col_b:
        st.warning("#### 3. Support-Masked BCE와 극단적 가중치\n"
                   "XCT 데이터가 아예 없는 빈 공간을 '정상'이라고 거짓말하지 않기 위해, 데이터가 확실히 존재하는 영역만 학습시켰습니다. "
                   "또한 전체의 1.7%에 불과한 치명적 불량을 잡기 위해 양성 타겟에 `pos_weight=100`의 가중치를 걸었습니다.")
        
        st.error("#### 4. Gated CBAM Fusion\n"
                 "파우더 도포 이미지와 레이저 조사 이미지를 단순 합치면 강한 신호가 약한 신호를 묻어버립니다. "
                 "따라서 CBAM(어텐션)을 달아주어 지금 이 결함이 **'어느 공정에서 기인했는지' 모델 스스로 가중치를 저울질**하도록 만들었습니다.")

    st.divider()
    
    st.header("성능 평가 및 산업적 가치")
    st.markdown("""
    우리의 Ground Truth는 완벽한 픽셀 마스크가 아닌 대략적인 덩어리(Weak Target)입니다. 따라서 픽셀 단위로 칼채점을 하는 
    F1-Score 지표는 낮게 나올 수밖에 없습니다.
    
    하지만 **객체 단위(Blob-level)**로 평가하면 완전히 이야기가 달라집니다. 
    듬성듬성한 XCT 데이터로만 학습시켰음에도, **숨겨진 치명적 결함 덩어리의 약 30%(Recall 28.2%)를 정확히 타격**하고 있습니다. 
    이는 실제 산업 현장에서 공정 중 불량 알람을 띄우는 용도로 충분한 가능성을 입증한 결과입니다.
    """)
    
    roc_path = "outputs/roc_prc_curves.png"
    if os.path.exists(roc_path):
        st.image(Image.open(roc_path), caption="픽셀 단위 평가 곡선 (보수적 기준의 정량 지표)", use_container_width=False, width=600)
