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
tab1, tab_arch, tab2, tab3, tab4 = st.tabs([
    "프로젝트 소개", 
    "모델 아키텍처 상세",
    "모델 성능 검증",
    "프로젝트 결과 분석",
    "테스트셋 기준 결함 모니터링"
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

# 탭 2: 모델 성능 검증
with tab2:
    st.header("모델 성능 검증 (변인 통제 기반 실험 설계)")
    st.markdown("""
    우리 프로젝트는 단순히 모델 결과를 나열하는 것을 넘어, **독립 변인을 철저히 통제한 실험(Ablation Study)**을 통해 Recall과 F1-Score가 어떻게, 그리고 왜 향상되었는지 합리적으로 증명합니다.
    """)
    
    st.divider()
    
    st.subheader("Step 1. [조작 변인: 공정 데이터] 두 공정의 융합은 필수적인가?")
    st.markdown("""
    - **통제 변인:** 모델 아키텍처 (ResNet 기반), 평가 기준 (2mm 객체 단위) 고정
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
          > **왜 B-only보다 성능(F1)이 우수한가?** 두 이미지를 융합하면 모델이 '시계열적 교차 검증'을 수행할 수 있습니다. 즉, 파우더 도포(A) 상태에서 미세한 이상이 있었던 위치에 불규칙한 용융(B)이 발생했을 때만 이를 '확실한 결함'으로 판단하게 됩니다. 이로 인해 불필요한 오답이 획기적으로 줄어들어, **F1-Score가 단일 모델 대비 약 2배(25.4%)로 수직 상승**하는 최적의 밸런스를 달성했습니다.
        """)
        
    st.divider()
    
    st.subheader("Step 2. [조작 변인: 평가 기준] 우리 모델은 정말 성능이 낮은 것일까?")
    st.markdown("""
    - **통제 변인:** 사용 모델 (최종 A+B Fusion) 완벽히 고정
    - **실험 목적:** 모델의 실제 결함 탐지 능력이 '픽셀 단위 칼채점'이라는 잘못된 평가 잣대 때문에 가려지고 있음을 증명
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
    st.markdown("*(※ 추가적인 ROC/PRC 곡선 등 픽셀 단위 분석의 한계점은 다음 분석 탭에서 이어집니다.)*")

# 탭 3: 프로젝트 결과 분석
with tab3:
    st.header("프로젝트 결과 분석 (핵심 설계 원리)")
    st.markdown("우리 프로젝트가 높은 성능을 달성할 수 있었던 핵심적인 영상 처리 기법과 모델 아키텍처 설계 의도를 상세히 분석합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Gated CBAM Attention 기반 특징 융합")
        cbam_path = "outputs/cbam_attention_layer242.png"
        if os.path.exists(cbam_path):
            st.image(Image.open(cbam_path), caption="Layer 242의 CBAM Attention Heatmap", use_container_width=True)
        st.markdown("""
        단순한 모델 결합(Concat)의 한계
        초기 실험에서는 파우더 도포 이미지(A)와 레이저 조사 이미지(B)를 단순히 합쳐서 모델에 넣었습니다. 하지만 이럴 경우, 두 이미지 간의 의미 없는 배경 노이즈까지 함께 섞여버리면서 오히려 단일 모델보다 성능이 떨어지는 간섭 현상이 발생했습니다.
        
        해결책: CBAM (Convolutional Block Attention Module)
        이를 해결하기 위해 인간이 시각적으로 중요한 곳에만 집중하듯, 모델 스스로 중요한 채널과 공간에 가중치를 부여하는 CBAM을 도입했습니다. 위 히트맵에서 붉게 칠해진 영역을 보면, 모델이 부품의 형태가 없는 배경은 철저히 무시하고, 부품의 모서리나 표면 텍스처가 크게 변한 결함 의심 스팟(Spatter)에만 핀포인트로 강한 어텐션을 주고 있음을 알 수 있습니다.
        """)
        
    with col2:
        st.subheader("2. 가우시안 타겟 릴렉세이션 및 Support Masking")
        dist_path = "outputs/defect_distribution_2d.png"
        if os.path.exists(dist_path):
            st.image(Image.open(dist_path), caption="2D 결함 타겟 분포", use_container_width=True)
        st.markdown("""
        좌표 오차 문제 (Calibration Drift)
        이 프로젝트의 가장 큰 어려움 중 하나는 XCT 정답지와 2D 카메라 이미지 간의 미세한 물리적 좌표 불일치였습니다. 3D 공간을 2D 픽셀로 변환하는 과정에서 기계적 오차가 발생하기 때문에, 완벽한 1픽셀 단위의 정답을 요구하면 모델이 혼란에 빠집니다.
        
        해결책: 가우시안 블러 (Gaussian Target Relaxation)
        해결책으로 정답 픽셀 주변에 sigma=2 수준의 가우시안 블러를 주어 공간적 관용을 부여했습니다. 즉, "정확히 이 픽셀을 맞춰라"가 아니라 "이 근방에 결함이 있다"고 가르친 것입니다.
        
        Support-Masked BCE 손실 함수
        또한 XCT 데이터가 제공되지 않는 구멍이나 빈 공간을 '정상'이라고 잘못 학습하는 것을 막기 위해, 데이터가 확실히 존재하는 부품 내부 영역(Support)에서만 오차를 계산(Masking)하도록 손실 함수를 재설계하여 학습 안정성을 극대화했습니다.
        """)

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
            
        st.subheader("🟢 실시간 라이브 시뮬레이터")
        st.markdown("슬라이더를 움직이거나 '라이브 재생' 버튼을 눌러 결함 탐지 과정을 실시간으로 모니터링하세요.")
        
        col_btn, col_slider = st.columns([1, 4])
        
        with col_btn:
            st.write("") # 정렬용
            button_label = "⏹️ 재생 중지" if st.session_state.is_playing else "▶️ 라이브 재생 시작"
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
            
        df_filtered = df[df['layer_z'] <= current_z]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter_3d(
                df_filtered, 
                x='machine_x_mm', y='machine_y_mm', z='layer_z',
                color='score_percent', size='score_percent',
                color_continuous_scale='YlOrRd',
                range_color=[80, 100],
                title=f"검출된 3D 이상 후보 위치 (Layer: {current_z})",
                labels={
                    'machine_x_mm': 'X 좌표',
                    'machine_y_mm': 'Y 좌표',
                    'layer_z': '층수',
                    'score_percent': '결함 확률'
                },
                hover_data=['primary_cause']
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title='Machine X',
                    yaxis_title='Machine Y',
                    zaxis_title='Layer',
                    xaxis=dict(range=[-20, 20]),
                    yaxis=dict(range=[-20, 20]),
                    zaxis=dict(range=[203, 250]),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=4)
                ),
                uirevision='constant' # 사용자가 마우스로 조작한 카메라 시점(회전, 줌)을 업데이트 후에도 유지
            )
            # 키를 제거하여 Plotly가 업데이트 시 화면 전체를 Unmount하지 않고 자연스럽게 데이터를 교체하도록 함
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("탐지된 결함 목록")
            st.markdown(f"**누적 {len(df_filtered)}개**의 결함 의심 구역 발견 (현재 층: {current_z})")
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
                st.info("현재 층수까지 발견된 결함이 없습니다.")

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

