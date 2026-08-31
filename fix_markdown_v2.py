import re

# Read with errors='replace' to fix the bad byte
with open('AMMT_프로젝트를_처음부터_이해하기.md', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# The corruption happened at Chapter 16 and Chapter 19.
# It looks like:
# 캘리브레이션 가정 하나만 바뀌어도 결함 부품 번호가 완전히 뒤바뀌는(Part agreement 0/240) ## 32. Multi-seed...
# ...
# 1% 마진을 가장 합리적인 실무 안전장치 커트라인으로 공식 채택했다.리즘 고도화(Refinement)로 방향을 선회했다.
# 
# ---
# 
# ## 20. 현재 설정(Config)의 엄격한 변경 금지 사항

# We need to cleanly fix this block.

# First, let's restore chapter 16 to 19 correctly.
chapter_16_end = """캘리브레이션 가정 하나만 바뀌어도 결함 부품 번호가 완전히 뒤바뀌는(Part agreement 0/240) 현상을 확인했다. 따라서 현재 모델 출력의 1차적 진실(Primary Truth)은 오직 '카메라 픽셀 좌표(x_pixel, y_pixel)' 뿐이며, 물리 좌표 변환은 반드시 별도의 독립된 캘리브레이션 모듈의 검증을 거쳐야만 유효함을 확정했다.

---

# Part V. 캘리브레이션 모호성과 독립 검증 체계

## 17. 임시 캘리브레이션 (Provisional Calibration)의 한계

**연구 목표 (Objective):** 
카메라 이미지(픽셀)를 실제 프린터 물리 공간(mm)으로 변환하는 변환 행렬(Homography)을 구하고자 했다.

**검증 과정 (Validation Process):**
부품의 화면 모서리를 잡고 가능한 192가지의 거울상/회전(Mirror/Rotation) 조합을 비교했다. 그 결과 90도 회전(Rank1)과 270도 회전(Rank2) 가설이 수학적 오차율(RMSE) 면에서 동점(Tie)을 이루는 모호성(Ambiguity)을 발견했다.

**결과 및 의의 (Result & Significance):**
카메라 내부 조명이나 시각적 단서(Local photometric evidence)를 근거로 임시로 Rank2(mirror_rotate_270)를 작업용(Working) 표준으로 두었으나, 이것이 독립적이고 절대적인 물리적 진실은 아님을 분명히 했다. (이 모호성은 이후 제 36장에서 XCT 오버레이를 통해 Rank2로 최종 타결된다.)

---

## 18. NIST 메타데이터를 통한 독립 검증 시도

**연구 목표 (Objective):** 
스크린 픽셀이 아닌 공식 계측용(Metrology) 레퍼런스 데이터를 활용하여 캘리브레이션 변환을 독립적이고 객관적으로 증명하고자 했다.

**검증 과정 (Validation Process):**
NIST에서 제공하는 DotGrid 이미지, 보조 카메라(SecondaryCamera)의 붉은 레이저 기준점(Red marker), 그리고 체커보드(Checkerboard) 데이터를 검증 테이블에 올렸다.

**결과 및 의의 (Result & Significance):**
DotGrid의 패턴은 선명히 인식되었으나, 붉은 레이저 점 1개만으로는 카메라 방향축(Orientation sign)을 수학적으로 결정할 수 없었다. 체커보드 역시 금속 가루 표면의 반사광을 체커보드 모서리로 오인하는 노이즈(수백 개의 False positive)가 발생하여, 공식 메타데이터를 활용한 직접적 변환은 아직 신뢰할 수 없는 보류(Hold) 상태로 판정했다.

---

## 19. DotGrid Method #2 격자 추적 알고리즘 보류

**연구 목표 (Objective):** 
NIST Method #2 (DotGrid 기반 변환)를 활용하기 위해 카메라 속 점(Dot)들과 실제 물리적 50x50 격자판을 1:1로 매칭(Correspondence)하는 알고리즘을 개발하고자 했다.

**검증 과정 (Validation Process):**
V1 알고리즘(PCA + 1D 클러스터링)을 통해 1,518개의 점을 찾아냈으나, 미리 약속해둔 5x5 홀드아웃(Held-out) 블록을 통한 잔차(Residual) 테스트를 수행했다.

**결과 및 의의 (Result & Significance):**
점은 찾았으나 그 점이 '몇 번째 줄(Row/Col)'에 있는지 인덱싱하는 과정에서 원본 닷 피치(Dot pitch) 대비 0.41배 이상의 과도한 오차(RMSE)가 발생하여 검증 게이트를 통과하지 못했다. 이로 인해 자동 캘리브레이션 적용을 안전하게 보류(Hold)하고 정교한 2D 그래프(Graph) 기반 추적 알고리즘 고도화(Refinement)로 방향을 선회했다."""

chapter_32_33 = """## 32. Multi-seed Comparison과 Fail-Fast Safety Gate의 도입

**연구 목표 (Objective):** 
새로운 Candidate 모델(A-only Temporal Difference)이 기존 Reference 모델(C32 Temporal Residual)보다 우연이 아닌 구조적 우위로 나은 성능을 내는지 5-seed 비교를 통해 검증하고, 이 과정에서 발생할 수 있는 데이터 누락 버그를 원천 차단하고자 했다.

**검증 과정 (Validation Process):**
초기 실행 중 경로 누락(`registered_xct_v1`)으로 인해 Loss가 `None`으로 계산되는 "침묵 속의 실패" 현상을 발견했다. 이를 막기 위해 경로 검증, 빈 데이터 사전 검사, Loss=None 런타임 중단이라는 3단계 Fail-Fast 방어막을 구축했다. 이후 올바른 경로에서 5개의 랜덤 시드(1001~1005)를 부여하여 두 모델 간의 Test Loss를 정밀 교차 검증했다.

**결과 및 의의 (Result & Significance):**
첫 번째 시드(1001) 연산에서 Test Loss가 0.0696에서 0.0683으로 감소(`-0.0013`)함을 확인한 데 이어, 전체 5개 시드 평균 오차 감소치 `-0.001989`를 달성하며 Candidate 모델이 5전 5승(전승)을 기록했다. 랜덤 가중치 초기화에 흔들리지 않고 **A-only Temporal Difference 모델의 우월함이 100% 증명(Sign consistency 1.0)** 됨에 따라 공식 뼈대를 전면 교체했다.

---

## 33. Margin-based Withholding Audit (안전 마진 시뮬레이션)

**연구 목표 (Objective):** 
1등 후보와 2등 후보 간의 점수 격차가 너무 적은 모호한(Near-tie) 상황일 때 해당 경보를 안전하게 기각(Withhold)하기 위한 최적의 '안전 마진(Safety Margin)' 임계값을 결정하고자 했다.

**검증 과정 (Validation Process):**
Temporal Difference 모델이 도출한 48개 테스트 레이어의 예측 맵을 분석하여 1, 2등 간 평균 점수 차이(Margin: 0.0353)와 물리적 거리(Peak Distance: 276.43 픽셀)를 측정했다. 이후 1%, 2%, 5%의 세 가지 마진 커트라인을 가상으로 적용하여 각각 몇 개의 레이어가 기각되는지 시뮬레이션했다.

**결과 및 의의 (Result & Significance):**
5% 마진(0.05)을 적용할 경우 전체 레이어의 81.25%(39/48)가 기각되어 시스템 기능이 마비됨을 확인했다. 반면 **1% 마진(0.01)**을 적용할 경우 점수 경합이 가장 불안정한 하위 25.0%(12/48)의 레이어만 선별적으로 차단할 수 있었다. 이에 따라 1% 마진을 가장 합리적인 실무 안전장치 커트라인으로 공식 채택했다."""

# Regex to find the corrupted section between "캘리브레이션 가정 하나만 바뀌어도 결함 부품 번호가 완전히 뒤바뀌는(Part agreement 0/240)"
# and "## 20. 현재 설정(Config)의 엄격한 변경 금지 사항"
start_str = "캘리브레이션 가정 하나만 바뀌어도 결함 부품 번호가 완전히 뒤바뀌는(Part agreement 0/240)"
end_str = "## 20. 현재 설정(Config)의 엄격한 변경 금지 사항"

start_idx = text.find(start_str)
end_idx = text.find(end_str)

if start_idx != -1 and end_idx != -1:
    fixed_text = text[:start_idx] + chapter_16_end + "\n\n---\n\n" + end_str + text[end_idx+len(end_str):]
    
    # Now we need to make sure chapters 32 and 33 are properly placed.
    # Where should they go? Before Chapter 34.
    ch34_str = "## 34. 시계열 정보량의 독립적 평가: A-only vs B-only 베이스라인"
    ch34_idx = fixed_text.find(ch34_str)
    
    # We first remove the corrupted chapter 32 and 33 if they are still lingering somewhere.
    # Actually, we completely replaced them because they were jammed in chapter 16-19.
    # But wait, are the OLD chapter 32 and 33 still there? 
    # Let's check if "## 32. Multi-seed Comparison과 Fail-Fast Safety Gate의 도입" exists in the text AFTER our replacement.
    
    # We will just replace everything from "## 32." to "## 34."
    ch32_old_str = "## 32. Multi-seed Comparison과 Fail-Fast Safety Gate의 도입"
    ch32_idx = fixed_text.find(ch32_old_str)
    
    if ch32_idx != -1 and ch34_idx != -1 and ch32_idx < ch34_idx:
        fixed_text = fixed_text[:ch32_idx] + chapter_32_33 + "\n\n---\n\n" + fixed_text[ch34_idx:]
    else:
        # If it wasn't there, insert before ch34
        fixed_text = fixed_text[:ch34_idx] + chapter_32_33 + "\n\n---\n\n" + fixed_text[ch34_idx:]
    
    with open('AMMT_프로젝트를_처음부터_이해하기.md', 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    print("Fixed corruption successfully.")
else:
    print("Could not find boundaries for corruption.")

