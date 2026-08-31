# AMMT 모델 핵심 PyTorch 구현 코드

다이어그램으로 설명해 드린 3가지 핵심 모델들이 파이썬(PyTorch) 코드로 어떻게 구현되어 있는지 핵심 클래스들을 발췌했습니다.

## 1. A-only / B-only 단일 모델 구조
조기 경보(A)와 사후 평가(B)에 각각 단독으로 사용되는 시계열 차분 모델입니다. K=4 프레임 중 과거 3프레임의 평균을 구하고, 이를 마지막 프레임(Endpoint)에서 빼서 **변화량(Difference)**을 추출해 내는 것이 핵심입니다.

```python
class AOnlyCausalTemporalDifferenceCandidateNet(nn.Module):
    def __init__(self, input_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        # 독립적인 프레임 인코더
        self.frame_encoder = nn.Sequential(
            ConvNormAct(input_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        # 차분(변화량) 인코더
        self.difference_encoder = nn.Sequential(
            ConvNormAct(base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        # 특징 융합 및 최종 디코더
        self.fusion = nn.Sequential(
            ConvNormAct(2 * base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
        )
        self.decoder = nn.Sequential(
            ConvNormAct(base_channels, base_channels),
            ConvNormAct(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward(self, history: Tensor) -> Tensor:
        # 1. 4프레임 각각 인코딩
        encoded = self.frame_encoder(history.reshape(batch * steps, channels, height, width))
        encoded_history = encoded.reshape(batch, steps, encoded.shape[1], height, width)
        
        # 2. 마지막 프레임과 과거 3프레임 평균 계산
        encoded_endpoint = encoded_history[:, -1]
        encoded_prior_mean = encoded_history[:, :-1].mean(dim=1)
        
        # 3. 핵심: 변화량(Difference) 추출
        difference_feature = encoded_endpoint - encoded_prior_mean
        encoded_difference = self.difference_encoder(difference_feature)
        
        # 4. 병합 후 디코딩하여 확률 맵(0~1) 출력
        fused_feature = self.fusion(torch.cat([encoded_endpoint, encoded_difference], dim=1))
        return torch.sigmoid(self.decoder(fused_feature))
```

---

## 2. A+B Gated Fusion 모델 (v3 최신형)
A(가루)와 B(쇳물)의 정보를 모두 사용합니다. 두 개의 브랜치(Branch)에서 따로 특징을 추출한 뒤, `GatedFusion` 모듈이 Attention 밸브를 통해 노이즈를 스스로 걸러냅니다. 최근 과적합을 잡기 위해 `nn.Dropout2d(p=0.3)`가 추가되었습니다.

```python
class GatedFusion(nn.Module):
    """공간 및 채널 단위로 A와 B 정보의 중요도를 스스로 조절하는 밸브 (Attention Gate)"""
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            ConvNormAct(2 * channels, channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=3, padding=1)
        )

    def forward(self, feat_a: Tensor, feat_b: Tensor) -> Tensor:
        concat = torch.cat([feat_a, feat_b], dim=1)
        
        # 1. 시그모이드를 통한 0~1 사이의 부드러운 차단 밸브(Gate) 맵 생성
        gates = torch.sigmoid(self.attention(concat))
        
        # 2. 추출된 특징에 밸브 값을 곱해 쓸모없는 노이즈를 0으로 소멸시킴
        return concat * gates


class ABGatedFusionRegularizedCausalTemporalDifferenceCandidateNet(nn.Module):
    def __init__(self, input_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        # 1. A와 B를 독립적으로 분석하는 두 개의 쌍둥이 브랜치
        self.branch_a = TemporalDifferenceBranch(input_channels, base_channels)
        self.branch_b = TemporalDifferenceBranch(input_channels, base_channels)
        
        # 2. 노이즈를 걸러내는 게이트 모듈
        self.gated_fusion = GatedFusion(base_channels)
        
        # 3. 과적합(Overfitting) 방지를 위한 강력한 Dropout이 적용된 디코더
        self.decoder = nn.Sequential(
            ConvNormAct(2 * base_channels, base_channels),
            nn.Dropout2d(p=0.3),  # Spatial Dropout 30%
            ConvNormAct(base_channels, base_channels),
            nn.Dropout2d(p=0.3),  # Spatial Dropout 30%
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward(self, history_a: Tensor, history_b: Tensor) -> Tensor:
        fused_a = self.branch_a(history_a)
        fused_b = self.branch_b(history_b)
        
        # 단순 병합(Concat) 대신 학습된 Gated CNN 밸브 적용
        gated_ab = self.gated_fusion(fused_a, fused_b)
        
        # 결과 출력
        return torch.sigmoid(self.decoder(gated_ab))
```
