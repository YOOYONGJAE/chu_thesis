import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta

STATE_DIM  = 10
HIDDEN_DIM = 16
LR         = 1e-3


class BetaPolicy(nn.Module):
    """state → Beta(α, β) 분포 파라미터"""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(HIDDEN_DIM, 1)
        self.beta_head  = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        h = self.shared(x)
        # softplus + 1 로 α, β > 1 보장 → unimodal Beta (학습 안정성)
        alpha = nn.functional.softplus(self.alpha_head(h)).squeeze(-1) + 1.0
        beta  = nn.functional.softplus(self.beta_head(h)).squeeze(-1)  + 1.0
        return alpha, beta


class BanditController:
    """
    Contextual Bandit 기반 echo 확률 컨트롤러
    - 정책: Beta(α(s), β(s))에서 p 샘플
    - 학습: 윈도우 동안 모은 (state, p) 쌍에 동일 reward로 policy gradient
    AQRERM의 p = T_est / T_max 를 학습형 정책으로 대체
    """
    def __init__(self):
        self.policy = BetaPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        # 100 tick 윈도우 동안 누적되는 (state, sampled_p) 쌍들
        self.buffer = []

    def predict(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            alpha, beta = self.policy(x)
            p = Beta(alpha, beta).sample()
        p_val = float(p)
        self.buffer.append((state, p_val))
        return p_val

    def train(self, reward):
        if not self.buffer:
            return

        states  = torch.tensor([s for s, _ in self.buffer], dtype=torch.float32)
        actions = torch.tensor([a for _, a in self.buffer], dtype=torch.float32)
        # Beta는 0/1에서 log_prob이 발산하므로 살짝 안쪽으로 클램프
        actions = actions.clamp(1e-4, 1.0 - 1e-4)

        alphas, betas = self.policy(states)
        log_probs = Beta(alphas, betas).log_prob(actions)

        # Bandit policy gradient: maximize r · log π(a|s)
        loss = -(log_probs * reward).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
