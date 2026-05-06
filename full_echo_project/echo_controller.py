import torch
import torch.nn as nn
import torch.optim as optim

STATE_DIM  = 10
HIDDEN_DIM = 16
LR_ACTOR   = 1e-3
LR_CRITIC  = 1e-3


class Actor(nn.Module):
    """state → p (0~1)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


class Critic(nn.Module):
    """state → V(s)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


class EchoController:
    """
    Actor-Critic 기반 echo 확률 컨트롤러
    AQRERM의 p = T_est / T_max 를 대체
    reward = -D_window (100 tick 평균 delivery time)
    """
    def __init__(self):
        self.actor  = Actor()
        self.critic = Critic()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 마지막 (state, p) 저장 — 100 tick 후 train에 사용
        self.last_state  = None
        self.last_action = None

    def predict(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            p = self.actor(x)
        p_val = float(p)
        self.last_state  = state
        self.last_action = p_val
        return p_val

    def train(self, reward, next_state):
        if self.last_state is None:
            return

        s  = torch.tensor(self.last_state, dtype=torch.float32)
        s_ = torch.tensor(next_state,      dtype=torch.float32)
        r  = torch.tensor(reward,          dtype=torch.float32)

        # Critic: advantage = r + V(s') - V(s)
        v      = self.critic(s)
        with torch.no_grad():
            v_next = self.critic(s_)
        advantage = r + v_next - v

        # Critic 업데이트
        critic_loss = advantage ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        p = self.actor(s)
        actor_loss = -torch.log(p + 1e-8) * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
