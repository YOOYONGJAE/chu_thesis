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

        # self.last_state  = None
        # self.last_action = None
        self.prev_state   = None   # 학습에 사용할 이전 상태 s
        self.latest_state = None   # 다음 상태 s'        

    def predict(self, state):
        # torch.tensor : 다차원 배열을 생성하는 함수. state 리스트를 PyTorch 텐서로 변환
        x = torch.tensor(state, dtype=torch.float32) 
        # with 블록의 뜻 : 기본적인 의미는 해당 블록의 특징 설정. 
        # 여기서는 torch.no_grad()로 감싸서 블록 내에서는 자동 미분 기능이 꺼짐. 
        # 즉, 이 블록에서 수행되는 연산은 그래디언트 계산에 포함되지 않음.
        with torch.no_grad():
            p = self.actor(x) # ACTOR 가 state를 입력받아 p값을 예측
        p_val = float(p) # 예측된 p값을 Python float로 변환하여 반환하기 전에 저장

        # self.last_state  = state
        # self.last_action = p_val
        self.latest_state = state
        if self.prev_state is None:
            self.prev_state = state        

        return p_val

    def train(self, reward):
        # if self.last_state is None:
        #     return

        # s  = torch.tensor(self.last_state, dtype=torch.float32)
        # s_ = torch.tensor(next_state,      dtype=torch.float32)

        if self.prev_state is None or self.latest_state is None:
            return
    
        s  = torch.tensor(self.prev_state,   dtype=torch.float32)
        s_ = torch.tensor(self.latest_state, dtype=torch.float32) 

        r  = torch.tensor(reward,          dtype=torch.float32)

        # Critic: advantage = r + V(s') - V(s)
        v      = self.critic(s)
        with torch.no_grad():
            v_next = self.critic(s_)

        # Advantage 계산: 현재 상태에서의 가치와 다음 상태에서의 가치를 비교하여 행동의 이점을 평가
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

        # 학습이 끝난 후, 현재 상태를 이전 상태로 업데이트하여 다음 학습에 사용
        self.prev_state = self.latest_state
