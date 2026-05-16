import math
import torch
import torch.nn as nn
import torch.optim as optim

NODE_STATE_DIM   = 10   # 노드별 state (EchoController, BanditController 용)
GLOBAL_STATE_DIM = 16   # 네트워크 글로벌 state (LTrainController 용)
HIDDEN_DIM       = 32
LR_ACTOR         = 1e-3
LR_CRITIC        = 1e-3

# 하위 호환을 위해 STATE_DIM 이름도 NODE_STATE_DIM 으로 유지
STATE_DIM = NODE_STATE_DIM


class Actor(nn.Module):
    """state → p (0~1). state_dim 인자로 입력 차원 지정 가능 (기본 노드 10차원)."""
    def __init__(self, state_dim=NODE_STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


class Critic(nn.Module):
    """state → V(s). state_dim 인자로 입력 차원 지정 가능 (기본 노드 10차원)."""
    def __init__(self, state_dim=NODE_STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
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


class CategoricalActor(nn.Module):
    """
    state → logits over action_dim (각 L 후보에 대한 비정규화 점수).
    softmax 는 사용 지점(Categorical 분포)에서 적용해서 수치적으로 더 안전하게.
    state_dim 인자로 입력 차원 지정 가능 (기본 노드 10차원, LTrainController 는 글로벌 16차원).
    """
    def __init__(self, action_dim, state_dim=NODE_STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# AQRERM_L_TRAIN: EchoController 대신 LTrainController 사용
class LTrainController:
    """
    Categorical policy 기반 L 컨트롤러 (글로벌 state 입력 버전).

    구조:
      global_state (16차원)  →  Linear(16→32) → ReLU → Linear(32→4)  →  logits
      logits  →  softmax  →  P(L=0), P(L=1), P(L=2), P(L=3)
      샘플링  →  action_idx  →  L = ACTIONS[action_idx]

    학습 (REINFORCE with critic baseline):
      advantage = reward + V(s') - V(s)
      actor_loss = -log P(선택된 action | s) * advantage
      critic_loss = advantage ** 2

    글로벌 state 는 simulator 가 매 stat_interval 마다 계산해서 set_window_L 로 주입.
    한 window (=100 tick) 동안 모든 노드가 같은 L 을 공유하고, window 끝의 reward 로
    그 L 선택을 평가해 actor 정책을 학습.

    인터페이스:
      set_window_L(global_state, current_tick) — simulator 가 stat_interval 끝에 호출
      predict(state, current_tick)            — 노드가 매 라우팅마다 호출 (cached_L 반환만)
      train(reward)                           — simulator 가 reward 와 함께 학습 트리거
    """
    ACTIONS = [0, 1, 3]   # L 후보 (사용자 지정)
    WINDOW  = 100             # L 을 새로 결정하는 주기 (tick) — stat_interval 과 맞춤

    def __init__(self):
        self.actor  = CategoricalActor(action_dim=len(self.ACTIONS),
                                       state_dim=GLOBAL_STATE_DIM)
        self.critic = Critic(state_dim=GLOBAL_STATE_DIM)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 학습용 (s, a, s') 추적 — 글로벌 state 기준
        self.prev_state        = None   # 직전 window 시작 시점의 state (학습의 s)
        self.latest_state      = None   # 현재 window 시작 시점의 state (학습의 s')
        self.prev_action_idx   = None   # prev_state 에서 선택했던 action 의 인덱스

        # window 단위 L 캐시
        self.cached_L          = self.ACTIONS[0]
        self.cached_action_idx = 0
        self.last_window_idx   = -1   # -1 로 시작 → 첫 set_window_L 호출에서 무조건 새 window 인식

        # 외부 plot 용 — 두 가지 분포를 동시에 추적
        self.last_L           = self.ACTIONS[0]
        self.L_history        = []   # per-routing: 매 라우팅 결정마다 cached_L 누적
        self.L_window_history = []   # per-window: 새 window 시작 시점에 결정된 L 한 번씩 누적

        # 첫 window (tick 0~WINDOW-1) 동안 사용할 L 을 dummy zero state 로 한 번 샘플
        # → random init 가중치에서 사실상 균등 분포로 뽑힘, window 0 의 baseline L 역할
        with torch.no_grad():
            dummy = torch.zeros(GLOBAL_STATE_DIM, dtype=torch.float32)
            logits = self.actor(dummy)
            distribution = torch.distributions.Categorical(logits=logits)
            action_idx = int(distribution.sample().item())
        self.cached_L          = self.ACTIONS[action_idx]
        self.cached_action_idx = action_idx
        # window 0 의 L 을 per-window history 의 첫 entry 로 기록
        self.L_window_history.append(self.cached_L)

    def set_window_L(self, global_state, current_tick):
        """
        시뮬레이터가 매 stat_interval 마다 호출해서 다음 window 의 L 을 결정.
        - 새 window 진입이면: 글로벌 state 로 actor 호출 → action 샘플링 → cached_L 갱신
        - 동시에 학습용 latest_state 갱신 (다음 train 의 s' 역할)
        - prev_state 가 아직 None 이면 (= 첫 호출) 같은 state 로 부트스트랩

        주의: 시뮬레이터는 train() 호출보다 먼저 이 메서드를 호출해야 합니다.
        그래야 train 의 advantage 계산에 fresh 한 latest_state 가 반영됨.
        """
        window_idx = current_tick // self.WINDOW
        if window_idx != self.last_window_idx:
            safe_state = [v if math.isfinite(v) else 0.5 for v in global_state]
            x = torch.tensor(safe_state, dtype=torch.float32)
            with torch.no_grad():
                logits = self.actor(x)
                distribution = torch.distributions.Categorical(logits=logits)
                action_idx = int(distribution.sample().item())
            L = self.ACTIONS[action_idx]

            self.cached_L          = L
            self.cached_action_idx = action_idx
            self.last_window_idx   = window_idx
            self.latest_state      = safe_state
            self.L_window_history.append(L)   # per-window: 새 window 의 L 한 번 기록

            # 첫 호출이면 prev 도 같은 state 로 부트스트랩 → train 의 첫 호출에서 안전
            if self.prev_state is None:
                self.prev_state      = safe_state
                self.prev_action_idx = action_idx

    def predict(self, state, current_tick):
        """
        노드가 매 라우팅 결정마다 호출. 현재는 cached_L 만 반환하고 L_history 누적.
        실제 L 선택(sampling)은 simulator 가 set_window_L 로 별도 처리.
        state 인자는 하위 호환 위해 받지만 사용 안 함.
        """
        self.last_L = self.cached_L
        self.L_history.append(self.cached_L)
        return self.cached_L

    def train(self, reward):
        if (self.prev_state is None or self.latest_state is None
                or self.prev_action_idx is None):
            return
        s  = torch.tensor(self.prev_state,   dtype=torch.float32) # 이전 상태 s
        s_ = torch.tensor(self.latest_state, dtype=torch.float32) # 다음 상태 s'
        r  = torch.tensor(reward, dtype=torch.float32) # simulator.py 에서 계산된 reward (score 변화량 기반)

        # Critic: advantage = r + V(s') - V(s)
        v = self.critic(s)
        with torch.no_grad():
            v_next = self.critic(s_)
        # r + v_next = 현재 action 선택으로 인해 얻은 보상과 다음 상태의 가치 → 이 값이 현재 상태의 가치 v 보다 크면 advantage > 0 → 행동이 개선된 것으로 간주
        # v = 현재 상태의 가치 → 이 값이 크면 현재 상태가 좋음.
        # r이 양수면 개선된 상태. 
        # 그 r에 다음 상태의 가치 v_next 를 더해서, 현재 상태의 가치 v 와 비교.
        # 비교 결과가 양수면 행동이 개선된 것.
        advantage = r + v_next - v        # 현재 코드는 γ가 없습니다.

        # Critic 업데이트입니다.
        # Critic의 목적은 advantage를 0에 가깝게 만드는 것입니다.
        critic_loss = advantage ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor (REINFORCE with critic baseline):
        # 현재 상태를 입력으로 받아서 각 행동에 대한 로짓을 출력. 
        # logits의 타입은 torch.Tensor로서, 각 행동에 대한 비정규화된 점수입니다. 
        # 예를 들어, ACTIONS가 [0, 1, 2, 3] 이고 logits가 [0.1, 0.5, -0.2, 0.3] 이라면, 행동 1이 가장 높은 점수를 가지고 있습니다.
        logits = self.actor(s) 
        # 그 로짓들을 CategoricalActor가 softmax를 내부적으로 적용하여 행동 선택 확률 분포로 변환합니다.
        # 예를 들어, 위의 logits에 softmax가 적용되면 음수 logit은 낮은 확률로, 양수 logit은 높은 확률로 변환됩니다.
        distribution = torch.distributions.Categorical(logits=logits)
        # log_prob는 predict에서 선택된 이전의 action_idx에 해당하는 행동의 로그 확률입니다.
        # log_prob는 항상 음수!!!! 이거나 0임. 예를 들어 0.999는 log_prob가 -0.001, 0.5는 log_prob가 -0.693, 0.1은 log_prob가 -2.302입니다.
        log_prob = distribution.log_prob(torch.tensor(self.prev_action_idx))

        # 선택한 행동의 로그 확률에 advantage를 곱함. 
    
        # 우선 actor_loss가 음수이든 양수이든 간에, 옵티마이저는 actor_loss를 줄이는 방향으로 업데이트를 진행합니다.
        # 여기서 "줄인다"는 것은 0으로 보낸다는 뜻이 아니라, 수직선 기준으로 더 작은 값이 되게 한다는 뜻입니다.
        # 예를 들어 actor_loss가 0.9이면 0.5, 0.2 쪽으로 줄이고, actor_loss가 -0.9이면 -1.2, -1.6 쪽으로 줄입니다.
        #
        # advantage가 양수면 선택한 행동이 예상보다 좋은 결과를 낸 것이므로
        # actor_loss는 -log_prob * 양수 형태가 됩니다.
        # 이때 log_prob는 선택된 action의 확률에 log를 씌운 값이고, 확률은 0~1 사이이므로 log_prob는 보통 음수입니다.
        # 따라서 -log_prob는 양수가 되고, actor_loss도 양수가 됩니다.
        #
        # 이 경우 옵티마이저는 actor_loss를 줄이는 방향으로 업데이트를 진행합니다.
        # 그리고 이 업데이트는 선택된 행동의 logit을 높이는 방향으로 작용합니다.
        # 왜냐하면 log_prob는 그 안에 들어 있는 선택 확률이 크면 클수록 0에 가까워지고,
        # log_prob가 0에 가까워질수록 -log_prob가 작아지며, 그에 따라 actor_loss도 작아지기 때문입니다.
        # 즉 좋은 결과를 낸 action은 다음에 더 높은 확률로 선택되도록 조정됩니다.
        #
        # 반대로 advantage가 음수면 선택한 행동이 예상보다 나쁜 결과를 낸 것이므로
        # actor_loss는 -log_prob * 음수 형태가 됩니다.
        # log_prob는 음수이고, -log_prob는 양수이므로, 여기에 음수 advantage가 곱해져 actor_loss는 음수가 됩니다.
        #
        # 마찬가지로 옵티마이저는 actor_loss를 줄이는 방향으로 업데이트를 진행합니다.
        # 이 경우는 선택된 행동의 logit을 낮추는 방향으로 작용합니다.
        # 왜냐하면 선택된 action의 확률이 낮아질수록 log_prob는 더 작은 음수가 되고,
        # advantage가 음수인 상황에서는 actor_loss도 더 작은 값이 되기 때문입니다.
        # 즉 나쁜 결과를 낸 action은 다음에 더 낮은 확률로 선택되도록 조정됩니다.
        
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 다음 train 을 위한 shift: prev := latest
        self.prev_state      = self.latest_state
        self.prev_action_idx = self.cached_action_idx


class LCloseController(LTrainController):
    """
    적응형 제어 규칙 기반 L 컨트롤러 (신경망 학습 X, hysteresis state machine).

    매 stat_interval 마다 (d_window, total_qlen, pending_count) 를 받아:
      1. 각 지표의 정상 baseline 을 EMA 로 온라인 추정
           D_base ← 0.95 * D_base + 0.05 * D_window  (qlen_base, pending_base 동일)
      2. baseline 대비 증가율로 instability score 계산
           instability = W_ADT  * ADT_증가율
                      + W_QLEN * 큐_증가율
                      + W_PEND * pending_증가율
      3. hysteresis state machine 으로 stressed flag 갱신
           - 진입: instability > ENTER_THRESHOLD 가 ENTER_PERSISTENCE 회 연속
           - 해제: instability < EXIT_THRESHOLD 이고 pending_delta <= 0 이 EXIT_PERSISTENCE 회 연속
      4. stressed=True → L = L_STRESS(0), 아니면 L = L_NORMAL(3)

    부모 LTrainController 의 actor/critic 은 인프라 호환 위해 상속하지만 L 결정엔
    사용 안 함. train() 은 no-op 으로 오버라이드. 부모의 set_window_L 도 오버라이드
    해서 global_state 무시하고 stressed flag 만 보고 cached_L 갱신.

    baseline EMA 는 stressed=False 일 때만 갱신 — 정상 상태의 평균만 학습해야
    이상 상태와 명확히 분리되어 진단되기 때문 (stress 중 baseline 이 따라가면
    "stress 가 정상"으로 학습돼 해제 불가능해짐).
    """
    # instability score 가중치 (가중치 합 ≈ 임계와 비교될 스케일)
    W_ADT     = 1.0
    W_QLEN    = 1.0
    W_PENDING = 1.0

    # hysteresis 임계 — 진입은 높게, 해제는 낮게 (dead zone 보장)
    ENTER_THRESHOLD = 0.3   # 가중합 증가율 30% 초과 → 진입 후보
    EXIT_THRESHOLD  = 0.1   # 10% 이하 → 해제 후보

    # 지속 카운트 (진입은 빠르게, 해제는 신중하게)
    ENTER_PERSISTENCE = 2   # 2 window 연속 → 진입
    EXIT_PERSISTENCE  = 4   # 4 window 연속 → 해제

    # baseline EMA 감쇠 계수: 0.95 * prev + 0.05 * current
    BASE_EMA_ALPHA = 0.05

    # L 값
    L_NORMAL = 3   # 평상시 사용
    L_STRESS = 0   # stress 진입 시 사용

    def __init__(self):
        super().__init__()
        # 온라인 학습되는 baseline (첫 호출에 첫 관측치로 초기화)
        self.D_base       = None
        self.qlen_base    = None
        self.pending_base = None
        # pending_delta 계산용
        self.prev_pending_count = 0
        # hysteresis state machine
        self.stressed       = False
        self.enter_counter  = 0
        self.exit_counter   = 0
        # 초기 cached_L 은 NORMAL (부모의 dummy sampling 결과를 덮어씀)
        self.cached_L          = self.L_NORMAL
        self.cached_action_idx = 0   # 사용 안 하지만 부모 호환 위해 채워둠
        # 부모가 L_window_history 에 dummy sample 결과를 넣었으니 NORMAL 로 교체
        self.L_window_history = [self.L_NORMAL]

    def set_window_L(self, global_state, current_tick):
        """
        부모와 시그니처 동일 — simulator 가 호출.
        global_state 는 무시하고, 이미 update_stress() 가 갱신한 stressed flag 만 보고
        cached_L 을 L_STRESS 또는 L_NORMAL 로 설정.
        """
        window_idx = current_tick // self.WINDOW
        if window_idx != self.last_window_idx:
            self.cached_L        = self.L_STRESS if self.stressed else self.L_NORMAL
            self.last_window_idx = window_idx
            self.L_window_history.append(self.cached_L)   # per-window 기록

    def update_stress(self, d_window, total_qlen, pending_count):
        """
        시뮬레이터가 매 stat_interval 마다 set_window_L 보다 먼저 호출.
        baseline EMA + instability score + hysteresis state machine 로 stressed 갱신.
        """
        # 첫 호출은 baseline 만 초기화 (아직 비교 대상이 없음)
        if self.D_base is None:
            self.D_base             = max(d_window,     1e-6)
            self.qlen_base          = max(total_qlen,   1e-6)
            self.pending_base       = max(pending_count, 1e-6)
            self.prev_pending_count = pending_count
            return

        # 1. baseline 대비 증가율 — "지금 값이 평소보다 얼마나 큰가" 비율
        d_rate       = (d_window      - self.D_base)       / max(self.D_base,       1e-6)
        qlen_rate    = (total_qlen    - self.qlen_base)    / max(self.qlen_base,    1e-6)
        pending_rate = (pending_count - self.pending_base) / max(self.pending_base, 1e-6)

        # 2. instability score — 가중합
        instability = (self.W_ADT     * d_rate
                     + self.W_QLEN    * qlen_rate
                     + self.W_PENDING * pending_rate)

        # 3. pending_delta — 해제 조건의 추가 안전장치 (회복 추세 확인)
        pending_delta = pending_count - self.prev_pending_count
        self.prev_pending_count = pending_count

        # 4. hysteresis state machine
        if not self.stressed:
            # 진입 검사 — instability 가 ENTER 임계 이상으로 지속되는지
            if instability > self.ENTER_THRESHOLD:
                self.enter_counter += 1
                if self.enter_counter >= self.ENTER_PERSISTENCE:
                    self.stressed      = True
                    self.enter_counter = 0
                    self.exit_counter  = 0
            else:
                self.enter_counter = 0
        else:
            # 해제 검사 — instability 가 EXIT 임계 이하 AND pending 이 안 늘고 있을 때
            if instability < self.EXIT_THRESHOLD and pending_delta <= 0:
                self.exit_counter += 1
                if self.exit_counter >= self.EXIT_PERSISTENCE:
                    self.stressed      = False
                    self.enter_counter = 0
                    self.exit_counter  = 0
            else:
                self.exit_counter = 0

        # 5. baseline EMA 갱신 — stressed=False 일 때만 (이상 상태가 baseline 오염하지 않게)
        if not self.stressed:
            a = self.BASE_EMA_ALPHA
            self.D_base       = (1 - a) * self.D_base       + a * d_window
            self.qlen_base    = (1 - a) * self.qlen_base    + a * total_qlen
            self.pending_base = (1 - a) * self.pending_base + a * pending_count

    def train(self, reward):
        """
        규칙 기반 컨트롤러라 actor/critic 학습 안 함 — 부모의 train 을 no-op 으로 차단.
        호출되어도 아무 일도 일어나지 않아 안전.
        """
        return
