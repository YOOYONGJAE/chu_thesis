import math
import random
from collections import deque
# from echo_controller import EchoController


# =====================================================================
# PFE (Point Full Echo) 알고리즘 상수
# - 노드 별 total_point (Full Echo 사용 예산) 의 초기값
# - 그 외 gr, B_max, C 는 self.params 에서 읽음 (main 스크립트에서 override 가능)
# =====================================================================
PFE_TOTAL_POINT_INITIAL = 0.0   # 시작 포인트. 0 이면 처음엔 Full Echo 불가, 모은 뒤 발동


# =====================================================================
# T_max 가속 감쇠 (위로 볼록한 감소 곡선)
#  - 매 tick: T_max ← max(T_est, T_max - coef · age_since_peak · T_max)
#  - peak 갱신 시 age_since_peak 리셋 → 직후엔 0 (감쇠 멈춤),
#    시간이 지날수록 per-tick 감쇠량이 age 에 비례해 커짐.
#  - 적분하면 drop ≈ (coef/2) · age² · T_max — age 의 2차 함수 (concave 감소).
#  - peak 가 진짜 위기였다면 T_est 가 빠르게 따라잡아 갱신 → age 리셋 되어
#    decay 가 다시 0 부터 시작. 한 번 찍힌 옛 peak 만 점점 빨리 잊는다.
#  - aqrerm_c_tdec / pfe_tdec 변형에서만 활성화.
# =====================================================================
TMAX_DECAY_COEF_DEFAULT = 1e-7  # ticks^-2 단위, params['tmax_decay_coef'] 로 override 가능


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _signed_ratio01(x, scale):
    """0 → 0.5, 양수 → >0.5, 음수 → <0.5 (tanh 기반)"""
    if scale == 0:
        return 0.5
    return 0.5 + 0.5 * math.tanh(x / scale)


class Packet:
    def __init__(self, src, dst, created_at):
        self.src = src
        self.dst = dst
        self.created_at = created_at
        self.queue_entry_tick = created_at  # 현재 큐 진입 tick (큐 이동마다 갱신) : 의미는 "이 tick까지 이 노드 큐에 있었음"
        self.route_memory = []              # 방문 노드 리스트 (AQRERM용)


class Node:
    def __init__(self, node_id, neighbors, algorithm, params, num_nodes):
        self.id = node_id
        self.neighbors = neighbors          # 인접 노드 ID 리스트
        self.algorithm = algorithm
        self.params = params

        self.queue = deque()               # 처리 대기 패킷
        self.incoming = []                 # 이번 tick 도착 패킷 (다음 tick에 queue로 이동)

        # Q 테이블: Q[dst][nbr] = 예상 전달 시간, 초기값 1.0
        self.Q = {
            d: {n: 1.0 for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }

        self.T_est = 0.0
        self.T_max = 1.0
        # T_max 가속 감쇠용 — peak 갱신 후 경과 tick 수
        self.age_since_peak = 0
        # decay 활성 여부 (변형 알고리즘 한정)
        self.tmax_decay_enabled = algorithm in (
            'aqrerm_c_tdec', 'pfe_tdec', 'aqrerm_c05_l0_tdec'
        )
        # decay 시작 tick — 0 이면 시뮬레이션 시작부터, memory_cut_tick 이면 절단 시점부터
        # aqrerm_c05_l0_tdec 만 link-cut 시점부터 Tdec 활성, 나머지는 0 (즉시)
        self.tmax_decay_start_tick = (
            params.get('memory_cut_tick', 0)
            if algorithm == 'aqrerm_c05_l0_tdec' else 0
        )

        # PFE per-tick 포인트 적립 활성 여부 (변형 한정)
        # 활성 시 simulator tick 루프가 모든 노드에 tick_accumulate_point() 호출 →
        # 큐가 비어 라우팅 안 하는 tick 에도 gr 만큼 적립이 진행됨.
        # 라우팅 함수 본문에서는 적립을 빼서 중복 적립 방지.
        self.tick_accum_enabled = algorithm in (
            'aqpace',
            'aqpace_l0',
            'aqpace_then_aqrerm',
            'aqpace_no_queue',
            'aqpace_no_pre_no_queue',
            'aqpace_no_pre',
            'aqpace_no_L',
        )

        # ΔQ_min: 목적지별 직전 Q_min 저장
        self.prev_Q_min = {d: 1.0 for d in range(num_nodes) if d != node_id}

        # TD_error_ema: TD 에러의 지수이동평균
        self.td_error_ema = 0.0
        self.td_ema_alpha = 0.1  # EMA 감쇠 계수

        # route_switching_recent: 최근 500 tick 내 y* 변경 기록 [(tick, y_star), ...]
        self.y_star_history = deque()

        # echo_age_avg: 목적지 d, 이웃 n별 마지막 echo tick
        self.last_echo_tick = {
            d: {n: 0 for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }

        # AQRERM_c: echo 응답 시 받은 이웃 큐 길이 캐시 (실시간 직접 읽기 대체)
        self.last_known_queue = {n: 0 for n in neighbors}

        # aqrerm_c_pre: 목적지별 이전 y* (random echo 시 무조건 포함)
        self.last_y_star = {}

        # PFE: 누적 포인트 (Full Echo 사용 예산). 시작값은 모듈 상수 PFE_TOTAL_POINT_INITIAL.
        self.total_point = PFE_TOTAL_POINT_INITIAL

        # PFE 진단 카운터 — simulator 가 stat_interval 시점에 읽고 0 으로 리셋
        # full_echo_ratio = pfe_window_full_echo_count / pfe_window_route_count
        # 포인트 게이트가 부하에 잘 반응해 열리는 빈도를 시간축으로 추적
        self.pfe_window_full_echo_count = 0
        self.pfe_window_route_count     = 0

        # AdE 진단 카운터 — Adv > 0 이벤트 (= fresh 정보로 y* 보다 더 좋은 이웃 발견) 추적
        # 라우팅 한 번 안에서 최선의 비-y_star (= 가장 작은 score_n) 만 기록
        self.pfe_window_adv_event_count  = 0   # Adv > 0 발생 라우팅 수
        self.pfe_window_adv_sum          = 0.0 # max Adv (= score_y - 최선 score_n) 누적합
        self.pfe_window_score_y_sum      = 0.0 # Adv 이벤트 시 score_y_star 누적합
        self.pfe_window_score_n_best_sum = 0.0 # Adv 이벤트 시 최선 score_n 누적합

        # AdE eta_n 통계 — 비-y_star 학습률 분포 (cap 도달 비율 / 평균 / 분산용 sum)
        self.pfe_window_eta_n_count      = 0   # eta_n 적용 횟수 (= 비-y_star update 수)
        self.pfe_window_eta_n_sum        = 0.0
        self.pfe_window_eta_n_sq_sum     = 0.0 # variance 계산용 (E[X^2])
        self.pfe_window_eta_n_clip_count = 0   # eta_n == eta cap 도달 횟수

        # y_star switching — 같은 dst 에 대한 y_star 가 이전 라우팅과 달라지면 카운트
        self.prev_y_star            = {}       # dst -> 직전 y_star
        self.pfe_window_switch_count = 0

    # -------------------------------------------------------------------------
    # T_est 업데이트 (AQFE / AQRERM)
    # T_est = 모든 목적지에 대해 min_y Q[d][y] 의 평균 (AQRERM 정의)
    # -------------------------------------------------------------------------
    def update_T_est(self):
        if not self.Q or not self.neighbors:
            self.T_est = 0.0
            return
        self.T_est = sum(
            min(self.Q[d][n] for n in self.neighbors)
            for d in self.Q
        ) / len(self.Q)
        if self.T_est > self.T_max:
            self.T_max = self.T_est
            # peak 갱신 시 가속 감쇠의 age 도 리셋 — 새 peak 직후엔 decay 0
            self.age_since_peak = 0

    # -------------------------------------------------------------------------
    # T_max 가속 감쇠 — 매 tick 모든 노드에 호출 (simulator tick 루프).
    # tmax_decay_enabled=False 인 노드는 즉시 return (기존 동작 유지).
    # 라우팅 안 하는 노드도 호출되어야 옛 peak 가 잊혀짐 — 이게 high-watermark 문제 해소의 핵심.
    # -------------------------------------------------------------------------
    def tick_decay_tmax(self, current_tick):
        if not self.tmax_decay_enabled:
            return
        # 시작 tick 이전엔 age 증가도 안 하고 decay 도 안 함 — start 시점부터 0 에서 출발
        if current_tick < self.tmax_decay_start_tick:
            return
        self.age_since_peak += 1
        if self.T_max > self.T_est:
            coef = self.params.get('tmax_decay_coef', TMAX_DECAY_COEF_DEFAULT)
            decrement = coef * self.age_since_peak * self.T_max
            self.T_max = max(self.T_est, self.T_max - decrement)

    # -------------------------------------------------------------------------
    # PFE per-tick 포인트 적립 — 매 tick 모든 노드에 호출 (simulator tick 루프).
    # tick_accum_enabled=False 인 노드는 즉시 return.
    # 라우팅 호출 여부와 무관하게 큐가 비어 있어도 적립 진행 →
    # 한가한 노드도 풀에코 예산을 천천히 축적 가능.
    # 활성 변형의 라우팅 함수는 본문에서 적립을 제거하여 중복 방지.
    # -------------------------------------------------------------------------
    def tick_accumulate_point(self, current_tick):
        if not self.tick_accum_enabled:
            return
        gr    = self.params.get('pfe_gr', 0.133)
        b_max = self.params.get('pfe_b_max', 0.5)
        self.total_point = min(b_max, self.total_point + gr)

    # -------------------------------------------------------------------------
    # 이 노드에서 dst까지의 최선 추정치 반환
    # exclude_node: 이 노드를 이웃 후보에서 제외 (Route Memory용)
    # -------------------------------------------------------------------------
    def best_estimate(self, dst, exclude_node=None):
        if dst == self.id:
            return 0.0
        if dst not in self.Q:
            return float('inf')
        candidates = {
            n: self.Q[dst][n]
            for n in self.neighbors
            if n != exclude_node
        }
        if not candidates:
            # exclude 후 후보가 비면 fallback: exclude_node 도 포함해서 다시 시도
            # (dead-end 노드 — 실제 라우팅도 같은 fallback 으로 작동하므로 일관됨)
            # 이 한 줄이 Q 테이블의 inf 폭주를 차단해서 평균/분산 통계까지 보호함
            candidates = {n: self.Q[dst][n] for n in self.neighbors}
        if not candidates:
            return float('inf')
        return min(candidates.values())

    # -------------------------------------------------------------------------
    # 라우팅: 패킷을 받아 다음 홉 반환, Q 테이블 업데이트
    # -------------------------------------------------------------------------
    def route(self, packet, current_tick, all_nodes):
        if self.algorithm == 'q_routing':
            return self._route_q(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqfe':
            return self._route_aqfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe':
            return self._route_pfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm':
            return self._route_aqrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_l0':
            # AQRERM + memory_cut_tick 이후 L=0 강제 (link_cut 시나리오용)
            return self._route_aqrerm_l0(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_no_mem':
            return self._route_aqrerm_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_no_L':
            # alias: AQRERM + Route Memory 완전 비활성 (aqrerm_no_mem 와 동일)
            return self._route_aqrerm_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_4000_no_L':
            # alias: AQRERM + memory_cut_tick 이후 L=0 (main_compare_PFE_link_cut.py 의
            # CUT_TICK=4000 기준 명명. 실제 전환 시점은 params['memory_cut_tick'] 값을 따름.)
            return self._route_aqrerm_l0(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c':
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_tdec':
            # AQRERM_c=0.5 + T_max 가속 감쇠 — 라우팅 로직은 동일, T_max 관리만 다름
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_ade':
            # AQRERM_c + Advantage-weighted eta2 (AdE)
            # echo_set 안의 비-y_star 만 차등 학습률, echo_set 밖은 기존 동작
            return self._route_aqrerm_c_ade(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_ade_l0':
            # AQRERM_c_AdE + memory_cut_tick (보통 7000) 이후 L=0 강제 (link_cut 시나리오용)
            return self._route_aqrerm_c_ade_l0(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_tdec':
            # PFE + T_max 가속 감쇠 — 라우팅 로직은 동일, T_max 관리만 다름
            return self._route_pfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_c':
            # PFE + AQRERM_c 식 큐 페널티 (c_q · last_known_queue) — params['c'] 그대로 사용
            return self._route_pfe_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_c03':
            # PFE_c 변형 — c=0.3 강제 (AQRERM_c03 와 동일한 override 패턴)
            return self._route_pfe_c03(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_c05_l0':
            # PFE_c=0.5 + memory_cut_tick (보통 7000) 이후 L=0 강제
            return self._route_pfe_c_l0(packet, current_tick, all_nodes, 0.5)
        elif self.algorithm == 'pfe_c03_l0':
            # PFE_c=0.3 + memory_cut_tick (보통 7000) 이후 L=0 강제
            return self._route_pfe_c_l0(packet, current_tick, all_nodes, 0.3)
        elif self.algorithm == 'pfe_c_ade':
            # PFE_c + Advantage-weighted eta2 (AdE)
            # Full Echo 직후 fresh Score 차이로 비-y_star 학습률 차등 적용
            return self._route_pfe_c_ade(packet, current_tick, all_nodes)
        
            # ★ PFE_c_pre_echo: PFE_c_AdE 의 변형 — echo 와 노드 선정의 순서를 뒤집음
        elif self.algorithm == 'aqpace_route_accum':
            # [AQPACE ablation] tick 기반 포인트 적립 → 라우팅 호출 시에만 적립으로 변경.
            # 포인트 게이트 / 선에코 / c·큐 페널티 / Route Memory 는 AQPACE 와 동일.
            # 패킷이 없어 라우팅 안 한 tick 에는 포인트가 쌓이지 않는 전신 버전.
            # 실험 의도: tick 단위 균등 적립(AQPACE)이 라우팅 빈도 기반 적립보다 유리한지 확인.
            return self._route_pfe_c_pre_echo(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_route_accum_no_L':
            # [AQPACE ablation] aqpace_route_accum + link_cut 시점(보통 7000 tick) 이후 L=0 강제.
            # 실험 의도: 링크 절단 시나리오에서 Route Memory 없이 라우팅 호출 기반 적립의 적응 속도 확인.
            return self._route_pfe_c_pre_echo_l0(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace':
            # [AQPACE 풀버전] 포인트 게이트 + 선에코(echo→선정→학습) + c·큐 페널티 + Route Memory(L).
            # tick_accumulate_point() 로 simulator 가 매 tick 모든 노드에 포인트 균등 적립.
            return self._route_aqpace(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_no_queue':
            # [AQPACE ablation] c·큐 페널티 항(c·queue_n) 제거.
            # y_star 선정 시 Score = t_n 만 사용 (큐 혼잡 정보 미반영).
            # 포인트 게이트 / 선에코 / Route Memory / tick 적립은 AQPACE 와 동일.
            # 실험 의도: 이웃 큐 길이 정보가 수렴 속도 및 SS ADT 개선에 얼마나 기여하는지 확인.
            return self._route_pfe_pre_echo_tick(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_no_pre_no_queue':
            # [AQPACE ablation] 선에코 순서 변경(select-first) + c·큐 페널티 제거.
            # stale Q 로 먼저 y_star 선정 → 포인트 충분 시 echo 실행 (AQPACE 의 반대 순서).
            # 큐 페널티도 없으므로 AQPACE 대비 두 요소 동시 제거.
            # 실험 의도: 선에코 순서와 큐 페널티를 모두 제거했을 때의 성능 하한 측정.
            return self._route_pfe_echo_tick(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_no_pre':
            # [AQPACE ablation] 선에코 순서만 변경(select-first), c·큐 페널티는 유지.
            # stale Q + c·last_known_queue 로 먼저 y_star 선정 → 포인트 충분 시 echo.
            # AQPACE 는 echo → fresh Score → 선정 순이지만 이 변형은 선정 → echo 순.
            # 실험 의도: 에코를 먼저 해 fresh 정보로 선정하는 것(AQPACE)의 우위 여부 확인.
            return self._route_pfe_c_echo_tick(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_pre':
            # AQRERM_c 의 pre-echo 변형 — 랜덤 echo (이전 y* 무조건 포함) 먼저 →
            # fresh t + c · fresh queue 로 y* 선택 → 차등 update.
            return self._route_aqrerm_c_pre(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_pre':
            # AQRERM 의 pre-echo 변형 (c 없음) — 랜덤 echo (이전 y* 보장) →
            # fresh t 만으로 y* 선택 → 차등 update.
            return self._route_aqrerm_pre(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqpace_no_point':
            # [AQPACE ablation] 포인트 게이트 시스템 완전 제거.
            # 매 라우팅마다 무조건 Full Echo 실행 (포인트 잔액 무관).
            # 선에코 순서 / c·큐 페널티 / Route Memory 는 AQPACE 와 동일.
            # 실험 의도: 포인트 게이트(에코 빈도 제어)가 없을 때 echo 비용 대비 성능 확인.
            #   echo 비용이 0인 현재 모델에서는 AQPACE 와 성능이 같거나 더 나아야 함 —
            #   차이가 있다면 포인트 게이트의 간접 효과(학습 속도 조절)를 의심할 수 있음.
            return self._route_fe_c_pre_echo(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_no_L':
            # [AQPACE ablation] Route Memory 완전 비활성 (L=0 항상 강제).
            # 포인트 게이트 / 선에코 / c·큐 페널티 / tick 적립은 AQPACE 와 동일.
            # 실험 의도: Route Memory(최근 L 홉 방문 노드 제외)가 루프 억제 및
            #   수렴 속도에 기여하는 정도 확인.
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_aqpace(packet, current_tick, all_nodes)
            finally:
                self.params['L'] = original_L

        elif self.algorithm == 'aqpace_l0':
            # [링크 절단 시나리오용] AQPACE + link_cut 시점(보통 7000 tick) 이후 L=0 강제.
            # 실험 의도: 링크 절단 후 Route Memory 가 대체 경로 재학습을 방해하는지 확인.
            return self._route_aqpace_l0(packet, current_tick, all_nodes)

        elif self.algorithm == 'aqpace_then_aqrerm':
            # [하이브리드] switch_tick 이전엔 AQPACE 로 초기 학습, 이후엔 AQRERM 으로 전환.
            # Q 테이블은 그대로 유지 → AQPACE 가 채운 추정치를 AQRERM 이 이어받아 사용.
            # 실험 의도: AQPACE 의 초기 수렴 속도 + AQRERM 의 안정 운영을 결합 시
            #   각 단계의 장점이 실제로 조합되는지 확인.
            switch_tick = self.params.get('switch_tick', 15000)
            if current_tick < switch_tick:
                return self._route_aqpace(packet, current_tick, all_nodes)
            else:
                return self._route_aqrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_c_ade_l0':
            # PFE_c_AdE + memory_cut_tick (보통 7000) 이후 L=0 강제
            # link_cut 시나리오용 — c 는 params['c'] (main_link_cut.py 에서 0.5) 그대로
            return self._route_pfe_c_ade_l0(packet, current_tick, all_nodes)
        elif self.algorithm == 'pfe_c01_ade_l0':
            # PFE_c_AdE_L0 변형 — c=0.1 강제
            return self._route_pfe_c_ade_l0(packet, current_tick, all_nodes, 0.1)
        elif self.algorithm == 'pfe_c10_ade_l0':
            # PFE_c_AdE_L0 변형 — c=1.0 강제
            return self._route_pfe_c_ade_l0(packet, current_tick, all_nodes, 1.0)
        elif self.algorithm == 'aqrerm_c_7000_no_mem':
            return self._route_aqrerm_c_7000_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_all_no_mem':
            return self._route_aqrerm_c_all_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_7000_no_c':
            return self._route_aqrerm_c_7000_no_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_7000_one_c':
            return self._route_aqrerm_c_7000_one_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_low_c':
            return self._route_aqrerm_c_low_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c03':
            return self._route_aqrerm_c03(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c07':
            return self._route_aqrerm_c07(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c_high_c':
            return self._route_aqrerm_c_high_c(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_c01_l0':
            return self._route_aqrerm_c_l0(packet, current_tick, all_nodes, 0.1)
        elif self.algorithm == 'aqrerm_c03_l0':
            return self._route_aqrerm_c_l0(packet, current_tick, all_nodes, 0.3)
        elif self.algorithm == 'aqrerm_c05_l0':
            return self._route_aqrerm_c_l0(packet, current_tick, all_nodes, 0.5)
        elif self.algorithm == 'aqrerm_c05_l0_tdec':
            # c=0.5 + 7000 부터 L=0 + 7000 부터 Tdec — 라우팅은 동일 헬퍼, T_max 관리만 다름
            return self._route_aqrerm_c_l0(packet, current_tick, all_nodes, 0.5)
        elif self.algorithm == 'aqrerm_c07_l0':
            return self._route_aqrerm_c_l0(packet, current_tick, all_nodes, 0.7)
        elif self.algorithm in ('aqrerm_c_l_train', 'aqrerm_c_l_close'):
            return self._route_aqrerm_c_l_train(packet, current_tick, all_nodes)
        elif self.algorithm in ('learned_aqrerm', 'bandit_aqrerm'):
            return self._route_learned_aqrerm(packet, current_tick, all_nodes)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _route_q(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])

        q = current_tick - packet.queue_entry_tick
        s = 1
        t = all_nodes[y_star].best_estimate(dst)

        self.Q[dst][y_star] += eta * (q + s + t - self.Q[dst][y_star])

        return y_star

    def _route_aqfe(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']

        self.update_T_est()

        # 모든 이웃의 t 값 수집 (Full Echo)
        t_values = {n: all_nodes[n].best_estimate(dst) for n in self.neighbors}

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = (self.T_est / self.T_max) * k if self.T_max > 0 else 0.0

        # 선택된 이웃: eta로 업데이트
        self.Q[dst][y_star] += eta * (q + s + t_values[y_star] - self.Q[dst][y_star])

        # 나머지 이웃: eta2로 업데이트
        for n in self.neighbors:
            if n != y_star:
                self.Q[dst][n] += eta2 * (q + s + t_values[n] - self.Q[dst][n])

        return y_star

    # -------------------------------------------------------------------------
    # PFE (Point Full Echo) — 포인트 예산 기반 Full Echo 게이트
    #
    # R_x(t)        = T_est / T_max   (현재 노드 불안정도, 0~1)
    # G_x(t)        = gr * R_x(t)     (이번 이벤트에서 적립할 포인트)
    # total_point  ← min(B_max, total_point + G_x(t))
    #
    # 매 라우팅 결정에서 — total_point >= C 면 Full Echo (AQFE 식, total_point -= C),
    # total_point <  C 면 Q-routing (y* 만 단일 업데이트).
    # 즉 echo 선택지가 2가지 (all-or-nothing), 확률적 부분 echo 없음.
    #
    # y* 는 단순 argmin Q (큐 페널티 X, route memory X) — 사용자 지정.
    # eta2 는 AQRERM 논문식 (T_est/T_max) * k (k=0.5).
    # gr / B_max / C 는 params 에서 읽어 (없으면 기본값) — main 스크립트 튜닝 가능.
    # -------------------------------------------------------------------------
    def _route_pfe(self, packet, current_tick, all_nodes):
        # 0. 진단: 윈도우 라우팅 호출 카운트 (Full Echo 비율 분모)
        self.pfe_window_route_count += 1

        # 1. 현재 패킷의 목적지
        dst = packet.dst

        # 2. Q값 갱신에 쓰는 기본 학습률
        eta = self.params['eta']

        # 3. Full Echo에서 선택되지 않은 이웃들을 얼마나 강하게 갱신할지 정하는 계수
        k = self.params['k']

        # 4. PFE 포인트 적립률
        #    R_x = T_est / T_max 만큼 계산된 위험도에 gr을 곱해서 포인트를 적립함
        gr = self.params.get('pfe_gr', 0.1)

        # 5. 포인트 잔고의 최대치
        b_max = self.params.get('pfe_b_max', 5.0)

        # 6. Route Memory 크기 (AQRERM 와 동일하게 params['L'] 사용)
        #    visited 필터링으로 warmup 의 ping-pong 차단
        L = self.params['L']

        # 7. 현재 노드의 T_est, T_max 갱신
        self.update_T_est()

        # 8. Full Echo 1회 사용 비용 — 동적 가격
        #    sale_pt = 1 - R_x 이라 위기 상태(R_x → 1)일수록 Full Echo 싸짐.
        #    T_est=T_max 인 경우 0 이 되어 Full Echo 무한 발동하는 corner case 보호 (최저 0.1)
        sale_pt = 1.0 - (self.T_est / self.T_max) if self.T_max > 0 else 1.0
        if sale_pt == 0:
            sale_pt = 0.1
        c_pt = self.params.get('pfe_c', sale_pt)
        # c_pt = self.params.get('pfe_c', 0.1)

        # 9. 현재 상태가 과거 최대 지연 추정치 대비 얼마나 나쁜지 계산 (R_x)
        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # 10. 포인트 적립
        # self.total_point = min(b_max, self.total_point + gr * R_x)
        self.total_point = min(b_max, self.total_point + gr)

        # 11. Route Memory: 방문한 노드 제외한 후보로 y_star 선택
        #     모든 이웃이 visited 면 fallback (전체 이웃 사용 — AQRERM 와 동일 패턴)
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(candidates, key=lambda n: self.Q[dst][n])

        # 12. 현재 패킷이 이 노드 큐에서 기다린 시간
        q = current_tick - packet.queue_entry_tick

        # 13. 링크 전송 시간. 현재 시뮬레이터에서는 1 tick 으로 고정
        s = 1

        # 14. 포인트가 충분하면 Full Echo 수행
        if self.total_point >= c_pt:
            self.total_point -= c_pt
            # 진단: 이번 윈도우에서 Full Echo 가 발동한 횟수 (분자)
            self.pfe_window_full_echo_count += 1

            # Full Echo 에서 선택되지 않은 이웃을 갱신할 때 쓰는 보조 학습률
            eta2 = R_x * k

            # 모든 이웃에게 "너를 통해 dst 까지 가면 남은 시간이 얼마냐?" 를 물어봄.
            # exclude_node=self.id 로 자기 자신은 답에서 빼게 함 (route memory 의 두 번째 효과).
            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}

            # 실제로 선택한 이웃 y_star 는 기본 학습률 eta 로 강하게 갱신
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )

            # 선택하지 않은 나머지 이웃들은 eta2 로 약하게 갱신
            for n in self.neighbors:
                if n != y_star:
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )

        # 15. 포인트가 부족하면 일반 Q-routing 처럼 선택 이웃만 갱신
        else:
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # 16. Route Memory 갱신 (L=0 이면 항상 빈 리스트, 그 외엔 최근 L 홉만 보관)
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        # 17. 실제 패킷을 보낼 다음 홉 반환
        return y_star

    # -------------------------------------------------------------------------
    # PFE_c — PFE 본체 + AQRERM_c 식 큐 페널티
    # - y_star 선택: argmin (Q[dst][n] + c_q · last_known_queue[n])
    # - Full Echo 발동 시: 모든 이웃의 큐 길이 캐시 갱신 (echo piggyback)
    # - Q-routing 모드 (포인트 부족) 시: y_star 큐 길이만 갱신
    # - c_q 는 params['c'] (AQRERM_c 과 동일 키), c_pt 는 PFE 의 동적 가격 (별개)
    # -------------------------------------------------------------------------
    def _route_pfe_c(self, packet, current_tick, all_nodes):
        # 0. 진단: 윈도우 라우팅 호출 카운트
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        gr    = self.params.get('pfe_gr', 0.1)
        b_max = self.params.get('pfe_b_max', 5.0)
        L     = self.params['L']
        c_q   = self.params['c']  # AQRERM_c 식 큐 페널티 계수 (PFE 의 c_pt 와 혼동 주의)

        self.update_T_est()

        # PFE 동적 가격 (Full Echo 1 회 비용)
        sale_pt = 1.0 - (self.T_est / self.T_max) if self.T_max > 0 else 1.0
        if sale_pt == 0:
            sale_pt = 0.1
        c_pt = self.params.get('pfe_c', sale_pt)

        # 포인트 적립
        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0
        # self.total_point = min(b_max, self.total_point + gr * R_x)
        self.total_point = min(b_max, self.total_point + gr )

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # y_star 선택 — AQRERM_c 식 큐 페널티 포함
        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
        )

        q = current_tick - packet.queue_entry_tick
        s = 1

        # Full Echo or Q-routing 분기
        if self.total_point >= c_pt:
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1
            eta2 = R_x * k

            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}

            # echo piggyback: 모든 이웃의 큐 길이 캐시 갱신
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            for n in self.neighbors:
                if n != y_star:
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            # Q-routing 모드: y_star 의 큐 길이만 갱신 (그 이웃에만 "접촉" 했으므로)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # PFE_c03 — PFE_c 본체에서 c=0.3 강제 (AQRERM_c03 와 동일한 override 패턴)
    # -------------------------------------------------------------------------
    def _route_pfe_c03(self, packet, current_tick, all_nodes):
        original_c = self.params['c']
        self.params['c'] = 0.3
        try:
            return self._route_pfe_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # PFE_c_L0: PFE_c 에 c=c_value 고정 + memory_cut_tick (보통 7000) 이후 L=0 전환
    # - AQRERM_C{X}_L0 (_route_aqrerm_c_l0) 와 동일한 override 패턴
    # - dispatcher 에서 c_value 인자로 0.3, 0.5 전달
    # - 모든 override 는 try/finally 로 즉시 원복
    # -------------------------------------------------------------------------
    def _route_pfe_c_l0(self, packet, current_tick, all_nodes, c_value):
        cut_tick = self.params.get('memory_cut_tick', 0)
        original_c = self.params['c']
        self.params['c'] = c_value
        if current_tick >= cut_tick:
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_pfe_c(packet, current_tick, all_nodes)
            finally:
                self.params['c'] = original_c
                self.params['L'] = original_L
        else:
            try:
                return self._route_pfe_c(packet, current_tick, all_nodes)
            finally:
                self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # PFE_c_AdE — PFE_c 본체 + Advantage-weighted eta2 (단순화 식)
    #
    # 핵심: Full Echo 직후 fresh Score 로 y_star 와 다른 이웃을 사후 비교 →
    #       y_star 보다 점수가 좋았던 이웃은 강하게, 나빴던 이웃은 약하게 학습.
    #
    # η₂,n = clip(η₂_base + α · (Score_y* − Score_n),  η_floor,  η)
    #   Score_n     = t_n + c_q · queue_n       (fresh 정보로)
    #   η₂_base     = R_x · η · k                (현 PFE_c uniform eta2)
    #   α (기본 0.05)    — 점수 차이 → 학습률 환산 계수
    #   η_floor (기본 0.01) — 학습 완전 정지 방지
    #   η (= 0.9)         — y_star 학습률, AdE 의 cap
    #
    # Q-routing 분기 (포인트 부족) 에선 echo 정보 없으니 기존 동작 그대로 (y_star 만 update).
    # -------------------------------------------------------------------------
    def _route_pfe_c_ade(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        gr    = self.params.get('pfe_gr', 0.1)
        b_max = self.params.get('pfe_b_max', 1.0)
        L     = self.params['L']
        c_q   = self.params['c']

        # AdE 파라미터
        alpha     = self.params.get('ade_alpha',     100)
        # eta_floor = self.params.get('ade_eta_floor', 0.01)

        self.update_T_est()

        sale_pt = 1.0 - (self.T_est / self.T_max) if self.T_max > 0 else 1.0
        if sale_pt == 0:
            sale_pt = 0.1
        c_pt = self.params.get('pfe_c', sale_pt)

        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0
        # self.total_point = min(b_max, self.total_point + gr * R_x)
        self.total_point = min(b_max, self.total_point + gr )

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # y_star 선택 — stale 캐시 기반 (기존 그대로)
        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
        )

        # y_star switching 추적 — 같은 dst 에 대한 이전 y_star 와 비교
        prev = self.prev_y_star.get(dst)
        if prev is not None and prev != y_star:
            self.pfe_window_switch_count += 1
        self.prev_y_star[dst] = y_star

        q = current_tick - packet.queue_entry_tick
        s = 1

        # Full Echo 발동 시: fresh t_n 수집 → y_star 대비 다른 이웃의 상대적 성능 평가 → AdE로 eta2 차등 적용
        if self.total_point >= c_pt:
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1
            eta2_base = R_x * k

            # t_values : 모든 이웃의 dst 까지 예상 시간 (자기 자신 제외, route memory 효과)
            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id) 
                        for n in self.neighbors}
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            # Fresh Score 산출 — y_star 도 fresh 로 재평가
            score_y_star = t_values[y_star] + c_q * self.last_known_queue[y_star]

            # y_star 는 항상 full eta 로 학습 (기존 그대로)
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )

            # 비-y_star: baseline 균등 학습 + Adv 양수일 때만 추가 boost
            # η₂,n = min(η, η₂_base + α · max(0, Score_y* − Score_n))
            # → Adv ≤ 0 이면 그대로 eta2_base (= 기존 PFE_c 동작), Adv > 0 만 boost
            best_adv      = 0.0
            best_score_n  = None
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    score_n = t_values[n] + c_q * self.last_known_queue[n]
                    # adv = max(0.0, (score_y_star - score_n) / (score_y_star + score_n))
                    adv = max(0.0, score_y_star - score_n)  # 절대 차이로도 시도해봄 (scale 민감도 낮추려고)
                    if adv > best_adv:
                        best_adv     = adv
                        best_score_n = score_n
                    # eta_n = min(eta, eta2_base + self.T_est/self.T_max * adv * 100)
                    # eta_n = min(eta, eta2_base + self.T_max * adv)
                    eta_n = min(eta, eta2_base + self.T_est * adv)
                    # eta_n = min(eta, eta2_base + alpha * adv)
                    # eta_n = min(eta, eta2_base + adv)
                else:
                    eta_n = eta2_base  # 도달 불가 이웃은 baseline 만
                # eta_n 통계 — clip rate / 평균 / 분산
                self.pfe_window_eta_n_count    += 1
                self.pfe_window_eta_n_sum      += eta_n
                self.pfe_window_eta_n_sq_sum   += eta_n * eta_n
                if eta_n >= eta - 1e-9:
                    self.pfe_window_eta_n_clip_count += 1
                self.Q[dst][n] += eta_n * (
                    q + s + t_values[n] - self.Q[dst][n]
                )

            # AdE 진단 — 이번 라우팅에서 Adv > 0 이 한 번이라도 있었으면 기록
            if best_score_n is not None:
                self.pfe_window_adv_event_count  += 1
                self.pfe_window_adv_sum          += best_adv
                self.pfe_window_score_y_sum      += score_y_star
                self.pfe_window_score_n_best_sum += best_score_n
        else:
            # Q-routing 모드: echo 정보 없음 — y_star 만 update
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # PFE_c_pre_echo — echo 와 노드 선정의 순서를 뒤집은 변형.
    #
    # 기존 PFE_c_AdE 와의 차이:
    #   - 선정 → echo → 학습 (기존)
    #   - echo → 선정 → 학습 (이 변형)
    #
    # 동작:
    #   포인트 적립 / 사용 방식은 기존 PFE 와 동일.
    #   total_point >= c_pt 이면:
    #     1) Full Echo 발동 → 모든 이웃의 fresh t_n + queue 수집
    #     2) Score = t_n + c_q * queue_n 으로 y_star 선정 (Route Memory visited 필터 적용)
    #     3) y_star: eta 로 update, 그 외 이웃: eta2 = R_x * k 로 균등 update
    #   포인트 부족 시 (Q-routing 모드):
    #     1) echo 미수행
    #     2) stale 캐시 (Q + c_q * last_known_queue) 로 y_star 선정
    #     3) y_star 만 update (PFE_c_AdE 의 Q-routing 모드와 동일)
    #
    # 주의: Pre-echo 모드에서는 y_star 가 fresh Score 기준 argmin 이므로
    #       어떤 비-y_star 이웃도 Score 가 y_star 보다 작을 수 없음 → Adv 항상 0.
    #       따라서 AdE 식 학습률 차등이 의미 없어지고 eta2 균등 적용.
    # -------------------------------------------------------------------------
    def _route_pfe_c_pre_echo(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        gr    = self.params.get('pfe_gr', 0.1)
        b_max = self.params.get('pfe_b_max', 1.0)
        L     = self.params['L']
        c_q   = self.params['c']

        self.update_T_est()

        # R_x 는 현재 노드의 불안정도 지표임. 0 이면 안정적, 1 에 가까울수록 불안정.
        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # PFE 동적 가격 (Full Echo 1 회 비용)
        # sale_pt = 1.0 - (self.T_est / self.T_max) if self.T_max > 0 else 1.0
        sale_pt = 1.0 - R_x if self.T_max > 0 else 1.0
        if sale_pt == 0:
            sale_pt = 0.1
        c_pt = self.params.get('pfe_c', sale_pt)


        # 포인트는 매 라우팅마다 적립됨.
        self.total_point = min(b_max, self.total_point + gr)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        if self.total_point >= c_pt:
            # === Pre-echo 모드 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            # 1차: 모든 이웃의 fresh t_n + queue 수집
            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            # 2차: fresh Score = t_n + c_q * queue_n 으로 y_star 선정
            def _score(n):
                if math.isfinite(t_values[n]):
                    return t_values[n] + c_q * self.last_known_queue[n]
                else:
                    return float('inf')

            y_star = min(candidates, key=_score)

            # 3차: y_star eta, 비-y_star eta2 균등 update
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === Q-routing 모드 (포인트 부족) ===
            # stale 캐시 기반 선정
            y_star = min(
                candidates,
                key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
            )
            # y_star 만 update
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQPACE — _route_pfe_c_pre_echo 변형.
    # 차이점: 라우팅 함수 본문에서 total_point += gr 줄을 제거.
    # 대신 simulator tick 루프가 tick_accumulate_point() 로 매 tick 모든 노드에
    # gr 만큼 적립. 큐가 비어 라우팅 안 하는 tick 에도 적립 진행됨.
    # -------------------------------------------------------------------------
    def _route_aqpace(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        b_max = self.params.get('pfe_b_max', 0.5)
        L     = self.params['L']
        c_q   = self.params['c']

        self.update_T_est()



        # ★ 본문 적립 제거 — 적립은 simulator 의 tick_accumulate_point() 가 담당
        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # PFE 동적 가격 (Full Echo 1 회 비용)
        sale_pt = max(0.1, 1.0 - R_x)

        c_pt = self.params.get('pfe_c', sale_pt)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        if self.total_point >= c_pt:
            # === Pre-echo 모드 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            def _score(n):
                if math.isfinite(t_values[n]):
                    return t_values[n] + c_q * self.last_known_queue[n]
                else:
                    return float('inf')

            y_star = min(candidates, key=_score)

            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === Q-routing 모드 (포인트 부족) ===
            y_star = min(
                candidates,
                key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
            )
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # PFE_c_pre_echo_L0 — PFE_c_pre_echo 본체 + memory_cut_tick (보통 7000) 이후 L=0 강제.
    # link_cut 시나리오용. c 는 params['c'] 그대로 사용.
    # -------------------------------------------------------------------------
    def _route_pfe_c_pre_echo_l0(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick >= cut_tick:
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_pfe_c_pre_echo(packet, current_tick, all_nodes)
            finally:
                self.params['L'] = original_L
        else:
            return self._route_pfe_c_pre_echo(packet, current_tick, all_nodes)

    # -------------------------------------------------------------------------
    # PFE_pre_echo_Tick — _route_aqpace 변형.
    # 차이점: 선택과 Q-routing 폴백 모두에서 큐 길이 항 (c · queue) 을 완전히 제거.
    # 결과적으로 PFE 의 즉시 큐 반응성을 버리고 t / Q 만으로 결정 → 후반 진동 완화.
    # 적립은 simulator 의 tick_accumulate_point() 가 매 tick 담당 (동일).
    # -------------------------------------------------------------------------
    def _route_pfe_pre_echo_tick(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        b_max = self.params.get('pfe_b_max', 1.0)
        L     = self.params['L']

        self.update_T_est()

        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # PFE 동적 가격 (Full Echo 1 회 비용)
        sale_pt = max(0.1, 1.0 - R_x)
        c_pt = self.params.get('pfe_c', sale_pt)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        if self.total_point >= c_pt:
            # === Pre-echo 모드 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}
            # 큐 길이는 통계 / 진단용으로만 캐시. 선택 식에는 안 들어감.
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            # 선택 식: t_values 만 사용 (큐 항 제거)
            def _score(n):
                if math.isfinite(t_values[n]):
                    return t_values[n]
                else:
                    return float('inf')

            y_star = min(candidates, key=_score)

            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === Q-routing 모드 (포인트 부족) ===
            # 선택 식: stale Q 만 사용 (큐 항 제거)
            y_star = min(candidates, key=lambda n: self.Q[dst][n])
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # PFE_echo_Tick — select-first 구조의 PFE 변형 (echo 순서 뒤로).
    # AQRERM 처럼 stale Q argmin 으로 y* 선택 → 그 다음 PFE 포인트 게이트 echo.
    # - 포인트 충분 (total_point >= c_pt) : 모든 이웃 풀에코, y* η + 비-y* η2 update
    # - 포인트 부족 : Q-routing fallback (y* 한 노드만 single echo, y* η update)
    # 큐 페널티 c 는 안 씀 (선택 식이 stale Q 만).
    # 적립은 simulator 의 tick_accumulate_point() 가 매 tick 담당.
    # -------------------------------------------------------------------------
    def _route_pfe_echo_tick(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        b_max = self.params.get('pfe_b_max', 1.0)
        L     = self.params['L']

        self.update_T_est()

        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # PFE 동적 가격 (Full Echo 1 회 비용)
        sale_pt = max(0.1, 1.0 - R_x)
        c_pt = self.params.get('pfe_c', sale_pt)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # 선택: stale Q argmin (큐 항 없음) — AQRERM 패턴
        y_star = min(candidates, key=lambda n: self.Q[dst][n])

        if self.total_point >= c_pt:
            # === 포인트 충분: 모든 이웃 풀에코 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}
            # 큐 길이도 같이 동기화 (진단/c 변형용)
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            # y* η 로 강하게 갱신
            if math.isfinite(t_values[y_star]):
                self.Q[dst][y_star] += eta * (
                    q + s + t_values[y_star] - self.Q[dst][y_star]
                )
            # 비-y* η2 로 약하게 갱신
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === 포인트 부족: Q-routing fallback (y* 한 노드만 single echo) ===
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # PFE_c_echo_Tick — PFE_echo_tick 변형, 선택 식에 큐 페널티 c 추가.
    # 선택: stale Q + c · last_known_queue (AQRERM_c 패턴)
    # 나머지 흐름은 PFE_echo_tick 과 동일.
    # -------------------------------------------------------------------------
    def _route_pfe_c_echo_tick(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        b_max = self.params.get('pfe_b_max', 1.0)
        L     = self.params['L']
        c_q   = self.params['c']

        self.update_T_est()

        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # PFE 동적 가격 (Full Echo 1 회 비용)
        sale_pt = max(0.1, 1.0 - R_x)
        c_pt = self.params.get('pfe_c', sale_pt)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # 선택: stale Q + c · last_known_queue (AQRERM_c 패턴)
        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
        )

        if self.total_point >= c_pt:
            # === 포인트 충분: 모든 이웃 풀에코 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                        for n in self.neighbors}
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            if math.isfinite(t_values[y_star]):
                self.Q[dst][y_star] += eta * (
                    q + s + t_values[y_star] - self.Q[dst][y_star]
                )
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === 포인트 부족: Q-routing fallback (y* 한 노드만 single echo) ===
            t = all_nodes[y_star].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # FE_c_pre_echo — aqpace 에서 PFE 포인트 시스템 제거.
    # 매 라우팅마다 무조건 full echo → fresh t + c · queue 로 y* 선택 →
    # y* η, 비-y* η2 차등 update. fallback / 포인트 게이트 없음.
    # AQFE + Route Memory + c 페널티 + pre-echo 선택의 조합.
    # -------------------------------------------------------------------------
    def _route_fe_c_pre_echo(self, packet, current_tick, all_nodes):
        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        L     = self.params['L']
        c_q   = self.params['c']

        self.update_T_est()

        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # 항상 full echo — 모든 이웃의 fresh t + fresh queue 수집
        t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                    for n in self.neighbors}
        for n in self.neighbors:
            self.last_known_queue[n] = len(all_nodes[n].queue)

        # 선택: fresh t + c · fresh queue (pre-echo 선택)
        def _score(n):
            if math.isfinite(t_values[n]):
                return t_values[n] + c_q * self.last_known_queue[n]
            else:
                return float('inf')
        y_star = min(candidates, key=_score)

        # y* η 로 강하게 update
        if math.isfinite(t_values[y_star]):
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
        # 비-y* η2 로 약하게 update
        for n in self.neighbors:
            if n == y_star:
                continue
            if math.isfinite(t_values[n]):
                self.Q[dst][n] += eta2 * (
                    q + s + t_values[n] - self.Q[dst][n]
                )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c_pre — AQRERM_c 의 pre-echo 변형.
    # 흐름:
    #   1) random echo set 구성 (각 이웃 확률 p = R_x + 이전 y* 무조건 포함)
    #   2) echo set 멤버에게서 fresh t, fresh queue 수집
    #   3) Route Memory 필터 후, 선택: argmin (fresh t + c · fresh queue) (AQRERM_c 패턴)
    #   4) y* η 로 update, 나머지 echo set η2 (= p · k) 로 update
    #   5) 새 y* 를 self.last_y_star[dst] 에 저장 (다음 결정에서 echo set 에 보장)
    # PFE 포인트 시스템 없음 (랜덤 echo 기반).
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_pre(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k   = self.params['k']
        L   = self.params['L']
        c_q = self.params['c']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = p * k

        # 1. Random echo set 구성
        echo_set = set()
        # 1-1. 이전 y* 항상 포함 (현재 neighbor 인 경우만)
        prev_y = self.last_y_star.get(dst)
        if prev_y is not None and prev_y in self.neighbors:
            echo_set.add(prev_y)
        # 1-2. 나머지 이웃은 확률 p 로 추가
        for n in self.neighbors:
            if n != prev_y and random.random() < p:
                echo_set.add(n)
        # 1-3. fallback: 비면 모든 이웃 포함 (첫 호출 + p≈0 인 경우)
        if not echo_set:
            echo_set = set(self.neighbors)

        # 2. echo set 멤버의 fresh t, fresh queue 수집
        t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                    for n in echo_set}
        for n in echo_set:
            self.last_known_queue[n] = len(all_nodes[n].queue)

        # 3. Route Memory 필터 (echo set 안에서만)
        visited = set(packet.route_memory)
        candidates = [n for n in echo_set if n not in visited]
        if not candidates:
            candidates = list(echo_set)

        # 4. 선택: argmin (fresh t + c · fresh queue)
        def _score(n):
            if math.isfinite(t_values[n]):
                return t_values[n] + c_q * self.last_known_queue[n]
            else:
                return float('inf')
        y_star = min(candidates, key=_score)

        # 5. y* η 로 update
        if math.isfinite(t_values[y_star]):
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
        # 6. 나머지 echo set 멤버 η2 로 update
        for n in echo_set:
            if n == y_star:
                continue
            if math.isfinite(t_values[n]):
                self.Q[dst][n] += eta2 * (
                    q + s + t_values[n] - self.Q[dst][n]
                )

        # 7. 새 y* 저장
        self.last_y_star[dst] = y_star

        # 8. Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_pre — AQRERM 의 pre-echo 변형 (c 없는 버전).
    # _route_aqrerm_c_pre 과 동일하나 선택 식에서 c · queue 항만 제거.
    # 흐름:
    #   1) random echo set 구성 (각 이웃 확률 p = R_x + 이전 y* 무조건 포함)
    #   2) echo set 멤버에게서 fresh t 수집
    #   3) Route Memory 필터 후, 선택: argmin (fresh t) — c 페널티 없음
    #   4) y* η 로 update, 나머지 echo set η2 (= p · k) 로 update
    #   5) 새 y* 를 self.last_y_star[dst] 에 저장
    # PFE 포인트 시스템 없음 (랜덤 echo 기반).
    # -------------------------------------------------------------------------
    def _route_aqrerm_pre(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k   = self.params['k']
        L   = self.params['L']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = p * k

        # 1. Random echo set 구성
        echo_set = set()
        # 1-1. 이전 y* 항상 포함 (현재 neighbor 인 경우만)
        prev_y = self.last_y_star.get(dst)
        if prev_y is not None and prev_y in self.neighbors:
            echo_set.add(prev_y)
        # 1-2. 나머지 이웃은 확률 p 로 추가
        for n in self.neighbors:
            if n != prev_y and random.random() < p:
                echo_set.add(n)
        # 1-3. fallback: 비면 모든 이웃 포함 (첫 호출 + p≈0 인 경우)
        if not echo_set:
            echo_set = set(self.neighbors)

        # 2. echo set 멤버의 fresh t 수집 (queue 캐시는 안 갱신 — c 안 쓰니까 필요 없음)
        t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=self.id)
                    for n in echo_set}

        # 3. Route Memory 필터 (echo set 안에서만)
        visited = set(packet.route_memory)
        candidates = [n for n in echo_set if n not in visited]
        if not candidates:
            candidates = list(echo_set)

        # 4. 선택: argmin (fresh t) — c 페널티 없음
        def _score(n):
            if math.isfinite(t_values[n]):
                return t_values[n]
            else:
                return float('inf')
        y_star = min(candidates, key=_score)

        # 5. y* η 로 update
        if math.isfinite(t_values[y_star]):
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
        # 6. 나머지 echo set 멤버 η2 로 update
        for n in echo_set:
            if n == y_star:
                continue
            if math.isfinite(t_values[n]):
                self.Q[dst][n] += eta2 * (
                    q + s + t_values[n] - self.Q[dst][n]
                )

        # 7. 새 y* 저장
        self.last_y_star[dst] = y_star

        # 8. Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQPACE_L0 — _route_aqpace 본체 + memory_cut_tick 이후 L=0 강제.
    # link_cut 시나리오에서 cut 시점부터 Route Memory 무효화하면서도 매 tick 적립 유지.
    # -------------------------------------------------------------------------
    def _route_aqpace_l0(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick >= cut_tick:
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_aqpace(packet, current_tick, all_nodes)
            finally:
                self.params['L'] = original_L
        else:
            return self._route_aqpace(packet, current_tick, all_nodes)

    # -------------------------------------------------------------------------
    # PFE_c_AdE_L0 — PFE_c_AdE 본체 + memory_cut_tick (보통 7000) 이후 L=0 강제.
    # _route_pfe_c_l0 와 동일한 try/finally override 패턴.
    # c_value=None 이면 params['c'] 그대로, 값 지정 시 그 값으로 override (c01/c10 변형용).
    # -------------------------------------------------------------------------
    def _route_pfe_c_ade_l0(self, packet, current_tick, all_nodes, c_value=None):
        cut_tick = self.params.get('memory_cut_tick', 0)
        original_c = self.params['c']
        if c_value is not None:
            self.params['c'] = c_value
        try:
            if current_tick >= cut_tick:
                original_L = self.params['L']
                self.params['L'] = 0
                try:
                    return self._route_pfe_c_ade(packet, current_tick, all_nodes)
                finally:
                    self.params['L'] = original_L
            else:
                return self._route_pfe_c_ade(packet, current_tick, all_nodes)
        finally:
            if c_value is not None:
                self.params['c'] = original_c


    def _route_aqrerm(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # Route Memory: 방문한 노드 제외
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(candidates, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        # y*는 항상 echo, 나머지는 확률 p로 echo
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            # Route Memory: y=n에서 t 추정 시 현재 노드 x 제외
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # Route Memory 갱신 (L=0 이면 항상 빈 리스트)
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_L0 — AQRERM 본체 + memory_cut_tick (보통 7000) 이후 L=0 강제.
    # link_cut 시나리오에서 절단 직후 route memory 무효화 비교용.
    # -------------------------------------------------------------------------
    def _route_aqrerm_l0(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick >= cut_tick:
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_aqrerm(packet, current_tick, all_nodes)
            finally:
                self.params['L'] = original_L
        else:
            return self._route_aqrerm(packet, current_tick, all_nodes)
        
    # -------------------------------------------------------------------------
    # AQRERM_no_mem: AQRERM에서 Route Memory만 끈 변형 (디버깅용)
    # - 방문 노드 후보 제외 X
    # - 이웃 t 추정 시 self.id 제외 X
    # - packet.route_memory도 갱신하지 않음
    # -------------------------------------------------------------------------
    def _route_aqrerm_no_mem(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c: AQRERM + 큐 길이 페널티를 라우팅 결정에 반영
    # score(n) = Q[dst][n] + c * len(neighbor.queue)
    # Q 갱신식과 echo 메커니즘은 AQRERM과 동일
    # -------------------------------------------------------------------------
    def _route_aqrerm_c(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']
        c = self.params['c']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # 캐시된 이웃 큐 길이로 score 계산 (echo 응답 때 받은 stale 값 사용)
        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            # echo 응답에 piggyback된 큐 길이 캐시 갱신
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # Route Memory 갱신 (L=0 이면 항상 빈 리스트)
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c_AdE — AQRERM_c 본체 + Advantage-weighted eta2 (단순화 식)
    #
    # PFE_c_AdE 와 같은 공식, 다만 적용 범위가 echo_set 안의 비-y_star 로 한정.
    # echo_set 밖 이웃은 기존 AQRERM_c 처럼 update X.
    # echo_set 가 {y_star} 만 있으면 (낮은 R_x) AdE 효과 0 — 자연스러운 부하 의존성.
    #
    # η₂,n = clip(η₂_base + α · (Score_y* − Score_n),  η_floor,  η)
    #   η₂_base = p · η · k  (= R_x · η · k, 기존 AQRERM_c eta2)
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_ade(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']
        c = self.params['c']

        # AdE 파라미터
        alpha     = self.params.get('ade_alpha',     0.5)
        eta_floor = self.params.get('ade_eta_floor', 0.01)

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2_base = p * k

        # echo_set 확률적 선택 (기존 AQRERM_c 그대로)
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        # 1차 패스: echo_set 멤버의 fresh t_n 수집 + queue 캐시 갱신
        t_values = {}
        for n in echo_set:
            t_values[n] = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)

        # Fresh Score_y_star — y_star 도 fresh 로 재평가
        score_y_star = t_values[y_star] + c * self.last_known_queue[y_star]

        # 2차 패스: Q 갱신
        for n in echo_set:
            if n == y_star:
                # y_star 는 항상 full eta
                self.Q[dst][n] += eta * (
                    q + s + t_values[n] - self.Q[dst][n]
                )
            else:
                # 비-y_star: baseline 균등 학습 + Adv 양수일 때만 추가 boost
                # η₂,n = min(η, η₂_base + α · max(0, Score_y* − Score_n))
                if math.isfinite(t_values[n]):
                    score_n = t_values[n] + c * self.last_known_queue[n]
                    # adv = max(0.0, (score_y_star - score_n) / (score_y_star + score_n))
                    adv = max(0.0, score_y_star - score_n)  # 절대 차이로도 시도해봄 (scale 민감도 낮추려고)
                    # eta_n = min(eta, eta2_base + self.T_est/self.T_max * adv * 100)
                    # eta_n = min(eta, eta2_base + self.T_max * adv)
                    eta_n = min(eta, eta2_base + self.T_est * adv)
                    # eta_n = min(eta, eta2_base + alpha * adv)
                    # eta_n = min(eta, eta2_base + adv)
                else:
                    eta_n = eta2_base
                self.Q[dst][n] += eta_n * (
                    q + s + t_values[n] - self.Q[dst][n]
                )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c_AdE_L0 — AQRERM_c_AdE 본체 + memory_cut_tick (보통 7000) 이후 L=0 강제.
    # link_cut 시나리오용 — c 는 params['c'] (main_link_cut.py 에서 0.5) 그대로.
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_ade_l0(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick >= cut_tick:
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_aqrerm_c_ade(packet, current_tick, all_nodes)
            finally:
                self.params['L'] = original_L
        else:
            return self._route_aqrerm_c_ade(packet, current_tick, all_nodes)

    # -------------------------------------------------------------------------
    # AQRERM_c_no_mem: 디버깅용
    # - current_tick < memory_cut_tick: 일반 AQRERM_c 그대로 (L = params['L'])
    # - current_tick >= memory_cut_tick: effective L = 0 으로 route memory 무효화
    #   * visited 필터링 비활성화 (모든 이웃이 후보)
    #   * packet.route_memory 항상 빈 리스트로 유지
    #   (best_estimate의 exclude_node=self.id 효과는 그대로 유지)
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_7000_no_mem(self, packet, current_tick, all_nodes):
        # memory_cut_tick (보통 7000) 이전엔 L=params['L'], 이후엔 L=0 으로 전환
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']
        memory_cut_tick = self.params.get('memory_cut_tick', 0)
        L = 0 if current_tick >= memory_cut_tick else self.params['L']

        if L == 0:
            packet.route_memory = []

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        if L > 0:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c_ALL_NO_MEM: 시뮬레이션 시작부터 끝까지 L=0 (route memory 영구 비활성)
    # - memory_cut_tick 분기 없음 — 항상 L=0
    # - packet.route_memory 항상 빈 리스트, visited 필터링 의미 없음
    # - 그 외 echo, Q 업데이트, last_known_queue 캐싱은 AQRERM_c 본체 그대로
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_all_no_mem(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']

        # L = 0 영구 강제 — route_memory 매번 빈 리스트로 유지
        packet.route_memory = []

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # visited 항상 비어 있으니 모든 이웃이 후보
        candidates = list(self.neighbors)
        if not candidates:
            return packet.dst   # 이웃 없으면 위로 (보호)

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # route_memory 는 갱신 없이 빈 리스트 유지
        return y_star

    # -------------------------------------------------------------------------
    # AQRERM_c_7000_NO_C: memory_cut_tick (보통 7000) 이전엔 일반 AQRERM_c
    # (c=params['c'], L=params['L']), 이후엔 c=0 + L=0 강제.
    #   - c=0: 큐 페널티 비활성화 (AQRERM 비슷한 큐 무시 라우팅)
    #   - L=0: route memory 비활성화 (방문 노드 필터링 X, 빈 리스트 유지)
    # 한 라우팅 결정 동안만 self.params['c']/['L'] 을 잠시 override 후
    # _route_aqrerm_c 호출 → try/finally 로 즉시 원복하여 다른 노드 라우팅에 영향 없음.
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_7000_no_c(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick < cut_tick:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        original_c = self.params['c']
        original_L = self.params['L']
        self.params['c'] = 0.0
        self.params['L'] = 0
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c
            self.params['L'] = original_L

    # -------------------------------------------------------------------------
    # AQRERM_c_7000_ONE_C: memory_cut_tick 이전엔 일반 AQRERM_c, 이후엔 c=1 + L=0 강제
    #   - c=1: 큐 페널티 강화 (기존 0.5 의 두 배, 부하 분산 더 적극적)
    #   - L=0: route memory 비활성화
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_7000_one_c(self, packet, current_tick, all_nodes):
        cut_tick = self.params.get('memory_cut_tick', 0)
        if current_tick < cut_tick:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        original_c = self.params['c']
        original_L = self.params['L']
        self.params['c'] = 3.0
        self.params['L'] = 0
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c
            self.params['L'] = original_L

    # -------------------------------------------------------------------------
    # AQRERM_c_LOW_C: 시뮬레이션 전체 구간에서 c=0.1 고정 (큐 페널티 약하게)
    # 일반 AQRERM_c 로직 그대로, c 값만 매 라우팅마다 0.1 로 override.
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_low_c(self, packet, current_tick, all_nodes):
        original_c = self.params['c']
        self.params['c'] = 0.1
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # AQRERM_C03: 시뮬레이션 전체 구간에서 c=0.3 고정
    # -------------------------------------------------------------------------
    def _route_aqrerm_c03(self, packet, current_tick, all_nodes):
        original_c = self.params['c']
        self.params['c'] = 0.3
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # AQRERM_C07: 시뮬레이션 전체 구간에서 c=0.7 고정
    # -------------------------------------------------------------------------
    def _route_aqrerm_c07(self, packet, current_tick, all_nodes):
        original_c = self.params['c']
        self.params['c'] = 0.7
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # AQRERM_c_HIGH_C: 시뮬레이션 전체 구간에서 c=1.0 고정 (큐 페널티 강하게)
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_high_c(self, packet, current_tick, all_nodes):
        original_c = self.params['c']
        self.params['c'] = 1.0
        try:
            return self._route_aqrerm_c(packet, current_tick, all_nodes)
        finally:
            self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # AQRERM_C{X}_L0: c=c_value 로 고정 + memory_cut_tick (보통 7000) 이후 L=0 전환
    # - 헬퍼 메서드. dispatcher 에서 c_value 인자로 0.1, 0.3, 0.5, 0.7 전달.
    # - 모든 override 는 try/finally 로 즉시 원복.
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_l0(self, packet, current_tick, all_nodes, c_value):
        cut_tick = self.params.get('memory_cut_tick', 0)
        original_c = self.params['c']
        self.params['c'] = c_value
        if current_tick >= cut_tick:
            # 절단 시점 이후: c=c_value 유지하면서 L=0 강제
            original_L = self.params['L']
            self.params['L'] = 0
            try:
                return self._route_aqrerm_c(packet, current_tick, all_nodes)
            finally:
                self.params['c'] = original_c
                self.params['L'] = original_L
        else:
            # 절단 이전: c 만 override, L 은 원래 값 유지
            try:
                return self._route_aqrerm_c(packet, current_tick, all_nodes)
            finally:
                self.params['c'] = original_c

    # -------------------------------------------------------------------------
    # AQRERM_c_L_TRAIN: AQRERM_c과 동일하되 route memory size L 을 글로벌 학습으로 결정
    # - y_star 선택: AQRERM_c의 큐 페널티 (Q + c * last_known_queue)
    # - echo 확률 p: AQRERM_c의 표준 p = T_est / T_max
    # - L: simulator 가 매 stat_interval 마다 글로벌 state 로 sampling 해서
    #      controller.cached_L 에 박아둔 값을 그대로 사용 (window 동안 일정)
    # - td_error_ema 만 글로벌 state 의 avg_TD_error 계산용으로 유지 갱신
    # - 그 외 보조 변수 (prev_Q_min, y_star_history, last_echo_tick) 는 갱신 생략
    # -------------------------------------------------------------------------
    def _route_aqrerm_c_l_train(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # L 은 컨트롤러의 window-cached 값을 직접 읽기 — actor 호출 없음
        controller = self.params['controller']
        L = controller.cached_L
        # L_history 누적 (per-routing 단위 진단 로그용)
        controller.L_history.append(L)

        # --- 이하 AQRERM_c 본체와 동일 ---
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # td_error_ema 갱신 — 글로벌 state 의 avg_TD_error 입력으로 쓰임
        td_error_ystar = q + s + all_nodes[y_star].best_estimate(dst, exclude_node=self.id) - self.Q[dst][y_star]
        if math.isfinite(td_error_ystar):
            self.td_error_ema = (1 - self.td_ema_alpha) * self.td_error_ema + self.td_ema_alpha * abs(td_error_ystar)

        # route_memory 갱신: 학습된 L 사용
        if L > 0:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory
        else:
            packet.route_memory = []

        return y_star


    # 역할
    # 다음 홉을 결정한다.
    # echo_set을 확률적으로 선정하여 Q 테이블 업데이트
    def _route_learned_aqrerm(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']

        self.update_T_est()

        # --- 상태값 계산 ---
        q_values = list(self.Q[dst].values())

        # Q_min, Q_avg, Q_spread, Q_variance
        Q_min      = min(q_values)
        Q_avg      = sum(q_values) / len(q_values)
        Q_spread   = max(q_values) - Q_min
        Q_variance = sum((v - Q_avg) ** 2 for v in q_values) / len(q_values)

        # ΔQ_min: 현재 Q_min - 직전 Q_min
        delta_Q_min = Q_min - self.prev_Q_min[dst]
        self.prev_Q_min[dst] = Q_min

        # TD_error_ema: 직전 업데이트의 EMA (아직 업데이트 전이므로 현재값 사용)
        TD_error_ema = self.td_error_ema

        # queue_len: 현재 큐 길이
        queue_len = len(self.queue)

        # route_switching_recent: 최근 500 tick 내 y* 변경 횟수
        self.y_star_history = deque(
            [(t, y) for t, y in self.y_star_history if current_tick - t <= 500]
        )
        route_switching_recent = sum(
            1 for i in range(1, len(self.y_star_history))
            if self.y_star_history[i][1] != self.y_star_history[i-1][1]
        )

        # echo_age_avg: 목적지 d 기준 이웃별 마지막 echo 이후 경과 tick 평균
        echo_age_avg = sum(
            current_tick - self.last_echo_tick[dst][n]
            for n in self.neighbors
        ) / len(self.neighbors)

        # T_ratio: AQRERM의 p와 동일한 값 (참고용 상태값)
        T_ratio = self.T_est / self.T_max if self.T_max > 0 else 0.0

        T_max = self.T_max if self.T_max > 0 else 1.0
        state = [
            _clamp01(Q_min              / T_max),
            _clamp01(Q_avg              / T_max),
            _clamp01(Q_spread           / T_max),
            _clamp01(Q_variance ** 0.5  / T_max),
            _signed_ratio01(delta_Q_min,  T_max),
            _clamp01(TD_error_ema       / T_max),
            _clamp01(queue_len          / 10),
            _clamp01(route_switching_recent / 10),
            _clamp01(echo_age_avg       / 500),
            T_ratio,
        ]

        # 에코를 위해 에코 컨트롤러로 p 계산. 
        # 에코 컨트롤러에서 ACTOR가 state를 입력받아 p값을 예측하여 반환
        # 진행 중에 last state 가 저장됨
        p = self.params['controller'].predict(state)

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(candidates, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        # 선정된 이웃 y*는 항상 echo, 나머지는 확률 p로 echo
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        # 확률적으로 선정된 echo_set에 대해 Q 테이블 업데이트
        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            td_error = q + s + t_n - self.Q[dst][n]
            if n == y_star:
                self.Q[dst][n] += eta * td_error
            else:
                self.Q[dst][n] += eta2 * td_error

            # last_echo_tick 갱신
            self.last_echo_tick[dst][n] = current_tick

        # TD_error_ema 갱신 (y*의 TD error 기준)
        td_error_ystar = q + s + all_nodes[y_star].best_estimate(dst, exclude_node=self.id) - self.Q[dst][y_star]
        self.td_error_ema = (1 - self.td_ema_alpha) * self.td_error_ema + self.td_ema_alpha * abs(td_error_ystar)

        # y_star_history 갱신
        self.y_star_history.append((current_tick, y_star))

        new_memory = packet.route_memory + [self.id]
        if len(new_memory) > L:
            new_memory = new_memory[-L:]
        packet.route_memory = new_memory

        # 라우팅 결정 (다음 홉 반환)
        return y_star
