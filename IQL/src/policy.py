import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import random 

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# stochastic policy function and deterministic policy function 

class GaussianPolicy(nn.Module):
    # 확률적정책 
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.epsilon = 1 # 초기 값 

    def forward(self, obs):\
        # 상태 obs 를 받아서 평균 mean 과 공분산 scale_tril 로 다변량 정규분포를 생성함 
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            # exploration 시에는 dist.sample() , evaluation 시에는 dist.mean() 사용 
            return dist.mean if deterministic else dist.sample()
     # ──────────────────────────────────────────
# GaussianPolicy 내부에 추가
# ──────────────────────────────────────────
    def new_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        exp_prob: float = 0.3,    # ε-탐색 확률
        noise_std: float = 10.0    # 기본 노이즈 세기
    ) -> torch.Tensor:
        """
        direct_action() 과 동일한 사용법:
            action = policy.new_action(torchify(obs),
                                    deterministic=False).cpu().numpy()
        - deterministic=True  → 항상 mean
        - deterministic=False → exp_prob 확률로 mean+노이즈 (노이즈 방향 = 오차 방향)
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)                     # MultivariateNormal(mean, Σ)
            mean_action = dist.mean              # [Q1_mean, Q2_mean]

            # ─── 평가(결정론) 모드 ──────────────────────────────
            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            # ─── ε-탐색 여부 결정 ───────────────────────────────
            if torch.rand(1).item() < exp_prob:
                # Δ = TSP − T  (obs = [T1,TSP1,T2,TSP2,…] 구조라면 인덱스 수정!)
                delta1 = obs[2] - obs[0]   # TSP1 - T1
                delta2 = obs[3] - obs[1]   # TSP2 - T2

                # 가우시안 노이즈 생성
                noise = torch.randn_like(mean_action) * noise_std

                # Q1 방향 보정
                if delta1 >= 0:   # 현재 온도 ↓  → Q1 ↑
                    noise[0] =  torch.abs(noise[0]) \
                                if torch.rand(1).item() > 0.2 else -torch.abs(noise[0]) / 2
                else:             # 현재 온도 ↑  → Q1 ↓
                    noise[0] = -torch.abs(noise[0]) \
                                if torch.rand(1).item() > 0.2 else  torch.abs(noise[0]) / 2
                # Q2 방향 보정
                if delta2 >= 0:
                    noise[1] =  torch.abs(noise[1]) \
                                if torch.rand(1).item() > 0.2 else -torch.abs(noise[1]) / 2
                else:
                    noise[1] = -torch.abs(noise[1]) \
                                if torch.rand(1).item() > 0.2 else  torch.abs(noise[1]) / 2

                action = mean_action + noise
            else:
                action = mean_action

            return torch.clamp(action, 0.0, 100.0)

    def new_action1(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        err_thr: float = 1.0,        # 오차 임계값 (°C)
        noise_std: float = 10.0      # 기본 노이즈 세기
    ) -> torch.Tensor:
        """
        - deterministic=True  → 항상 mean
        - deterministic=False → |TSP-T|이 err_thr보다 크면 탐색(mean+noise)
                                작으면 mean만 사용
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean          # [Q1_mean, Q2_mean]

            # ── 평가(결정론) 모드 ───────────────────────────────
            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            # ── 오차 계산 ───────────────────────────────────────
            delta1 = obs[2] - obs[0]         # TSP1 - T1
            delta2 = obs[3] - obs[1]         # TSP2 - T2
            explore = (torch.abs(delta1) > err_thr) or (torch.abs(delta2) > err_thr)

            if explore:
                # 가우시안 노이즈 생성
                noise = torch.randn_like(mean_action) * noise_std
                

                # Q1 방향 보정
                if delta1 >= 0:   # 현재 온도 낮음 → Q1 ↑
                    noise[0] =  torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[0]) / 2
                else:             # 현재 온도 높음 → Q1 ↓
                    noise[0] = -torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else  torch.abs(noise[0]) / 2

                # Q2 방향 보정
                if delta2 >= 0:   # 현재 온도 낮음 → Q2 ↑
                    noise[1] =  torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[1]) / 2
                else:             # 현재 온도 높음 → Q2 ↓
                    noise[1] = -torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else  torch.abs(noise[1]) / 2

                action = mean_action + noise
            else:
                action = mean_action   # 안전 구간 → 보수적

            return torch.clamp(action, 0.0, 100.0)

    def new_action2(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        exp_prob: float = 0.5,          # ε-탐색 확률 (0.5가 best였다!)
        k_std: float = 4.0,             # noise_std = k_std * error
        min_std: float = 4.0,           # noise_std 하한
        max_std: float = 20.0           # noise_std 상한
    ) -> torch.Tensor:
        """
        - deterministic=True  → 항상 mean
        - deterministic=False → exp_prob 확률로 탐색
            · 탐색 시 noise_std = k_std * error  (clip[min_std, max_std])
            · 노이즈 부호는 오차 방향(Δ = TSP − T)에 맞춰 조정
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean                       # [Q1_mean, Q2_mean]

            # ── 평가(결정론) 모드 ────────────────────────────────
            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            # ── ε-탐색 여부 결정 ─────────────────────────────────
            if torch.rand(1).item() < exp_prob:
                # Δ = TSP − T  (obs = [T1,TSP1,T2,TSP2,…])
                delta1 = obs[2] - obs[0]                  # TSP1 - T1
                delta2 = obs[3] - obs[1]                  # TSP2 - T2
                error  = torch.sqrt(delta1 ** 2 + delta2 ** 2)

                # ―― 오차 비례 noise_std 계산 ―――――――――――――――――
                noise_std = torch.clamp(k_std * error, min=min_std, max=max_std)

                # 가우시안 노이즈 생성 (채널별 동일 std)
                noise = torch.randn_like(mean_action) * noise_std

                # Q1 방향 보정
                if delta1 >= 0:   # 현재 온도 낮음 → Q1 ↑
                    noise[0] =  torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[0]) / 2
                else:             # 현재 온도 높음 → Q1 ↓
                    noise[0] = -torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else  torch.abs(noise[0]) / 2

                # Q2 방향 보정
                if delta2 >= 0:   # 현재 온도 낮음 → Q2 ↑
                    noise[1] =  torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[1]) / 2
                else:             # 현재 온도 높음 → Q2 ↓
                    noise[1] = -torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else  torch.abs(noise[1]) / 2

                action = mean_action + noise
            else:
                action = mean_action                     # 보수적(mean) 모드

            return torch.clamp(action, 0.0, 100.0)

# 안전 범위 내 탐색 
    def error_act(
        self,
        obs, 
        deterministic: bool = False,
        enable_grad: bool = False,
        err_thr: float = 1.0,        # |Tsp−T| ≤ err_thr 일 때만 노이즈 추가
        noise_std: float = 10.0       # 가우시안 노이즈 표준편차 (PWM %)
    ):
        """
        1 + 4 탐색 전략:
        - 큰 오차(|Tsp - T| > err_thr) → 학습된 policy의 평균(mean)만 사용
        - 작은 오차(|Tsp - T| ≤ err_thr) → mean + N(0, noise_std²) 로 탐색
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
        #    print("obs shape:", obs.shape)
            mean_action = dist.mean                      # [Q1_mean, Q2_mean]

            if deterministic:
                # 평가 모드: 항상 평균
                return torch.clamp(mean_action, 0.0, 100.0)

            # ── 현재 오차 계산 ───────────────────────────────────────
            delta1 = obs[2] - obs[0]   # TSP1 - T1
            delta2 = obs[3] - obs[1]   # TSP2 - T2

            in_safe_region = (
                torch.abs(delta1) <= err_thr and
                torch.abs(delta2) <= err_thr
            )
            if in_safe_region:
                std = torch.full_like(mean_action, noise_std)
                noise = torch.normal(mean=torch.zeros_like(mean_action), std=std)
                action = mean_action + noise
                print(f"탐색 (noise added) : err1= {delta1:.2f}, err2= {delta2 :.2f}")
            else:
                action = mean_action
              #  print(f"보수적 행동: err1={delta1:.2f}, err2={delta2:.2f}")

            return torch.clamp(action, 0.0, 100.0)
        
    def direct_action(
            self,
            obs, 
            deterministic: bool = False,
            enable_grad: bool = False,
            err_thr: float = 5.0,         # 이 이상이면 탐색
            base_scale: float = 1.0,      # 오차에 곱해줄 계수 (탐색 강도)
            min_std: float = 1.0,         # 최소 noise std
            max_std: float = 20.0         # 최대 noise std
        ):
            """
            오차가 클 때만 탐색:
            - noise_std ∝ 오차 크기
            - noise 방향 ∝ TSP - T (오차 방향)
            """
            with torch.set_grad_enabled(enable_grad):
                dist = self(obs)
                mean_action = dist.mean  # [Q1_mean, Q2_mean]

                if deterministic:
                    return torch.clamp(mean_action, 0.0, 100.0)

                # 🔁 오차 계산
                delta1 = obs[2] - obs[0]   # TSP1 - T1
                delta2 = obs[3] - obs[1]   # TSP2 - T2
                error = torch.sqrt(delta1 ** 2 + delta2 ** 2)

                if error > err_thr:
                    # ✅ noise std (강도): 오차 크기에 비례
                    noise_std = torch.clamp(base_scale * error, min=min_std, max=max_std)

                    # ✅ noise 방향: 오차 방향 따라 부호 결정
                    noise = torch.zeros_like(mean_action)
                    noise[0] = torch.abs(torch.randn(1)) * noise_std
                    noise[1] = torch.abs(torch.randn(1)) * noise_std

                    if delta1 < 0:  # T1 > TSP1 → Q1 줄여야
                        noise[0] *= -1
                    if delta2 < 0:  # T2 > TSP2 → Q2 줄여야
                        noise[1] *= -1

                    action = mean_action + noise
                    print(f"탐색: err={error:.2f}, noise_std={noise_std:.2f}, noise={noise}")
                else:
                    action = mean_action
                    print(f"보수적 행동: err={error:.2f} (<= {err_thr})")

                return torch.clamp(action, 0.0, 100.0)

    def hybrid_reverse_act(
        self,
        obs, 
        deterministic: bool = False,
        enable_grad: bool = False,
        err_thr: float = 5.0,         # 이 이상이면 탐색
        base_scale: float = 3.0,      # 오차에 곱해줄 계수 (탐색 강도)
        min_std: float = 2.0,         # 최소 noise std
        max_std: float = 20.0         # 최대 noise std
    ):
        """
        오차가 클수록 탐색 강도를 높이는 반전된 1+4 탐색 전략:
        - |Tsp - T| > err_thr → mean + noise (noise_std ∝ 오차)
        - 작을 때는 보수적(mean만 사용)
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean  # [Q1_mean, Q2_mean]

            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            delta1 = obs[2] - obs[0]   # TSP1 - T1
            delta2 = obs[3] - obs[1]   # TSP2 - T2
            error = torch.sqrt(delta1 ** 2 + delta2 ** 2)

            if error > err_thr:
                # 가변적 noise std 계산
                noise_std = torch.clamp(base_scale * error, min=min_std, max=max_std)
                std = torch.full_like(mean_action, noise_std)
                noise = torch.normal(mean=torch.zeros_like(mean_action), std=std)
                action = mean_action + noise
                print(f"탐색: err={error:.2f}, noise_std={noise_std:.2f}")
            else:
                action = mean_action
                print(f"보수적 행동: err={error:.2f} (<= {err_thr})")

            return torch.clamp(action, 0.0, 100.0)

    def reverse_error_act( # 오차가 기준치 보다 크면 거기서 탐색 진행 
            self,
            obs, 
            deterministic: bool = False,
            enable_grad: bool = False,
            err_thr: float = 10,        # |Tsp−T| > err_thr 이면 탐색
            noise_std: float = 10.0     # 가우시안 노이즈 표준편차 (PWM %)
        ):
            """
            반전된 1 + 4 탐색 전략:
            - 큰 오차(|Tsp - T| > err_thr) → mean + noise (탐색)
            - 작은 오차(|Tsp - T| ≤ err_thr) → mean만 사용 (보수적)
            """
            with torch.set_grad_enabled(enable_grad):
                dist = self(obs)
                mean_action = dist.mean  # [Q1_mean, Q2_mean]

                if deterministic:
                    return torch.clamp(mean_action, 0.0, 100.0)

                delta1 = obs[2] - obs[0]   # TSP1 - T1
                delta2 = obs[3] - obs[1]   # TSP2 - T2

                # 오차가 큰 경우에만 탐색
                in_explore_region = (
                    torch.abs(delta1) > err_thr or
                    torch.abs(delta2) > err_thr
                )

                if in_explore_region:
                    std = torch.full_like(mean_action, noise_std)
                    noise = torch.normal(mean=torch.zeros_like(mean_action), std=std)
                    action = mean_action + noise
                    print(f"탐색 (noise added): err1={delta1:.2f}, err2={delta2:.2f}")
                else:
                    action = mean_action
                    print(f"보수적 행동: err1={delta1:.2f}, err2={delta2:.2f}")

                return torch.clamp(action, 0.0, 100.0)

    def directional_override_act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        err_thr: float = 1.2
    ):
        """
        ε-greedy + directional override 기반 탐색 정책
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean  # [Q1, Q2]

            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            delta1 = obs[2] - obs[0]
            delta2 = obs[3] - obs[1]

            if delta1 > err_thr:
                Q1 = 100.0
                print(f"🔥 Q1 = 100 (delta1 = {delta1.item():.2f} > {err_thr})")
            elif delta1 < -err_thr:
                Q1 = 0.0
                print(f"❄️ Q1 = 0 (delta1 = {delta1.item():.2f} < -{err_thr})")
            else:
                Q1 = mean_action[0]
                print(f"✅ Q1 = mean ({Q1.item():.2f}) (|delta1| <= {err_thr})")

            if delta2 > err_thr:
                Q2 = 100.0
                print(f"🔥 Q2 = 100 (delta2 = {delta2.item():.2f} > {err_thr})")
            elif delta2 < -err_thr:
                Q2 = 0.0
                print(f"❄️ Q2 = 0 (delta2 = {delta2.item():.2f} < -{err_thr})")
            else:
                Q2 = mean_action[1]
                print(f"✅ Q2 = mean ({Q2.item():.2f}) (|delta2| <= {err_thr})")

            action = torch.tensor([Q1, Q2], device=obs.device)
            return torch.clamp(action, 0.0, 100.0)

    def guided_act(
            self,
            obs: torch.Tensor,
            deterministic: bool = False,
            enable_grad: bool = False,
            err_thr: float = 1.0,
            max_noise_std: float = 10.0,
            bias_scale: float = 0.5
        ):
        """
        TSP에 빠르게 수렴하기 위한 guided exploration 함수.

        - 큰 오차일수록 더 큰 탐색(std) + 오차 방향으로 bias 추가
        - 작은 오차는 학습된 policy의 평균(mean)만 사용
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean  # [Q1, Q2]

            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            # 오차 계산
            delta1 = obs[2] - obs[0]  # TSP1 - T1
            delta2 = obs[3] - obs[1]  # TSP2 - T2
            err_norm = torch.sqrt(delta1**2 + delta2**2)

            # 노이즈 std는 오차에 비례
            scaled_std = min(err_norm.item() * 5.0, max_noise_std)
            noise = torch.normal(
                mean=torch.zeros_like(mean_action),
                std=torch.full_like(mean_action, scaled_std)
            )

            # 오차 방향으로 bias 추가
            bias = torch.tensor([
                bias_scale * delta1.item(),
                bias_scale * delta2.item()
            ], device=obs.device)

            # 최종 행동 = mean + bias + noise
            action = mean_action + bias + noise
            print(action[0], action[1])
            return torch.clamp(action, 0.0, 100.0)

    # def act(self, obs, deterministic=False, enable_grad=False, exploration_std=0.03):
    #     """
    #     IQL 논문 스타일의 탐색:
    #     - deterministic: 평균만 사용
    #     - exploration: mean + N(0, σ²) (σ=0.03 추천)
    #     """
    #     with torch.set_grad_enabled(enable_grad):
    #         dist = self(obs)
    #         mean = dist.mean

    #         if deterministic:
    #             return torch.clamp(mean, 0.0, 100.0)  # 안전 범위 내 clip (선택)

    #         # IQL 논문식 exploration: mean + Gaussian noise
    #         noise = torch.normal(
    #             mean=torch.zeros_like(mean),
    #             std=exploration_std
    #         )
    #         action = mean + noise
    #         return torch.clamp(action, 0.0, 100.0)  # clamp 없으면 히터가 음수일 수도 있음

    #def online_act (self, obs, deterministic=False, enable_grad = False, )
    def online_act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        err_thr: float = 4.0,        # |Tsp−T| ≤ err_thr 일 때만 노이즈 추가
        noise_std: float = 10.0       # 가우시안 노이즈 표준편차 (PWM %)
    ):
        """
        1 + 4 탐색 전략:
        - 큰 오차(|Tsp - T| > err_thr) → 학습된 policy의 평균(mean)만 사용
        - 작은 오차(|Tsp - T| ≤ err_thr) → mean + N(0, noise_std²) 로 탐색
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean                      # [Q1_mean, Q2_mean]

            if deterministic:
                # 평가 모드: 항상 평균
                return torch.clamp(mean_action, 0.0, 100.0)

            # ── 현재 오차 계산 ───────────────────────────────────────
            delta1 = obs[2] - obs[0]   # TSP1 - T1
            delta2 = obs[3] - obs[1]   # TSP2 - T2

            in_safe_region = (
                torch.abs(delta1) <= err_thr and
                torch.abs(delta2) <= err_thr
            )

            if in_safe_region:
                noise = torch.normal(
                    mean=torch.zeros_like(mean_action),
                    std=noise_std,
                    device=obs.device
                )
                action = mean_action + noise
            else:
                action = mean_action

            return torch.clamp(action, 0.0, 100.0)



class DeterministicPolicy(nn.Module):
    # 결정론적 정책 
    # 상태를 받아서 직접 행동을 출력함 
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

 #### 
    def forward(self, obs):
        out = self.net(obs)         # ∈ [-1, 1]
        return 50.0 * (out + 1.0)   # → ∈ [0, 100]
        # return (self.net(obs))

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)