import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import random 

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

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

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()

    def epsilon_act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False,
        exp_prob: float = 0.5,    
        noise_std: float = 10.0 
    ) -> torch.Tensor:

        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)                   
            mean_action = dist.mean            

            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            if torch.rand(1).item() < exp_prob:
                delta1 = obs[2] - obs[0]   # TSP1 - T1
                delta2 = obs[3] - obs[1]   # TSP2 - T2

                noise = torch.randn_like(mean_action) * noise_std

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


    def reverse_error_act( # 오차가 기준치 보다 크면 거기서 탐색 진행 
            self,
            obs, 
            deterministic: bool = False,
            enable_grad: bool = False,
            err_thr: float = 10,      
            noise_std: float = 10.0     
        ):

            with torch.set_grad_enabled(enable_grad):
                dist = self(obs)
                mean_action = dist.mean 

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

        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            mean_action = dist.mean  # [Q1, Q2]

            if deterministic:
                return torch.clamp(mean_action, 0.0, 100.0)

            delta1 = obs[2] - obs[0]
            delta2 = obs[3] - obs[1]

            if delta1 > err_thr:
                Q1 = 100.0
           #     print(f"🔥 Q1 = 100 (delta1 = {delta1.item():.2f} > {err_thr})")
            elif delta1 < -err_thr:
                Q1 = 0.0
           #     print(f"❄️ Q1 = 0 (delta1 = {delta1.item():.2f} < -{err_thr})")
            else:
                Q1 = mean_action[0]
           #     print(f"✅ Q1 = mean ({Q1.item():.2f}) (|delta1| <= {err_thr})")

            if delta2 > err_thr:
                Q2 = 100.0
           #     print(f"🔥 Q2 = 100 (delta2 = {delta2.item():.2f} > {err_thr})")
            elif delta2 < -err_thr:
                Q2 = 0.0
           #     print(f"❄️ Q2 = 0 (delta2 = {delta2.item():.2f} < -{err_thr})")
            else:
                Q2 = mean_action[1]
           #     print(f"✅ Q2 = mean ({Q2.item():.2f}) (|delta2| <= {err_thr})")

            action = torch.tensor([Q1, Q2], device=obs.device)
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