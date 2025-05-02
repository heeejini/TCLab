import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

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
        
    #def online_act (self, obs, deterministic=False, enable_grad = False, )
            
    def online_act(self, obs, deterministic=False, enable_grad=False, eps=0.05):
        
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            if deterministic:
                action = dist.mean
            else:
                if torch.rand(1).item() < eps:
                    print("----random----")
                    # 완전한 랜덤 액션
                    action = torch.FloatTensor(2).uniform_(0, 100).to(obs.device)
                   # print(action)
                else:
                    print("no random")
                    action = dist.sample()
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