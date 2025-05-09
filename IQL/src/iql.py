import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

EXP_ADV_MAX = 100


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    @staticmethod
    def _two_step(optimizer, loss_closure):
        """
        loss_closure() -> torch.Tensor
        └ 한번의 호출로 loss 계산·backward 까지 수행해야 한다.
        • SAM   : first_step → second_step  (2× forward/backward)
        • 일반 옵티마이저 : step() 한 번
        return (loss_val, loss_val_second)  # 후자는 SAM일 때만 유효
        """
        loss = loss_closure()
        loss.backward()
        if hasattr(optimizer, "first_step"):          # SAM 여부 판정
            optimizer.first_step(zero_grad=True)

            loss_second = loss_closure()              # perturbed weights
            loss_second.backward()
            optimizer.second_step(zero_grad=True)
            return loss.detach(), loss_second.detach()
        else:                                         # Adam / SGD 등
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return loss.detach(), None
    # ------------------------------------------------------------------ #
    def update(self, observations, actions, next_observations, rewards, terminals):
        # ---------- 0. 타깃 계산 (no_grad) ---------------------------------
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v   = self.vf(next_observations)
            targets  = rewards + (1. - terminals.float()) * self.discount * next_v

        # ---------- 1. V 업데이트 ------------------------------------------
        def v_loss_fn():
            v   = self.vf(observations)
            adv = target_q - v
            return asymmetric_l2_loss(adv, self.tau)

        v_loss, _ = self._two_step(self.v_optimizer, v_loss_fn)

        # ---------- 2. Q 업데이트 ------------------------------------------
        def q_loss_fn():
            qs = self.qf.both(observations, actions)
            return sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        q_loss, _ = self._two_step(self.q_optimizer, q_loss_fn)

        # ---------- 3. 타깃 Q EMA -----------------------------------------
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # ---------- 4. Policy 업데이트 -------------------------------------
        def policy_loss_fn():
            # advantage 재계산 (V가 갱신됐으므로 새로 평가)
            with torch.no_grad():
                v   = self.vf(observations)
                adv = target_q - v
                exp_adv = torch.exp(self.beta * adv).clamp(max=EXP_ADV_MAX)

            policy_out = self.policy(observations)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(actions)
            else:  # deterministic
                bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)

            return torch.mean(exp_adv * bc_losses)

        policy_loss, _ = self._two_step(self.policy_optimizer, policy_loss_fn)
        self.policy_lr_schedule.step()

        return {
            "v_loss":      v_loss.item(),
            "q_loss":      q_loss.item(),
            "policy_loss": policy_loss.item(),
            "reward_mean": rewards.mean().item(),
        }