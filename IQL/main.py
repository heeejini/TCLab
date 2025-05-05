from pathlib import Path

import numpy as np
import torch
from tqdm import trange
import wandb
from src.sam import SAM 

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy_sim, evaluate_policy_tclab
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_env_and_dataset(log, npz_path, max_episode_steps = None):
    
    log(f"Loading offline dataset from {npz_path}")
    print(f"Loading offline dataset from {npz_path}") 

    data = np.load(npz_path)
    dataset = {k: torchify(v) for k, v in data.items()}
    
    for k, v in dataset.items():
        log(f"  {k:17s} shape={tuple(v.shape)} dtype={v.dtype}")
    return None, dataset 

def build_optimizer_factory(args):
    if args.sam:
        # SAM( params, base_optimizer, **base_opt_kwargs, rho=... )
        return lambda params: SAM(params,
                                  torch.optim.Adam,
                                  lr=args.learning_rate,
                                  betas=(0.9, 0.999),
                                  rho=args.sam_rho)
    else:
        return lambda params: torch.optim.Adam(params,
                                               lr=args.learning_rate)



def main(args):
    torch.set_num_threads(1)
    wandb.init(
        project="tclab-project",
        name=args.exp_name,
        entity="jhj0628",
        config=vars(args)  # 모든 하이퍼파라미터 자동 저장
    )
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, args.npz_path, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)


    # def eval_policy():
    #     if args.method == "simulator":
    #         result = evaluate_policy_sim(policy, args)
    #     else:
    #         result = evaluate_policy_tclab(policy, args)

    #     log.row(result)

    def eval_policy( policy, args):
        if args.method == "simulator":
            return evaluate_policy_sim(policy, args)
        else:
            return evaluate_policy_tclab(policy, args)


        # eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
        #                          for _ in range(args.n_eval_episodes)])
        # normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        # log.row({
        #     'return mean': eval_returns.mean(),
        #     'return std': eval_returns.std(),
        #     'normalized return mean': normalized_returns.mean(),
        #     'normalized return std': normalized_returns.std(),
        # })

    # def eval_policy  안에 실제/simul class 로 가져오기 

    optimizer_factory = build_optimizer_factory(args)

        
    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim,
                        hidden_dim=args.hidden_dim,
                        n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=optimizer_factory,
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    with torch.no_grad():
            obs = dataset['observations'][:5000]
            act = dataset['actions'][:5000]
            adv = iql.qf(obs, act) - iql.vf(obs)
    print("[Init Advantage] mean:", adv.mean().item(),
            "std:", adv.std().item())
        
    best_total_error = float('inf')
    best_total_return = -float('inf')
    best_q_loss = float('inf')
    best_v_loss = float('inf')
    best_policy_loss = float('inf')

    best_step = -1
    # early stopping 파라미터 
    patience        = 6         
    min_delta_err   = 0.5         # error 가 0.5° 이하로만 개선되면 무시
    min_delta_ret   = 1.0         # return 이 1.0 이상 개선돼야 의미
    no_improve_cnt  = 0           # 연속 미개선 카운터
    stop_step       = None        # 조기 종료 시점 기록 

    for step in trange(args.n_steps):
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))

        if (step + 1) % 5_000 == 0:
            with torch.no_grad():
                test_obs = dataset['observations'][:5000]
                act = dataset['actions'][:5000]
                adv = iql.qf(test_obs, act) - iql.vf(test_obs)
        
                if isinstance(iql.policy, DeterministicPolicy):
                    test_act = iql.policy(test_obs).cpu().numpy()
                else:
                    dist = iql.policy(test_obs)
                    test_act = dist.sample().cpu().numpy()
            print(f"[Step {step+1}] Advantage mean={adv.mean():.4f}, std={adv.std():.4f}")
            print(f"[Debug {step+1}] action range : {test_act.min():.3f}  ~  {test_act.max():.3f}")

        if (step + 1) % args.eval_period == 0:
            metrics = eval_policy(iql.policy, args)
            metrics.update({'step': step + 1})

            full_log = loss_dict.copy()
            full_log.update(metrics)

            prev_best_err = best_total_error
            prev_best_ret = best_total_return

            # float 처리
            for k, v in full_log.items():
                if isinstance(v, torch.Tensor):
                    full_log[k] = v.item() if v.numel() == 1 else float(v.mean().item())
                elif isinstance(v, (np.ndarray, list)):
                    full_log[k] = float(np.mean(v))
                elif not isinstance(v, (int, float)):
                    full_log[k] = str(v)

            print(f"\n[Step {step+1}] Evaluation:")
            for k, v in full_log.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")

            # total error 계산 
            try:
                total_error = full_log['E1'] + full_log['E2'] + full_log['Over'] + full_log['Under']
            except KeyError:
                total_error = np.inf                       # 값이 없으면 무한대
            full_log['total_error'] = total_error

            # Best total_return 
            if total_error < best_total_error:
                best_total_error = total_error
                best_step = step + 1
                torch.save(iql.state_dict(), log.dir / 'best.pt')
                print(f"✅ [Step {step+1}] Best model saved!")

            if full_log.get('total_return', -1e9) > best_total_return:
                best_total_return = full_log['total_return']
                torch.save(iql.state_dict(), log.dir/'best_return.pt')
                print(f"✅ [Step {step+1}] Best model (Total Return) saved!")

            # Best q_loss 
            if full_log.get('q_loss', 1e9) < best_q_loss:
                best_q_loss = full_log['q_loss']
                torch.save(iql.state_dict(), log.dir/'best_q.pt')
                print(f"✅ [Step {step+1}] Best model (Q Loss) saved!")

            # Best v_loss 
            if full_log.get('v_loss', 1e9) < best_v_loss:
                best_v_loss = full_log['v_loss']
                torch.save(iql.state_dict(), log.dir/'best_v.pt')
                print(f"✅ [Step {step+1}] Best model (V Loss) saved!")

            # Best policy_loss 
            if full_log.get('policy_loss', 1e9) < best_policy_loss:
                best_policy_loss = full_log['policy_loss']
                torch.save(iql.state_dict(), log.dir/'best_policy.pt')
                print(f"✅ [Step {step+1}] Best model (Policy Loss) saved!")

            log.row(full_log)
            wandb.log(full_log)

            # early stopping 체크 
            improved = False

            if (prev_best_err - total_error) > min_delta_err:
                improved = True
                best_total_error = total_error   
                no_improve_cnt = 0
            elif (full_log['total_return'] - prev_best_ret) > min_delta_ret:
                improved = True
                best_total_return = full_log['total_return'] 
                no_improve_cnt = 0
            if not improved:
                no_improve_cnt += 1
                print(f"Early‑stop patience {no_improve_cnt}/{patience}")

            if no_improve_cnt >= patience:
                stop_step = step + 1
                print(f"\nEarly‑Stopping triggered at step {stop_step} !")
                break

    torch.save(iql.state_dict(), log.dir/'final.pt')
    if stop_step is not None:
        with open(log.dir / 'early_stop.txt', 'w') as f:
            f.write(f"Stopped early at step {stop_step} (no improvement for {patience} evals)\n")

    with open(log.dir/'best_info.txt', 'w') as f:
        f.write(f"Best Step (Total Error 기준): {best_step}\n")
        f.write(f"Best Total Error: {best_total_error:.3f}\n")
        f.write(f"Best Total Return: {best_total_return:.3f}\n")
        f.write(f"Best Q Loss: {best_q_loss:.3f}\n")
        f.write(f"Best V Loss: {best_v_loss:.3f}\n")
        f.write(f"Best Policy Loss: {best_policy_loss:.3f}\n")

    wandb.finish()
    log.close()
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default='tclab-mpc-iql')
    parser.add_argument('--log-dir', default='./sam')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**5 *3) # 50만 step 만 돌아도 충분히 수렴
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--stochastic-policy', action='store_false', dest='deterministic_policy')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1200)  # 20분 
    parser.add_argument('--sample_interval', type=float, default=5.0)
    parser.add_argument('--exp_name', default='iql_default')
    #C:\Users\User\TCLab\Data\MPC\first_reward.npz
    parser.add_argument('--npz-path', default="C:\\Users\\User\\TCLab\\dataset\\outputs\\first_reward.npz")
    parser.add_argument('--scaler')
    # main.py 맨 위 argparse 부분
    parser.add_argument("--sam", action="store_true",
                        help="Sharpness-Aware Minimization 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.03,
                        help="SAM perturbation half-width (ρ)")
    parser.add_argument('--method', default='simulator') # eval 시에 어떤 것을 통해서 할 지
    #C:\\Users\\Developer\\TCLab\\Data\\next_reward_timeerror_scaler.pkl
    #C:\\Users\\Developer\\TCLab\\Data\\next_reward_timeerror_scaler.pkl
    #  C:\\Users\\Developer\\TCLab\\Data\\next_reward_scaler.pkl
    main(parser.parse_args())
    #"C:/Users/Developer/TCLab/Data/next_reward_timeerror_scaler.pkl"
    # "C:/Users/Developer/TCLab/Data/next_reward_scaler.pkl" 
    # first_reward
    # "C:/Users/Developer/TCLab/Data/first_reward.pkl" 
    # main.py 맨 위 argparse 부분



"""
    평가기준:
        (1) E1|목표온도 – 측정온도| +  E2|목표온도 – 측정온도|  ← 주 평가기준
        (2) Over 평가용: max(0, 측정온도 - 목표온도) 
        (3) Under 평가용: max(0, 목표온도 - 측정온도) 
"""