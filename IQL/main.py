from pathlib import Path

import numpy as np
import torch
from tqdm import trange

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


def main(args):
    torch.set_num_threads(1)
    
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

    

        
    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    for step in trange(args.n_steps):
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))

        # ───────────────────────────────────────────────────────────────
        # ① 5 k step마다 정책 출력 범위 확인
        # ───────────────────────────────────────────────────────────────
        if (step + 1) % 5_000 == 0:         # ← 주기 원하는 대로
            with torch.no_grad():
                test_obs = dataset['observations'][:1000]        # 1 000개 샘플
                test_act = iql.policy(test_obs).cpu().numpy()    # 네트워크 **현재** 출력
            print(f"[Debug {step+1}] action range : {test_act.min():.3f}  ~  {test_act.max():.3f}")

        # ───────────────────────────────────────────────────────────────
        # ② (선택) 평가 & 로그
        # ───────────────────────────────────────────────────────────────
        if (step + 1) % args.eval_period == 0:
            metrics = eval_policy(iql.policy, args)
            metrics.update({'step': step + 1})

            # → loss_dict 병합
            full_log = loss_dict.copy()
            full_log.update(metrics)

            # float 처리 (안전하게)
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

            log.row(full_log)

        # reward 뽑아야될 듯 
        # if (step + 1) % 1000 == 0 : 
        #     # 1000 step 마다 로그 출력 
        #     print(f"[Step {step+1}] "
        #       f"V_loss: {loss_dict['v_loss']:.4f}, "
        #       f"Q_loss: {loss_dict['q_loss']:.4f}, "
        #       f"Policy_loss: {loss_dict['policy_loss']:.4f}",
        #       f"R̄:{loss_dict['reward_mean']:.3f}")
        #     loss_dict.update({'step': step + 1})
        #     log.row(loss_dict)


    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default='tclab-mpc-iql')
    parser.add_argument('--log-dir', default='./runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**5 * 5 ) # 50만 step 만 돌아도 충분히 수렴
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy',default=True, action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1200)  # 20분 
    parser.add_argument('--sample_interval', type=float, default=5.0)
    parser.add_argument('--npz-path', default='C:\\Users\\Developer\\TCLab\\Data\\mpc_dataset.npz')
    parser.add_argument('--method', default='simulator') # eval 시에 어떤 것을 통해서 할 지
    main(parser.parse_args())


"""
    평가기준:
        (1) E1|목표온도 – 측정온도| +  E2|목표온도 – 측정온도|  ← 주 평가기준
        (2) Over 평가용: max(0, 측정온도 - 목표온도) 
        (3) Under 평가용: max(0, 목표온도 - 측정온도) 
"""