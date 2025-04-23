from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy_sim, evaluate_policy_tclab


def get_env_and_dataset(log, npz_path, max_episode_steps = None):
    
    log(f"Loading offline dataset from {npz_path}")
    print(f"Loading offline dataset from {npz_path}") 

    data = np.load(npz_path)
    dataset = {k: torchify(v) for k, v in data.items()}
    
    for k, v in dataset.items():
        log(f"  {k:17s} shape={tuple(v.shape)} dtype={v.dtype}")
    return None, dataset 



# def get_env_and_dataset(log, env_name, max_episode_steps):
#     env = gym.make(env_name)
#     dataset = d4rl.qlearning_dataset(env)

#     if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
#         min_ret, max_ret = return_range(dataset, max_episode_steps)
#         log(f'Dataset returns have range [{min_ret}, {max_ret}]')
#         dataset['rewards'] /= (max_ret - min_ret)
#         dataset['rewards'] *= max_episode_steps
#     elif 'antmaze' in env_name:
#         dataset['rewards'] -= 1.

#     for k, v in dataset.items():
#         dataset[k] = torchify(v)

#     return env, dataset


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

        """
        평가기준:
            (1) E1|목표온도 – 측정온도| +  E2|목표온도 – 측정온도|  ← 주 평가기준
            (2) Over 평가용: max(0, 측정온도 - 목표온도) 
            (3) Under 평가용: max(0, 목표온도 - 측정온도) 
        """
    def eval_policy():
        if args.method == "simulator" :
            # 시뮬레이터 evaluate_policy_sim 호출 
            evaluate_policy_sim()

        else :
            # 실제 키트 eval 함수 호출 
            evaluate_policy_tclab()

        return 
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
         
        if (step+1) % args.eval_period == 0:
            eval_policy()

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
    parser.add_argument('--n-steps', type=int, default=10**4 * 5 ) # 50만 step 만 돌아도 충분히 수렴
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--npz-path', default='C:\\Users\\Developer\\TCLab\\Data\\mpc_dataset.npz')
    parser.add_argument('--method', description='simulator or tclab', default='simulator') # eval 시에 어떤 것을 통해서 할 지
    main(parser.parse_args())