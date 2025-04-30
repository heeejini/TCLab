import torch
import numpy as np
from pathlib import Path
from tqdm import trange
import joblib
import wandb

from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify, Log, set_seed, sample_batch, evaluate_policy_sim


def rollout_simulator(policy, buffer, reward_scaler, args):
    from src.eval_policy import generate_random_tsp
    from tclab import setup

    lab = setup(connected=False)
    env = lab(synced=False)
    env.Q1(0); env.Q2(0)
    env._T1 = env._T2 = args.ambient

    steps = int(args.max_episode_steps / args.sample_interval)
    Tsp1 = generate_random_tsp(args.max_episode_steps, args.sample_interval)
    Tsp2 = generate_random_tsp(args.max_episode_steps, args.sample_interval)

    T1, T2 = env.T1, env.T2
    policy.eval()

    for k in trange(steps, desc="rollout"):
        obs = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=args.deterministic_policy).cpu().numpy()

        Q1 = float(np.clip(action[0], 0, 100))
        Q2 = float(np.clip(action[1], 0, 100))
        env.Q1(Q1); env.Q2(Q2)

        next_T1, next_T2 = env.T1, env.T2
        next_obs = np.array([next_T1, next_T2, Tsp1[k], Tsp2[k]], dtype=np.float32)

        err1 = Tsp1[k] - next_T1
        err2 = Tsp2[k] - next_T2
        raw_reward = -np.sqrt(err1**2 + err2**2)
        reward = reward_scaler.transform([[raw_reward]])[0][0]

        done = (k == steps - 1)

        buffer['observations'].append(obs)
        buffer['actions'].append([Q1, Q2])
        buffer['next_observations'].append(next_obs)
        buffer['rewards'].append(reward)
        buffer['terminals'].append(done)

        T1, T2 = next_T1, next_T2


def online_finetune(args):
    torch.set_num_threads(1)
    wandb.init(project="tclab-project", name=args.exp_name, config=vars(args))
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    set_seed(args.seed)

    obs_dim, act_dim = 4, 2
    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden)

    qf = TwinQ(obs_dim, act_dim, args.hidden_dim, args.n_hidden)
    vf = ValueFunction(obs_dim, args.hidden_dim, args.n_hidden)
    iql = ImplicitQLearning(
        qf=qf, vf=vf, policy=policy,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.learning_rate),
        max_steps=args.n_steps, tau=args.tau, beta=args.beta,
        discount=args.discount, alpha=args.alpha)


    iql.load_state_dict(torch.load(args.pt_path))
    print(f"Loaded pretrained IQL model from: {args.pt_path}")

    reward_scaler = joblib.load(args.scaler)


    buffer = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": []
    }

    for episode in range(args.n_episodes):
        rollout_simulator(iql.policy, buffer, reward_scaler, args)

        dataset = {
            k: torchify(np.array(v, dtype=np.float32))
            for k, v in buffer.items()
        }

        for _ in range(args.update_per_episode):
            batch = sample_batch(dataset, args.batch_size)
            loss_dict = iql.update(**batch)

        metrics = evaluate_policy_sim(iql.policy, args)
        metrics.update({"episode": episode})
        metrics.update(loss_dict)

        log.row(metrics)
        wandb.log(metrics)

    torch.save(iql.state_dict(), log.dir/'final_online.pt')
    print(f"Final fine-tuned model saved to: {log.dir/'final_online.pt'}")
    wandb.finish()
    log.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt-path',  default="C:\\Users\\Developer\\TCLab\\IQL\\cum_reward\\tclab-mpc-iql\\04-30-25_11.09.02_usmn\\best.pt")
    parser.add_argument('--scaler', default="C:\\Users\\Developer\\TCLab\\Data\\first_reward.pkl")
    parser.add_argument('--exp_name', default="online_ft")
    parser.add_argument('--env-name', default="tclab-online")
    parser.add_argument('--log-dir', default="./logs_online")
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--sample_interval', type=float, default=5.0)
    parser.add_argument('--n-steps', type=int, default=500) # 50만 step 만 돌아도 충분히 수렴
    parser.add_argument('--n-episodes', type=int, default=250)
    parser.add_argument('--update_per_episode', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ambient', type=float, default=29.0)
    parser.add_argument('--stochastic-policy', action='store_false', dest='deterministic_policy')
    
    args = parser.parse_args()

    online_finetune(args)
