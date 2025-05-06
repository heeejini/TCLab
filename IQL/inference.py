import torch
import argparse
import joblib
from pathlib import Path
from src.policy import GaussianPolicy, DeterministicPolicy
from src.eval_policy import simulator_policy

def load_policy(pt_path, obs_dim, act_dim, hidden_dim, n_hidden, deterministic):
    if deterministic:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim, n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim, n_hidden)
    policy.load_state_dict(torch.load(pt_path))
    return policy

def main(args):
    obs_dim, act_dim = 4, 2  # [T1, T2, TSP1, TSP2], [Q1, Q2]
    policy = load_policy(args.pt_path, obs_dim, act_dim, args.hidden_dim, args.n_hidden, args.deterministic_policy)
    policy.eval()

    all_results = []
    for episode_id in range(args.n_eval_episodes):
        for seed in range(args.n_seeds):
            result = simulator_policy(
                policy=policy,
                total_time_sec=args.total_time_sec,
                dt=args.sample_interval,
                log_root=args.log_dir / f"ep{episode_id}",
                seed=seed,
                ambient=args.ambient,
                deterministic=args.deterministic_policy,
                scaler=args.scaler
            )
            result.update({"episode": episode_id, "seed": seed})
            all_results.append(result)

    # 결과 요약
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df[["E1", "E2", "Over", "Under", "total_return"]].mean()
    print("\n[평가 요약]")
    print(summary)
    df.to_csv(args.log_dir / "inference_summary.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("./inference_logs"))
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--total-time-sec", type=int, default=1200)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--ambient", type=float, default=29.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--deterministic-policy", action="store_true")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main(args)
import torch
import argparse
import joblib
from pathlib import Path
from src.policy import GaussianPolicy, DeterministicPolicy
from src.eval_policy import simulator_policy

def load_policy(pt_path, obs_dim, act_dim, hidden_dim, n_hidden, deterministic):
    if deterministic:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim, n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim, n_hidden)
    policy.load_state_dict(torch.load(pt_path))
    return policy

def main(args):
    obs_dim, act_dim = 4, 2  # [T1, T2, TSP1, TSP2], [Q1, Q2]
    policy = load_policy(args.pt_path, obs_dim, act_dim, args.hidden_dim, args.n_hidden, args.deterministic_policy)
    policy.eval()

    all_results = []
    for episode_id in range(args.n_eval_episodes):
        for seed in range(args.n_seeds):
            result = simulator_policy(
                policy=policy,
                total_time_sec=args.total_time_sec,
                dt=args.sample_interval,
                log_root=args.log_dir / f"ep{episode_id}",
                seed=seed,
                ambient=args.ambient,
                deterministic=args.deterministic_policy,
                scaler=args.scaler
            )
            result.update({"episode": episode_id, "seed": seed})
            all_results.append(result)

    # 결과 요약
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df[["E1", "E2", "Over", "Under", "total_return"]].mean()
    print("\n[평가 요약]")
    print(summary)
    df.to_csv(args.log_dir / "inference_summary.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("./inference_logs"))
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--total-time-sec", type=int, default=1200)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--ambient", type=float, default=29.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--deterministic-policy", action="store_true")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main(args)
