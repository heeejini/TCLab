from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import csv
import numpy as np
import torch
import pandas as pd
from tqdm import trange
import wandb

import matplotlib.pyplot as plt

# Project modules --------------------------------------------------------------
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import (
    torchify,
    evaluate_policy_sim,
    evaluate_policy_tclab,
    set_seed,
)
from src.eval_mpc import simulate_mpc_episode, generate_random_tsp, collect_mpc_real_episode

def plot_single_rollout(result_dict, save_path: Path, dt: float = 5.0):
    T1, T2 = result_dict["T1"], result_dict["T2"]
    TSP1, TSP2 = result_dict["Tsp1"], result_dict["Tsp2"]
    Q1, Q2 = result_dict["Q1"], result_dict["Q2"]
    steps = len(T1)
    t = np.arange(steps) * dt

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(t, T1, label="T1"); ax[0].plot(t, TSP1, "--", label="TSP1")
    ax[0].plot(t, T2, label="T2"); ax[0].plot(t, TSP2, ":", label="TSP2")
    ax[0].grid(); ax[0].legend(); ax[0].set_ylabel("Temp (°C)")
    ax[1].plot(t, Q1, label="Q1"); ax[1].plot(t, Q2, label="Q2")
    ax[1].grid(); ax[1].legend(); ax[1].set_ylabel("Heater (%)"); ax[1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def build_iql_skeleton(
    obs_dim: int,
    act_dim: int,
    pt_path: Path,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    deterministic_policy: bool = True,
) -> ImplicitQLearning:
    if deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim, n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim, n_hidden)

    qf = TwinQ(obs_dim, act_dim, hidden_dim, n_hidden)
    vf = ValueFunction(obs_dim, hidden_dim, n_hidden)

    dummy_opt = lambda params: torch.optim.Adam(params, lr=1e-3)

    iql = ImplicitQLearning(
        qf=qf,
        vf=vf,
        policy=policy,
        optimizer_factory=dummy_opt,
        max_steps=0,
        tau=0.0,
        beta=0.0,
        alpha=0.0,
        discount=0.99,
    )

    ckpt = torch.load(pt_path, map_location=torch.device("cpu"))
    iql.load_state_dict(ckpt, strict=False)
    iql.eval()

    return iql

def run_inference_multiple_seeds(args: argparse.Namespace):
    wandb.init(
        project="tclab-project1",
        name=f"infer_{args.method}",
        config=vars(args),
    )

    ckpt = torch.load(args.pt_path, map_location="cpu")
    first_layer_key = next(k for k in ckpt if k.startswith("policy.net.0.weight"))
    obs_dim = ckpt[first_layer_key].shape[1]
    act_dim = ckpt["policy.log_std"].shape[0]

    iql = build_iql_skeleton(
        obs_dim,
        act_dim,
        args.pt_path,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        deterministic_policy=args.deterministic_policy,
    )
    policy = iql.policy
    args.scaler = args.scaler_path

    # all_results = []

    # for tsp_seed in args.epi_gen_seeds:
    #     tmp_args = argparse.Namespace(**vars(args))
    #     tmp_args.seed = tsp_seed

    #     if args.method == "simulator":
    #         result = evaluate_policy_sim(policy, tmp_args)
    #     elif args.method == "real":
    #         result = evaluate_policy_tclab(policy, tmp_args)
    #     else:
    #         raise ValueError(f"Unknown method: {args.method}")

    #     result.update({"tsp_seed": tsp_seed})
    #     all_results.append(result)

    #     seed_dir = args.log_dir / f"seed{tsp_seed}"
    #     seed_dir.mkdir(parents=True, exist_ok=True)
    #     plot_single_rollout(result, save_path=seed_dir / "rollout.png", dt=args.sample_interval)

    # plot_average_rollout(all_results, save_path=args.log_dir / "avg_rollout.png", dt=args.sample_interval)

    # df = pd.DataFrame(all_results)
    # summary = df[["E1", "E2", "Over", "Under", "total_return"]].mean()

    #     # wandb 로깅
    # wandb.log({f"IQL/{k}": v for k, v in summary.items()})


    # print("\n──────── Evaluation Summary (Averaged over TSP seeds) ────────")
    # print(summary)
    # print("────────────────────────────────────────────────────────────")

    #df.to_csv(args.log_dir / "inference_summary.csv", index=False)

    print("\n🔍 Running MPC evaluation for the same seeds...")
    mpc_results = []
    for seed in args.epi_gen_seeds:
        if args.method == "simulator":
            mpc_result = simulate_mpc_episode(seed=seed, args=args)
        else:
            mpc_result = collect_mpc_real_episode(seed=seed, args=args)

        mpc_result["seed"] = seed
        mpc_results.append(mpc_result)

        seed_dir = args.log_dir / f"mpc_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        plot_single_rollout(mpc_result, save_path=seed_dir / "rollout.png", dt=args.sample_interval)
        
                # ✅ CSV 저장
        csv_path = seed_dir / "rollout.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "T1", "T2", "Q1", "Q2", "TSP1", "TSP2"])
            t = np.arange(len(mpc_result["T1"])) * args.sample_interval
            for i in range(len(t)):
                writer.writerow([
                    t[i],
                    mpc_result["T1"][i],
                    mpc_result["T2"][i],
                    mpc_result["Q1"][i],
                    mpc_result["Q2"][i],
                    mpc_result["Tsp1"][i],
                    mpc_result["Tsp2"][i],
                ])
    mpc_df = pd.DataFrame(mpc_results)
    mpc_df.to_csv(args.log_dir / "mpc_summary.csv", index=False)

    mpc_avg = mpc_df[["E1", "E2", "Over", "Under", "total_return"]].mean()
    wandb.log({f"MPC/{k}": v for k, v in mpc_avg.items()})

    print("\n[MPC Average Summary over seeds]")
    print(mpc_avg)

    with open(args.log_dir / "mpc_result.txt", "w") as f:
        for k, v in mpc_avg.items():
            f.write(f"{k}: {v:.3f}\n")

def plot_average_rollout(results_list, save_path: Path, dt: float = 5.0):
    T1 = np.mean([r["T1"] for r in results_list], axis=0)
    T2 = np.mean([r["T2"] for r in results_list], axis=0)
    TSP1 = np.mean([r["Tsp1"] for r in results_list], axis=0)
    TSP2 = np.mean([r["Tsp2"] for r in results_list], axis=0)
    Q1 = np.mean([r["Q1"] for r in results_list], axis=0)
    Q2 = np.mean([r["Q2"] for r in results_list], axis=0)
    steps = len(T1)
    t = np.arange(steps) * dt

    fig, ax = plt.subplots(2,1,figsize=(10,8))
    ax[0].plot(t, T1, label="T1"); ax[0].plot(t, TSP1, "--", label="TSP1")
    ax[0].plot(t, T2, label="T2"); ax[0].plot(t, TSP2, ":", label="TSP2")
    ax[0].grid(); ax[0].legend(); ax[0].set_ylabel("Temp (°C)")
    ax[1].plot(t, Q1, label="Q1"); ax[1].plot(t, Q2, label="Q2")
    ax[1].grid(); ax[1].legend(); ax[1].set_ylabel("Heater (%)"); ax[1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def main():
    parser = argparse.ArgumentParser()
    # C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/online_directional_override_act_err_thr1.0_05-12-1740/05-12-25_17.40.25_kdyy/best.pt 
    # online tuning best 
    parser.add_argument('--pt-path', default="C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/online_directional_override_act_err_thr1.0_05-12-1740/05-12-25_17.40.25_kdyy/best.pt")
    #parser.add_argument('--pt-path', default="C:/Users/Developer/TCLab/IQL/sam/tclab-mpc-iql/05-12-25_10.45.27_eemw/best.pt")
    parser.add_argument("--scaler-path", default="C:/Users/Developer/TCLab/Data/future.pkl")
    parser.add_argument("--log-dir", type=Path, default=Path("./inference_results_online_simul"))
    parser.add_argument("--method", choices=["simulator", "real"], default="simulator")
    parser.add_argument("--epi-gen-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--reward-type", type=int, default=3)
    parser.add_argument("--ambient", type=float, default=29.0)

    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)  # policy 고정
    run_inference_multiple_seeds(args)
    wandb.finish()

if __name__ == "__main__":
    main()
