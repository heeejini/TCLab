from pathlib import Path
import os
import numpy as np
import torch
from tqdm import trange
import wandb
import copy 

from src.sam import SAM
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (return_range, set_seed, Log, sample_batch, torchify,
                      evaluate_policy_sim, evaluate_policy_tclab, evaluate_extra_seeds, MetricTracker, EarlyStopping)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_env_and_dataset(log, npz_path, max_episode_steps=None):
    log(f"Loading offline dataset from {npz_path}")
    print(f"Loading offline dataset from {npz_path}")

    data = np.load(npz_path)
    dataset = {k: torchify(v) for k, v in data.items()}

    for k, v in dataset.items():
        log(f"  {k:17s} shape={tuple(v.shape)} dtype={v.dtype}")
    return None, dataset

def build_optimizer_factory(args):
    if args.sam:  # Sharpness-Aware Minimization
        return lambda params: SAM(
            params,
            torch.optim.Adam,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            rho=args.sam_rho,
        )
    else:
        return lambda params: torch.optim.Adam(
            params,
            lr=args.learning_rate,
        )

def main(args):
    torch.set_num_threads(1)

    wandb.init(
        project="tclab-project1",
        name=args.exp_name,
        config=vars(args),
    )



    log = Log(Path(args.log_dir) / args.env_name, vars(args))
    log(f"Log dir: {log.dir}")


    env, dataset = get_env_and_dataset(log, args.npz_path, args.max_episode_steps)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    set_seed(args.seed, env=env)


    if args.deterministic_policy:
        policy = DeterministicPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden
        )
    else:
        policy = GaussianPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden
        )


    def eval_policy(policy, args):
        if args.method == "simulator":
            return evaluate_policy_sim(policy, args)
        elif args.method == "real":
            return evaluate_policy_tclab(policy, args)

    optimizer_factory = build_optimizer_factory(args)
    iql = ImplicitQLearning(
        qf=TwinQ(
            obs_dim,
            act_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        ),
        vf=ValueFunction(
            obs_dim,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
        ),
        policy=policy,
        optimizer_factory=optimizer_factory,
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
    )

    with torch.no_grad():
        obs = dataset["observations"][:5000]
        act = dataset["actions"][:5000]
        adv = iql.qf(obs, act) - iql.vf(obs)
    print(
        "[Init Advantage] mean:",
        adv.mean().item(),
        "std:",
        adv.std().item(),
    )

    stop_step = None
    tracker = MetricTracker(log.dir)
    early   = EarlyStopping(patience=6, min_delta_err=0.5, min_delta_ret=1.0)

    stop_step = None                     
    for step in trange(args.n_steps):
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))

        if (step + 1) % 5_000 == 0:
            with torch.no_grad():
                obs = dataset["observations"][:5000]
                act = dataset["actions"][:5000]
                adv = iql.qf(obs, act) - iql.vf(obs)
            print(f"[{step+1}] Advantage μ={adv.mean():.4f}, σ={adv.std():.4f}")

        if (step + 1) % args.eval_period == 0:
            metrics = eval_policy(iql.policy, args)           # E1,E2,total_return …
            metrics.update(loss_dict)                         # q_loss,v_loss,policy_loss
            metrics["step"] = step + 1
            metrics["total_error"] = metrics.get("E1", 0) + metrics.get("E2", 0)

            wandb.log(metrics);      log.row(metrics)

            tracker.update_best(metrics, step+1, iql.state_dict())

            stop, _ = early.step(metrics["total_error"], metrics["total_return"])
            if stop:
                stop_step = step + 1
                print(f" Early-Stopping at step {stop_step}")
                break

    torch.save(iql.state_dict(), log.dir / "final.pt")
    if stop_step:
        (log.dir / "early_stop.txt").write_text(
            f"Stopped at step {stop_step} (no improvement for {early.patience} evals)\n"
        )

    with open(log.dir / "best_info.txt", "w") as f:
        f.write(f"Best step           : {tracker.best_step}\n")
        f.write(f"Best total_error    : {tracker.best_total_error:.3f}\n")
        f.write(f"Best total_return   : {tracker.best_total_return:.3f}\n")
        f.write(f"Best q_loss         : {tracker.best_q_loss:.3f}\n")
        f.write(f"Best v_loss         : {tracker.best_v_loss:.3f}\n")
        f.write(f"Best policy_loss    : {tracker.best_policy_loss:.3f}\n")

    if args.eval_seeds:
        evaluate_extra_seeds(
            policy=iql.policy,
            args=args,
            log=log,
            eval_fn=eval_policy,       # → simulator / real 판단 포함
            filename="extra_eval.csv"
        )

    wandb.finish()
    log.close()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="tclab-mpc-iql")
    parser.add_argument("--log-dir", default="./new")
    parser.add_argument("--seed", type=int, default=3)

    # 모델 & 학습 파라미터
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=10 ** 5 * 3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument(
        "--stochastic-policy", action="store_false", dest="deterministic_policy"
    )

    parser.add_argument("--eval-period", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--sample_interval", type=float, default=5.0)

    parser.add_argument("--exp_name", default="iql_default")
    parser.add_argument("--npz-path", default="C:\\Users\\Developer\\TCLab\\Data\\reward\\first.npz")
    parser.add_argument("--scaler", default="C:\\Users\\Developer\\TCLab\\Data\\reward\\first.pkl")

    parser.add_argument("--sam", action="store_true", help="Sharpness-Aware Minimization 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.03, help="SAM perturbation half-width (ρ)")

    parser.add_argument("--method", default="simulator")
    parser.add_argument("--reward_type", type=int, default=1)

    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2],
        help="추가 평가용 random seed 목록 (예: --eval-seeds 0 1 2 )",
    )


    main(parser.parse_args())
