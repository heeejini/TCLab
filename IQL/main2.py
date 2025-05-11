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
                      evaluate_policy_sim, evaluate_policy_tclab)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------------------------------------------------------------- #
# 데이터 로딩 유틸
# --------------------------------------------------------------------------- #
def get_env_and_dataset(log, npz_path, max_episode_steps=None):
    log(f"Loading offline dataset from {npz_path}")
    print(f"Loading offline dataset from {npz_path}")

    data = np.load(npz_path)
    dataset = {k: torchify(v) for k, v in data.items()}

    for k, v in dataset.items():
        log(f"  {k:17s} shape={tuple(v.shape)} dtype={v.dtype}")
    return None, dataset


# --------------------------------------------------------------------------- #
# 옵티마이저 팩토리
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #
def main(args):
    torch.set_num_threads(1)

    # wandb 세팅
    wandb.init(
        project="tclab-project1",
        name=args.exp_name,
        entity="jhj0628",
        config=vars(args),
    )

    # 로그 객체
    log = Log(Path(args.log_dir) / args.env_name, vars(args))
    log(f"Log dir: {log.dir}")

    # 환경 & 데이터
    env, dataset = get_env_and_dataset(log, args.npz_path, args.max_episode_steps)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    set_seed(args.seed, env=env)

    # 정책
    if args.deterministic_policy:
        policy = DeterministicPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden
        )
    else:
        policy = GaussianPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden
        )

    # 평가 함수
    def eval_policy(policy, args):
        if args.method == "simulator":
            return evaluate_policy_sim(policy, args)
        elif args.method == "real":
            return evaluate_policy_tclab(policy, args)

    # IQL
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

    # Advantage 디버깅
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

    # best 값들 (Early-stopping용)
    best_total_error = float("inf")
    best_total_return = -float("inf")
    best_q_loss = float("inf")
    best_v_loss = float("inf")
    best_policy_loss = float("inf")
    best_step = -1

    # Early-stopping 파라미터
    patience = 6
    min_delta_err = 0.5
    min_delta_ret = 1.0
    no_improve_cnt = 0
    stop_step = None

    # ------------------------------- 학습 루프 ------------------------------- #
    for step in trange(args.n_steps):
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))

        # 디버깅 출력
        if (step + 1) % 5_000 == 0:
            with torch.no_grad():
                test_obs = dataset["observations"][:5000]
                act = dataset["actions"][:5000]
                adv = iql.qf(test_obs, act) - iql.vf(test_obs)

                if isinstance(iql.policy, DeterministicPolicy):
                    test_act = iql.policy(test_obs).cpu().numpy()
                else:
                    dist = iql.policy(test_obs)
                    test_act = dist.sample().cpu().numpy()

            print(
                f"[Step {step+1}] Advantage mean={adv.mean():.4f}, "
                f"std={adv.std():.4f}"
            )
            print(
                f"[Debug {step+1}] action range : "
                f"{test_act.min():.3f}  ~  {test_act.max():.3f}"
            )

        # 주기적 평가
        if (step + 1) % args.eval_period == 0:
            metrics = eval_policy(iql.policy, args)
            metrics.update({"step": step + 1})

            full_log = loss_dict.copy()
            full_log.update(metrics)

            # 텐서/배열 → float 변환
            for k, v in full_log.items():
                if isinstance(v, torch.Tensor):
                    full_log[k] = v.item() if v.numel() == 1 else float(v.mean().item())
                elif isinstance(v, (np.ndarray, list)):
                    full_log[k] = float(np.mean(v))
                elif not isinstance(v, (int, float)):
                    full_log[k] = str(v)

            # total_error 계산
            try:
                total_error = (
                    full_log["E1"]
                    + full_log["E2"]
                    + full_log["Over"]
                    + full_log["Under"]
                )
            except KeyError:
                total_error = np.inf
            full_log["total_error"] = total_error

            # 콘솔 출력
            print(f"\n[Step {step+1}] Evaluation:")
            for k, v in full_log.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")

            prev_best_error = best_total_error
            prev_best_return = best_total_return

            # 모델 저장
            if total_error < best_total_error:
                best_total_error = total_error
                best_step = step + 1
                torch.save(iql.state_dict(), log.dir / "best.pt")
                print(f"✅ [Step {step+1}] Best model saved!")

            if full_log.get("total_return", -1e9) > best_total_return:
                best_total_return = full_log["total_return"]
                torch.save(iql.state_dict(), log.dir / "best_return.pt")

            if full_log.get("q_loss", 1e9) < best_q_loss:
                best_q_loss = full_log["q_loss"]
                torch.save(iql.state_dict(), log.dir / "best_q.pt")

            if full_log.get("v_loss", 1e9) < best_v_loss:
                best_v_loss = full_log["v_loss"]
                torch.save(iql.state_dict(), log.dir / "best_v.pt")

            if full_log.get("policy_loss", 1e9) < best_policy_loss:
                best_policy_loss = full_log["policy_loss"]
                torch.save(iql.state_dict(), log.dir / "best_policy.pt")

            log.row(full_log)
            wandb.log(full_log)
            
            # early-stopping 체크
            improved = False
            if (prev_best_error - total_error) > min_delta_err:
                improved = True
                no_improve_cnt = 0
            elif (full_log["total_return"] - prev_best_return) > min_delta_ret:
                improved = True
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
                print(f"Early-stop patience {no_improve_cnt}/{patience}")

            if no_improve_cnt >= patience:
                stop_step = step + 1
                print(f"\nEarly-Stopping triggered at step {stop_step} !")
                break

    # --------------------- 학습 종료 후: 모델 저장 --------------------- #
    torch.save(iql.state_dict(), log.dir / "final.pt")
    if stop_step is not None:
        with open(log.dir / "early_stop.txt", "w") as f:
            f.write(
                f"Stopped early at step {stop_step} "
                f"(no improvement for {patience} evals)\n"
            )

    with open(log.dir / "best_info.txt", "w") as f:
        f.write(f"Best Step (Total Error 기준): {best_step}\n")
        f.write(f"Best Total Error: {best_total_error:.3f}\n")
        f.write(f"Best Total Return: {best_total_return:.3f}\n")
        f.write(f"Best Q Loss: {best_q_loss:.3f}\n")
        f.write(f"Best V Loss: {best_v_loss:.3f}\n")
        f.write(f"Best Policy Loss: {best_policy_loss:.3f}\n")

    # ---------------------- Extra evaluation (seed 2/3/4) ---------------------- #
    if args.eval_seeds:
        print("\n🔍 Extra evaluation with seeds:", args.eval_seeds)

        extra_rows = []
        for s in args.eval_seeds:
            tmp_args = copy.copy(args)
            tmp_args.seed = s
            metrics = eval_policy(iql.policy, tmp_args)

            total_error = (metrics["E1"] + metrics["E2"] +
                        metrics["Over"] + metrics["Under"])
            metrics.update({"total_error": total_error, "seed": s})

            # 콘솔 출력
            print(f"\n  Seed {s}:")
            print(f"    total_return = {metrics['total_return']:.3f}")
            print(f"    total_error  = {metrics['total_error']:.3f}")

            extra_rows.append(metrics)          # ✅ 먼저 리스트에 누적
            log.row({f"extra_s{s}_{k}": v for k, v in metrics.items()})
            wandb.log({f"extra_s{s}_{k}": v for k, v in metrics.items()})

        # --- 모든 시드 수집 후 Table 한 번 생성 ---
            tbl = wandb.Table(columns=["seed", "total_error", "total_return"])

            for r in extra_rows:
                seed = r.get("seed")
                err  = r.get("total_error")
                ret  = r.get("total_return")

                if isinstance(err, (int, float)) and isinstance(ret, (int, float)):
                    tbl.add_data(seed, err, ret)
                else:
                    print(f"⚠️ 테이블 생략: seed={seed}, error={err}, return={ret}")

            wandb.log({"extra_eval_table": tbl})


        avg_return = np.mean([m["total_return"] for m in extra_rows])
        avg_error = np.mean([m["total_error"] for m in extra_rows])

        # 요청: avg_error 값을 total_error 로 간주
        avg_metrics = {
            "seed": "avg",
            "total_return": avg_return,
            "total_error": avg_error,
        }

        print(
            f"\n📊 Avg over seeds {args.eval_seeds}: "
            f"total_return = {avg_return:.3f},  total_error = {avg_error:.3f}"
        )

        wandb.log({"extra_avg_total_return": avg_return, "extra_avg_total_error": avg_error})
        log.row(avg_metrics)

        # CSV 저장
        import csv

        csv_path = log.dir / "extra_eval.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(extra_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extra_rows)
            writer.writerow(avg_metrics)
        print(f"✅ Extra-evaluation results saved to {csv_path}")

    # ----------------------------------------------------------------------- #
    wandb.finish()
    log.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="tclab-mpc-iql")
    parser.add_argument("--log-dir", default="./sam")
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

    # 평가 주기
    parser.add_argument("--eval-period", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--sample_interval", type=float, default=5.0)

    # 경로
    parser.add_argument("--exp_name", default="iql_default")
    
    #first_reward / next_reward_scaler
    parser.add_argument("--npz-path", default="C:/Users/User/tclab1/Data/next_reward_scaler.npz")
    parser.add_argument("--scaler", default="C:/Users/User/tclab1/Data/next_reward_scaler.pkl")

    # SAM
    parser.add_argument("--sam", action="store_true", help="Sharpness-Aware Minimization 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.03, help="SAM perturbation half-width (ρ)")

    # 평가 방식
    parser.add_argument("--method", default="simulator")
    parser.add_argument("--reward_type", type=int, default=2)

    # 📌 추가: extra evaluation seeds
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=None,
        help="추가 평가용 random seed 목록 (예: --eval-seeds 1 2 3 )",
    )

    main(parser.parse_args())
