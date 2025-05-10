import torch
import numpy as np
from pathlib import Path
from tqdm import trange
import joblib
import wandb
import copy 
from pathlib import Path 
from datetime import datetime
import json 
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify, Log, set_seed, sample_batch, evaluate_policy_sim, evaluate_policy_tclab
from src.sam import SAM 
from src.eval_policy import compute_reward 

def eval_policy( policy, args):
    if args.method == "simulator":
        return evaluate_policy_sim(policy, args)
    else:
        return evaluate_policy_tclab(policy, args)



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


def rollout_tclab(policy, buffer, reward_scaler, args):
    """
    온라인 학습용 데이터를 실제 TCLab 보드에서 수집하여 buffer 에 추가한다.
    시뮬레이터용 rollout 과 동일하게
      - 관측 = [T1, T2, TSP1, TSP2]
      - 행동  = [Q1, Q2]  (0‑100 %)
      - 리워드 = -sqrt(err1² + err2²)  (optionally scaled)
    를 기록한다.
    """
    import time
    import numpy as np
    import joblib
    from tclab import TCLab
    from tqdm import trange
    from src.eval_policy import generate_random_tsp
    from src.util import torchify

    dt       = args.sample_interval      
    steps    = int(args.max_episode_steps / dt)
    ambient  = 29.0             
    policy.eval()

    # ── set‑point 프로파일 ──────────────────────────────────────
    Tsp1 = generate_random_tsp(args.max_episode_steps, dt)
    Tsp2 = generate_random_tsp(args.max_episode_steps, dt)

    # ── 보드 연결 ────────────────────────────────────────────────
    with TCLab() as arduino:
        arduino.LED(100)
        arduino.Q1(0); arduino.Q2(0)

        # (선택) 냉각 대기
        while arduino.T1 > ambient or arduino.T2 > ambient:
            time.sleep(10)

        T1, T2 = arduino.T1, arduino.T2  # 초기 센서값

        for k in trange(steps, desc="rollout‑tclab"):
            loop_start = time.time()

            obs = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)
            with torch.no_grad():
                action = policy.act(torchify(obs), deterministic=args.deterministic_policy).cpu().numpy()

            Q1 = float(np.clip(action[0], 0, 100))
            Q2 = float(np.clip(action[1], 0, 100))
            arduino.Q1(Q1); arduino.Q2(Q2)

            time.sleep(dt)                    # dt 초 대기 
            next_T1, next_T2 = arduino.T1, arduino.T2
            
            if args.reward_type == 1:               # 현재 오차
                err1 = Tsp1[k] - T1
                err2 = Tsp2[k] - T2
            else:                                   # 다음 오차 (default=2)
                if k < steps - 1:
                    err1 = Tsp1[k + 1] - next_T1
                    err2 = Tsp2[k + 1] - next_T2
                else:
                    err1 = Tsp1[k] - next_T1
                    err2 = Tsp2[k] - next_T2
            reward = compute_reward(err1, err2, reward_scaler)
            done   = (k == steps - 1)

            next_obs = np.array(
                [next_T1, next_T2,
                Tsp1[min(k + 1, steps - 1)],
                Tsp2[min(k + 1, steps - 1)]],
                dtype=np.float32
            )

            buffer.add_transition(obs, [Q1, Q2], next_obs, reward, done)
            T1, T2 = next_T1, next_T2
        arduino.Q1(0); arduino.Q2(0)


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
        ### 
        env.update(t=k*args.sample_interval)
        #### 
        obs = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)
        with torch.no_grad():
            action = policy.directional_override_act(torchify(obs), deterministic=args.deterministic_policy).cpu().numpy()

        
        Q1 = float(np.clip(action[0], 0, 100))
        Q2 = float(np.clip(action[1], 0, 100))
        print(f"😀 Q1 가열율 : {Q1}, Q2 가열율 : {Q2}" )
        env.Q1(Q1); env.Q2(Q2)

        env.update(t=(k + 1) * args.sample_interval)
        next_T1, next_T2 = env.T1, env.T2

    
        if args.reward_type == 1:
            err1 = Tsp1[k] - T1
            err2 = Tsp2[k] - T2
        else:
            if k < steps - 1:
                err1 = Tsp1[k + 1] - next_T1
                err2 = Tsp2[k + 1] - next_T2
            else:
                err1 = Tsp1[k] - next_T1
                err2 = Tsp2[k] - next_T2
        reward = compute_reward(err1, err2, reward_scaler)
        done   = (k == steps - 1)
        next_obs = np.array(
            [next_T1, next_T2,
            Tsp1[min(k + 1, steps - 1)],
            Tsp2[min(k + 1, steps - 1)]],
            dtype=np.float32
        )

        ### 버퍼에 추가하기 
        buffer.add_transition(obs, [Q1,Q2],next_obs, reward, done)

        T1, T2 = next_T1, next_T2


def online_finetune(args):
    torch.set_num_threads(1)
    
    from replay_buffer import ExperienceBufferManager 
    buffer = ExperienceBufferManager()   
   
    log_dir = Path(args.log_dir)                

    if args.resume:
        print(f"Resuming into existing log dir: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%m-%d-%H%M")
        log_dir = log_dir / args.env_name / f"{args.exp_name}_{timestamp}"

    # ← 여기서부터 기존대로
    log = Log(log_dir, vars(args))

    wandb_id = ""
    resume_path = log_dir / "resume_info.json"
    set_seed(args.seed)
    optimizer_factory = build_optimizer_factory(args)

    
    obs_dim, act_dim = 4, 2
    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden)

    qf = TwinQ(obs_dim, act_dim, args.hidden_dim, args.n_hidden)
    vf = ValueFunction(obs_dim, args.hidden_dim, args.n_hidden)
    iql = ImplicitQLearning(
        qf=qf, vf=vf, policy=policy,
        optimizer_factory=optimizer_factory,
        max_steps=args.n_steps, tau=args.tau, beta=args.beta,
        discount=args.discount, alpha=args.alpha)
    
    resume_path = log_dir / "resume_info.json"
    start_episode = 0

        
    if args.resume and resume_path.exists():
        with open(resume_path) as f:
            resume_data = json.load(f)
        start_episode = resume_data.get("last_episode", 0) + 1
        wandb_id      = resume_data.get("wandb_id", "")
        iql.load_state_dict(torch.load(log_dir / "last.pt"))
        print(f"✅ Resumed from {start_episode} with last.pt")
    else:
        iql.load_state_dict(torch.load(args.pt_path))
        print(f"🔁 Loaded pretrained IQL model from: {args.pt_path}")
        reward_scaler = joblib.load(args.scaler)
    wandb.init(
        project="tclab-project1",
        name=args.exp_name,
        config=vars(args),
        id=wandb_id or wandb.util.generate_id(),
        resume="allow"
    )

    # buffer = {
    #     "observations": [],
    #     "actions": [],
    #     "next_observations": [],
    #     "rewards": [],
    #     "terminals": []
    # }

    # if args.init_buffer : 
    #     npz = np.load(args.init_buffer)
    #     for k in buffer.keys():
    #         buffer[k] = npz[k].tolist()
    #     print(f"Pre-loaded buffer from {args.init_buffer}"
    #           f"(size : {len(buffer['observations'])})")
    
    if args.init_buffer : 
        buffer.load(args.init_buffer)

    buffer.summary()

    best_total_error = float("inf")
    best_state = None

    for episode in range(args.n_episodes):
        if args.type == "simulator" :
            rollout_simulator(iql.policy, buffer, reward_scaler, args)
        elif args.type == "real" : 
            rollout_tclab(iql.policy, buffer, reward_scaler, args)


        dataset = buffer.to_torch()

        if episode >= args.warmup_episodes : 
            for _ in range(args.update_per_episode):
                batch = sample_batch(dataset, args.batch_size)
                loss_dict = iql.update(**batch)
                iql.policy.epsilon = max(0.05, iql.policy.epsilon * 0.995)
        else : 
            loss_dict = {"policy_loss": 0, "q_loss": 0, "v_loss": 0}

        if episode + 1 == args.warmup_episodes:
            print("✅ Warm-up 종료. 현재 버퍼 상태 요약:")
            if hasattr(buffer, 'summary'):
                buffer.summary()
          #  exit()
        if args.type == "simulator" : 
            metrics = evaluate_policy_sim(iql.policy, args)
        elif args.type == "real" :
            metrics = evaluate_policy_tclab(iql.policy, args)

        metrics.update({"episode": episode})
        metrics.update(loss_dict)

        try:
            total_error = (
                metrics["E1"] + metrics["E2"] +
                metrics["Over"] + metrics["Under"]
            )
            metrics["total_error"] = total_error
        except KeyError:
            print("⚠️ Warning: total_error 계산 실패 (E1, E2, Over, Under 누락)")
            total_error = None

        if total_error is not None and total_error < best_total_error:
            best_total_error = total_error
            best_state = copy.deepcopy(iql.state_dict())
            torch.save(iql.state_dict(), log.dir / 'best.pt')
            print(f"[EP {episode}] ✅ Best model 저장됨 (total_error={total_error:.4f})")

        log.row(metrics)
        wandb.log(metrics)

        if args.save_buffer_path and args.save_buffer_every > 0 \
            and ((episode + 1) % args.save_buffer_every == 0):
            # logs_online_realkit/exp_name_타임스탬프/buffer_ep5.npz 형태로 저장됨
            buf_file = log.dir / f"buffer_ep{episode+1}.npz"
            buffer.save(buf_file)

    if best_state is not None:
        iql.load_state_dict(best_state)
    torch.save(iql.state_dict(), log.dir / 'final_online.pt')
    print(f"최적 파라미터 저장 완료: {log.dir / 'final_online.pt'}")

    if args.save_buffer_path:
        final_buf_path = log.dir / Path(args.save_buffer_path).name
        buffer.save(final_buf_path)
    print(f"replay buffer 저장 완료 : {final_buf_path}")

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
            tbl.add_data(r["seed"], r["total_error"], r["total_return"])
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

    
    wandb.finish()
    log.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # sam optimizer / 512 hidden dim 
    # 6개 obs C:/Users/Developer/TCLab/IQL/sam/tclab-mpc-iql/05-07-25_13.28.11_gigk/best.pt
    
    #parser.add_argument('--pt-path', default = "C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/05-07-25_18.08.30_lzec/ep10.pt")
    parser.add_argument('--pt-path', default="C:\\Users\\User\\tclab1\\IQL\\sam\\tclab-mpc-iql\\05-10-25_18.14.24_tplg\\best.pt" )
    # 오프라인 학습으로 가장 성능 좋은 pt 파일 , 경로 아래 
    #parser.add_argument('--pt-path',  default="C:\\Users\\Developer\\TCLab\\IQL\\cum_reward\\tclab-mpc-iql\\04-30-25_11.09.02_usmn\\best.pt")
    parser.add_argument('--scaler', default="C:\\Users\\User\\tclab1\\Data\\first_reward.pkl")
    parser.add_argument('--exp_name', default="online_ft")
    parser.add_argument('--env-name', default="tclab-online")
    parser.add_argument('--log-dir', default="./logs_online_realkit")
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--sample_interval', type=float, default=5.0)
    #n_episodes=100, update_per_episode=60
    # 1000
# 탐색이 있는 온라인 fine-tune, 20~30 epoch 목표
    parser.add_argument('--n-episodes',         type=int, default=250)   # 총 episode
    parser.add_argument('--update_per_episode', type=int, default=60)   # episode마다 60 update
    parser.add_argument('--n-steps',            type=int, default=9500) # 9 000 step보다 약간 크게

    parser.add_argument('--warmup-episodes', type=int, default=100,
                    help='초기 rollout만 수행하고 업데이트는 생략할 에피소드 수')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--ambient', type=float, default=29.0)
    parser.add_argument('--stochastic-policy', action='store_false', dest='deterministic_policy')
    parser.add_argument("--sam", action="store_true",
                        help="SAM(Sharpness‑Aware Minimization) 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.05,
                        help="SAM perturbation 반경 ρ")
    parser.add_argument("--init-buffer", default='', help="시작 시 불러올 .npz 버퍼 경로")
    parser.add_argument("--type", default="simulator", help="rollout 종류 설정 (simulator / tclab kit)")
    parser.add_argument("--save-buffer-path", default="./saved_buffer.npz",
                    help="누적 rollout 을 저장할 .npz 경로 (빈 문자열이면 저장하지 않음)")
    parser.add_argument("--save-buffer-every", type=int, default=0,
                        help="N 에피소드마다 버퍼를 저장 (0이면 마지막에만 저장)")
    parser.add_argument("--resume", action="store_true", help="이전 학습 이어서 재개할지 여부")
    parser.add_argument("--reward_type", type=int, default=2)

    # 📌 추가: extra evaluation seeds
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=None,
        help="추가 평가용 random seed 목록 (예: --eval-seeds 1 2 3 )",
    )


    args = parser.parse_args()
    print(args.scaler)
    online_finetune(args)