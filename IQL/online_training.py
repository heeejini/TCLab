import torch
import numpy as np
from pathlib import Path
from tqdm import trange
import joblib
import wandb
import copy 

from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify, Log, set_seed, sample_batch, evaluate_policy_sim, evaluate_policy_tclab
from src.sam import SAM 

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

            if k == steps - 1:
                TSP1_next, TSP2_next = Tsp1[k], Tsp2[k]
                TSP1_mean, TSP2_mean = Tsp1[k], Tsp2[k]
            else:
                TSP1_next, TSP2_next = Tsp1[k + 1], Tsp2[k + 1]
                TSP1_mean = 0.5 * (Tsp1[k] + Tsp1[k + 1])
                TSP2_mean = 0.5 * (Tsp2[k] + Tsp2[k + 1])

            next_obs = np.array([next_T1, next_T2, TSP1_next, TSP2_next],
                                dtype=np.float32)

            # ─ 리워드 계산 (next 기반)
            err1 = TSP1_next - next_T1
            err2 = TSP2_next - next_T2
            raw_reward = -np.sqrt(err1**2 + err2**2)
            reward = reward_scaler.transform([[raw_reward]])[0][0]


            done = (k == steps - 1)

            buffer["observations"].append(obs)
            buffer["actions"].append([Q1, Q2])
            buffer["next_observations"].append(next_obs)
            buffer["rewards"].append(reward)
            buffer["terminals"].append(done)

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
            action = policy.act(torchify(obs), deterministic=args.deterministic_policy).cpu().numpy()

        Q1 = float(np.clip(action[0], 0, 100))
        Q2 = float(np.clip(action[1], 0, 100))
        env.Q1(Q1); env.Q2(Q2)

        next_T1, next_T2 = env.T1, env.T2

        if k == steps - 1:
            TSP1_next = Tsp1[k]
            TSP2_next = Tsp2[k]
            TSP1_mean = Tsp1[k]
            TSP2_mean = Tsp2[k]
        else:
            TSP1_next = Tsp1[k + 1]
            TSP2_next = Tsp2[k + 1]
            TSP1_mean = (Tsp1[k] + Tsp1[k + 1]) / 2
            TSP2_mean = (Tsp2[k] + Tsp2[k + 1]) / 2

        next_obs = np.array([next_T1, next_T2, TSP1_next, TSP2_next], dtype=np.float32)

        # ─ 리워드 계산 (next 기반)
        err1 = TSP1_next - next_T1
        err2 = TSP2_next - next_T2
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

    best_total_error = float("inf")
    best_state = None

    for episode in range(args.n_episodes):
        if args.type == "simulator" :
            rollout_simulator(iql.policy, buffer, reward_scaler, args)
        elif args.type == "real" : 
            rollout_tclab(iql.policy, buffer, reward_scaler, args)

        dataset = {
            k: torchify(np.array(v, dtype=np.float32))
            for k, v in buffer.items()
        }

        for _ in range(args.update_per_episode):
            batch = sample_batch(dataset, args.batch_size)
            loss_dict = iql.update(**batch)

        if args.type == "simulator" : 
            metrics = evaluate_policy_sim(iql.policy, args)
        elif args.type == "real" :
            metrics = evaluate_policy_tclab(iql.policy, args)

        metrics.update({"episode": episode})
        metrics.update(loss_dict)

        # total_error 계산
        try:
            total_error = (
                metrics["E1"] + metrics["E2"] +
                metrics["Over"] + metrics["Under"]
            )
            metrics["total_error"] = total_error
        except KeyError:
            print("⚠️ Warning: total_error 계산 실패 (E1, E2, Over, Under 누락)")
            total_error = None

        if total_error is not None:
            if total_error < best_total_error:
                best_total_error = total_error
                best_state = copy.deepcopy(iql.state_dict())
                print(f"[EP {episode}] 성능 개선 total_error={total_error:.4f}")
            else:
                if best_state is not None:
                    iql.load_state_dict(best_state)
                    print(f"[EP {episode}] 성능 악화. 이전 best(total_error={best_total_error:.4f})로 롤백")

        log.row(metrics)
        wandb.log(metrics)

    if best_state is not None:
        iql.load_state_dict(best_state)
    torch.save(iql.state_dict(), log.dir / 'final_online.pt')
    print(f"최적 파라미터 저장 완료: {log.dir / 'final_online.pt'}")
    wandb.finish()
    log.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # sam optimizer / 512 hidden dim 
    parser.add_argument('--pt-path',  default="C:\\Users\\Developer\\TCLab\\IQL\\cum_reward\\tclab-mpc-iql\\04-30-25_11.09.02_usmn\\best.pt")
    parser.add_argument('--scaler', default="C:\\Users\\Developer\\TCLab\\Data\\first_reward.pkl")
    parser.add_argument('--exp_name', default="online_ft")
    parser.add_argument('--env-name', default="tclab-online")
    parser.add_argument('--log-dir', default="./logs_online")
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--sample_interval', type=float, default=5.0)
    #n_episodes=100, update_per_episode=60
    # 1000
    parser.add_argument('--n-episodes', type=int, default=500)
    parser.add_argument('--update_per_episode', type=int, default=60)
    parser.add_argument('--n-steps', type=int, default=30000) 

    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ambient', type=float, default=29.0)
    parser.add_argument('--stochastic-policy', action='store_false', dest='deterministic_policy')
    # argparse 영역 맨 아래쯤
    parser.add_argument("--sam", action="store_true",
                        help="SAM(Sharpness‑Aware Minimization) 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.05,
                        help="SAM perturbation 반경 ρ")
    parser.add_argument("--type", default="simulator", help="rollout 종류 설정 (simulator / tclab kit)")
    args = parser.parse_args()

    online_finetune(args)