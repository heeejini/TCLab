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

            err1 = TSP1_mean - T1
            err2 = TSP2_mean - T2
            raw_reward = -np.sqrt(err1**2 + err2**2)
            reward = reward_scaler.transform([[raw_reward]])[0][0]

            done = (k == steps - 1)

            # buffer["observations"].append(obs)
            # buffer["actions"].append([Q1, Q2])
            # buffer["next_observations"].append(next_obs)
            # buffer["rewards"].append(reward)
            # buffer["terminals"].append(done)

            ### 버퍼에 추가하기 
            buffer.add_transition(obs, [Q1,Q2],next_obs, reward, done)

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

        err1 = TSP1_mean - T1
        err2 = TSP2_mean - T2
        raw_reward = -np.sqrt(err1**2 + err2**2)
        reward = reward_scaler.transform([[raw_reward]])[0][0]

        done = (k == steps - 1)

        ### 버퍼에 추가하기 
        buffer.add_transition(obs, [Q1,Q2],next_obs, reward, done)

        T1, T2 = next_T1, next_T2


def online_finetune(args):
    torch.set_num_threads(1)
    
    from replay_buffer import ExperienceBufferManager 
    buffer = ExperienceBufferManager()   



    # ✅ 항상 먼저 선언
    log_dir = Path(args.log_dir)

    if args.resume:
        with open(log_dir / "resume_info.json") as f:
            resume_data = json.load(f)
        wandb_id = resume_data.get("wandb_id", "")
        print(f"완디비 resume 재개 {wandb_id}")
    else:
        wandb_id = ""

    wandb.init(
        project="tclab-project",
        name=args.exp_name,
        config=vars(args),
        id=wandb_id or wandb.util.generate_id(),
        resume="allow"
    )
    timestamp = datetime.now().strftime("%m-%d-%H%M")
    log_dir = Path(args.log_dir) / args.env_name / f"{args.exp_name}_{timestamp}"
    log = Log(log_dir, vars(args))
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
            wandb_id = resume_data.get("wandb_id", "")
        iql.load_state_dict(torch.load(log_dir / 'last.pt'))
        print(f"✅ Resumed from {start_episode} with last.pt")
    else:
        wandb_id = ""
        iql.load_state_dict(torch.load(args.pt_path))
        print(f"🔁 Loaded pretrained IQL model from: {args.pt_path}")

    reward_scaler = joblib.load(args.scaler)


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
    
    wandb.init(project="tclab-project", name=args.exp_name, config=vars(args), id=wandb_id or wandb.util.generate_id(), resume="allow")
    wandb_id = wandb.run.id

    if args.init_buffer : 
        buffer.load(args.init_buffer)

    best_total_error = float("inf")
    best_state = None

    for episode in range(args.n_episodes):
        if args.type == "simulator" :
            rollout_simulator(iql.policy, buffer, reward_scaler, args)
        elif args.type == "real" : 
            rollout_tclab(iql.policy, buffer, reward_scaler, args)

        # dataset = {
        #     k: torchify(np.array(v, dtype=np.float32))
        #     for k, v in buffer.items()
        # }

        dataset = buffer.to_torch()

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

        if total_error is not None and total_error < best_total_error:
            best_total_error = total_error
            best_state = copy.deepcopy(iql.state_dict())
            torch.save(iql.state_dict(), log.dir / 'best.pt')
            print(f"[EP {episode}] ✅ Best model 저장됨 (total_error={total_error:.4f})")

        torch.save(iql.state_dict(), log.dir / 'last.pt')
        with open(log.dir / "resume_info.json", "w") as f:
            json.dump({"last_episode": episode, "wandb_id": wandb_id}, f)
            # else:
            #     if best_state is not None:
            #         iql.load_state_dict(best_state)
            #         print(f"[EP {episode}] 성능 악화. 이전 best(total_error={best_total_error:.4f})로 롤백")

        log.row(metrics)
        wandb.log(metrics)
        # 5episode 마다 저장 

        if (episode + 1) % 5 == 0:
            model_path = log.dir / f"ep{episode+1}.pt"
            torch.save(iql.state_dict(), model_path)
            print(f"[EP {episode+1}] 🔄 5회차마다 저장됨: {model_path.name}")

        ### 주기적 버퍼 저장 
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

    
    wandb.finish()
    log.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # sam optimizer / 512 hidden dim 
    # 6개 obs C:/Users/Developer/TCLab/IQL/sam/tclab-mpc-iql/05-07-25_13.28.11_gigk/best.pt
    
    #parser.add_argument('--pt-path', default = "C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/05-07-25_18.08.30_lzec/ep10.pt")
    parser.add_argument('--pt-path', default="C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/realkit_save_buffer_05-08-1453/05-08-25_14.53.43_gbqn/best.pt" )
    # 오프라인 학습으로 가장 성능 좋은 pt 파일 , 경로 아래 
    #parser.add_argument('--pt-path',  default="C:\\Users\\Developer\\TCLab\\IQL\\cum_reward\\tclab-mpc-iql\\04-30-25_11.09.02_usmn\\best.pt")
    parser.add_argument('--scaler', default="C:/Users/Developer/TCLab/Data/first_reward.pkl")
    parser.add_argument('--exp_name', default="online_ft")
    parser.add_argument('--env-name', default="tclab-online")
    parser.add_argument('--log-dir', default="./logs_online_realkit")
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--sample_interval', type=float, default=5.0)
    #n_episodes=100, update_per_episode=60
    # 1000
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--update_per_episode', type=int, default=240)
    parser.add_argument('--n-steps', type=int, default=8000) 

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
    parser.add_argument("--sam", action="store_true",
                        help="SAM(Sharpness‑Aware Minimization) 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.05,
                        help="SAM perturbation 반경 ρ")
    parser.add_argument("--init-buffer", default='', help="시작 시 불러올 .npz 버퍼 경로")
    parser.add_argument("--type", default="simulator", help="rollout 종류 설정 (simulator / tclab kit)")
    parser.add_argument("--save-buffer-path", default="./saved_buffer.npz",
                    help="누적 rollout 을 저장할 .npz 경로 (빈 문자열이면 저장하지 않음)")
    parser.add_argument("--save-buffer-every", type=int, default=5,
                        help="N 에피소드마다 버퍼를 저장 (0이면 마지막에만 저장)")
    parser.add_argument("--resume", action="store_true", help="이전 학습 이어서 재개할지 여부")
    
    args = parser.parse_args()
    print(args.scaler)
    online_finetune(args)