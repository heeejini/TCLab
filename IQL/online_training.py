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
from src.replay_buffer import ExperienceBufferManager 
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify, Log, set_seed, sample_batch, evaluate_policy_sim, evaluate_policy_tclab, evaluate_extra_seeds
from src.sam import SAM 
from src.eval_policy import compute_reward 
from collections import deque
N_STEP_REWARD = 5

def eval_policy( policy, args):
    if args.type == "simulator":
        return evaluate_policy_sim(policy, args)
    else:
        return evaluate_policy_tclab(policy, args)


def build_optimizer_factory(args):
    if args.sam:
        return lambda params: SAM(params,
                                  torch.optim.Adam,
                                  lr=args.learning_rate,
                                  betas=(0.9, 0.999),
                                  rho=args.sam_rho)
    else:
        return lambda params: torch.optim.Adam(params,
                                               lr=args.learning_rate)
import time
def generate_random_tsp1(
    total_time_sec: int = 1200,
    dt: float = 5.0,  # sample / control interval
    low: float = 25.0,
    high: float = 65.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Set-point 구간별 정보를 로그로 출력.
    각 구간 길이: 평균 480초, σ 100초, 최소 160초, 최대 800초
    """
    seed = int((time.time() * 1e6) % 1e9)  
    np.random.seed(seed)
    print(f"[🧪 TSP 생성용 시드: {seed}]")

    n_steps = int(total_time_sec / dt)
    tsp = np.zeros(n_steps)
    i = 0
    seg_id = 1  # 구간 번호

    print(f"\n--- Set-point 프로파일 생성 (총 시간: {total_time_sec}초, 총 step: {n_steps}) ---")
    while i < n_steps:
        dur_sec = int(np.clip(np.random.normal(480, 100), 160, 800))
        dur_steps = max(1, int(dur_sec / dt))
        end = min(i + dur_steps, n_steps)

        temp = round(np.random.uniform(low, high), 2)
        tsp[i:end] = temp

        start_time = int(i * dt)
        end_time = int((end - 1) * dt)
        print(f"구간 {seg_id}: step {i:>3} ~ {end-1:>3} (시간 {start_time:>4}s ~ {end_time:>4}s) → 목표 온도: {temp:.2f}°C")

        i = end
        seg_id += 1
    print("-----------------------------------------------------------\n")
    return tsp
from collections import deque           # ★ 추가

def rollout_tclab(policy, buffer, reward_scaler, args):
    """
      reward_type = 1 :  |TSP_t − T_t|
      reward_type = 2 :  |TSP_t − T_{t+1}|
      reward_type = 3 :  |TSP_t − T_{t+N}|   (N = args.n_step)
    """
    import time, joblib
    from tclab import TCLab
    from tqdm import trange
    from src.util import torchify

    dt    = args.sample_interval
    steps = int(args.max_episode_steps / dt)
    N     = getattr(args, "n_step", 5)      
    ambient = 29.0

    Tsp1 = generate_random_tsp1(args.max_episode_steps, dt)
    Tsp2 = generate_random_tsp1(args.max_episode_steps, dt)

    obs_q, act_q, tsp_q, temp_q = deque(), deque(), deque(), deque()

    with TCLab() as arduino:
        arduino.LED(100)
        arduino.Q1(0); arduino.Q2(0)

        while arduino.T1 > ambient or arduino.T2 > ambient:
            time.sleep(10)

        policy.eval()

        for k in trange(steps, desc="rollout-tclab"):
            loop_start = time.time()

            T1, T2 = arduino.T1, arduino.T2
            obs    = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)

            with torch.no_grad():
                act = policy.new_action(torchify(obs),
                                 deterministic=args.deterministic_policy).cpu().numpy()
            Q1, Q2 = float(np.clip(act[0], 0, 100)), float(np.clip(act[1], 0, 100))
            arduino.Q1(Q1); arduino.Q2(Q2)

            time.sleep(max(0.0, dt - (time.time() - loop_start)))

            next_T1, next_T2 = arduino.T1, arduino.T2
            next_obs = np.array(
                [next_T1, next_T2,
                 Tsp1[min(k+1, steps-1)], Tsp2[min(k+1, steps-1)]],
                dtype=np.float32
            )

            obs_q.append(obs)
            act_q.append([Q1, Q2])
            tsp_q.append([Tsp1[k], Tsp2[k]])
            temp_q.append([T1, T2])

            if args.reward_type in (1, 2):
                if args.reward_type == 1:
                    err1, err2 = Tsp1[k] - T1, Tsp2[k] - T2
                else:  
                    err1, err2 = Tsp1[min(k+1, steps-1)] - next_T1, \
                                 Tsp2[min(k+1, steps-1)] - next_T2
                reward = compute_reward(err1, err2, reward_scaler)
                done   = (k == steps-1)
                buffer.add_transition(obs, [Q1, Q2], next_obs, reward, done)

            elif args.reward_type == 3:
                if len(obs_q) >= N:
                    s_t      = obs_q.popleft()
                    a_t      = act_q.popleft()
                    tsp_now  = tsp_q.popleft() 
                    t_future = temp_q.pop()      # N step 뒤 온도

                    err1 = tsp_now[0] - t_future[0]
                    err2 = tsp_now[1] - t_future[1]
                    reward = compute_reward(err1, err2, reward_scaler)
                    done   = (k >= steps-1)
                    buffer.add_transition(s_t, a_t, next_obs, reward, done)
            else:
                raise ValueError("reward_type must be 1, 2, or 3")

        arduino.Q1(0); arduino.Q2(0)


from collections import deque

def rollout_simulator(policy, buffer, reward_scaler, args):
    from tclab import setup
    lab = setup(connected=False)
    env = lab(synced=False)
    env.Q1(0); env.Q2(0)
    env._T1 = env._T2 = args.ambient

    steps = int(args.max_episode_steps / args.sample_interval)

    Tsp1 = generate_random_tsp1(args.max_episode_steps, args.sample_interval)
    Tsp2 = generate_random_tsp1(args.max_episode_steps, args.sample_interval)

    policy.eval()

    N   = getattr(args, "n_step", 5)             
    obs_q, act_q, tsp_q, temp_q = deque(), deque(), deque(), deque()

    for k in trange(steps, desc="rollout"):
        env.update(t=k * args.sample_interval)
        T1, T2 = env.T1, env.T2

        obs = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)
        with torch.no_grad():
            act, std = policy.act(torchify(obs), deterministic=True, return_std=True)
            act = act.cpu().numpy()
            std = std.cpu().numpy()
        print("action : ", act)
        print("std : ", std)

        Q1, Q2 = float(np.clip(act[0], 0, 100)), float(np.clip(act[1], 0, 100))
        env.Q1(Q1); env.Q2(Q2)

        # 1-step next 관측
        env.update(t=(k + 1) * args.sample_interval) ####
        next_T1, next_T2 = env.T1, env.T2
        next_obs = np.array(
            [next_T1, next_T2,
             Tsp1[min(k+1, steps-1)], Tsp2[min(k+1, steps-1)]],
            dtype=np.float32
        )

        if args.reward_type == 1:
            err1, err2 = Tsp1[k]                 - T1, \
                         Tsp2[k]                 - T2
            reward = compute_reward(err1, err2, reward_scaler)
            done   = (k == steps - 1)
            buffer.add_transition(obs, [Q1, Q2], next_obs, reward, done)

        elif args.reward_type == 2:
            err1, err2 = Tsp1[min(k+1, steps-1)] - next_T1, \
                         Tsp2[min(k+1, steps-1)] - next_T2
            reward = compute_reward(err1, err2, reward_scaler)
            done   = (k == steps - 1)
            buffer.add_transition(obs, [Q1, Q2], next_obs, reward, done)

        elif args.reward_type == 3:
            obs_q.append(obs);   act_q.append([Q1, Q2])
            tsp_q.append([Tsp1[k], Tsp2[k]])
            temp_q.append([T1, T2])

            if len(obs_q) >= N:
                s_t      = obs_q.popleft()
                a_t      = act_q.popleft()
                tsp_now  = tsp_q.popleft()       # 맨 앞 요소 제거  / 왼쪽 방향
                t_future = temp_q.pop()          # 맨 뒤 요소 제거  / 오른쪽 방향 

                err1 = tsp_now[0] - t_future[0]
                err2 = tsp_now[1] - t_future[1]
                reward = compute_reward(err1, err2, reward_scaler)
                done   = (k >= steps - 1)            # 마지막 pop 판별
                buffer.add_transition(s_t, a_t, next_obs, reward, done)
        else:
            raise ValueError("reward_type must be 1, 2, or 3")

    env.Q1(0); env.Q2(0)


def online_finetune(args):
    torch.set_num_threads(1)
    
    buffer = ExperienceBufferManager()   
   
    log_dir = Path(args.log_dir)                

    if args.resume:
        print(f"Resuming into existing log dir: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%m-%d-%H%M")
        log_dir = log_dir / args.env_name / f"{args.exp_name}_{timestamp}"

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
      
        if args.type == "simulator" : 
            metrics = evaluate_policy_sim(iql.policy, args)
        elif args.type == "real" :
            metrics = evaluate_policy_tclab(iql.policy, args)

        metrics.update({"episode": episode})
        metrics.update(loss_dict)

        try:
            total_error = (
                metrics["E1"] + metrics["E2"] 
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

    if args.eval_seeds:
        evaluate_extra_seeds(
            policy=iql.policy,
            args=args,
            log=log,
            eval_fn=eval_policy,       
            filename="extra_eval.csv"
        )

    wandb.finish()
    log.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--pt-path', default=r"C:\Users\Developer\TCLab\IQL\IQLtrained_model\05-17-25_00.08.59_xxao\best.pt")
    parser.add_argument("--scaler", default="C:/Users/Developer/TCLab/Data/reward/first.pkl")
    parser.add_argument('--exp_name', default="online_ft")
    parser.add_argument('--env-name', default="tclab-online")
    parser.add_argument('--log-dir', default="./logs_online_realkit")
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--sample_interval', type=float, default=5.0)
    parser.add_argument('--n-episodes',         type=int, default=10)   
    parser.add_argument('--update_per_episode', type=int, default=30)  
    parser.add_argument('--n-steps',            type=int, default=240 * 10  ) 

    parser.add_argument('--warmup-episodes', type=int, default=0,
                    help='초기 rollout만 수행하고 업데이트는 생략할 에피소드 수')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--ambient', type=float, default=29.0)
    parser.add_argument('--stochastic-policy', action='store_false', dest='deterministic_policy')
    parser.add_argument("--sam", action="store_true",
                        help="SAM(Sharpness‑Aware Minimization) 사용 여부")
    parser.add_argument("--sam-rho", type=float, default=0.05,
                        help="SAM perturbation 반경 ρ")

    parser.add_argument("--init-buffer", default="C:/Users/Developer/TCLab/Data/reward/first.npz",
                         help="시작 시 불러올 .npz 버퍼 경로")
    parser.add_argument("--type", default="simulator", help="rollout 종류 설정 (simulator / real)")
    parser.add_argument("--save-buffer-path", default="./saved_buffer.npz",
                    help="누적 rollout 을 저장할 .npz 경로 (빈 문자열이면 저장하지 않음)")
    parser.add_argument("--save-buffer-every", type=int, default=2,
                        help="N 에피소드마다 버퍼를 저장 (0이면 마지막에만 저장)")
    parser.add_argument("--resume", action="store_true", help="이전 학습 이어서 재개할지 여부")
    parser.add_argument("--err_thr", required=False, default=5)
    parser.add_argument("--noise_std" , required=False, default=5)

    parser.add_argument("--reward_type", type=int, default=1)
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2],
        help="추가 평가용 random seed 목록 (예: --eval-seeds 0 1 2 )",
    )

    args = parser.parse_args()
    print(args.scaler)
    online_finetune(args)



