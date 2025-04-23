# %%
# Policy evaluation notebook cell
#
# ① 경로 설정 ───────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\Developer\TCLab\IQL\logs\mpc-iql\04-23-25_14.45.10_ukeq\final.pt"
EVAL_LOG_ROOT = r"C:\Users\Developer\TCLab\IQL\eval_sim_logs"
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ② 필요한 라이브러리 import ───────────────────────────────────────────
import os, time, csv, math, torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tclab import setup                    # BYU TCLab 시뮬레이터
# util.py 에 이미 존재한다고 가정
from .util import torchify, set_seed

from .policy import GaussianPolicy

# ③ sim_evaluate_policy 함수 정의 ─────────────────────────────────────
def generate_random_tsp(
    steps: int,
    label: str = "TSP",
    low: float = 25.0,
    high: float = 65.0,
    min_time: float = 160.0,   # ← 그대로 두되 사용하지 않는다
    dt: float = 1.0,
) -> np.ndarray:
    """
    데이터 수집 스크립트(MPCDataCollector)와 동일한 방식으로
    set-point 프로파일을 만든다.

    - 각 구간 길이 ~ 𝒩(480 s, 100² s²), 단 160 s–800 s 범위로 clip
    - dt 간격으로 steps 만큼 채울 때까지 반복
    - 온도는 [low, high] 균등 분포
    """
    tsp = np.zeros(steps)
    i = 0
    seg_id = 1
    print(f"[{label}] --- Set-point 프로파일 생성 (총 step={steps}, dt={dt}s) ---")
    while i < steps:
        # 구간 길이(초) 샘플링 후 step 단위로 변환
        dur_sec   = float(np.clip(np.random.normal(480, 100), 160, 800))
        dur_steps = max(1, int(dur_sec / dt))
        end       = min(i + dur_steps, steps)

        # 목표 온도 설정
        temp = round(np.random.uniform(low, high), 2)
        tsp[i:end] = temp

        # 로그 출력
        start_t = int(i * dt)
        end_t   = int((end - 1) * dt)
        print(
            f"[{label}] 구간 {seg_id}: step {i:>4} ~ {end-1:>4} "
            f"(시간 {start_t:>4}s ~ {end_t:>4}s) → 목표 온도 {temp:.2f}°C"
        )

        i = end
        seg_id += 1
    print(f"[{label}] -----------------------------------------------------------\n")
    return tsp


def sim_evaluate_policy(
    policy,
    max_steps=1200,
    log_root="./eval_logs",
    seed=1,
    ambient=29.0,
    deterministic=True,
):
    set_seed(seed)
    run_dir = Path(log_root) / f"sim_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lab = setup(connected=False)
    env = lab(synced=False)
    env.T_amb = 29.0  # 직접 ambient 설정 (속성명은 T_amb 또는 ambient일 수 있음)

    env.Q1(0); env.Q2(0)


    Tsp1 = generate_random_tsp(max_steps, 'TSP1', dt=1.0)
    Tsp2 = generate_random_tsp(max_steps, 'TSP2', dt=1.0)

    t  = np.arange(max_steps)
    T1 = np.zeros(max_steps); T2 = np.zeros(max_steps)
    Q1 = np.zeros(max_steps); Q2 = np.zeros(max_steps)

    total_return = 0.0
    e1 = e2 = over = under = 0.0

    policy.eval()

    for k in range(max_steps):
        env.update(t=k)

        T1[k] = env.T1
        T2[k] = env.T2
        obs = np.array([T1[k], T2[k], Tsp1[k], Tsp2[k]], dtype=np.float32)

        with torch.no_grad():
            act = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()

        Q1[k] = float(np.clip(act[0], 0, 100))
        Q2[k] = float(np.clip(act[1], 0, 100))
        env.Q1(Q1[k]); env.Q2(Q2[k])

        reward = -math.hypot(T1[k] - Tsp1[k], T2[k] - Tsp2[k])
        total_return += reward

        err1 = Tsp1[k] - T1[k]
        err2 = Tsp2[k] - T2[k]
        e1 += abs(err1);  e2 += abs(err2)
        over  += max(0, -err1) + max(0, -err2)
        under += max(0,  err1) + max(0,  err2)

    env.Q1(0); env.Q2(0)

    # CSV 저장
    csv_path = run_dir / "rollout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time","T1","T2","Q1","Q2","TSP1","TSP2"])
        for k in range(max_steps):
            w.writerow([t[k], T1[k], T2[k], Q1[k], Q2[k], Tsp1[k], Tsp2[k]])

    # 그래프 저장
    fig, ax = plt.subplots(2,1,figsize=(10,8))
    ax[0].plot(t, T1, label="T1"); ax[0].plot(t, Tsp1, "--", label="TSP1")
    ax[0].plot(t, T2, label="T2"); ax[0].plot(t, Tsp2, ":", label="TSP2")
    ax[0].set_ylabel("Temp (°C)"); ax[0].legend(); ax[0].grid()

    ax[1].plot(t, Q1, label="Q1"); ax[1].plot(t, Q2, label="Q2")
    ax[1].set_ylabel("Heater (%)"); ax[1].set_xlabel("Time (s)")
    ax[1].legend(); ax[1].grid()
    plt.tight_layout()
    plt.savefig(run_dir / "rollout.png")
    plt.show()

    metrics = dict(return_sum=total_return, E1=e1, E2=e2, Over=over, Under=under)
    return metrics

# ④ 정책 네트워크 구성 & 가중치 로드 ──────────────────────────────────
policy_net = GaussianPolicy(obs_dim=4, act_dim=2)
ckpt = torch.load(MODEL_PATH, map_location="cpu")

# 'policy.' 접두사만 추출
subdict = {k.replace("policy.", ""): v for k, v in ckpt.items() if k.startswith("policy.")}
policy_net.load_state_dict(subdict)

# ⑤ 시뮬레이터 평가 실행 ───────────────────────────────────────────
metrics = sim_evaluate_policy(
    policy=policy_net,
    max_steps=1200,
    log_root=EVAL_LOG_ROOT,
    seed=1,
    ambient=29.0,
    deterministic=True
)

print("Evaluation metrics:", metrics)

