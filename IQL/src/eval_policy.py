"""eval_policy.py
시뮬레이터(TCLabModel) 및 실제 TCLab 장치에서 학습된 정책을 평가하는 함수 모음.
두 함수 모두 공통적으로 다음 dict를 반환한다.
    {
        'T1'           : np.ndarray[…],
        'T2'           : np.ndarray[…],
        'Tsp1'         : np.ndarray[…],
        'Tsp2'         : np.ndarray[…],
        'Q1'           : np.ndarray[…],
        'Q2'           : np.ndarray[…],
        'total_return' : float,
        'E1'           : float,   # Σ|Tsp1 − T1|
        'E2'           : float,   # Σ|Tsp2 − T2|
        'Over'         : float,   # Σmax(0, Tm − Tsp)
        'Under'        : float    # Σmax(0, Tsp − Tm)
    }
"""
from __future__ import annotations
import os, csv, math, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import trange
from collections import deque

from tclab import setup, TCLab  # 시뮬레이터 & 실제

# def compute_reward(e1, e2):
#     return -math.hypot(e1, e2)

# def compute_reward(acc_error1, acc_error2, E1=1.0, E2=1.0):
#     return - (E1 * acc_error1 + E2 * acc_error2)

EPS = 1e-6
REWARD_MIN = -26.0 
REWARD_MAX = -6.0  

E1, E2 = 1.0,1.0


from sklearn.preprocessing import StandardScaler
import joblib
# ★ 학습 시 사용한 scaler를 따로 저장하거나, 같은 방식으로 다시 fit 해야 일관성 유지 가능
reward_scaler = StandardScaler()
#reward_scaler = joblib.load("C:\\Users\\Developer\\TCLab\\Data\\reward_scaler.pkl")
#reward_scaler = joblib.load("C:\\Users\\Developer\\TCLab\Data\\reward_scaler_time.pkl")

# def compute_reward(e1, e2):
#     return -math.hypot(e1, e2)
def compute_reward(acc_error1, acc_error2, reward_scaler):
    raw_reward = - np.sqrt(acc_error1**2 + acc_error2 ** 2)
    #reward = - np.sqrt((E1 * acc_error1)**2 + (E2 * acc_error2)**2)
    scaled = reward_scaler.transform([[raw_reward]])[0][0]
    return scaled


# def compute_reward(acc_error1, acc_error2, time_sec: float, reward_scaler):

#     alpha = 0.05
#     raw_reward = - np.sqrt(acc_error1**2 + acc_error2**2) - alpha * time_sec

#     # 스케일러 적용 (주의: scaler는 학습된 상태여야 함)
#     scaled = reward_scaler.transform([[raw_reward]])[0][0]
#     return scaled

def generate_random_tsp(total_time_sec : int = 1200,                   
                        dt: float          = 5.0,      # sample / control interval
                        low: float         = 25.0,
                        high: float        = 65.0,
                        verbose: bool      = False,) -> np.ndarray:
    """
    Set-point 구간별 정보를 로그로 출력.
    각 구간 길이: 평균 480초, σ 100초, 최소 160초, 최대 800초
    """
    # 시간 관련 변수 
    n_steps = int(total_time_sec/ dt)
    tsp = np.zeros(n_steps)
    i = 0
    seg_id = 1  # 구간 번호

    print(f"\n--- Set-point 프로파일 생성 (총 시간: {total_time_sec}초, 총 step: {n_steps}) ---")
    while i < n_steps:
        dur_sec = int(np.clip(np.random.normal(480, 100), 160, 800))
        dur_steps = max(1, int(dur_sec / dt))
        end = min(i + dur_steps, n_steps)
    
        # set-point 값 설정
        temp = round(np.random.uniform(low, high), 2)
        tsp[i:end] = temp
    
        # 로그 출력
        start_time = int(i * dt)
        end_time = int((end - 1) * dt)
        print(f"구간 {seg_id}: step {i:>3} ~ {end-1:>3} (시간 {start_time:>4}s ~ {end_time:>4}s) → 목표 온도: {temp:.2f}°C")
    
        i = end
        seg_id += 1
    print("-----------------------------------------------------------\n")
    return tsp

# ────────────────────────────────
# 1) 시뮬레이터 평가 함수
# ────────────────────────────────

def simulator_policy(
    policy,
    total_time_sec: int = 1200,
    dt: float = 5.0,
    log_root: str | Path = "./eval_sim_logs",
    seed: int = 0,
    ambient: float = 29.0, # start_temp 
    deterministic: bool = True, 
    scaler : str| Path = '',
    reward_type : int = 3
):
    from .util import torchify, set_seed
    steps = int(total_time_sec/dt)
    """TCLab 시뮬레이터(setup(connected=False))에서 정책 평가"""
    set_seed(seed)
    run_dir = Path(log_root) / f"sim_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- 스케일러 설정
    print(f"scaler 경로 : {scaler}")
    reward_scaler = joblib.load(scaler)

    # --- 시뮬레이터 환경 생성 ---
    lab = setup(connected=False)
    env = lab(synced=False)
    if hasattr(env, "T_amb"):
        env.T_amb = ambient  # ambient 설정 지원 시
    env.Q1(0); env.Q2(0)

    env._T1 = 29.0
    env._T2 = 29.0

    # --- set‑point 시퀀스 ---
    Tsp1 = generate_random_tsp(total_time_sec, dt)
    Tsp2 = generate_random_tsp(total_time_sec, dt)

    # --- 버퍼 초기화 ---
    t  = np.arange(steps) * dt
    T1 = np.zeros(steps); T2 = np.zeros(steps)
    Q1 = np.zeros(steps); Q2 = np.zeros(steps)

    total_ret = e1 = e2 = over = under = 0.0
    policy.eval()

    for k in trange(steps, desc="sim"):  # 1 s per step
        env.update(t=k * dt)
        T1[k] = env.T1   
        T2[k] = env.T2

        obs = np.array([T1[k], T2[k], Tsp1[k], Tsp2[k]], dtype=np.float32)
        with torch.no_grad():
            act = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        Q1[k] = float(np.clip(act[0], 0, 100))
        Q2[k] = float(np.clip(act[1], 0, 100))
        env.Q1(Q1[k])
        env.Q2(Q2[k])
        # ▒▒ 에러 계산 분기 ▒▒
        if reward_type == 1:
            # 현재 시점 기준
            print(f"reward type : {reward_type}")
            err1 = Tsp1[k] - T1[k]
            err2 = Tsp2[k] - T2[k]
        elif reward_type == 2:
            # 다음 시점 기준
            print(f"reward type : {reward_type}")
            next_T1, next_T2 = env.T1, env.T2
            if k < steps - 1:
                err1 = Tsp1[k + 1] - next_T1
                err2 = Tsp2[k + 1] - next_T2
            else:
                err1 = Tsp1[k] - next_T1
                err2 = Tsp2[k] - next_T2
        elif reward_type == 3:
            # n-step future 기준: TSP_t - T_{t+n}
            n = 5  
            j = min(k + n, steps - 1)
            env.update(t=j * dt)  # t+n 시점까지 환경 갱신
            future_T1, future_T2 = env.T1, env.T2
            err1 = Tsp1[k] - future_T1
            err2 = Tsp2[k] - future_T2
                    
        else:
            raise ValueError(f"Invalid reward_type: {reward_type} (must be 1 or 2)")

        reward = compute_reward(err1, err2, reward_scaler)
        total_ret += reward
        e1 += abs(err1); e2 += abs(err2)
        over += max(0, -err1) + max(0, -err2)
        under += max(0, err1) + max(0, err2)

    # CSV & 그래프 저장
    csv_path = run_dir / "rollout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time","T1","T2","Q1","Q2","TSP1","TSP2"])
        for k in range(steps):
            w.writerow([t[k], T1[k], T2[k], Q1[k], Q2[k], Tsp1[k], Tsp2[k]])

    fig, ax = plt.subplots(2,1,figsize=(10,8))
    ax[0].plot(t, T1, label="T1"); ax[0].plot(t, Tsp1, "--", label="TSP1")
    ax[0].plot(t, T2, label="T2"); ax[0].plot(t, Tsp2, ":", label="TSP2")
    ax[0].grid(); ax[0].legend(); ax[0].set_ylabel("Temp (°C)")
    ax[1].plot(t, Q1, label="Q1"); ax[1].plot(t, Q2, label="Q2")
    ax[1].grid(); ax[1].legend(); ax[1].set_ylabel("Heater (%)"); ax[1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.savefig(run_dir / "rollout.png"); plt.close()

    return dict(T1=T1, T2=T2, Tsp1=Tsp1, Tsp2=Tsp2, Q1=Q1, Q2=Q2,
                total_return=total_ret, E1=e1, E2=e2, Over=over, Under=under)

# ────────────────────────────────
# 2) 실제 장비(TCLab) 평가 함수
# ────────────────────────────────
def tclab_policy(
    policy,
    total_time_sec: int = 1200,
    dt: float = 5.0,
    log_root: str | Path = "./eval_real_logs",
    seed: int = 0,
    ambient: float = 29.0,
    deterministic: bool = True,
    scaler: str | Path = '',
    reward_type: int = 2  # 1: 현재 T, 2: next T 기준
):
    from .util import torchify, set_seed
    steps = int(total_time_sec / dt)
    set_seed(seed)
    n_step = 5
    run_dir = Path(log_root) / f"real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"scaler 경로 : {scaler}")
    reward_scaler = joblib.load(scaler) if scaler else None

    with TCLab() as arduino:
        arduino.LED(100)
        arduino.Q1(0); arduino.Q2(0)

        while arduino.T1 > ambient or arduino.T2 > ambient:
            time.sleep(10)

        Tsp1 = generate_random_tsp(total_time_sec, dt)
        Tsp2 = generate_random_tsp(total_time_sec, dt)

        t  = np.arange(steps) * dt
        T1 = np.zeros(steps); T2 = np.zeros(steps)
        Q1 = np.zeros(steps); Q2 = np.zeros(steps)

        total_ret = e1 = e2 = over = under = 0.0
        policy.eval()

        if reward_type == 3:
            future_q = deque(maxlen=n_step+1)   # (T1,T2) 최근 n+1개
            
        for k in trange(steps, desc="real"):
            loop_start = time.time()

            T1[k] = arduino.T1
            T2[k] = arduino.T2

            obs = np.array([T1[k], T2[k], Tsp1[k], Tsp2[k]], dtype=np.float32)
            with torch.no_grad():
                act = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()

            Q1[k] = float(np.clip(act[0], 0, 100))
            Q2[k] = float(np.clip(act[1], 0, 100))
            arduino.Q1(Q1[k]); arduino.Q2(Q2[k])

            time.sleep(max(0.0, dt - (time.time() - loop_start)))

            if reward_type == 3:
                future_q.append((T1[k], T2[k]))   # 현재 T 저장

            # ── 행동 실행 후 next 관측 읽기 ───────────────────────
            time.sleep(max(0.0, dt - (time.time() - loop_start)))
            next_T1, next_T2 = arduino.T1, arduino.T2
            # 리워드 계산 기준 분기
            if reward_type == 1:
                err1, err2 = Tsp1[k] - T1[k], Tsp2[k] - T2[k]
            elif reward_type == 2 and k < steps - 1:
                err1, err2 = Tsp1[k + 1] - next_T1, Tsp2[k + 1] - next_T2
            elif reward_type == 3: 
                # n step 미래 T가 준비됐을 떄만 reward 계산 . 
                if len(future_q) == n_step + 1 :
                    T_future1, T_future2 = future_q.popleft()
                    err1 = Tsp1[k-n_step] - T_future1 
                    err2 = Tsp2[k-n_step] - T_future2
                else : 
                    err1 = err2 = 0.0 #큐가 아직 부족할 땐 reward 0 
            else:
                err1, err2 = 0.0, 0.0  # 마지막 step은 reward 없음

            reward = compute_reward(err1, err2, reward_scaler)
            total_ret += reward

            e1 += abs(err1);  e2 += abs(err2)
            over += max(0, -err1) + max(0, -err2)
            under += max(0,  err1) + max(0,  err2)

        arduino.Q1(0); arduino.Q2(0)

    csv_path = run_dir / "rollout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time","T1","T2","Q1","Q2","TSP1","TSP2"])
        for k in range(steps):
            w.writerow([t[k], T1[k], T2[k], Q1[k], Q2[k], Tsp1[k], Tsp2[k]])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(t, T1, label="T1"); ax[0].plot(t, Tsp1, "--", label="TSP1")
    ax[0].plot(t, T2, label="T2"); ax[0].plot(t, Tsp2, ":", label="TSP2")
    ax[0].grid(); ax[0].legend(); ax[0].set_ylabel("Temp (°C)")

    ax[1].plot(t, Q1, label="Q1"); ax[1].plot(t, Q2, label="Q2")
    ax[1].grid(); ax[1].legend(); ax[1].set_ylabel("Heater (%)")
    ax[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(run_dir / "rollout.png")
    plt.close()

    return dict(
        T1=T1, T2=T2, Tsp1=Tsp1, Tsp2=Tsp2,
        Q1=Q1, Q2=Q2,
        total_return=total_ret,
        E1=e1, E2=e2, Over=over, Under=under
    )
