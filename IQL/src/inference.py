from util import torchify
from tclab import TCLab
import numpy as np
import time
import matplotlib.pyplot as plt
from src.eval_policy import generate_random_tsp

def run_policy_on_tclab(policy, total_time=1200, dt=5.0, ambient=29.0):
    steps = int(total_time / dt)

    T1 = np.zeros(steps)
    T2 = np.zeros(steps)
    Q1 = np.zeros(steps)
    Q2 = np.zeros(steps)
    Tsp1 = generate_random_tsp(total_time, dt)  # 너가 이미 정의한 setpoint 생성 함수
    Tsp2 = generate_random_tsp(total_time, dt)
    t = np.arange(steps) * dt

    with TCLab() as lab:
        print(lab.version)
        lab.LED(100)

        for k in range(steps):
            T1[k] = lab.T1
            T2[k] = lab.T2

            obs = np.array([T1[k], T2[k], Tsp1[k], Tsp2[k]], dtype=np.float32)
            act = policy.act(torchify(obs), deterministic=True).cpu().numpy()

            Q1[k] = float(np.clip(act[0], 0, 100))
            Q2[k] = float(np.clip(act[1], 0, 100))
            lab.Q1(Q1[k])
            lab.Q2(Q2[k])

            print(f"[{k*dt:4.0f}s] T1={T1[k]:.2f}, T2={T2[k]:.2f} | Q1={Q1[k]:.2f}, Q2={Q2[k]:.2f}")
            time.sleep(dt)

        lab.Q1(0); lab.Q2(0)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(t, T1, label="T1"); plt.plot(t, Tsp1, "--", label="TSP1")
    plt.plot(t, T2, label="T2"); plt.plot(t, Tsp2, "--", label="TSP2")
    plt.legend(); plt.grid(); plt.title("TCLab Real-world Policy Execution")
    plt.xlabel("Time (s)"); plt.ylabel("Temperature (°C)")
    plt.show()
