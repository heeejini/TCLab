import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import time

try:
    from tclab import setup, TCLab
    TCLAB_AVAILABLE = True
except ImportError:
    TCLAB_AVAILABLE = False

from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

MODEL_PATH  = r"C:\Users\Developer\TCLab\IQL\logs_online_realkit\tclab-online\new_action_epsilon0.15_05-16-1545\05-16-25_15.45.55_ygbw\best.pt"
SCALER_PATH = "C:\\Users\\Developer\\TCLab\\Data\\reward\\first.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# 1. 헬퍼: 세그먼트 기반 랜덤 TSP 생성 함수 (사용자 제공 스펙 그대로)            
# ──────────────────────────────────────────────────────────────────────────────

def generate_random_tsp(
    total_time_sec: int = 1200,
    dt: float = 5.0,
    low: float = 25.0,
    high: float = 65.0,
    verbose: bool = False,
) -> np.ndarray:
    """Set‑point 시퀀스를 구간 단위로 생성한다.

    \- 구간 길이는 total_time_sec 값에 따라 다른 정규분포를 사용한다.
    \- 각 구간의 온도는 `low` ~ `high` 범위에서 균등 샘플링한다.
    """
    n_steps = int(total_time_sec / dt)
    tsp = np.zeros(n_steps)
    i = 0
    seg_id = 1

    # duration 분포 설정
    if total_time_sec == 600:
        mean_dur, std_dur, min_dur, max_dur = 240, 50, 80, 400
    elif total_time_sec == 900:
        mean_dur, std_dur, min_dur, max_dur = 360, 75, 120, 600
    else:  # 1200s 기본
        mean_dur, std_dur, min_dur, max_dur = 480, 100, 160, 800

    if verbose:
        print(
            f"--- Set‑point 프로파일 생성 (총 {total_time_sec}s, step={n_steps}) ---"
        )

    while i < n_steps:
        duration = int(
            np.clip(np.random.normal(mean_dur, std_dur), min_dur, max_dur)
        )
        dur_steps = max(1, int(duration / dt))
        end = min(i + dur_steps, n_steps)

        temp = round(np.random.uniform(low, high), 2)
        tsp[i:end] = temp

        if verbose:
            print(
                f"구간 {seg_id}: step {i:>3} ~ {end-1:>3} → {temp:.2f}°C"  # noqa: E501
            )
        i = end
        seg_id += 1

    if verbose:
        print("-----------------------------------------------------------\n")
    return tsp

# ──────────────────────────────────────────────────────────────────────────────
# 2. 스트림릿 대시보드
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")
st.title("🌡️ TCLab ‑ IQL 실시간 제어")

mode = st.radio("🧪 실행 환경", ["Simulator", "Real Kit"], horizontal=True)
method = st.selectbox("🎯 TSP 생성 방식", ["사용자 지정", "Random", "Sin graph"])
show_preview = st.checkbox("📈 TSP 미리보기", value=True)

# 사용자 입력 위젯 ------------------------------------------------------------
if method == "사용자 지정":
    c1, c2 = st.columns(2)
    with c1:
        temp1_str = st.text_input("TSP1 (쉼표 구분)", "35,45,55")
    with c2:
        temp2_str = st.text_input("TSP2 (쉼표 구분)", "40,50,60")
elif method == "Random":
    c1, c2 = st.columns(2)
    with c1:
        low1, high1 = st.slider("TSP1 범위", 25, 70, (30, 60))
    with c2:
        low2, high2 = st.slider("TSP2 범위", 25, 70, (30, 60))
else:  # 사인 함수
    c1, c2 = st.columns(2)
    with c1:
        base1 = st.slider("TSP1 기준", 30, 65, 40)
        amp1 = st.slider("TSP1 진폭", 1, 15, 10)
    with c2:
        base2 = st.slider("TSP2 기준", 30, 65, 45)
        amp2 = st.slider("TSP2 진폭", 1, 15, 10)

# 실험 파라미터 ---------------------------------------------------------------
dt = 5.0
horizon_s = 1200
steps = int(horizon_s / dt)
obs_dim = 4
act_dim = 2

# TSP 시퀀스 생성 함수 ---------------------------------------------------------

def create_tsp():
    """UI 선택에 따라 TSP1, TSP2 시퀀스를 반환한다."""
    if method == "사용자 지정":
        try:
            v1 = [float(x) for x in temp1_str.split(",")]
            v2 = [float(x) for x in temp2_str.split(",")]
        except ValueError:
            st.error("숫자를 쉼표로 구분해 입력하세요.")
            return None, None

        seg_len1 = steps // len(v1)
        seg_len2 = steps // len(v2)
        T1 = np.concatenate([np.full(seg_len1, v) for v in v1])
        T2 = np.concatenate([np.full(seg_len2, v) for v in v2])
        T1 = (
            T1[:steps]
            if T1.size >= steps
            else np.pad(T1, (0, steps - T1.size), "edge")
        )
        T2 = (
            T2[:steps]
            if T2.size >= steps
            else np.pad(T2, (0, steps - T2.size), "edge")
        )
    elif method == "Random":
        T1 = generate_random_tsp(horizon_s, dt, low1, high1)
        T2 = generate_random_tsp(horizon_s, dt, low2, high2)
    else:  # 사인 함수
        t = np.arange(steps) * dt
        T1 = base1 + amp1 * np.sin(2 * np.pi * t / horizon_s)
        T2 = base2 + amp2 * np.cos(2 * np.pi * t / horizon_s)
    return T1, T2

Tsp1, Tsp2 = create_tsp()

# ─── TSP 미리보기 -------------------------------------------------------------
if show_preview and Tsp1 is not None:
    st.subheader("TSP 미리보기")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(Tsp1, label="TSP1")
    ax.plot(Tsp2, label="TSP2")
    ax.set_xlabel("Step")
    ax.set_ylabel("°C")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# ─── 실행 버튼 ----------------------------------------------------------------
run = st.button("🚀 제어 시작")

# IQL Policy 로딩 (캐시) ------------------------------------------------------

@st.cache_resource
def load_iql_policy():
    policy = GaussianPolicy(obs_dim, act_dim, 256, 2)
    qf = TwinQ(obs_dim, act_dim, 256, 2)
    vf = ValueFunction(obs_dim, 256, 2)
    dummy_opt = lambda p: torch.optim.Adam(p, lr=1e-3)
    iql = ImplicitQLearning(
        qf,
        vf,
        policy,
        dummy_opt,
        max_steps=7500,
        tau=0.8,
        beta=3.0,
        alpha=0.005,
        discount=0.99,
    )
    iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    scaler = joblib.load(SCALER_PATH)
    return iql.policy.eval(), scaler

# ─── 제어 루프 ----------------------------------------------------------------
if run and Tsp1 is not None:
    policy, reward_scaler = load_iql_policy()

    # 환경 선택
    if mode == "Simulator":
        if not TCLAB_AVAILABLE:
            st.error("tclab 패키지가 설치되지 않았습니다.")
            st.stop()
        lab_cls = setup(connected=False)
        env = lab_cls(synced=False)
    else:
        if not TCLAB_AVAILABLE:
            st.error("TCLab 하드웨어 라이브러리가 없습니다.")
            st.stop()
        try:
            env = TCLab()
        except Exception as ex:
            st.error(f"TCLab 연결 오류: {ex}")
            st.stop()

    env.Q1(0)
    env.Q2(0)
    if hasattr(env, "_T1"):
        env._T1 = env._T2 = 29.0

    T1_list, T2_list, Q1_list, Q2_list = [], [], [], []
    total_ret = 0.0
    E1 = E2 = Over = Under = 0.0

    prog = st.progress(0.0)

    live_plot = st.empty()

    for k in range(steps):
        now = k * dt
        if hasattr(env, "update"):
            env.update(t=now)

        T1, T2 = env.T1, env.T2
        obs = torchify(np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32))
        with torch.no_grad():
            action = policy.act(obs, deterministic=True).cpu().numpy()
        Q1 = float(np.clip(action[0], 0.0, 100.0))
        Q2 = float(np.clip(action[1], 0.0, 100.0))
        env.Q1(Q1); env.Q2(Q2)

        T1_list.append(T1); T2_list.append(T2)
        Q1_list.append(Q1); Q2_list.append(Q2)

        err1 = Tsp1[k] - T1
        err2 = Tsp2[k] - T2
        raw_r = -np.sqrt(err1**2 + err2**2)
        reward = reward_scaler.transform([[raw_r]])[0, 0]
        total_ret += reward
        E1 += abs(err1); E2 += abs(err2)
        Over  += max(0, -err1) + max(0, -err2)
        Under += max(0,  err1) + max(0,  err2)

        if k % 5 == 0 or k == steps - 1:
            df_tmp = pd.DataFrame({"T1": T1_list, "T2": T2_list, "TSP1": Tsp1[:k+1], "TSP2": Tsp2[:k+1]})
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df_tmp["T1"], label="T1"); ax.plot(df_tmp["T2"], label="T2")
            ax.plot(df_tmp["TSP1"], "--", label="TSP1"); ax.plot(df_tmp["TSP2"], ":", label="TSP2")
            ax.set_ylabel("°C"); ax.set_xlabel("Step"); ax.grid(); ax.legend(ncol=4, fontsize=8)
            live_plot.pyplot(fig)

        prog.progress((k+1) / steps)
        time.sleep(dt if mode == "Real Kit" else 0.01)

    env.Q1(0); env.Q2(0)

    st.subheader("✅ 제어 완료")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Return", f"{total_ret:.2f}")
    c2.metric("Total Error", f"{E1+E2:.2f}")
    c3.metric("Over | Under", f"{Over:.1f} / {Under:.1f}")

    df_out = pd.DataFrame({
        "T1": T1_list, "T2": T2_list,
        "Q1": Q1_list, "Q2": Q2_list,
        "TSP1": Tsp1, "TSP2": Tsp2
    })
    st.download_button("📥 CSV 다운로드", df_out.to_csv(index=False).encode("utf-8"), file_name="rollout")
