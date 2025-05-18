import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import time

# 👉 실제 키트가 연결되지 않은 환경에서도 실행되도록 try‑except
try:
    from tclab import setup, TCLab
    TCLAB_AVAILABLE = True
except ImportError:
    TCLAB_AVAILABLE = False

from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify

# 📌 Font (Windows 한글)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\User\tclab2\IQL\logs_online_realkit\tclab-online\present_reward_online_tuning_act()_05-18-0009\05-18-25_00.09.56_zzed\best.pt"
SCALER_PATH = r"C:/Users/User/tclab2/Data/reward/first.pkl"

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")

st.title("🌡️ TCLab ‑ IQL 실시간 제어")

mode = st.radio("🧪 실행 환경", ["Simulator", "Real Kit"], horizontal=True)

method = st.selectbox("🎯 TSP 생성 방식", ["사용자 지정", "랜덤", "사인 함수"])
show_preview = st.checkbox("📈 TSP 미리보기", value=True)

# 입력 위젯 --------------------------------------------------------------------
if method == "사용자 지정":
    c1, c2 = st.columns(2)
    with c1:
        temp1_str = st.text_input("TSP1 (쉼표 구분)", "35,45,55")
    with c2:
        temp2_str = st.text_input("TSP2 (쉼표 구분)", "40,50,60")
elif method == "랜덤":
    c1, c2 = st.columns(2)
    with c1:
        low1, high1 = st.slider("TSP1 범위", 30, 70, (35, 55))
    with c2:
        low2, high2 = st.slider("TSP2 범위", 30, 70, (40, 60))
else:  # 사인
    c1, c2 = st.columns(2)
    with c1:
        base1 = st.slider("TSP1 기준", 30, 65, 40)
        amp1  = st.slider("TSP1 진폭", 1, 15, 10)
    with c2:
        base2 = st.slider("TSP2 기준", 30, 65, 45)
        amp2  = st.slider("TSP2 진폭", 1, 15, 10)

# 공통 파라미터 ---------------------------------------------------------------
dt        = 5.0
horizon_s = 1200
steps     = int(horizon_s / dt)        # 240 step
obs_dim   = 4
act_dim   = 2

# TSP 생성 --------------------------------------------------------------------

def create_tsp():
    if method == "사용자 지정":
        try:
            v1 = [float(x) for x in temp1_str.split(",")]
            v2 = [float(x) for x in temp2_str.split(",")]
        except ValueError:
            st.error("숫자를 쉼표로 구분해 입력하세요.")
            return None, None
        seg_len = steps // len(v1)
        T1 = np.concatenate([np.full(seg_len, v) for v in v1])[:steps]
        T2 = np.concatenate([np.full(seg_len, v) for v in v2])[:steps]
    elif method == "랜덤":
        T1 = np.random.uniform(low1, high1, steps)
        T2 = np.random.uniform(low2, high2, steps)
    else:  # 사인
        t = np.arange(steps) * dt
        T1 = base1 + amp1 * np.sin(2 * np.pi * t / horizon_s)
        T2 = base2 + amp2 * np.cos(2 * np.pi * t / horizon_s)
    return T1, T2

Tsp1, Tsp2 = create_tsp()

if show_preview and Tsp1 is not None:
    st.subheader("TSP 미리보기")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(Tsp1, label="TSP1")
    ax.plot(Tsp2, label="TSP2")
    ax.set_xlabel("Step"); ax.set_ylabel("°C")
    ax.grid(); ax.legend()
    st.pyplot(fig)

run = st.button("🚀 제어 시작")

# ────────────────────────────────────────────────────────────────────────────────
# 모델 & 스케일러 로드 (캐시)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_iql_policy():
    policy = GaussianPolicy(obs_dim, act_dim, 256, 2)
    qf     = TwinQ(obs_dim, act_dim, 256, 2)
    vf     = ValueFunction(obs_dim, 256, 2)
    dummy_opt = lambda p: torch.optim.Adam(p, lr=1e-3)
    iql = ImplicitQLearning(qf, vf, policy, dummy_opt, max_steps=7500, tau=0.8, beta=3.0, alpha=0.005, discount=0.99)
    iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    scaler = joblib.load(SCALER_PATH)
    return iql.policy.eval(), scaler

# 실행 ------------------------------------------------------------------------
if run and Tsp1 is not None:
    policy, reward_scaler = load_iql_policy()

    # 환경 초기화
    if mode == "Simulator":
        if not TCLAB_AVAILABLE:
            st.error("tclab 패키지가 설치되지 않았습니다. Simulator 모드를 사용할 수 없습니다.")
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

    # 초기 상태
    env.Q1(0); env.Q2(0)
    if hasattr(env, "_T1"):  # 시뮬레이터 초기화
        env._T1 = env._T2 = 29.0

    # 로그 저장용 리스트
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

        # 기록
        T1_list.append(T1); T2_list.append(T2)
        Q1_list.append(Q1); Q2_list.append(Q2)

        # 보상 및 에러
        err1 = Tsp1[k] - T1
        err2 = Tsp2[k] - T2
        raw_r = -np.sqrt(err1**2 + err2**2)
        reward = reward_scaler.transform([[raw_r]])[0, 0]
        total_ret += reward
        E1 += abs(err1); E2 += abs(err2)
        Over  += max(0, -err1) + max(0, -err2)
        Under += max(0,  err1) + max(0,  err2)

        # 실시간 그래프 업데이트
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
