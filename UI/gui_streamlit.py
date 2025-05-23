import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from tclab import setup, TCLab
import matplotlib.pyplot as plt
import matplotlib 
import time

from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify


# 📌 한글 폰트 설정 (Windows 기준)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# 고정 경로
MODEL_PATH = "C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/05-07-25_18.08.30_lzec/ep10.pt"
SCALER_PATH = "C:/Users/Developer/TCLab/Data/first_reward.pkl"

# 페이지 설정
st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>input {
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌡️ TCLab IQL 실시간 제어 대시보드")

# 사용자 입력
col1, col2 = st.columns(2)
with col1:
    temp1_str = st.text_input("🎯 목표 온도 TSP1 (쉼표로 입력)", "35,45,55")
with col2:
    temp2_str = st.text_input("🎯 목표 온도 TSP2 (쉼표로 입력)", "40,50,60")
mode = st.radio("🧪 실행 환경을 선택하세요", ["Simulator", "Real Kit"], horizontal=True)
run_button = st.button("🚀 제어 시작")

# 설정값
dt = 5.0
total_time = 1200
obs_dim = 4
act_dim = 2

@st.cache_resource
def load_model_and_scaler():
    policy = GaussianPolicy(obs_dim, act_dim, 256, 2)
    qf = TwinQ(obs_dim, act_dim, 256, 2)
    vf = ValueFunction(obs_dim, 256, 2)
    dummy_opt = lambda params: torch.optim.Adam(params, lr=1e-3)
    iql = ImplicitQLearning(
        qf=qf, vf=vf, policy=policy,
        optimizer_factory=dummy_opt,
        max_steps=7500, tau=0.8, beta=3.0,
        alpha=0.005, discount=0.99,
    )
    iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    scaler = joblib.load(SCALER_PATH)
    return iql.policy.eval(), scaler

policy, reward_scaler = load_model_and_scaler()

def generate_tsp(values, total_time=1200, dt=5.0):
    steps = int(total_time / dt)
    num_seg = len(values)
    seg_len = steps // num_seg
    tsp = np.zeros(steps)
    for i, val in enumerate(values):
        start = i * seg_len
        end = steps if i == num_seg - 1 else (i + 1) * seg_len
        tsp[start:end] = val
    return tsp

if run_button:
    try:
        values1 = [float(v) for v in temp1_str.split(",")]
        values2 = [float(v) for v in temp2_str.split(",")]
        assert all(29 <= t <= 65 for t in values1 + values2)
    except:
        st.error("❌ 온도는 29~65 사이의 숫자만 쉼표로 입력해야 합니다.")
        st.stop()

    Tsp1 = generate_tsp(values1)
    Tsp2 = generate_tsp(values2)

    steps = len(Tsp1)
    T1_list, T2_list = [], []
    Q1_list, Q2_list = [], []
    total_ret = 0.0
    E1 = E2 = Over = Under = 0.0

    if mode == "Simulator":
        lab = setup(connected=False)
        env = lab(synced=False)
    else:
        env = TCLab()

    env.Q1(0)
    env.Q2(0)
    env._T1 = env._T2 = 29.0

    progress = st.progress(0)
    plot = st.empty()

    for k in range(steps):
        if hasattr(env, "update"):
            env.update(t=k*dt)

        T1 = env.T1
        T2 = env.T2
        obs = np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32)
        with torch.no_grad():
            act = policy.act(torchify(obs), deterministic=True).cpu().numpy()

        Q1 = float(np.clip(act[0], 0, 100))
        Q2 = float(np.clip(act[1], 0, 100))
        env.Q1(Q1)
        env.Q2(Q2)

        T1_list.append(T1)
        T2_list.append(T2)
        Q1_list.append(Q1)
        Q2_list.append(Q2)

        err1 = Tsp1[k] - T1
        err2 = Tsp2[k] - T2
        raw_reward = -np.sqrt(err1**2 + err2**2)
        reward = reward_scaler.transform([[raw_reward]])[0][0]
        total_ret += reward

        E1 += abs(err1)
        E2 += abs(err2)
        Over += max(0, -err1) + max(0, -err2)
        Under += max(0, err1) + max(0, err2)

        df = pd.DataFrame({
            "T1": T1_list, "T2": T2_list,
            "TSP1": Tsp1[:k+1], "TSP2": Tsp2[:k+1]
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["T1"], label="T1")
        ax.plot(df["T2"], label="T2")
        ax.plot(df["TSP1"], "--", label="TSP1")
        ax.plot(df["TSP2"], ":", label="TSP2")
        ax.legend(); ax.grid(); ax.set_ylabel("온도 (°C)")
        ax.set_xlabel("시간 Step")
        plot.pyplot(fig)
        progress.progress((k+1)/steps)
        time.sleep(dt if mode == "Real Kit" else 0.01)

    env.Q1(0); env.Q2(0)
    st.success("✅ 제어 종료 완료")

    st.markdown(f"### 📊 평가 지표")
    st.metric("Total Return", f"{total_ret:.2f}")
    st.metric("Total Error", f"{E1+E2+Over+Under:.2f}")
    st.write(f"- **E1**: {E1:.3f}, **E2**: {E2:.3f}")
    st.write(f"- **Over**: {Over:.3f}, **Under**: {Under:.3f}")

    df_out = pd.DataFrame({
        "T1": T1_list, "T2": T2_list,
        "Q1": Q1_list, "Q2": Q2_list,
        "TSP1": Tsp1, "TSP2": Tsp2
    })
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("📥 결과 CSV 다운로드", data=csv, file_name="rollout.csv")
