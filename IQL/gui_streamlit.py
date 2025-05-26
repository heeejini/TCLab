import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
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

MODEL_PATH = r"C:\Users\Developer\tclab-refactor\TCLab\IQL\best.pt"
SCALER_PATH = r"C:\Users\Developer\TCLab\Data\reward\first.pkl"


def generate_tsp(method,
                 low1=None, high1=None, low2=None, high2=None,
                 base1=None, amp1=None, base2=None, amp2=None,
                 num_cycles1=None, num_cycles2=None,
                 t2_max=65, steps=240, dt=5.0, horizon_s=1200):
    if method == "사용자 지정":
        seg1 = steps // len(low1)
        seg2 = steps // len(high2)
        T1 = np.concatenate([np.full(seg1, v) for v in low1])
        T2 = np.concatenate([np.full(seg2, min(v, t2_max)) for v in high2])
        T1 = T1[:steps] if T1.size >= steps else np.pad(T1, (0, steps - T1.size), "edge")
        T2 = T2[:steps] if T2.size >= steps else np.pad(T2, (0, steps - T2.size), "edge")
    elif method == "Random":
        def gen_random(low, high):
            n = int(horizon_s / dt)
            tsp = np.zeros(n)
            i = 0
            while i < n:
                dur = int(np.clip(np.random.normal(480, 100), 160, 800))
                dur_steps = max(1, int(dur / dt))
                end = min(i + dur_steps, n)
                val = round(np.random.uniform(low, high), 2)
                tsp[i:end] = val
                i = end
            return tsp
        T1 = gen_random(low1, high1)
        T2 = np.clip(gen_random(low2, high2), None, t2_max)
    else:
        t = np.arange(steps) * dt
        T1 = base1 + amp1 * np.sin(2 * np.pi * num_cycles1 * t / horizon_s)
        T2 = base2 + amp2 * np.cos(2 * np.pi * num_cycles2 * t / horizon_s)
        T2 = np.clip(T2, None, t2_max)
    return T1, T2

st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")
st.title("🌡️ TCLab - IQL 실시간 제어")

mode = st.radio("🥪 실행 환경", ["Simulator", "Real Kit"], horizontal=True)
method = st.selectbox("🌟 TSP 생성 방식", ["사용자 지정", "Random", "Sin graph"])
show_preview = st.checkbox("📈 TSP 미리보기", value=True)

dt, horizon_s = 5.0, 1200
steps = int(horizon_s / dt)
t2_max = 55 if mode == "Simulator" else 65

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
        low2, high2 = st.slider("TSP2 범위", 25, t2_max, (30, min(60, t2_max)))
elif method == "Sin graph":
    c1, c2 = st.columns(2)
    with c1:
        base1 = st.slider("TSP1 기준", 30, 65, 40)
        amp1 = st.slider("TSP1 진폭", 1, 15, 10)
    with c2:
        base2 = st.slider("TSP2 기준", 30, t2_max, 45)
        amp2_max = max(1, min(15, t2_max - base2))
        amp2 = st.slider("TSP2 진폭", 1, amp2_max, min(10, amp2_max))
    c1, c2 = st.columns(2)
    with c1:
        num_cycles1 = st.slider("🔁 TSP1 반복 횟수", 1, 10, 1)
    with c2:
        num_cycles2 = st.slider("🔁 TSP2 반복 횟수", 1, 10, 1)

# TSP 생성 및 상태 관리
if method == "사용자 지정":
    try:
        list1 = [float(x) for x in temp1_str.split(",")]
        list2 = [float(x) for x in temp2_str.split(",")]
    except:
        st.error("숫자를 쉼표로 구분해 입력하세요.")
        st.stop()
    T1_temp, T2_temp = generate_tsp(method, low1=list1, high2=list2, t2_max=t2_max, steps=steps, dt=dt, horizon_s=horizon_s)
elif method == "Random":
    T1_temp, T2_temp = generate_tsp(method, low1=low1, high1=high1, low2=low2, high2=high2, t2_max=t2_max, steps=steps, dt=dt, horizon_s=horizon_s)
else:
    T1_temp, T2_temp = generate_tsp(method, base1=base1, amp1=amp1, base2=base2, amp2=amp2, num_cycles1=num_cycles1, num_cycles2=num_cycles2, t2_max=t2_max, steps=steps, dt=dt, horizon_s=horizon_s)

regen = False
if "method_prev" not in st.session_state or st.session_state["method_prev"] != method:
    regen = True
if st.button("🔄 TSP 새로 생성"):
    regen = True

if regen or "Tsp1" not in st.session_state:
    st.session_state["Tsp1"] = T1_temp
    st.session_state["Tsp2"] = T2_temp
    st.session_state["method_prev"] = method

Tsp1 = st.session_state["Tsp1"]
Tsp2 = st.session_state["Tsp2"]

if show_preview:
    st.subheader("TSP 미리보기")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(Tsp1, label="TSP1")
    ax.plot(Tsp2, label="TSP2")
    ax.set_xlabel("Step"); ax.set_ylabel("°C"); ax.grid(); ax.legend()
    st.pyplot(fig)
    plt.close(fig)

run = st.button("🚀 제어 시작")

@st.cache_resource
def load_iql_policy():
    policy = GaussianPolicy(4, 2, 256, 2)
    qf = TwinQ(4, 2, 256, 2)
    vf = ValueFunction(4, 256, 2)
    opt = lambda p: torch.optim.Adam(p, lr=1e-3)
    iql = ImplicitQLearning(qf, vf, policy, opt, max_steps=7500, tau=0.8, beta=3.0, alpha=0.005, discount=0.99)
    iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    scaler = joblib.load(SCALER_PATH)
    return iql.policy.eval(), scaler

if run and Tsp1 is not None:
    policy, reward_scaler = load_iql_policy()

    if mode == "Simulator":
        if not TCLAB_AVAILABLE:
            st.error("tclab 패키지가 없습니다."); st.stop()
        env = setup(connected=False)(synced=False)
    else:
        if not TCLAB_AVAILABLE:
            st.error("TCLab 라이브러리가 없습니다."); st.stop()
        try:
            env = TCLab()
        except Exception as e:
            st.error(f"TCLab 연결 오류: {e}"); st.stop()

    env.Q1(0); env.Q2(0)
    if hasattr(env, "_T1"): env._T1 = env._T2 = 29.0

    T1_log, T2_log, Q1_log, Q2_log = [], [], [], []
    total_ret = E1 = E2 = Over = Under = 0.0
    prog = st.progress(0.0); live = st.empty()

    for k in range(steps):
        if hasattr(env, "update"): env.update(t=k * dt)

        T1, T2 = env.T1, env.T2
        obs = torchify(np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32))
        with torch.no_grad():
            act = policy.act(obs, deterministic=True).cpu().numpy()
        Q1, Q2 = float(np.clip(act[0], 0, 100)), float(np.clip(act[1], 0, 100))
        env.Q1(Q1); env.Q2(Q2)

        T1_log.append(T1); T2_log.append(T2)
        Q1_log.append(Q1); Q2_log.append(Q2)

        err1, err2 = Tsp1[k] - T1, Tsp2[k] - T2
        raw_r = -np.sqrt(err1**2 + err2**2)
        total_ret += reward_scaler.transform([[raw_r]])[0, 0]
        E1 += abs(err1); E2 += abs(err2)
        Over  += max(0, -err1) + max(0, -err2)
        Under += max(0,  err1) + max(0,  err2)

        if k % 5 == 0 or k == steps - 1:
            df = pd.DataFrame({"T1":T1_log,"T2":T2_log,
                               "TSP1":Tsp1[:k+1],"TSP2":Tsp2[:k+1]})
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(df["T1"], label="T1"); ax.plot(df["T2"], label="T2")
            ax.plot(df["TSP1"], "--", label="TSP1"); ax.plot(df["TSP2"], ":", label="TSP2")
            ax.set_xlabel("Step"); ax.set_ylabel("°C"); ax.grid(); ax.legend()
            live.pyplot(fig)
            plt.close(fig)

        prog.progress((k+1)/steps)
        time.sleep(dt if mode=="Real Kit" else 0.01)

    env.Q1(0); env.Q2(0)

    st.subheader("✅ 제어 완료")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Return", f"{total_ret:.2f}")
    c2.metric("Total Error",  f"{E1+E2:.2f}")
    c3.metric("Over | Under", f"{Over:.1f} / {Under:.1f}")

    df_out = pd.DataFrame({"T1":T1_log,"T2":T2_log,
                           "Q1":Q1_log,"Q2":Q2_log,
                           "TSP1":Tsp1,"TSP2":Tsp2})
    st.download_button("📥 CSV 다운로드",
                       df_out.to_csv(index=False).encode("utf-8"),
                       file_name="rollout.csv")