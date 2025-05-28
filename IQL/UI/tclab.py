import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, torch, os, sys
from random import uniform
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tclab import setup, TCLab
from IQL.src.policy import GaussianPolicy
from IQL.src.value_functions import TwinQ, ValueFunction
from IQL.src.iql import ImplicitQLearning
from IQL.src.util import torchify

MODEL_PATH_MPC = r"C:\Users\Developer\TCLab\IQL\src\best.pt"
MODEL_PATH_PID = "UI/model/PID_based_RL.pt"

st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")
st.title("🌡️ TCLab-실시간 제어")

def connect_env(mode: str):
    if mode == "RealKit":
        return TCLab()
    else:
        lab = setup(connected=False)
        return lab(synced=False)

def disconnect_env():
    env = st.session_state.get("env", None)
    if env is not None:
        try:
            env.close()
        except Exception as e:
            st.warning(f"환경 close 중 오류: {e}")
    st.session_state["env"] = None
    st.session_state["env_mode"] = None


col_left, col_div, col_right = st.columns([1, 0.03, 2.5])


with col_left:
    st.header("🧩 그래프 설정")
    x_axis_duration = st.slider("Duration (s)", 300, 1500, 1200)
    graph_type      = st.selectbox("TSP 생성 방식", ["Random", "Custom", "Sinusoidal"])

    # 1 초 해상도 배열
    xs = np.arange(0, x_axis_duration, 1)
    if "Tsp1" not in st.session_state:
        st.session_state.Tsp1 = np.zeros_like(xs)
        st.session_state.Tsp2 = np.zeros_like(xs)

    Tsp1 = st.session_state.Tsp1
    Tsp2 = st.session_state.Tsp2

    if graph_type == "Random":
        mean_d = st.number_input("Mean Section Dur (s)", 1, x_axis_duration, int(x_axis_duration*0.4))
        std_d  = st.number_input("Std Dur", 0, 200, int(x_axis_duration*25//300))
        min_d  = x_axis_duration*40//300
        max_d  = x_axis_duration*200//300
        tmin   = st.number_input("Temp Min", 0, 100, 25)
        tmax   = st.number_input("Temp Max", 0, 100, 65)

        if st.button("Generate"):
            cur = 0
            while cur < x_axis_duration:
                dur = int(np.clip(np.random.normal(mean_d, std_d), min_d, max_d))
                dur = min(dur, x_axis_duration - cur)
                Tsp1[cur:cur+dur] = uniform(tmin, tmax)
                cur += dur
            cur = 0
            while cur < x_axis_duration:
                dur = int(np.clip(np.random.normal(mean_d, std_d), min_d, max_d))
                dur = min(dur, x_axis_duration - cur)
                Tsp2[cur:cur+dur] = uniform(tmin, tmax)
                cur += dur

    elif graph_type == "Custom":
        st.write("### Custom 설정")
        sec = st.slider("구간 수", 1, 10, 3)
        Tsp1[:] = 29; Tsp2[:] = 29
        idx1 = idx2 = 0
        for i in range(sec):
            with st.expander(f"Section {i+1}", expanded=(i==0)):
                d1 = st.number_input("Dur1", 0, x_axis_duration-idx1, x_axis_duration//sec, key=f"d1{i}")
                t1 = st.number_input("Temp1", 0, 100, 25, key=f"t1{i}")
                d2 = st.number_input("Dur2", 0, x_axis_duration-idx2, x_axis_duration//sec, key=f"d2{i}")
                t2 = st.number_input("Temp2", 0, 100, 30, key=f"t2{i}")
            Tsp1[idx1:idx1+d1] = t1; idx1 += d1
            Tsp2[idx2:idx2+d2] = t2; idx2 += d2

    else:
        amp1 = st.slider("Amp1", 1, 50, 20); freq1 = st.slider("Freq1", 1, 10, 2)
        amp2 = st.slider("Amp2", 1, 50, 15); freq2 = st.slider("Freq2", 1, 10, 3)
        off1 = st.number_input("Offset1", 0, 100, 40); off2 = st.number_input("Offset2", 0, 100, 40)
        Tsp1[:] = amp1*np.sin(2*np.pi*freq1*(xs/x_axis_duration))+off1
        Tsp2[:] = amp2*np.sin(2*np.pi*freq2*(xs/x_axis_duration))+off2
    
    st.markdown("---")
    st.header("⚙️ 실행 설정")
    mode  = st.radio("실행 환경", ["Simulator", "RealKit"], horizontal=True)
    model = st.radio("모델 선택", ["MPC", "IQL"], horizontal=True)


    if st.button("🔌 연결 해제"):
        disconnect_env()
        st.success("연결 해제 완료")

with col_div:
    st.markdown("<div style='border-left:1px solid lightgray;height:150vh;'></div>", unsafe_allow_html=True)

with col_right:
    st.subheader("📉 TSP Preview")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(xs, Tsp1, label="TSP1"); ax.plot(xs, Tsp2, label="TSP2")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("°C"); ax.grid(); ax.legend()
    st.pyplot(fig)

    run = st.button("🚀 제어 시작")
    
    @st.cache_resource
    def load_iql(path, obs_dim, act_dim):
        pol = GaussianPolicy(obs_dim, act_dim, 256, 2)
        qf  = TwinQ(obs_dim, act_dim, 256, 2)
        vf  = ValueFunction(obs_dim, 256, 2)
        opt = lambda p: torch.optim.Adam(p, lr=1e-3)
        iql = ImplicitQLearning(qf, vf, pol, opt,
                                max_steps=7500, tau=0.8, beta=3.0,
                                alpha=0.005, discount=0.99)
        iql.load_state_dict(torch.load(path, map_location="cpu"))
        return iql.policy.eval()

    if run:
       
        env     = st.session_state.get("env", None)
        curmode = st.session_state.get("env_mode", None)

        if env is None or curmode != mode:
            # 다른 모드로 이미 연결돼 있으면 끊어주기
            disconnect_env()
            try:
                env = connect_env(mode)
                st.session_state["env"] = env
                st.session_state["env_mode"] = mode
                st.success(f"{mode} 연결 완료")
            except Exception as e:
                st.error(f"{mode} 연결 실패: {e}")
                st.stop()
        
        if model == "희진":
            MODEL_PATH, obs_dim, dt = MODEL_PATH_MPC, 4, 1.0
        elif model == "MPC":
            MODEL_PATH, obs_dim, dt = None, 0, 5.0
        else:
            MODEL_PATH, obs_dim, dt = None, 0, 1.0

        steps = int(x_axis_duration / dt)

        # ── 컨트롤러 준비 ------------------------------------------------------
        if model in ["IQL"]:
            policy = load_iql(MODEL_PATH, obs_dim, 2)
        elif model == "MPC":
            from mpc_lib import mpc_init
            mpc_ctl = mpc_init()
            ctrl_Tsp1 = Tsp1[::5][:steps]
            ctrl_Tsp2 = Tsp2[::5][:steps]
        else:
            from pid_controller import PIDController
            pid1 = PIDController(2.0,0.1,0.05,setpoint=Tsp1[0])
            pid2 = PIDController(2.0,0.1,0.05,setpoint=Tsp2[0])

        # ── 제어 루프 ----------------------------------------------------------
        env.Q1(0); env.Q2(0)
        T1_list, T2_list, Q1_list, Q2_list = [], [], [], []
        total_ret=E1=E2=Over=Under=0.0
        prog = st.progress(0.0); live = st.empty()

        for k in range(steps):
            if mode == "Simulator": env.update(t=k*dt)
            else: time.sleep(dt)

            T1,T2 = env.T1, env.T2
            idx   = k if model!="MPC" else k   

            
            if model in ["IQL"]:
                if model=="IQL":
                    obs = torchify(np.array([T1,T2,Tsp1[k],Tsp2[k]],dtype=np.float32))
                else:
                    dT1 = (T1 - T1_list[-4]) if len(T1_list)>=4 else 0
                    dT2 = (T2 - T2_list[-4]) if len(T2_list)>=4 else 0
                    obs = torchify(np.array([T1,Tsp1[k],dT1,T2,Tsp2[k],dT2],dtype=np.float32))
                act = policy.act(obs,deterministic=True).cpu().numpy()
                Q1,Q2 = np.clip(act,0,100)

            elif model == "MPC":
                Q1,Q2 = mpc_ctl.step([T1,T2],[ctrl_Tsp1[k],ctrl_Tsp2[k]],dt)

            else:  
                pid1.setpoint = Tsp1[k]; pid2.setpoint = Tsp2[k]
                Q1 = np.clip(pid1.update(T1,dt),0,100)
                Q2 = np.clip(pid2.update(T2,dt),0,100)

            env.Q1(Q1); env.Q2(Q2)

            T1_list.append(T1); T2_list.append(T2)
            Q1_list.append(Q1); Q2_list.append(Q2)
            ref1,ref2 = (ctrl_Tsp1[k], ctrl_Tsp2[k]) if model=="MPC" else (Tsp1[k],Tsp2[k])
            err1,err2 = ref1-T1, ref2-T2
            total_ret -= np.sqrt(err1**2+err2**2)
            E1+=abs(err1); E2+=abs(err2)
            Over+=max(0,-err1)+max(0,-err2)
            Under+=max(0,err1)+max(0,err2)

            if k%max(1,steps//240)==0 or k==steps-1:
                f,a = plt.subplots(figsize=(8,3))
                a.plot(T1_list,label="T1"); a.plot(T2_list,label="T2")
                if model=="MPC":
                    a.plot(ctrl_Tsp1[:k+1],"--",label="TSP1")
                    a.plot(ctrl_Tsp2[:k+1],":", label="TSP2")
                else:
                    a.plot(Tsp1[:k+1],"--",label="TSP1")
                    a.plot(Tsp2[:k+1],":", label="TSP2")
                a.grid(); a.legend(ncol=4,fontsize=8)
                live.pyplot(f)

            prog.progress((k+1)/steps)

        env.Q1(0); env.Q2(0)
        st.success("✅ 제어 완료")
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Return",f"{total_ret:.2f}")
        c2.metric("Total Error",f"{E1+E2:.2f}")
        c3.metric("Over | Under",f"{Over:.1f}/{Under:.1f}")

        df_out = pd.DataFrame({
            "T1":T1_list,"T2":T2_list,"Q1":Q1_list,"Q2":Q2_list,
            "TSP1": (ctrl_Tsp1 if model=="MPC" else Tsp1)[:len(T1_list)],
            "TSP2": (ctrl_Tsp2 if model=="MPC" else Tsp2)[:len(T1_list)]
        })
        st.download_button("📥 CSV 다운로드",
                           df_out.to_csv(index=False).encode("utf-8"),
                           file_name="rollout.csv")
