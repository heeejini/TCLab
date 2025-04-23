import numpy as np
import pandas as pd
from pathlib import Path

# 설정C:\Users\Developer\TCLab\csv
csv_folder = Path("C:\Users\Developer\TCLab\csv")
csv_files = sorted(csv_folder.glob("mpc_episode_*_data.csv"))

E1, E2 = 1.0, 1.0  # 가중치 설정

all_observations = []
all_actions = []
all_next_observations = []
all_rewards = []
all_dones = []

for file in csv_files:
    df = pd.read_csv(file)
    for i in range(len(df) - 1):
        curr = df.iloc[i]
        next_ = df.iloc[i + 1]
        
        state = [curr["T1"], curr["T2"], curr["TSP1"], curr["TSP2"]]
        action = [curr["Q1"], curr["Q2"]]
        next_state = [next_["T1"], next_["T2"], next_["TSP1"], next_["TSP2"]]
        
        # L2 norm 기반 reward
        error_vec = np.array([curr["TSP1"] - curr["T1"], curr["TSP2"] - curr["T2"]])
        reward = -np.linalg.norm(error_vec)
        
        done = (i == len(df) - 2)

        all_observations.append(state)
        all_actions.append(action)
        all_next_observations.append(next_state)
        all_rewards.append(reward)
        all_dones.append(done)

# numpy 배열로 변환
dataset = {
    "state": np.array(all_observations, dtype=np.float32),
    "actions": np.array(all_actions, dtype=np.float32),
    "next_state": np.array(all_next_observations, dtype=np.float32),
    "rewards": np.array(all_rewards, dtype=np.float32),
    "terminals": np.array(all_dones, dtype=bool),
}

# 저장
output_path_l2 = csv_folder.parent / "mpc_dataset.npz"
np.savez(output_path_l2, **dataset)

