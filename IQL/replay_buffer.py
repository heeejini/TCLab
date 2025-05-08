from pathlib import Path
import numpy as np
import torch
import json 
class ExperienceBufferManager:
    def __init__(self):
        self.buffer = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "terminals": []
        }

    def add_transition(self, obs, action, next_obs, reward, done):
        self.buffer["observations"].append(obs)
        self.buffer["actions"].append(action)
        self.buffer["next_observations"].append(next_obs)
        self.buffer["rewards"].append(reward)
        self.buffer["terminals"].append(done)

    def save(self, save_path: str | Path):
        np.savez(
            save_path,
            observations=np.array(self.buffer["observations"], dtype=np.float32),
            actions=np.array(self.buffer["actions"], dtype=np.float32),
            next_observations=np.array(self.buffer["next_observations"], dtype=np.float32),
            rewards=np.array(self.buffer["rewards"], dtype=np.float32),
            terminals=np.array(self.buffer["terminals"], dtype=bool)
        )
        print(f"Buffer saved to {save_path}")

        save_path = Path(save_path)
        save_path = save_path.with_suffix("npz")
        metadata_path = save_path.with_name(save_path.stem + "_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "reward_scaled": True,
            "observation_dim": len(self.buffer["observations"][0]),
            "action_dim": len(self.buffer["actions"][0]),
            "num_transitions": len(self)
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")

    def load(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.buffer = {k: data[k].tolist() for k in data.files}
        print(f"Buffer loaded from {npz_path} (size: {len(self.buffer['observations'])})")

    def to_torch(self):
        return {
            k: torch.tensor(v, dtype=torch.float32) if k != "terminals" else torch.tensor(v, dtype=torch.bool)
            for k, v in self.buffer.items()
        }

    def __len__(self):
        return len(self.buffer["observations"])
