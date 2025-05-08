from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QCheckBox, QMessageBox, QProgressDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer

import numpy as np
import torch
import joblib
import sys
import time
from pathlib import Path
from tclab import setup, TCLab

from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify


MODEL_PATH = "C:/Users/Developer/TCLab/IQL/logs_online_realkit/tclab-online/05-07-25_18.08.30_lzec/ep10.pt"
SCALER_PATH = "C:/Users/Developer/TCLab/Data/first_reward.pkl"


class IQLGuiApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCLab IQL 실시간 제어 GUI (QTimer 기반)")
        self.resize(800, 600)

        self.dt = 5.0
        self.step = 0
        self.total_steps = 0
        self.env = None

        self.T1 = []
        self.T2 = []
        self.TSP1 = []
        self.TSP2 = []
        self.Q1 = []
        self.Q2 = []

        self.total_e1 = 0
        self.total_e2 = 0
        self.total_over = 0
        self.total_under = 0

        self.timer = QTimer()
        self.timer.setInterval(int(self.dt * 1000))
        self.timer.timeout.connect(self.control_step)

        self.temps_input = QLineEdit()
        self.sim_checkbox = QCheckBox("시뮬레이터 사용")
        self.start_btn = QPushButton("▶ 시작")
        self.save_btn = QPushButton("💾 결과 저장")
        self.status_label = QLabel("상태: 초기화됨")

        self.canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("실시간 온도 제어")

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("목표 온도들 (쉼표로):"))
        hlayout.addWidget(self.temps_input)
        hlayout.addWidget(self.sim_checkbox)
        hlayout.addWidget(self.start_btn)
        hlayout.addWidget(self.save_btn)

        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(self.status_label)
        self.setLayout(vlayout)

        self.load_model_and_scaler()
        self.start_btn.clicked.connect(self.init_control)
        self.save_btn.clicked.connect(self.save_results)

    def load_model_and_scaler(self):
        obs_dim = 4
        act_dim = 2
        policy = GaussianPolicy(obs_dim, act_dim, 256, 2)
        qf = TwinQ(obs_dim, act_dim, 256, 2)
        vf = ValueFunction(obs_dim, 256, 2)

        def dummy_opt(params): return torch.optim.Adam(params, lr=1e-3)

        self.iql = ImplicitQLearning(
            qf=qf, vf=vf, policy=policy,
            optimizer_factory=dummy_opt,
            max_steps=7500, tau=0.8, beta=3.0,
            alpha=0.005, discount=0.99,
        )
        self.iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.policy = self.iql.policy
        self.policy.eval()
        self.scaler = joblib.load(SCALER_PATH)
        self.status_label.setText("✅ 모델/스케일러 로드 완료")

    def init_control(self):
        try:
            temp_list = [float(x) for x in self.temps_input.text().split(",")]
        except:
            QMessageBox.critical(self, "입력 에러", "숫자만 쉼표로 입력하세요")
            return
        if not all(29 <= t <= 65 for t in temp_list):
            QMessageBox.warning(self, "온도 범위 오류", "29~65도 사이로 입력하세요")
            return

        self.TSP1 = self.generate_tsp(temp_list).tolist()
        self.TSP2 = self.generate_tsp(temp_list).tolist()
        self.total_steps = len(self.TSP1)
        self.step = 0

        self.T1.clear(); self.T2.clear()
        self.Q1.clear(); self.Q2.clear()

        self.total_e1 = self.total_e2 = self.total_over = self.total_under = 0

        if self.sim_checkbox.isChecked():
            lab = setup(connected=False)
            self.env = lab(synced=False)
        else:
            self.env = TCLab()

        self.env.Q1(0); self.env.Q2(0)
        self.env._T1 = self.env._T2 = 29.0

        self.timer.start()
        self.status_label.setText("🚀 제어 시작...")

    def control_step(self):
        if self.step >= self.total_steps:
            self.timer.stop()
            self.env.Q1(0); self.env.Q2(0)
            total_error = self.total_e1 + self.total_e2 + self.total_over + self.total_under
            self.status_label.setText(f"✅ 제어 완료 (Total Error: {total_error:.2f})")
            return

        if hasattr(self.env, "update"):
            self.env.update(t=self.step * self.dt)

        T1 = self.env.T1
        T2 = self.env.T2
        tsp1 = self.TSP1[self.step]
        tsp2 = self.TSP2[self.step]
        obs = np.array([T1, T2, tsp1, tsp2], dtype=np.float32)
        with torch.no_grad():
            act = self.policy.act(torchify(obs), deterministic=True).cpu().numpy()
        Q1 = float(np.clip(act[0], 0, 100))
        Q2 = float(np.clip(act[1], 0, 100))
        self.env.Q1(Q1); self.env.Q2(Q2)

        self.T1.append(T1); self.T2.append(T2)
        self.Q1.append(Q1); self.Q2.append(Q2)

        e1 = abs(tsp1 - T1)
        e2 = abs(tsp2 - T2)
        over = max(0, T1 - tsp1) + max(0, T2 - tsp2)
        under = max(0, tsp1 - T1) + max(0, tsp2 - T2)

        self.total_e1 += e1
        self.total_e2 += e2
        self.total_over += over
        self.total_under += under

        self.update_plot()
        self.step += 1

    def generate_tsp(self, values):
        steps = int(1200 / self.dt)
        num_seg = len(values)
        seg_len = steps // num_seg
        tsp = np.zeros(steps)
        for i, v in enumerate(values):
            start = i * seg_len
            end = steps if i == num_seg - 1 else (i + 1) * seg_len
            tsp[start:end] = v
        return tsp

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.T1, label="T1")
        self.ax.plot(self.T2, label="T2")
        self.ax.plot(self.TSP1, "--", label="TSP1")
        self.ax.plot(self.TSP2, ":", label="TSP2")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Temp (°C)")
        self.ax.legend()
        self.canvas.draw()

    def save_results(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV로 저장", "rollout.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["T1", "T2", "Q1", "Q2", "TSP1", "TSP2"])
            for row in zip(self.T1, self.T2, self.Q1, self.Q2, self.TSP1, self.TSP2):
                writer.writerow(row)
        self.status_label.setText(f"✅ 결과 저장됨: {Path(path).name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = IQLGuiApp()
    gui.show()
    sys.exit(app.exec_())
