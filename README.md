# 🔥 TCLab Project: Control + Offline RL + Evaluation

본 프로젝트는 **Temperature Control Lab (TCLab)** 키트를 기반으로, 실제 물리 시스템을 제어하며 **모델 예측 제어(MPC)** 데이터를 수집하고, 이를 바탕으로 **Offline Reinforcement Learning (Implicit Q-Learning, IQL)**을 학습시키는 실험을 수행합니다.

---

## 📌 What is TCLab?

[TCLab (Temperature Control Lab)](https://apmonitor.com/pdc/index.php/Main/ArduinoTemperatureControl) 는 BYU(Brigham Young University)에서 제작한 저비용 실험 키트로, **2개의 히터(Q1, Q2)**와 **2개의 온도 센서(T1, T2)**를 통해 **MIMO 제어 시스템**을 학습하고 테스트할 수 있는 물리적 장치입니다.

---

## 🧪 Project Overview

### 🔧 Step 1. **MPC 기반 데이터 수집**
- **수집 방식**: 모델 예측 제어(MPC)를 활용해 매 step마다 히터 출력을 계산
- **입력 상태**: `[T1, T2, TSP1, TSP2]`
- **출력 행동**: `[Q1, Q2]`
- **보상 정의**: `-√((T1-TSP1)^2 + (T2-TSP2)^2)`  
- **제어 주기 (dt)**: 5초  
- **총 실험 시간**: 1200초 (20분)
- **총 데이터 포인트**: 240개 per episode

### 🧠 Step 2. **Offline RL 학습 (IQL)**
- 수집된 MPC 데이터를 기반으로 Implicit Q-Learning 학습
- 정책은 GaussianPolicy 또는 DeterministicPolicy 형태
- 학습 후, policy의 roll-out 결과를 simulator 또는 실제 키트를 통해 평가

---

## 🗓️ 프로젝트 일정

| 기간 | 내용 |
|------|------|
| **4/21 ~ 4/25** | PID 및 MPC 기반 데이터셋 생성 |
| **4/28 ~ 5/2**  | IQL 모델 Offline 학습 및 시뮬레이터 성능 평가 (1차) |
| **5/7 ~ 5/16**  | 실제 TCLab에서 Online Roll-out 및 성능 평가 (2차) |
