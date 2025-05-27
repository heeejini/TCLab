# 🔥 TCLab Project
TCLab 키트로 수집한 MPC 데이터를 기반으로 IQL 모델을 학습하고, 이를 활용하여 실제 환경에서도 적절한 온도 제어 RL 모델 구현하는 것이 본 실험의 목적


---

## 📌 What is TCLab?
[TCLab (Temperature Control Lab)](https://apmonitor.com/pdc/index.php/Main/ArduinoTemperatureControl)는 BYU(Brigham Young University)에서 제작한 실험 키트로, 2개의 히터(Q1, Q2)와 2개의 온도 센서(T1, T2)를 통해 MIMO 제어 시스템을 학습하고 테스트할 수 있는 물리적 장치이다.

---

## 🧪 Project Overview

### Step 1. **MPC 기반 데이터 수집**
- **수집 방식**: 모델 예측 제어(MPC)를 활용해 매 step마다 히터 출력을 계산
- **입력 상태**: `[T1, T2, TSP1, TSP2]`
- **출력 행동**: `[Q1, Q2]`
- **보상 정의**: `-√((T1-TSP1)^2 + (T2-TSP2)^2)`  
- **제어 주기 (dt)**: 5초  (MPC 를 APMonitor를 통해서 계산하는데, 이 때 한번의 연산에 5초가 걸림)
- **총 실험 시간**: 1200초 (20분)
- **총 데이터 포인트**: 240개 per episode

### Step 2. **Offline RL 학습 (IQL)**
- 수집된 MPC 데이터를 기반으로 Implicit Q-Learning 학습
- 이 때, IQL 모델은 https://github.com/gwthomas/IQL-PyTorch.git 구현을 사용함 

### Step 3. **Online 튜닝 (IQL)**
- Offline 학습한 IQL 모델을 통해서 Online fine-tuning 진행 
- 이 때, 탐색 실험은 시뮬레이터를 통해서 실험을 진행
- 최적의 방법론을 시뮬레이터를 통해서 찾은 뒤, 실제 키트와 시뮬레이터에서 각각 Online 튜닝 진행 

### Step 4. **평가**
- 평가 메트릭으로 설정한 두 가지 지표를 통해서 성능 평가 
- Average total return : 평가용 에피소드 3개에서 얻은 총 리턴의 평균
- Average total error : 평가용 에피소드 3개에서 계산된 총 오차 (E1 + E2) 의 평균



  ![image](https://github.com/user-attachments/assets/cef62cc8-6da4-48af-b92e-0528ad8f6c9e)


---
## 결과 
- Inference 시에 3개의 Inference 용 에피소드로 성능 측정 
- Online tuning >  MPC > Offline training 순의 결과 
<div align="center">
  <img src="https://github.com/user-attachments/assets/078fe0b8-600b-464e-b33e-74c187296045" width="400"/>
</div>


- Online tuning 으로 환경에 대한 데이터를 직접 수집하여, 환경에 대해 더욱 잘 적응한 모델을 구축 


---
## 🗓️ 프로젝트 일정

| 기간 | 내용 |
|------|------|
| **4/21 ~ 4/25** | MPC 기반 데이터셋 생성 |
| **4/28 ~ 5/2**  | IQL 모델 Offline 학습 |
| **5/7 ~ 5/16**  | IQL 모델 Online 튜닝  |
