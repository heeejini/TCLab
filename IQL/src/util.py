import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn
from .eval_policy import simulator_policy, tclab_policy

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    """
    여러 배치를 concat해서 한 번에 처리한 후, 원래 모양대로 다시 split해주는 함수
    주로 critic_target, actor_target처럼 batched 추론할 때 사용
    """
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    # polyak averaging , 안정적인 학습을 위해서 
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps): # 리턴 범위 계산 : 보상 정규화 용
    #주어진 데이터셋 (rewards, terminals) 로 부터 에피소드 별 return range 계산 
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns) # min, max 로 추출해서 보상 범위 확인 


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size): 
    #offline rl 에서는 replay buffer 대신 고정된 dataset 을 사용하므로, 전체 dataset 에서 batch_size 만큼 무작위 샘플링해서 mini-batch 구성
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}


# def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
#     # 학습된 정책을 실제로 평가해서 총 reward 를 측정, offline 학습 이후에 policy의 성능을 평가할 때 사용
#     obs = env.reset()
#     total_reward = 0.
#     for _ in range(max_episode_steps):
#         with torch.no_grad():
#             action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
#         next_obs, reward, done, info = env.step(action)
#         total_reward += reward
#         if done:
#             break
#         else:
#             obs = next_obs
#     return total_reward
####
# simluator 랑 tclab 실제에서의 evaluate_policy 가 필요할 것 



def evaluate_policy_sim(policy, args):

    return simulator_policy(
        policy=policy,
        total_time_sec=args.max_episode_steps,
        dt=args.sample_interval if hasattr(args, 'sample_interval') else 5.0,
        log_root="./eval_sim_logs",
        seed=args.seed,
        ambient=29.0,
        deterministic=args.deterministic_policy,
        scaler=args.scaler 
    )

def evaluate_policy_tclab(policy, args):
    return tclab_policy(
        policy=policy,
        total_time_sec=args.max_episode_steps,
        dt=args.sample_interval if hasattr(args, 'sample_interval') else 5.0,
        log_root="./eval_real_logs",
        seed=args.seed,
        ambient=29.0,
        deterministic=args.deterministic_policy, 
        scaler=args.scaler 
    )



def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        # ndarray나 list 값을 float로 변환
        for k, v in dict.items():
            if isinstance(v, (np.ndarray, list)):
                dict[k] = float(np.mean(v))

        # CSV writer 초기화
        if self.csv_file is None:
            self.fieldnames = sorted(dict.keys())  # 알파벳 순으로 고정
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        # 누락된 key는 빈칸으로 채움 (dict가 fieldnames를 항상 포함하도록)
        row_data = {key: dict.get(key, "") for key in self.fieldnames}

        self(str(row_data))
        self.csv_writer.writerow(row_data)
        if self.flush:
            self.csv_file.flush()



    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()