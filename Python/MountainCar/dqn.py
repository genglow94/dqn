import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# -----------------------------
# 1. 環境作成
# -----------------------------
env = gym.make("MountainCar-v0", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# -----------------------------
# 2. Qネットワーク
# -----------------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.fc(x)

q_net = QNet()
target_net = QNet()
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# 3. リプレイバッファ
# -----------------------------
replay_buffer = deque(maxlen=10000)

# -----------------------------
# 4. ハイパーパラメータ
# -----------------------------
gamma = 0.99
batch_size = 64
epsilon = 0.1
epsilon_decay = 0.995
min_epsilon = 0.01
target_update = 10  # エピソードごとにターゲットネットワーク更新

# -----------------------------
# 5. 学習ループ
# -----------------------------
n_episodes = 300  # 学習回数

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(obs_tensor)
            action = torch.argmax(q_values).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # -----------------------------
        # 報酬シェイピング
        # -----------------------------
        # next_obs[0] = 車の位置（-1.2 ~ 0.6）
        shaped_reward = reward + (next_obs[0] + 1.2)  # 左端-1.2を0に補正
        total_reward += shaped_reward

        # リプレイバッファに保存
        replay_buffer.append((obs, action, shaped_reward, next_obs, done))
        obs = next_obs

        # -----------------------------
        # バッチ学習
        # -----------------------------
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target = rewards + gamma * next_q * (1 - dones)
            loss = criterion(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # -----------------------------
    # ターゲットネットワーク更新
    # -----------------------------
    if (episode + 1) % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())

    # ε 減衰
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode+1}: Total shaped reward = {total_reward:.2f}, epsilon = {epsilon:.3f}")

env.close()
