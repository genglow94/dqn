import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from q_network import QNet
from replay_buffer import ReplayBuffer
import random

# --- 環境作成 ---
env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=1000)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# --- ネットワーク & ターゲットネット ---
q_net = QNet(obs_dim, n_actions)
target_net = QNet(obs_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=2e-3)
buffer = ReplayBuffer(10000)

# --- ハイパーパラメータ ---
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
target_update_freq = 10
num_episodes = 1000

# --- 学習ループ ---
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy で行動選択
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state_tensor)
            action = q_values.argmax().item()

        # 環境を1ステップ進める
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 報酬シェイピング（右方向に進むと少し報酬追加）
        shaped_reward = reward + (next_state[0] + 0.5)

        # バッファに保存
        buffer.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward

        # --- Qネットワーク更新 ---
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

            # Q(s, a)
            q_values = q_net(states_tensor).gather(1, actions_tensor)
            # max_a' Q'(s', a')
            with torch.no_grad():
                target_q_values = target_net(next_states_tensor).max(1, keepdim=True)[0]
                target = rewards_tensor + gamma * target_q_values * (1 - dones_tensor)

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- エピソード終了後 ---
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode+1}: Total reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

env.close()
