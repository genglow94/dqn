import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from q_network import QNet
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
import matplotlib.pyplot as plt
import time

# 時間計測開始
start_time = time.time()

# 環境作成（CartPole）
env = gym.make("CartPole-v1", render_mode=None)  # 学習中はNone、可視化したいときは"human"
obs_dim = env.observation_space.shape[0]  # 状態数（4）
n_actions = env.action_space.n            # 行動数（2）

# Qネットワーク作成
q_net = QNet(obs_dim, n_actions)
target_net = QNet(obs_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())  # 同期

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = PrioritizedReplayBuffer(50000)

# パラメータ設定
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
target_update_freq = 10
num_episodes = 1000
alpha = 0.6
beta = 0.4
beta_increment_per_episode = 0.001

# 学習曲線
episode_rewards = []

# 学習ループ
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy方策
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()

        # 環境1ステップ
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 経験を保存
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 学習ステップ
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size, beta)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1)

            # Q(s, a)
            q_values = q_net(states_tensor).gather(1, actions_tensor)

            # Double DQNターゲット
            with torch.no_grad():
                next_actions = q_net(next_states_tensor).argmax(1, keepdim=True)
                next_q_values = target_net(next_states_tensor).gather(1, next_actions)
                target = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

            td_errors = (q_values - target).detach().cpu().numpy().squeeze()
            buffer.update_priorities(indices, td_errors)

            # 損失 (IS重み付き)
            loss = (weights_tensor * (q_values - target).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

    # ε と β 更新
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    beta = min(1.0, beta + beta_increment_per_episode)

    # ターゲットネット更新
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    # 記録
    episode_rewards.append(total_reward)
    avg100 = np.mean(episode_rewards[-100:])

    print(f"Episode {episode+1:4d} | Reward: {total_reward:5.1f} | Avg100: {avg100:6.2f} | Epsilon: {epsilon:.3f}")

env.close()

# 学習時間表示
end_time = time.time()
print(f"\n総学習時間: {end_time - start_time:.2f} 秒")

# 学習曲線の描画
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Episode reward")
if len(episode_rewards) >= 100:
    avg100_list = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
    plt.plot(avg100_list, label="Avg100")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("CartPole-v1 DQN Learning Curve")
plt.legend()
plt.grid()
plt.show()
