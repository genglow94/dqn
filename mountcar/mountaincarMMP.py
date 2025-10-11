import gymnasium as gym
import torch
import torch.optim as optim #学習の重みを更新するためのライブラリ
import torch.nn.functional as F
import numpy as np
from MMP import MaxMinQNet
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
import matplotlib.pyplot as plt
import time

#時間計測開始
start_time = time.time()

#環境作成
env =gym.make("MountainCar-v0",render_mode = None,max_episode_steps=200)
obs_dim = env.observation_space.shape[0] #space:連続値
n_actions = env.action_space.n #.n離散値

#MMPネットワーク，ターゲットネット
q_net = MaxMinQNet(obs_dim,n_actions)
target_net = MaxMinQNet(obs_dim,n_actions)
target_net.load_state_dict(q_net.state_dict())#q_netをコピー
optimizer = optim.Adam(q_net.parameters(),lr=5e-4) #重みを更新
#buffer = ReplayBuffer(20000)
buffer = PrioritizedReplayBuffer(20000)

#パラメータ
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01
target_update_freq = 5
num_episodes = 3000
alpha = 0.6   # 優先度の影響度
beta = 0.4    # IS重みの補正度
beta_increment_per_episode = 0.001

# 学習曲線用
episode_rewards = []
last_100_rewards = []

#学習ループ
for episode in range(num_episodes):
    state,info =env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()#ランダム生成
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)#状態を多次元配列に変換
            q_values = q_net(state_tensor) #各行動のq値を計算
            action = q_values.argmax().item()#q値が最大なものを選ぶ

        #環境を1step進める
        next_state,reward,terminated,truncated,info = env.step(action)
        done = terminated or truncated #terminated:終了条件で終わったか，truncated:最大ステップで終わったか

        #報酬シェイピング
        shaped_reward = reward + 0.1 * next_state[1]

        #バッファに保存
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        #Qネットワーク更新
        if len(buffer) >= batch_size:
            states,actions,rewards,next_states,dones, indices, weights = buffer.sample(batch_size,beta)
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)#unsqueezeは1次元のテンソルを列ベクトルに変換
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1)

            #Q(s,a)
            q_values = q_net(states_tensor).gather(1,actions_tensor)#選択した行動を取り出す
            #max_a' Q'(s',a')
            with torch.no_grad():#勾配を計算しない
              next_actions = q_net(next_states_tensor).argmax(1, keepdim=True)
              next_q_values = target_net(next_states_tensor).gather(1, next_actions)
              target = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
              #target_q_values = target_net(next_states_tensor).max(1,keepdim=True)[0]#.max(1,keepdim=True)[0]:各行の最大のq値
              #target = rewards_tensor +gamma*target_q_values*(1- dones_tensor)

            td_errors = (q_values - target).detach().cpu().numpy().squeeze()
            buffer.update_priorities(indices, td_errors)
            # IS重み付き損失
            loss = (weights_tensor * (q_values - target).pow(2)).mean()

            #loss = F.mse_loss(q_values,target)#平均二乗誤差
            optimizer.zero_grad()#前回までの勾配をリセット
            loss.backward()#勾配計算
            #torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)#勾配クリッピング
            optimizer.step()#勾配を使ってパラメータ更新

    # 制限付き min-plus 正規化を適用
    if episode % 10 == 0 and len(states) > 0:
        states_batch, _, _, _, _, _, _ = buffer.sample(batch_size, beta)
        D = torch.FloatTensor(states_batch)   # 制限集合 D
        f_func = lambda x: q_net.max1(q_net.fc_in(x))# f_func: MaxLinear 層の出力 
        g_func = lambda x: q_net.min1(f_func(x))# g_func: MinLinear 層の出力  
        q_net.min1.restricted_normalize(D, f_func, g_func)

    #エピソード終了後
    epsilon = max(min_epsilon,epsilon*epsilon_decay)#探索率を減らす
    if episode % target_update_freq == 0:#一定エピソードごとに実行
        target_net.load_state_dict(q_net.state_dict())#ターゲットネットワークに Q ネットワークの重みをコピー

    # 学習曲線用に報酬を記録
    episode_rewards.append(total_reward)  

    print(f"Episode {episode+1}: Total reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

env.close()

#学習時間表示
end_time = time.time()
print(f"\n総学習時間: {end_time - start_time:.2f} 秒")

#学習曲線の描画
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("MMP Learning Curve")
plt.legend()
plt.grid()
plt.show()
