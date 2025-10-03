import gymnasium as gym

# 環境を作成
env = gym.make("MountainCar-v0", render_mode="human")

# 初期化
obs, info = env.reset()

for _ in range(1000):
    # ランダムに行動を選ぶ（0:左, 1:止まる, 2:右）
    action = env.action_space.sample()

    # 環境に行動を適用
    obs, reward, terminated, truncated, info = env.step(action)

    # エピソード終了判定
    if terminated or truncated:
        obs, info = env.reset()

env.close()
