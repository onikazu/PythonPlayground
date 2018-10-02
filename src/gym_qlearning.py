"""
http://neuro-educator.com/rl1/
https://book.mynavi.jp/manatee/detail/id=88997
gymでqlearningを行う

"""

# ライブラリインポート
import gym
from gym import wrappers
import numpy as np
import time


# [1]観測の離散化
def bins(clip_min, clip_max, num):
    # 特定範囲を一定間隔で区切りarrayを生成する。[1:-1]は最初から2番めから最後から2番目の意味
    # すなわち分けるときの区切りをarrayで返してくれる
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation

    # np.digitizeはbins（array）の何番目の範囲に第一引数が入るかを算出する
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_digitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_digitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_digitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_digitized))
    ]

    # (numdigitized)進数で変換している
    return sum(x * (num_digitized ** i) for i, x in enumerate(digitized))


# [2]actionを求める(右or左) ε-greedy法
def get_action(next_state, episode):
    # エピソードが進むごとにεは小さくなる
    epsilon = 0.5 * (1 / (episode + 1))

    # uniform で第一引数から第二匹数までの範囲の一様乱数を生成
    if epsilon <= np.random.uniform(0, 1):
        # 最大要素のインデックスを返す
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])

    return next_action


def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5

    # 次の状態のQ値の最大を算出
    next_Max = max(q_table[next_state][0], q_table[next_state][1])
    # ?
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * next_Max)

    return q_table


env = gym.make("CartPole-v0")
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 2000
goal_average_reward = 195
num_digitized = 6
q_table = np.random.uniform(-1, 1, size=(num_digitized ** 4, env.action_space.n))
total_reward_vec = np.zeros(num_consecutive_iterations)
final_x = np.zeros((num_episodes, 1))  # 学習後、各試行のt=200でのｘの位置を格納
islearned = 0
isrender = 0

for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):
        if islearned:
            env.render()
            time.sleep(0.1)
            # カートのx座標を出力
            print(observation[0])

        observation, reward, done, info = env.step(action)

        # 終了時の処理
        if done:
            if t < 195:
                reward = -100
            else:
                reward = 1

        else:
            reward = 1

        episode_reward += reward

        # 離散状態求めて、Q値を更新
        next_state = digitize_state(observation)
        q_table = update_Qtable(q_table, state, action, reward, next_state)

        action = get_action(next_state, episode)

        state = next_state

        # 終了時の処理
        if done:
            print("{0:d} Episode finished after {1:.4f} time steps /mean {2:.4f}".format(episode, t + 1,
                                                                                         total_reward_vec.mean()))
            # np.hstack は水平方向の連結
            # 先頭の要素を抜き取り後ろに総報酬を付け加える
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))
            if islearned:
                final_x[episode, 0] = observation[0]
            break

    if total_reward_vec.mean() >= goal_average_reward:
        print("success at {0:d} episode".format(episode))
        islearned = 1
        if isrender == 0:
            isrender = 1

if islearned == 1:
    np.savetxt("final_x.csv", final_x, delimiter=",")
