"""
https://qiita.com/sugulu/items/7a14117bbd3d926eb1f2

gymでSARSAを用いて学習を行う
Qlearningと違うところはQの更新方法

Q: Qtableの更新（最大のQ値で）　その後ε-greedyで行動決定
そのため実際に行った行動と、更新に用いた行動が異なることがある

SARSA: ε-greedyで行動を決定してから、そのQ値で更新
行動を決定してから更新を行う。使用する行動は同一

"""

import gym
from gym import wrappers
import numpy as np
import time


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_digitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_digitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_digitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_digitized)),
    ]

    return sum([x * (num_digitized ** i) for i, x in enumerate(digitized)])


def get_actiion(next_state, episode):
    # εはどんどん小さくなっていく
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


def update_Qtable_sarsa(q_table, state, action, reward, next_state, next_action):
    gamma = 0.99
    alpha = 0.5
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])
    return q_table


env = gym.make("CartPole-v0")
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 2000
goal_average_reward = 195
num_digitized = 6
q_table = np.random.uniform(-1, 1, size=(num_digitized**4, env.action_space.n))
final_x = np.zeros((num_episodes, 1))
total_reward_vec = np.zeros(num_consecutive_iterations)
islearned = 0
isrender = 0


for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):
        if islearned == 1:
            env.render()
            time.sleep(0.1)
            print(observation[0])

        observation, reward, done, info = env.step(action)

        if done:
            if t < 195:
                reward = -200
            else:
                reward = 1
        else:
            reward = 1

        episode_reward += reward

        next_state = digitize_state(observation)

        next_action =get_actiion(next_state, episode)
        q_table = update_Qtable_sarsa(q_table, state, action, reward, next_state, next_action)

        action = next_action
        state = next_state

        if done:
            print("{0:d} episode finished after {1:.4f} time steps / mean {2:.4f}".format(episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))

            if islearned:
                final_x[episode, 0] = observation[0]
            break


    if total_reward_vec.mean() >= goal_average_reward:
        print("episode {0:d} train agent successfuly".format(episode))
        islearned = 1
        if isrender == 0:
            isrender = 1

if islearned:
    np.savetxt("final_x_sarsa.csv", final_x, delimiter=",")
