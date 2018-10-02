"""
http://neuro-educator.com/rl2/
DQN  をgymで
いいところは入力データを離散化させる必要がないこと

ポイント
experience replay
逐次学習せずに{s(t), a(t), r(t), s(t+1)}をたくさんメモリに保持しておいて、あとでランダムに学習する

fixed target Q-network
学習時に少し前に固定してあったQネットワークを使用する
(experience replay)を行えば自然に実現される

報酬のcliping
各ステップの報酬を-1, 0, 1に固定してやる

誤差関数にhuber関数を利用

"""

import gym
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers
from keras import backend as K
import tensorflow as tf


# 損失関数
# おそらく損失関数などといったkerasの内部を弄るにはバックエンド(tensorflow)が必要
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    # bool型のtensorを返す
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みを学習している部分
    # ミニバッチの2次元配列をその都度毎回放り込んでいる
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))

        # memoryからbatch_sizeの分ランダムに取り込んでいる部分
        # memoryには{s(t), a(t), r(t), s(t+1)}がたくさん詰まっている
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            # inputsのi番目を変えている
            inputs[i:i+1] = state_b
            target = reward_b

            # 教師信号の使い分け
            # 次の状態が終了状態でなければ教師信号を以下のようにしてやる
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # ？model.predictの出力は二次元の配列と思われる
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma *targetQN.model.predict(next_state_b)[0][next_action]

            # 次の状態が終了状態ならばrewardをそのまま教師信号にしてやる
            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target
            self.model.fit(inputs, targets, epochs=1, verbose=0)


class Memory:
    def __init__(self, max_size=1000):
        # 配列の強いやつ
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # 0~len(buffer)(1000) の中から重複せずに数を選びbatch_sizeの長さ分の配列を生成
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        # バッファの中から生成したインデックスのデータを抜き出し配列に保存、返す
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


class Actor:
    def get_action(self, state, episode, targetQN):
        epsilon = 0.001 + 0.9/(1.0 + episode)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = targetQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)
        else:
            action = np.random.choice([0, 1])

        return action

DQN_MODE = 1
LENDER_MODE = 1

env = gym.make('CartPole-v1')
num_episodes = 299
max_num_of_steps = 200
goal_average_reward = 195
num_consecutive_iteration = 10
total_reward_vec = np.zeros(num_consecutive_iteration)
gamma = 0.99
islearned = 0
isrender = 0
hidden_size = 16
learning_rate = 0.00001
memory_size = 10000
batch_size = 32

mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
memory = Memory(max_size=memory_size)
actor = Actor()

for episode in range(num_episodes):
    # 初期化
    env.reset()
    # 1stepは適当な行動を取る
    state, reward, done, _ = env.step(env.action_space.sample())
    # 行列変換
    state = np.reshape(state, [1, 4])
    episode_reward = 0

    # targetQNをエピソード学習前のmainQNと同じにする
    targetQN = mainQN

    for t in range(max_num_of_steps + 1):
        if islearned and LENDER_MODE:
            env.render()
            time.sleep(0.1)
            print(state[0, 0])

        action = actor.get_action(state, episode, mainQN)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        if done:
            next_state = np.zeros(state.shape)
            if t < 195:
                reward = -1
            else:
                reward = 1
        else:
            reward = 0

        episode_reward += 1

        memory.add((state, action, reward, next_state))
        state = next_state

        # DQNは1step前のネットワークを重み学習に使用
        # DDQNは1episode前のネットワークを重み学習に使用
        if memory.len() > batch_size and not islearned:
            mainQN.replay(memory, batch_size, gamma, targetQN)

        # DQNなら
        if DQN_MODE:
            targetQN = mainQN

        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))
            print("{0:d} episode finished after {1:.4f} time steps / mean {2:.4f}".format(episode, t + 1, total_reward_vec.mean()))
            break

    if total_reward_vec.mean() >= goal_average_reward:
       print("{0:d} episode train agent successfuly".format(episode))
       islearned = 1
       if isrender == 0:
           isrender = 1





