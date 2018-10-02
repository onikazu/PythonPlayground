"""
https://www.jonki.net/entry/2016/05/05/174519
6の部屋から指定の部屋に移動するタスク
ε-greedyは取り入れていない。
（runGreedy というのはスタート位置をあらゆるところに(greedyに)設定しテストしている関数
"""


import numpy as np
import random
import sys


# Reward matrix
R = np.array([
[-1, -1, -1, -1,  0,  -1],
[-1, -1, -1,  0, -1, 100],
[-1, -1, -1,  0, -1,  -1],
[-1,  0,  0, -1,  0,  -1],
[ 0, -1, -1,  0, -1, 100],
[-1,  0, -1, -1,  0, 100]
])

# Initial Q-value
Q = np.zeros((6, 6))

LEARNING_COUNT = 1000
GAMMA = 0.8
GOAL_STATE = 5


class QLearning:
    def __init__(self):
        return

    def learn(self):
        # set a state randomly
        print("start learning")
        state = self._getRandomState()
        for i in range(LEARNING_COUNT):
            # extract possible actions in state
            possible_actions = self._getPossibleActionsFromState(state)

            # choose action from possible actions randomly
            action = random.choice(possible_actions)

            # update Q-Value
            next_state = action
            next_possible_actions = self._getPossibleActionsFromState(next_state)
            max_Q_next_s_a = self._getMaxQvalueFromStateAndPossibleActions(next_state, next_possible_actions)
            Q[state][action] = R[state][action] + GAMMA * max_Q_next_s_a

            state = next_state

            if state == GOAL_STATE:
                state = self._getRandomState()




    # アンダーバーは外部からの参照を受けない関数であることを明示することが多いらしい
    def _getRandomState(self):
        # generate random integer whose range is between a and b(0 <= int <= 5)
        return random.randint(0, R.shape[0] - 1)

    def _getPossibleActionsFromState(self, state):
        """
        :param state: int 0~5までの行番号
        :return:
        """
        # inspect strange state
        if state < 0 or state >= R.shape[0]: sys.exit("invalid state: {0:s}".format(state))
        # np.where: 条件に合う要素のインデックスを抽出する(型はnd array)
        # 今いる部屋(state)から移動できるのはどこか探している
        # [0]としているのはnp.whereの仕様上の関係（型がnd array）。意味的にはほとんど無視してOK
        return list(np.where(np.array(R[state] != -1)))[0]

    def _getMaxQvalueFromStateAndPossibleActions(self, state, possible_actions):
        return max([Q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        print(Q.astype(int)) # convert float to int for readability

    def runGreedy(self, start_state):
        print("======start======")
        state = start_state
        while state != GOAL_STATE:
            print("current state: {0:d}".format(state))
            possible_actions = self._getPossibleActionsFromState(state)

            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:
                if Q[state][a] > max_Q:
                    best_action_candidates = [a]
                    max_Q = Q[state][a]
                elif Q[state][a] == max_Q:
                    best_action_candidates.append(a)

            best_action = random.choice(best_action_candidates)
            print(">choose action : {0:d}".format(best_action))
            state = best_action

        print("stete is {0:d}".format(state))
        print("======end======")


if __name__ == "__main__":
    QL = QLearning()
    QL.learn()

    QL.dumpQvalue()

    for s in range(Q.shape[0] - 1):
        QL.runGreedy(s)


