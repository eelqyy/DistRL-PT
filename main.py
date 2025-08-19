import numpy as np
import pandas as pd
from qrdqn_agent import QR_DQN_Agent
from env_case import Case27


if __name__ == '__main__':
    case = Case27()
    agents = QR_DQN_Agent(case)

    episode_num = 30000
    epsilon = 0.9

    episode = 0
    Reward = np.zeros(episode_num+1)
    state_all = np.eye(case.T)
    Action = np.zeros((episode_num+1, case.T))
    while episode <= episode_num:
        state = state_all[0]
        total_reward = 0
        episode_record = [[] for i in range(5)]
        for t in range(case.T):
            action = agents.act(state, epsilon)
            episode_record[0].append(state)
            episode_record[1].append(action)
            if t < case.T-1:
                next_state = state_all[t+1]
                episode_record[4].append(False)
            else:
                next_state = state_all[0]
                episode_record[4].append(True)
            episode_record[3].append(next_state)
            state = next_state
        action_all = episode_record[1]
        reward_all = case.step_forward(action_all)[0]
        episode_record[2] = reward_all
        for t in range(case.T):
            agents.step(episode_record[0][t], episode_record[1][t], episode_record[2][t],
                        episode_record[3][t], episode_record[4][t])
        Reward[episode] = np.sum(reward_all)
        Action[episode] = action_all
        if episode % 100 == 0 and episode != 0:
            epsilon = epsilon * 0.98
            agents.lr_decay()
            print(f"Episode: {episode}")
        episode += 1
