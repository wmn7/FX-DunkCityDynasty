'''
@Author: WANG Maonan
@Date: 2023-10-29 14:29:38
@Description: Teat Wrapper for ray
@LastEditTime: 2023-10-29 14:35:09
'''
import math
import numpy as np

from DunkCityDynasty.env.gym_env import GymEnv
from PlayerAgent.env_wrapper.ray_wrapper import RayWrapper
from PlayerAgent.utils.read_config import read_config

class RandomAgent():
    def __init__(self):
        pass

    def get_actions(self, states):
        action = {}
        for agent_id, agent_info in states.items():
            action[agent_id] = 0
        return action


def main() -> None:
    # env config
    config = read_config()
    
    wrapper = RayWrapper()
    env = GymEnv(config, wrapper=wrapper)
    agent = RandomAgent()
    user_name = config['user_name']
    states, infos = env.reset(user_name=user_name, render=True)
    print(states.keys())
    # 首先利用 state 初始化不同的 player
    while True:
        actions = agent.get_actions(states)
        states, rewards, dones, truncated, infos = env.step(actions)
        print(actions)
        if dones['__all__']:
            break

if __name__ == '__main__':
    main()