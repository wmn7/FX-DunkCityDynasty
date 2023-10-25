'''
@Author: WANG Maonan
@Date: 2023-10-25 19:43:18
@Description: Base Wrapper, only for testing the env
@LastEditTime: 2023-10-25 19:58:00
'''
import numpy as np

class RuleWrapper():
    def __init__(self, config):
        self.config = config
        self.observation_space = None
        self.action_space = None

    def states_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states[key] = state_infos[key][1]
            action_mask = np.array(state_infos[key][-1])
            infos[key]['action_mask'] = action_mask
        return states, infos

    def rewards_wrapper(self, states):
        rewards = {}
        for key in states.keys():
            rewards[key] = 0
        return rewards