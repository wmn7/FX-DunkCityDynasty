'''
@Author: WANG Maonan
@Date: 2023-10-22 23:01:05
@Description: 游戏环境测试
@LastEditTime: 2023-10-25 20:18:56
'''
import math
import numpy as np

from DunkCityDynasty.env.gym_env import GymEnv
from PlayerAgent.env_wrapper.rule_wrapper import RuleWrapper
from PlayerAgent.utils.read_config import read_config

class RandomAgent():
    def __init__(self):
        self.player2actoin = {
            2: 12, # shoot
            9: 16,
            1: 22, # james' shoot
            82: 12, # shoot
            3: 26, # catch & shoot
        } # 不同球员 id 对应的动作

    # ###########################
    # ball status = 2, ball free
    # ###########################
    def calculate_angle(self, x1, z1, x2, z2):
        dx = x2 - x1
        dz = z2 - z1
        angle = math.atan2(dz, dx) * 180 / math.pi
        return angle

    def move_to_ball(self, ball_position_x, ball_position_y, position_x, position_y):
        angle = self.calculate_angle(position_x, position_y, ball_position_x, ball_position_y)
        if angle >= -22.5 and angle < 22.5:
            return 4  # Move: 0
        elif angle >= 22.5 and angle < 67.5:
            return 5  # Move: 135
        elif angle >= 67.5 and angle < 112.5:
            return 1  # Move: 90
        elif angle >= 112.5 and angle < 157.5:
            return 6  # Move: 45
        elif angle >= 157.5 or angle < -157.5:
            return 8  # Move: 315
        elif angle >= -157.5 and angle < -112.5:
            return 7  # Move: 225
        elif angle >= -112.5 and angle < -67.5:
            return 3  # Move: 180
        elif angle >= -67.5 and angle < -22.5:
            return 2  # Move: 270
        else:
            return 0  # Noop
    
    # ##############
    # Choose Action
    # ##############
    def choose_action(self, global_state, self_state):
        ball_status = global_state["ball_status"]

        ball_position_x = global_state["ball_position_x"]
        ball_position_y = global_state["ball_position_y"]
        position_x = self_state['position_x']
        position_y = self_state['position_y']

        # 如果持球
        if ball_status == 4:
            return np.random.randint(0,8)
        else:
            action_index = self.move_to_ball(
                ball_position_x=ball_position_x, 
                ball_position_y=ball_position_y, 
                position_x=position_x, 
                position_y=position_y
            )
            return action_index

    def get_actions(self, states):
        action = {}
        for agent_id, agent_info in states.items():
            global_state = agent_info['global_state']
            self_state = agent_info['self_state']
            new_action = self.choose_action(global_state=global_state, self_state=self_state)
            action[agent_id] = new_action
        return action


def main() -> None:
    # env config
    config = read_config()
    
    wrapper = RuleWrapper(config={})
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