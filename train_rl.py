'''
@Author: WANG Maonan
@Date: 2023-11-01 21:05:16
@Description: Train for RL
@LastEditTime: 2023-11-01 21:09:10
'''
import sys,os
sys.path.append(os.getcwd())
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import override

from DunkCityDynasty.env.ray_env import RayEnv

from PlayerAgent.env_wrapper import ray_wrapper
from PlayerAgent.policy.policy import Network
from PlayerAgent.utils.read_config import read_config

if __name__ == '__main__':
    ray.shutdown()
    ray.init()
    wrapper = ray_wrapper({'concated_states':True})
    ModelCatalog.register_custom_model("my_model", Network)
    register_env("my_env", lambda config: RayEnv(config, wrapper))

    env_config = read_config()

    obs_space = wrapper.observation_space
    act_space = wrapper.action_space
    config = (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1,
            # rollout_fragment_length='auto',
            enable_connectors=False,
            recreate_failed_workers=True,
        )
        .environment(
            env = "my_env",
            env_config = env_config,
            observation_space = obs_space,
            action_space = act_space,
        )
        .training(
            model = {
                "custom_model": "my_model",
            },
            train_batch_size=2048,
            lr=0.0001,
        )
        .multi_agent(
            policies = {
                "default_policy": PolicySpec(observation_space=obs_space,action_space=act_space,config={}),
                "shared_policy": PolicySpec(observation_space=obs_space,action_space=act_space,config={}),
            },
            policy_mapping_fn= lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .debugging(
            log_level="INFO",
        )
    )
    algo = config.build()
    while True:
        algo.train()

    # you can use the following code to compute actions from raw states
    # import json
    # with open('states_example.json', 'r') as f:
    #     state_infos = json.load(f)
    # states,_ = wrapper.states_wrapper(state_infos)
    # print("action",algo.compute_actions(states))