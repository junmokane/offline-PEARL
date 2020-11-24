from rlkit.envs.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import gym
import d4rl

halfcheetah_task_name = ['halfcheetah-random-v0',
                         'halfcheetah-medium-v0',
                         'halfcheetah-expert-v0',
                         'halfcheetah-medium-replay-v0',
                         'halfcheetah-medium-expert-v0']

ant_task_name = ['ant-medium-expert-v0',
                 'ant-random-expert-v0',
                 'ant-medium-replay-v0',
                 'ant-medium-v0',
                 'ant-random-v0',
                 'ant-expert-v0',
                 ]

i = 2
env = gym.make(ant_task_name[i])
dataset = env.get_dataset()
print(dataset['observations'].shape) # An N x dim_observation Numpy array of observations (N = 1e6)
print(dataset['actions'].shape)
print(dataset['rewards'].shape)

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
#dataset = d4rl.qlearning_dataset(env)
#print(dataset)

_ = env.reset()
while True:
    env.render(mode='human')
    env.step(env.action_space.sample())