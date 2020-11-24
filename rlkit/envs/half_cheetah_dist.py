import numpy as np
import d4rl

from .half_cheetah import HalfCheetahEnv
from . import register_env
import gym
import d4rl

@register_env('cheetah-dist')
class HalfCheetahDistEnv(HalfCheetahEnv):
    """
    The tasks are generated by sampling target distributions from a
    Bernoulli distribution on {-1, 1} with parameter 0.5
    (-1: mixed(medium-replay), 1: medium-expert}
    """
    def __init__(self, task={}, n_tasks=2, randomize_tasks=False):
        distributions = [-1, 1]
        self.tasks = [{'distributions': distribution} for distribution in distributions]
        self._task = task
        self._goal_dir = task.get('distributions', 1)  # just 1
        self._goal = self._goal_dir

        self.env_cheetah_0 = gym.make('halfcheetah-medium-replay-v0')
        self.env_cheetah_1 = gym.make('halfcheetah-medium-expert-v0')
        self.d0 = self.env_cheetah_0.get_dataset()
        self.d1 = self.env_cheetah_1.get_dataset()
        super(HalfCheetahDistEnv, self).__init__()

    def sample_tasks(self, num_tasks):
        distributions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'distributions': distribution} for distribution in distributions]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_dir = self._task['distributions']
        self._goal = self._goal_dir
        self.reset()