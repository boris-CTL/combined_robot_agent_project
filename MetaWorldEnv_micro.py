import gymnasium as gym
import numpy as np
import metaworld
import random

class MetaWorldMultiTaskEnv(gym.Env):
    def __init__(self, env_name, num_tasks, seed=None):
        super().__init__()
        self.training_envs = []
        if env_name == "mt10":
            assert num_tasks % 10 == 0, "Number of tasks must be a multiple of 10 for MetaWorld MT10"
            assert num_tasks <= 500, "Number of tasks must be less than or equal to 500 for MetaWorld MT10"
            mt10 = metaworld.MT10(seed=seed)
            for name, env_cls in mt10.train_classes.items():
                env = env_cls()
                tasks = random.choices([task for task in mt10.train_tasks
                                        if task.env_name == name], k=num_tasks//10)
                for task in tasks:
                    env.set_task(task)
                    self.training_envs.append(env)
                # shuffle the tasks
                random.shuffle(self.training_envs)
        else:
            mt1 = metaworld.MT1(env_name, seed=seed) 
            # choose n tasks
            tasks = random.sample(mt1.train_tasks, k=num_tasks)
            num_unique_tasks = len(set(tasks))
            print(f'select {num_unique_tasks} tasks from {len(mt1.train_tasks)} tasks')
            assert num_unique_tasks == num_tasks, "Number of tasks must be equal to the number of unique tasks"
            for task in tasks:
                env = mt1.train_classes[env_name]()
                env.set_task(task)
                self.training_envs.append(env)
        self.current_task = 0
        self.env = self.training_envs[self.current_task]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print(f"MetaWorldMultiTaskEnv initialized with {num_tasks} tasks")
        # print(f"Training Environments: {self.training_envs}")

    def reset(self, seed=None):
        self.current_task = (self.current_task + 1) % len(self.training_envs)
        self.env = self.training_envs[self.current_task]
        return self.env.reset(seed=seed)

    def step(self, action):
        # observation, reward, terminated, truncated, info = self.env.step(action)
        # The episode is terminated if the task is successful
        # terminated = terminated or info.get('success') 
        # return observation, reward, terminated, truncated, info
        return self.env.step(action)

    def render(self, mode='None'):
        if mode != 'None':
            self.env.render_mode = mode
        return self.env.mujoco_renderer.render(render_mode=self.env.render_mode, camera_id=2) # 2 or 4
    
    def seed(self, seed):
        self.env.seed(seed)
