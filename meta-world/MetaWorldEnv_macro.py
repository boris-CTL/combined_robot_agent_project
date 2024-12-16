import gymnasium as gym
import numpy as np
import metaworld
import random

import random
import time
import importlib
from gains import P_GAINS
from metaworld.policies.policy import move
from metaworld.policies.action import Action
from metaworld.policies import SawyerDrawerOpenV1Policy, SawyerDrawerOpenV2Policy, SawyerReachV2Policy
import numpy as np

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
            env = mt1.train_classes[env_name]()
            # choose n tasks
            tasks = random.choices(mt1.train_tasks, k=num_tasks)
            for task in tasks:
                env.set_task(task)
                self.training_envs.append(env)
        self.current_task = 0
        self.env = self.training_envs[self.current_task]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env_name = env_name
        print(f"MetaWorldMultiTaskEnv initialized with {num_tasks} tasks")

        # step() may cause ValueError raising, therefore we need to save the following
        self.prev_reward = None
        self.prev_terminated = None
        self.prev_info = None


        # load the scripted policy
        if env_name=='peg-insert-side-v2':
            module = importlib.import_module(f"metaworld.policies.sawyer_peg_insertion_side_v2_policy")
            self._policy = getattr(module, f"SawyerPegInsertionSideV2Policy")()
        else:
            module = importlib.import_module(f"metaworld.policies.sawyer_{env_name.replace('-','_')}_policy")
            # print(module)
            self._policy = getattr(module, f"Sawyer{env_name.title().replace('-','')}Policy")()
            # print(self._policy)
        self.p_control_time_out = 20 # timeout of the position tracking (for convergnece of P controller)
        self.p_control_threshold = 1e-4 # the threshold for declaring goal reaching (for convergnece of P controller)
        self._current_observation = None



    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation

    @property
    def _current_pos(self):
        """ Curret position of the hand. """
        return self._policy._parse_obs(self.current_observation)['hand_pos']

    @property
    def mw_policy(self):
        return self._policy


    def p_control(self, action):
        """ Compute the desired control based on a position target (action[:3])
        using P controller provided in Metaworld."""
        assert len(action)==4
        p_gain = P_GAINS[type(self.mw_policy)]
        if type(self.mw_policy) in [type(SawyerDrawerOpenV1Policy), type(SawyerDrawerOpenV2Policy)]:
            # This needs special cares. It's implemented differently.
            o_d = self.mw_policy._parse_obs(self.current_observation)
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                p_gain = 4.0
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                p_gain = 4.0
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                p_gain= 50.0

        control = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        control["delta_pos"] = move(self._current_pos, to_xyz=action[:3], p=p_gain)
        control["grab_effort"] = action[3]
        return control.array    


    # 要配合current_observation的初始化
    # def reset(self, seed=None):
    #     self._current_observation, info = self.env.reset(seed=seed)
    #     return self._current_observation, info    

    
    def reset(self, seed=None):
        self.current_task = (self.current_task + 1) % len(self.training_envs)
        self.env = self.training_envs[self.current_task]
        return self.env.reset(seed=seed)

    
    def pure_reset(self, seed=None):
        self._current_observation, info = self.env.reset(seed=seed)
        return self._current_observation, info

    
    def reverse_pos_from_action(self, action, p_gain):
        # need to call reset() earlier, otherwiese self.current_observation will return None
        from_xyz = self._policy._parse_obs(self.current_observation)['hand_pos']
        assert type(action) == np.ndarray
        scaled_action = action[:3] / p_gain
        to_xyz = list(scaled_action + from_xyz)
        to_xyz.extend([action[-1]])
        return to_xyz

    
    # def step(self, action):
    #     return self.env.step(action)

    # here, we have to "reverse" the conceived [x, y, z, gripper_state]
    # called `desired_pos_plus_gripper`
    def step(self, action):
        try:
            p_gain = P_GAINS[type(self.mw_policy)]
            desired_pos_plus_gripper = self.reverse_pos_from_action(np.array(action), p_gain)
            assert type(desired_pos_plus_gripper) == list
            assert len(desired_pos_plus_gripper) == 4
            
            previous_pos = self._current_pos  # the position of the hand before moving
            
            # traj_list_micro = []
            # traj_list_macro = []
            
            for i in range(self.p_control_time_out):
                control = self.p_control(desired_pos_plus_gripper)
                observation, reward, terminated, truncated, info = self.env.step(control)
                # self.env.render()
                
                # if (i == 0):
                #     traj_list_macro.append(control)
                # if (i == self.p_control_time_out - 1):
                #     traj_list_macro.append(reward)
                #     traj_list_macro.append(observation)
                    
                # traj_list_micro.append(control)
                # traj_list_micro.append(reward)
                # traj_list_micro.append(observation)
                

                # to cache values
                self._current_observation = observation
                self.prev_reward = reward
                self.prev_terminated = terminated
                self.prev_info = info
                
                desired_pos = desired_pos_plus_gripper[:3]
                if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                    break
                # if ((reward == 10) or (bool(info['success']) == True)):
                #     # 表示這時候雖然可能不是(i == self.p_control_time_out - 1)，但macro還是要存
                #     traj_list_macro.append(reward)
                #     traj_list_macro.append(observation)
                #     break
    
            terminated = terminated or info.get('success')
            info['success'] = bool(info['success'])
            # info['traj_list_micro'] = traj_list_micro
            # info['traj_list_macro'] = traj_list_macro
            return observation, reward, terminated, truncated, info
        except ValueError:
            # if raising `ValueError("You must reset the env manually once truncate==True")`,
            # cannot execute self.env.step(control) 20 times fully.
            print("Entering except block. ValueError seems to be raised.")
            truncated = True
            # other than `truncated`, return previously cached values
            return self._current_observation, self.prev_reward, self.prev_terminated, truncated, self.prev_info
            

    
    def render(self, mode='None'):
        if mode != 'None':
            self.env.render_mode = mode
        return self.env.render()
    
    def seed(self, seed):
        self.env.seed(seed)
