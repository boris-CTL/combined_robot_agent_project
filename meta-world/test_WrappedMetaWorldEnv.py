import gymnasium as gym
import numpy as np
import time
from MetaWorldEnv import MetaWorldMultiTaskEnv
from WrappedMetaWorldEnv import WrappedMetaWorldMultiTaskEnv


def parse_traj_list(global_traj_list_):
    all_state_list = []
    all_action_list = []
    all_reward_list = []
    for i in range(len(global_traj_list_)):
        if (i % 3 == 0):
            # it's a state
            all_state_list.append(global_traj_list_[i])
        elif (i % 3 == 1):
            # it's an action
            all_action_list.append(global_traj_list_[i])
        elif (i % 3 == 2):
            # it's a reward
            all_reward_list.append(global_traj_list_[i])
    return all_state_list, all_action_list, all_reward_list


# Initialize WrappedMetaWorldMultiTaskEnv
env_name = 'basketball-v2'
n_tasks = 50
seed = 111
env = MetaWorldMultiTaskEnv(env_name, n_tasks, seed=seed)
env.seed(seed)
wrapped_env = WrappedMetaWorldMultiTaskEnv(env, env_name, seed=seed)

obs, info = wrapped_env.reset()

# global_traj_list_micro is used to save every micro (s,a,r), i.e., it will save (s1, a1, r1, s2, a2, .., aN, rN, s(N+1))
global_traj_list_micro = [obs]

# let's say, if (s1, a1, r1, s2, a2, ..., a20, r20, s21) are produced for every WrappedMetaWorldMultiTaskEnv().step(), 
# global_traj_list_macro is used to save (s1, a1, r20, s21, ...)
global_traj_list_macro = [obs]

for _ in range(20):
    observation, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.env.action_space.sample())

    # information of (s,a,r) will be saved into info['traj_list_micro'] and info['traj_list_macro'] when WrappedMetaWorldMultiTaskEnv().step() 
    # is called.
    global_traj_list_micro.extend(info['traj_list_micro'])
    global_traj_list_macro.extend(info['traj_list_macro'])
    start_time = time.time()

    # for visualisation purpose
    while True:
        wrapped_env.render()
    
        # holding for 1 sec
        if time.time() - start_time >= 1:
            break

all_s_macro, all_act_macro, all_r_macro = parse_traj_list(global_traj_list_macro)

print("All the macro rewards are : ")
for i in range(len(all_r_macro)):
    print(all_r_macro[i])

assert len(all_r_macro) == 20
print("Testing correctly done.")