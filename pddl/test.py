import pddlgym
import imageio
from stable_baselines3 import PPO
import gym
import numpy as np
from PIL import Image
import gymnasium
import random
from .wrapper import RenderObservationWrapper

from pddlgym.demo_planning import create_replanning_policy, create_single_plan_policy

try:
    from pddlgym_planners.fd import FD
    from pddlgym_planners.planner import PlanningFailure
except ModuleNotFoundError:
    raise Exception("To run this demo file, install the " + \
        "PDDLGym Planners repository (https://github.com/ronuchit/pddlgym_planners)")
        
import wandb
from wandb.integration.sb3 import WandbCallback
is_wandb = True
if is_wandb:
    wandb.init(project="llm-prl-pddl", sync_tensorboard=True, id='231222')

verbose = True
render=True
problem_index = 0
env_name = "blocks_single"
video_path="output/pddlgym/video.gif"
env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
obs, _ = env.reset()
env.action_space.all_ground_literals(obs)
env = RenderObservationWrapper(env, obs)
env.fix_problem_index(problem_index)
policy = lambda s : env.action_space.sample(s)
images = []
obs, _ = env.reset()
max_num_steps=50
# cnn model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./output")

model.learn(total_timesteps=500000, callback=WandbCallback())
if is_wandb:
    wandb.finish()
planner = FD(alias_flag="--alias lama-first", )
policy = create_single_plan_policy(env, planner)
obs, _ = env.reset()
for t in range(max_num_steps):
    if render:
        images.append(env.render())
    action = policy(obs)
    print("Act:", action)

    obs, reward, done, _ = env.step(action[t])
    env.render()

    print("Rew:", reward)

    if done:
        break

if verbose:
    print("Final obs:", obs)
    print()

if render:
    images.append(env.render())
    imageio.mimwrite(video_path, images, fps=6)
    print("Wrote out video to", video_path)

env.close()
# if check_reward:
#     assert tot_reward > 0
if verbose:
    input("press enter to continue to next problem")