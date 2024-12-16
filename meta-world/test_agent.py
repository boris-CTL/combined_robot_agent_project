import metaworld
import gym
import torch
import random
from stable_baselines3 import PPO  # Assuming you're using Stable Baselines3 or a similar library

# Set random seed and environment name
seed = 45
env_name = "drawer-close-v2"
postfix="_0314"


# Set random seed for reproducibility
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Initialize Meta-World for a specific task (e.g., "reach-v2")
mt1 = metaworld.MT1(env_name, seed=seed)
env = mt1.train_classes[env_name]()
# print(len(mt1.train_tasks))
task = random.choice(mt1.train_tasks)
print(task)
env.set_task(task)
env.render_mode = "human"  # Set to "rgb_array" for image-based rendering
env.seed(seed)

# Load Model
model = PPO.load(f"ppo_{env_name}{postfix}")

# Test
obs, info = env.reset(seed=seed)
print(obs)
done = False
while not done:
    env.render()
    action, _states = model.predict(obs, deterministic=True)  # For Stable Baselines3
    print(action)  # Print the action that the agent takes
    # For PyTorch, you would use your model to predict based on obs, which might involve tensor conversion
    obs, reward, _, done, info = env.step(action)
    print('reward:', reward)
    print('done:', done)   
    print('info:', info)
