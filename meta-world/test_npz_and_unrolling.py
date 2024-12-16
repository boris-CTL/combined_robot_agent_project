import metaworld
import numpy as np

SEED_NUMBER = 333
task_idx = 12

mt1 = metaworld.MT1('reach-v2', seed = SEED_NUMBER) # Construct the benchmark, sampling tasks
env = mt1.train_classes['reach-v2']()  # Create an environment 
task = mt1.train_tasks[task_idx]
env.set_task(task)  # Set task
obs = env.reset()
env.render_mode = "human"  # Set to "rgb_array" for image-based rendering

# Load NPZ file
# test if npz can be loaded and data can be used as actions 
loaded_data = np.load(f'./ReachV2_4trajs/ReachV2_4traj_seed{SEED_NUMBER}_idx{task_idx}.npz', allow_pickle=True)

# Accessing individual lists
all_traj = list(loaded_data['all_traj'])
all_states = list(loaded_data['all_states'])
all_actions = list(loaded_data['all_actions'])
all_rewards = list(loaded_data['all_rewards'])

# Check if the initial states are the same
if (np.array_equal(obs[0], all_traj[0])):
    print("The initial states are the same.")
else:
    print("The initial states are not the same. Something went wrong. They should be the same.")

for i in range(len(all_actions)):
    action = list(all_actions[i])
    env.render()
    try:
        observation, reward, terminated, truncated, info= env.step(action)
        print("reward (executed) :", reward)
        print("reward (saved) :", all_rewards[i])
    except ValueError:
        pass