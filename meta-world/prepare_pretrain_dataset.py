import metaworld
import numpy as np
from torch.utils.data.dataset import Dataset

# SEED_NUMBER = 333
# task_idx = 12

seed_lists = [111, 222, 333, 444]
task_per_seed = 50
env_name = "reach-v2"

expert_observations = np.empty((0, 39))
expert_actions = np.empty((0, 4))

# Record the rewards
rewards = []
traj_lengths = []

success_counter = 0

for SEED_NUMBER in seed_lists:
    for task_idx in range(task_per_seed):

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

        # Make it numpy array
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        # Check if the initial states are the same
        if (np.array_equal(obs[0], all_traj[0])):
            print("The initial states are the same.")

            # Add the data to the dataset (T, D)
            # For states, do not append the last state
            expert_observations = np.append(expert_observations, all_states[:-1], axis=0)
            expert_actions = np.append(expert_actions, all_actions, axis=0)
            rewards.append(all_rewards[-1])
            traj_lengths.append(len(all_actions))
            success = (all_rewards[-1] == 10.0)
            success_counter += success
            print(all_states.shape)
            print(all_actions.shape)
            print(all_rewards[-1])
            print("Success: ", success)
        else:
            print("The initial states are not the same. Something went wrong. They should be the same.")
        
        # for i in range(len(all_actions)):
        #     # action = list(all_actions[i])
        #     # env.render()
        #     action = all_actions[i]
        #     try:
        #         observation, reward, terminated, truncated, info= env.step(action)
        #         print("reward (executed) :", reward)
        #         print("reward (saved) :", all_rewards[i])
        #     except ValueError:
        #         pass


print("Dataset info:")
print("Environment: ", env_name)
print("Shape of expert_observations: ", expert_observations.shape)
print("Shape of expert_actions: ", expert_actions.shape)
print("Number of trajectories: ", len(traj_lengths))
print("Average reward: ", np.mean(rewards))
print("Success rate: ", success_counter / len(traj_lengths))
print("Average trajectory length: ", np.mean(traj_lengths))

# Output the dataset info to a file
with open('expert_dataset_info.txt', 'w') as f:
    print("Dataset info:", file=f)
    print("Environment: ", env_name, file=f)
    print("Shape of expert_observations: ", expert_observations.shape, file=f)
    print("Shape of expert_actions: ", expert_actions.shape, file=f)
    print("Number of trajectories: ", len(traj_lengths), file=f)
    print("Average reward: ", np.mean(rewards), file=f)
    print("Success rate: ", success_counter / len(traj_lengths), file=f)
    print("Average trajectory length: ", np.mean(traj_lengths), file=f)

# Save the dataset
# np.savez('expert_dataset.npz', expert_observations=expert_observations, expert_actions=expert_actions)