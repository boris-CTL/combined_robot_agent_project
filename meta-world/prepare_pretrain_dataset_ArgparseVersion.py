import metaworld
import numpy as np
from torch.utils.data.dataset import Dataset
import argparse
import os
import glob
import re

# function-ized
def find_matching_npzs_paths(ENV_NAME : str) -> list:
    folder_path = f"outdir_{ENV_NAME}/{ENV_NAME}"
    # pattern = rf"{ENV_NAME}_4traj_seed\d+_idx\d+_micro_before_max\.npz"
    pattern = rf"{ENV_NAME}_4traj_seed\d+_idx\d+_micro\.npz"

    current_terminal_path = os.getcwd()
    
    # Use glob.glob() 
    matching_files = glob.glob(os.path.join(current_terminal_path, folder_path, "*"))

    # Use re 
    matching_files = [file for file in matching_files if re.match(pattern, os.path.basename(file))]
    return matching_files


def filter_traj_by_reward(matching_files : list) -> tuple:
    npzs_reward_list = []
    rew_eq_ten = dict()
    rew_one_to_ten = dict()
    rew_below_one = dict()
    
    for i in range(len(matching_files)):
        loaded_data = np.load(f'{matching_files[i]}', allow_pickle=True)
        all_rewards = list(loaded_data['all_rewards'])
        npzs_reward_list.append(all_rewards[-1])
        if (all_rewards[-1] == 10):
            rew_eq_ten[matching_files[i]] = all_rewards[-1]
        elif (all_rewards[-1] < 1):
            rew_below_one[matching_files[i]] = all_rewards[-1]
        else:
            # one to ten
            rew_one_to_ten[matching_files[i]] = all_rewards[-1]

    assert len(npzs_reward_list) == len(rew_eq_ten) + len(rew_one_to_ten) + len(rew_below_one)
    return rew_eq_ten, rew_one_to_ten, rew_below_one


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env_name', type=str, default='pick-place-v2')
    argparser.add_argument('--seed_lists', type=str, default='[111, 222, 333, 444, 555, 666]')
    argparser.add_argument('--reward_interval', '-r', type=str, default='eq_ten', dest='reward_interval')
    args = argparser.parse_args()

    # 改變至termianl path
    current_terminal_path = os.getcwd()
    os.chdir(current_terminal_path)

    seed_lists = eval(args.seed_lists)
    task_per_seed = 50
    env_name = args.env_name
    ENV_NAME = ''.join(part.capitalize() for part in env_name.split('-'))
    reward_interval = args.reward_interval

    expert_observations = np.empty((0, 39))
    expert_actions = np.empty((0, 4))

    # Record the rewards
    rewards = []
    traj_lengths = []

    success_counter = 0

    matching_files = find_matching_npzs_paths(ENV_NAME)
    rew_eq_ten, rew_one_to_ten, rew_below_one = filter_traj_by_reward(matching_files)

    if (reward_interval == 'eq_ten'):
        key_list = [key for key in rew_eq_ten]
    elif (reward_interval == 'one_to_ten'):
        key_list = [key for key in rew_one_to_ten]
    elif (reward_interval == 'below_one'):
        key_list = [key for key in rew_below_one]
    elif (reward_interval == 'above_one'):
        # 結合one_to_ten和eq_ten
        key_list = [key for key in rew_eq_ten]
        for key in rew_one_to_ten:
            key_list.append(key)
    else:
        raise NotImplementedError("This mode is not yet implemented")


    folder_path = f"outdir_{ENV_NAME}"
    count_list = []

    for SEED_NUMBER in seed_lists:
        for task_idx in range(task_per_seed):

            # find the matching file. if not found, simply pass
            # npz_file_path = f'{ENV_NAME}/{ENV_NAME}_4traj_seed{SEED_NUMBER}_idx{task_idx}_micro_before_max.npz'
            npz_file_path = f'{ENV_NAME}/{ENV_NAME}_4traj_seed{SEED_NUMBER}_idx{task_idx}_micro.npz'
            npz_file_path = os.path.join(current_terminal_path, folder_path, npz_file_path)
            print(npz_file_path)
            if npz_file_path in key_list:
                # 有找到就表示是符合我們要的traj, 所以要開環境
                # for checking only
                count_list.append((SEED_NUMBER, task_idx))

                mt1 = metaworld.MT1(env_name, seed = SEED_NUMBER) # Construct the benchmark, sampling tasks
                env = mt1.train_classes[env_name]()  # Create an environment 
                task = mt1.train_tasks[task_idx]
                env.set_task(task)  # Set task
                obs = env.reset()
                env.render_mode = "human"  # Set to "rgb_array" for image-based rendering

                # Load NPZ file
                # test if npz can be loaded and data can be used as actions 
                loaded_data = np.load(npz_file_path, allow_pickle=True)

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
                    assert np.array_equal(obs[0], all_traj[0])

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
                    raise AssertionError("obs[0] and all_traj[0] is not the same.")
            else:
                #表示不是符合我們要的traj, 所以simply pass
                print(SEED_NUMBER, task_idx, "not found.")
                pass

    # 它們必須一樣長，否則有錯
    assert len(count_list) == len(key_list)



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
    outdir_for_expert = f"./outdir_{ENV_NAME}/{ENV_NAME}_expert_dataset"
    # Create output directory if it does not already exist
    if not os.path.exists(outdir_for_expert):
        os.makedirs(outdir_for_expert)

    np.savez(f'{outdir_for_expert}/expert_dataset_{reward_interval}.npz', expert_observations=expert_observations, expert_actions=expert_actions)