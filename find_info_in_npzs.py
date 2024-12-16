import os
import glob
import re
import numpy as np

# function-ized
def find_matching_npzs_paths(ENV_NAME : str) -> list:
    folder_path = f"outdir_{ENV_NAME}/{ENV_NAME}"
    pattern = rf"{ENV_NAME}_4traj_seed\d+_idx\d+_micro_before_max\.npz"

    current_terminal_path = os.getcwd()
    
    # Use glob.glob() 
    matching_files = glob.glob(os.path.join(current_terminal_path, folder_path, "*"))
    print(os.path.join(current_terminal_path, folder_path))

    
    # Use re 
    matching_files = [file for file in matching_files if re.match(pattern, os.path.basename(file))]
    return matching_files



if __name__ == '__main__':
    
    
    env_name = 'pick-place-v2'
    ENV_NAME = ''.join(part.capitalize() for part in env_name.split('-'))

    matching_files = find_matching_npzs_paths(ENV_NAME)


    for i in range(len(matching_files)):
        loaded_data = np.load(f'{matching_files[i]}', allow_pickle=True)
        try:
            info = loaded_data['last_info']
            print(i, ":", info.item()['success'])
        except:
            print(f"When i = {i}, there's no info.")