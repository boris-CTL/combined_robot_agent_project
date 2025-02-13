# The Combined Robot Agent: An LLM+DRL Approach

This project tackles 3D continuous robotic environments (e.g., Meta-World) by leveraging the embedded knowledge in Large Language Models (LLMs) with reinforcement learning (RL) agents.

## Installation

### Step 0: Install LLF-Bench and Python Environment

1. Install [`llfbench`](https://github.com/microsoft/LLF-Bench) by following the instructions at [LLF-Bench GitHub](https://github.com/microsoft/LLF-Bench).

2. Install the Python environment using `environment.yml`:
    ```sh
    conda env create -f environment.yml
    conda activate your_env_name
    ```

## Project Workflow

The project is divided into three main steps:

### Step 1: Generate Trajectories Using LLMs

For tasks in the Meta-World environment, use LLMs to generate continuous actions step-by-step through prompting. Execute these actions in the environment to produce the next state, and continue this process to generate complete trajectories.

1. **Example**: Run the following command to use GPT-3.5 for the `window-close-v2` task:
    ```sh
    python3 gpt35_WindowCloseV2_ArgparseVersion.py
    ```
    This will generate trajectories and save them in the [outdir_WindowCloseV2](https://github.com/boris-CTL/combined_robot_agent_project/tree/main/outdir_WindowCloseV2) folder as `.npz` files.

    **Note**: Set up your OpenAI key and organization key in the script:
    ```python
    config['api_key'] = "your api key here"
    config['openai_org'] = "your_organization_key_here"
    ```

2. Similarly, you can run:
    ```sh
    python3 gpt35_PickPlaceV2_ArgparseVersion.py
    ```

3. The [llm_generated_trajectories](https://github.com/boris-CTL/combined_robot_agent_project/tree/main/llm_generated_trajectories) folder contains trajectories for four tasks: `basketball-v2`, `pick-place-v2`, `reach-v2`, and `window-close-v2`. These can be used directly for constructing the expert dataset in Step 2.

### Step 2: Construct Expert Dataset and Pretrain RL Agent

Aggregate the "expert" trajectories generated in Step 1 to create an expert dataset and use it to pretrain an RL agent.

1. **Example**: Run the following command to integrate trajectories for `window-close-v2` with seed `111`:
    ```sh
    python3 prepare_pretrain_dataset_ArgparseVersion.py --env_name 'window-close-v2' --seed_list '[111]'
    ```

2. **Example**: Run the following script to pretrain the RL agent using the expert dataset:
    ```sh
    sh pretrain_agent_with_expert_data.sh
    ```
    **Note**: Adjust the `envs` and `pretrain_dataset` paths accordingly.

### Step 3: Test Pretrained RL Agent

Perform inference using the pretrained RL agent on Meta-World tasks.

1. **Example**: Run the following script to test the agent:
    ```sh
    python3 test_agent.py
    ```
    **Note**: Adjust the model path, `env_name`, seed, and other parameters accordingly.

## Additional Information

- Ensure you have set up your OpenAI API key and organization key in the relevant scripts.


## License

This project is licensed under the MIT License. See the LICENSE file for details.








<!-- ## Installation of LLF-Bench's environment
Please follow the instruction in https://github.com/microsoft/LLF-Bench

## Run step-by-step unrolling by using LLMs
First, set up OpenAI key by modifying 
```python
config['api_key'] = "your api key here"
``` 
in `run_MetaWorld_reach_v2_gpt35.py`.

Second, simply run
```bash
python3 run_MetaWorld_reach_v2_gpt35.py
```

## To test WrappedMetaWorldEnv
Under the llm-prl/meta-world/ directory, simply run
```bash
python3 test_WrappedMetaWorldEnv.py
``` -->


