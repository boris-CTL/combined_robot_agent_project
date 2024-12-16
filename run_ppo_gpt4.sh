export  OPENAI_API_KEY=[your key]

task="cleanHouse"
# harvester fourCorners

CUDA_VISIBLE_DEVICES="2" python3 main_ppo_gpt4.py --env_task ${task} -a PPO --configfile cfg.py -o output/${task} -p step2e5_epo10_demo20_nondeter_prev_llm -v --total_timesteps 2e5 --collect_program_rollouts --num_demo_per_program 20 --offline_timesteps 100 --api_key [your key] --openai_org org-VFSaZQgJUGtyglfUzEzesHAC --input_height 14 --input_width 22
