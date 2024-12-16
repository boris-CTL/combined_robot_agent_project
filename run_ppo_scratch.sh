task="topOff"
# harvester fourCorners topOff_sparse

CUDA_VISIBLE_DEVICES="3" python3 main_ppo_gpt4.py --env_task ${task} -a PPO --configfile cfg.py -o output/${task} -p step2e5_epo10 -v --total_timesteps 2e5 
# --input_height 14 --input_width 22
# --collect_program_rollouts --num_demo_per_program 20 --offline_timesteps 100 --api_key not-my-key --openai_org org-VFSaZQgJUGtyglfUzEzesHAC

#-p step2e5_epo10_interactive_v12 -v --total_timesteps 2e5 --collect_program_rollouts --offline_timesteps 0 

# 0.73, 0.41