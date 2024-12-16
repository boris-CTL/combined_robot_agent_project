task="harvester"

CUDA_VISIBLE_DEVICES="2" python3 main.py --env_task ${task} -a DQN --configfile cfg.py -o output/${task} -p step2e5_epo10_demo20_nondeter_prev_llm -v --total_timesteps 2e5 --collect_program_rollouts --num_demo_per_program 20 --offline_timesteps 0 --api_key [YOUR_API_KEY] --openai_org [YOUR_ORG_ID]

#-p step2e5_epo10_interactive_v12 -v --total_timesteps 2e5 --collect_program_rollouts --offline_timesteps 0 
