task="fourCorners"
# harvester fourCorners

CUDA_VISIBLE_DEVICES="1,2,3" python3 main_ppo.py --env_task ${task} -a PPO --configfile cfg.py -o output/${task} -p step2e5_epo10_demo20_nondeter_prev_llama7b -v --total_timesteps 2e5 --collect_program_rollouts --num_demo_per_program 20 --offline_timesteps 100 

#-p step2e5_epo10_interactive_v12 -v --total_timesteps 2e5 --collect_program_rollouts --offline_timesteps 0 
