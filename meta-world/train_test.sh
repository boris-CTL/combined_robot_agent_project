# envs="assembly-v2 push-v2 pick-place-v2 door-open-v2 drawer-open-v2 drawer-close-v2 button-press-topdown-v2 peg-insert-side-v2 window-open-v2 window-close-v2 hammer-v2"
envs="pick-place-v2"
postfix="_0418_epo4000_bs5000_nt50_ei20_vi100_nopretrain"

for env in $envs
do
    xvfb-run -a python3 train_agent.py --env_name $env --epochs 4000 --batch_size 5000 --eval_interval 20 --visual_interval 100 --n_tasks 50 --do_pretrain False --pretrain_dataset None --pretrain_trajs 0 --postfix $postfix --log_wandb True &> logs/${env}${postfix}.log 
done