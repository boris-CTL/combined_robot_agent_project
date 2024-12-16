# envs="assembly-v2 push-v2 pick-place-v2 door-open-v2 drawer-open-v2 drawer-close-v2 button-press-topdown-v2 peg-insert-side-v2 window-open-v2 window-close-v2 hammer-v2"
envs="reach-v2"

for env in $envs
do
    python3 train_agent.py --env_name $env --total_steps 2e5 --n_tasks 50 --do_pretrain True --pretrain_dataset expert_dataset.npz --postfix _0321_steps2e5_with_pretrain --log_wandb True
done