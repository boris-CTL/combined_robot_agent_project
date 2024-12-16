
#!/bin/bash

envs = "window-open-v2"

for env in $envs
do
for idx in $(seq 1 49);
do
    python3 gpt35_interact.py --env_name $env --seed 111 --task_idx $idx
done
done