
#!/bin/bash


for idx in $(seq 1 49);
do
    # Copy the original file to a new filename with the corresponding index
    cp "gpt35_WindowCloseV2_Seed111_idx0.py" "gpt35_WindowCloseV2_Seed111_idx${idx}.py"
    
    # Modify the line with the task_idx in the newly created file
    sed -i "s/task_idx = 0/task_idx = ${idx}/g" "gpt35_WindowCloseV2_Seed111_idx${idx}.py"
done