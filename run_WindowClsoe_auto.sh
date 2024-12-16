

#!/bin/bash


for idx in $(seq 0 49);
do
    # Copy the original file to a new filename with the corresponding index
    python3 "gpt35_WindowCloseV2_Seed111_idx${idx}.py"
    
done