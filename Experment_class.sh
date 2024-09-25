#!/bin/bash

# Source the parameters file
source params.sh

# Iterate over different backbones (models)
for backbone in "${backbones[@]}"; do
    # CLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        python base_main_class.py --savename='baselines' --dataset=${dataset} --mode=clip --model_size=${backbone}
    done

    # DCLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        python base_main_class.py --savename='baselines' --dataset=${dataset} --mode=dclip --model_size=${backbone}
    done

    # Ours + filtered
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        k=${k_list[$i]}
        shot=${best_shot_list[$i]}
        python base_main_class.py --savename="ours_filtered" --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered --model_size=${backbone}
    done
done