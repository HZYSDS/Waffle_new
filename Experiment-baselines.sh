source params.sh

for backbone in "${backbones[@]}"; do
    # CLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        python base_main.py --savename='baselines' --dataset=${dataset} --mode=clip --model_size=${backbone}
    done

    # CLIP + concepts
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        if [[ -n $concept ]]; then
            python base_main.py --savename='baselines' --dataset=${dataset} --mode=clip --model_size=${backbone} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # DCLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        python base_main.py --savename='baselines' --dataset=${dataset} --mode=dclip --model_size=${backbone}
    done

    # DCLIP + concepts
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        if [[ -n $concept ]]; then
            python base_main.py --savename='baselines' --dataset=${dataset} --mode=dclip --model_size=${backbone} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # WaffleCLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        python base_main.py --savename='baselines' --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${backbone}
    done

    # WaffleCLIP + concepts
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        if [[ -n $concept ]]; then
            python base_main.py --savename='baselines' --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${backbone} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # WaffleCLIP + GPT + concepts
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        if [[ -n $concept ]]; then
            python base_main.py --savename='baselines' --dataset=${dataset} --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=${backbone} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done