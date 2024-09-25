#!/bin/bash
source params.sh

for model_size in "${backbones[@]}"; do
    # clip
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        savename="clip"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size}

        if [[ -n $concept ]]; then
            savename="clip_concept"
            python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # dclip
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        savename="dclip"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size}

        if [[ -n $concept ]]; then
            savename="dclip_concept"
            python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # WaffleCLIP
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        savename="waffle"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${model_size}

        if [[ -n $concept ]]; then
            savename="waffle_concept"
            python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # ours (comparative)
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        savename="ours"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size}

        if [[ -n $concept ]]; then
            savename="ours_concept"
            python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # ours + filtering
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        k=${k_list[$i]}

        case $model_size in
        "ViT-B/32")
            shot="${best_shot_list[$i]}"
            ;;
        "ViT-L/14")
            shot="${ViT_L_14_shot_list[$i]}"
            ;;
        "RN50")
            shot="${RN50_shot_list[$i]}"
            ;;
        *)
            echo "model_size error: $model_size"
            shot=""
            ;;
        esac

        savename="ours_filtered"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered --reps=5 --model_size=${model_size}

        if [[ -n $concept ]]; then
            savename="ours_filtered_concept"
            python base_main_class.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered --reps=5 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done