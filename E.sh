#!/bin/bash
source params.sh

# 使用指定的 backbones 和只运行 pets 数据集
for model_size in "${backbones[@]}"; do
    # clip
    dataset="dtd"
    concept=${concepts[${#concepts[@]}-1]} # 如果需要，可以指定 concept
    savename="clip"
    python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size}

    if [[ -n $concept ]]; then
        savename="clip_concept"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
    fi

    # dclip
    savename="dclip"
    python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size}

    if [[ -n $concept ]]; then
        savename="dclip_concept"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
    fi

    # WaffleCLIP
    savename="waffle"
    python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${model_size}

    if [[ -n $concept ]]; then
        savename="waffle_concept"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=waffle --waffle_count=15 --reps=7 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
    fi

    # ours (comparative)
    savename="ours"
    python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size}

    if [[ -n $concept ]]; then
        savename="ours_concept"
        python base_main_class.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
    fi

    # ours + filtering
    k=${k_list[${#k_list[@]}-1]} # 如果需要，可以根据pets设置合适的k值
    case $model_size in
    "ViT-B/32")
        shot="${best_shot_list[${#best_shot_list[@]}-1]}"
        ;;
    "ViT-L/14")
        shot="${ViT_L_14_shot_list[${#ViT_L_14_shot_list[@]}-1]}"
        ;;
    "RN50")
        shot="${RN50_shot_list[${#RN50_shot_list[@]}-1]}"
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