source params.sh

savename='filter_existing_method' # apply our filtering process to DCLIP descr.
backbone="ViT-B/32"

for ((i=0; i<${#datasets[@]}; i++));
do
    dataset=${datasets[$i]}
    concept=${concepts[$i]}
    k=5
    shot=${max_shot_list[$i]}

    # for seed in "${seed_list[@]}"
    # do
    #     python filter_descriptors.py --dataset=${dataset} --k=${k} --shot=${shot} --seed=${seed} --loaddir=dclip_descriptors --savedir=filtered_dclip_descriptors --model_size=${backbone}
    # done

    python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_dclip --reps=5 --model_size=${backbone}

    if [[ -n $concept ]]; then
        python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_dclip --reps=5 --model_size=${backbone} --label_before_text="A photo of a ${concept}: a "
    fi
done