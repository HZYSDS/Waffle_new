source params.sh

savename="equal_number" # use equal number of descriptors
backbone="ViT-B/32"

k=5

##### random selection
for ((i=0; i<${#datasets[@]}; i++));
do
    dataset=${datasets[$i]}
    shot=${max_shot_list[$i]}

    # for seed in "${seed_list[@]}"
    # do
    #     python random_select.py --dataset=${dataset} --k=${k} --seed=${seed} --loaddir=dclip_descriptors --savedir=random_dclip_descriptors &
    #     python random_select.py --dataset=${dataset} --k=${k} --seed=${seed} --loaddir=comparative_descriptors --savedir=random_comparative_descriptors &
    #     wait
    # done

    python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --mode=random_selection_dclip --reps=5 --model_size=${backbone}
    python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --mode=random_selection_comparative --reps=5 --model_size=${backbone}
done

##### filtering
for ((i=0; i<${#datasets[@]}; i++));
do
    dataset=${datasets[$i]}
    shot=${max_shot_list[$i]}

    # for seed in "${seed_list[@]}"
    # do
    #     python filter_descriptors.py --dataset=${dataset} --k=${k} --shot=${shot} --seed=${seed} --loaddir=dclip_descriptors --savedir=filtered_dclip_descriptors --model_size=${backbone} &
    #     python filter_descriptors.py --dataset=${dataset} --k=${k} --shot=${shot} --seed=${seed} --loaddir=comparative_descriptors --savedir=equal_descriptors --model_size=${backbone} &
    #     wait
    # done

    python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_dclip --reps=5 --model_size=${backbone}
    python base_main_new.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_equal_number --reps=5 --model_size=${backbone}
done