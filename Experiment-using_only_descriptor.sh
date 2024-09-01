source params.sh

 # use only descriptor for text input
backbone="ViT-B/32"

##### Before executing this shell, you should comment out #296 of tools.py #####

for ((i=0; i<${#datasets[@]}; i++));
do
    dataset=${datasets[$i]}
    savename='using_only_descriptor_dclip'
    python base_main_new.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${backbone}
    savename='using_only_descriptor_comparative'
    python base_main_new.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${backbone}
done