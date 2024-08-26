source params.sh

for model_size in "${backbones[@]}";
do
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}

        python extract_image_features.py --dataset=${dataset} --model_size=${model_size}
    done
done