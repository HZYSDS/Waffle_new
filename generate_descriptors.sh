source params.sh

##### Generate DCLIP (baseline) descriptors
for dataset in "${datasets[@]}"
do
    python generate_batch_jsonl.py --mode=dclip --dataset=${dataset}
done

# generate batch output: https://platform.openai.com/batches

python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_bmj16oKnBNvssf6nr0WuTzD8 --dataset=imagenet
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_bmj16oKnBNvssf6nr0WuTzD8 --dataset=imagenetv2
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_aUPDuEmvabxAfdDsHoBFEBR0 --dataset=caltech256
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_pI7wxDTMZkFvMYuE9bQTSqtV --dataset=cifar100
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_vkkWuDt6SQvrKWqKPHzVb7XX --dataset=cub
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_6E32lIujYZTkECJ53UugT3KT --dataset=eurosat
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_K1TKAPopyKSGaRcvMLyIBcHH --dataset=places365
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_m3pOALnbKkt6WIO3vnCdZEBZ --dataset=food101
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_yeWyDHuogdyH5KCiKbWioeKO --dataset=pets
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_5nfUv2oVGWFaQmGnzXr7fH3V --dataset=dtd
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_dIp1ZiuDu3ZllaxpGO8ykjSf --dataset=flowers102
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_yt6i0obAERzm2AB9DnJ26iFi --dataset=fgvcaircraft
python jsonl_to_descriptor.py --mode=dclip --batch_id=batch_c3jTgwgkPnb4VfDvv96y3v6b --dataset=cars

##### Generate Comparative (ours) descriptors
for dataset in "${datasets[@]}"
do
    python generate_batch_jsonl.py --mode=comparative --dataset=${dataset}
done

# generate batch output: https://platform.openai.com/batches

python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_7VC1XZOO6XyYsAVxpEqTINM8 --dataset=imagenet
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_7VC1XZOO6XyYsAVxpEqTINM8 --dataset=imagenetv2
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_551IIMtYnCIIU1kpRWj6GidL --dataset=caltech256
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_zIO0kRC41mPIrMUj0l02elD9 --dataset=cifar100
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_mQOwSOjXnbyXfrhydFbWgos3 --dataset=cub
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_RcqrAYRv1qgNDErsLos8LrwW --dataset=eurosat
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_IQqH6GHeelXUazQxfAnMczHl --dataset=places365
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_D0Nx867mFqBbSWVl2zP4B9Dg --dataset=food101
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_1tCZD7UIFqe0ujUH2skUcqBe --dataset=pets
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_LLZ7NkXowNjQsDWLSWkFyMBo --dataset=dtd
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_MrOIIo18TJTMO38iyjgH8UB7 --dataset=flowers102
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_h1apqET7hcskzkPeFSU0QeP6 --dataset=fgvcaircraft
python jsonl_to_descriptor.py --mode=comparative --batch_id=batch_7VfSyyosHae91ZsoBQgvviqW --dataset=cars

for ((i=0; i<${#datasets[@]}; i++));
do
    dataset=${datasets[$i]}
    concept=${concepts[$i]}

    python check_clip_context_length.py --dataset=${dataset} --concept="${concept}"
done