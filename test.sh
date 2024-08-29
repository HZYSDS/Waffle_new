python base_main_new.py --savename='ours' --dataset=cars --mode=comparative --model_size=ViT-L/14 --label_before_text="A photo of a car: a "

python base_main_new.py --savename='ours' --dataset=cars --mode=comparative --model_size=ViT-B/32 --label_before_text="A photo of a car: a "

python base_main_new.py --savename='ours' --dataset=cars --mode=comparative --model_size=RN50 --label_before_text="A photo of a car: a "

python base_main_new.py --savename='ours' --dataset=cars  --k=10 --shot=16 --mode=filtered --reps=5 --model_size=ViT-B/32 --label_before_text="A photo of a car: a "

python base_main_new.py --savename='ours' --dataset=cars  --k=10 --shot=16 --mode=filtered --reps=5 --model_size=ViT-L/14 --label_before_text="A photo of a car: a "

python base_main_new.py --savename='ours' --dataset=cars  --k=10 --shot=16 --mode=filtered --reps=5 --model_size=RN50 --label_before_text="A photo of a car: a "