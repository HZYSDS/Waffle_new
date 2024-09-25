# %%
import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import clip
import numpy as np
import pickle
from termcolor import colored
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import tqdm
import tools
import csv

# %%
parser = argparse.ArgumentParser()
# Base arguments
parser.add_argument('--mode', type=str, default='clip', choices=tools.METHODS, help='VLM extension to use.')
parser.add_argument('--seed', type=int, default=1, help='Replication seed.')
parser.add_argument('--batch_size', type=int, default=640, help='Batchsize, mainly used to compute image embeddings.')
parser.add_argument('--dataset', type=str, default='imagenetv2', choices=tools.DATASETS, help='Dataset to evaluate on.')
parser.add_argument('--model_size', type=str, default='ViT-B/32', choices=tools.BACKBONES, help='Pretrained CLIP model to use.')
parser.add_argument('--aggregate', type=str, default='mean', choices=('mean', 'max'), help='How to aggregate similarities of multiple language embeddings.')
parser.add_argument('--label_before_text', type=str, default='A photo of a ', help='Prompt-part going at the very beginning.')
parser.add_argument('--label_after_text', type=str, default='.', help='Prompt-part going at the very end.')
parser.add_argument('--pre_descriptor_text', type=str, default='', help='Text that goes right before the descriptor.')
parser.add_argument('--descriptor_separator', type=str, default=', ', help='Text separating descriptor part and classname.')
parser.add_argument('--dont_apply_descriptor_modification', action='store_true', help='Flag. If set, will not use "which (is/has/etc)" before descriptors.')
parser.add_argument('--merge_predictions', action='store_true', help='Optional flag to merge generated embeddings before computing retrieval scores.')
parser.add_argument('--save_model', type=str, default='', help='Set to a non-empty filename to store generated language embeddings & scores in a pickle file for all seed-repetitions.')
parser.add_argument('--randomization_budget', type=int, default=15, help='Budget w.r.t. to DCLIP for randomization ablations')
parser.add_argument('--waffle_count', type=int, default=15, help='For WaffleCLIP: Number of randomized descriptor pairs to use')
parser.add_argument('--reps', type=int, default=1, help='Number of repetitions to run a method for with changing randomization. Default value should be >7 for WaffleCLIP variants.')
parser.add_argument('--savename', type=str, default='result', help='Name of csv-file in which results are stored.')
parser.add_argument('--shot', type=int, default=0, help='[0, 1, 2, 4, 8, 16, 32, 64]')
parser.add_argument('--k', type=int, default=0, help='5, 10, 15, 20')
parser.add_argument('--vmf_scale', type=float, default=1)
opt = parser.parse_args()
opt.apply_descriptor_modification = not opt.dont_apply_descriptor_modification

# %% Get dataloader and load model.
tools.seed_everything(opt.seed)
opt, dataset = tools.setup(opt)

print(colored(f"\nLoading model [{opt.model_size}] for dataset [{opt.dataset}] ...\n", "yellow", attrs=["bold"]))

opt.device = device = torch.device('cuda')
model, preprocess = clip.load(opt.model_size, device=device, jit=False)
model.eval()
model.requires_grad_(False)

# %% Compute image embeddings if not already precomputed.
precomputed_encs_folder = 'precomputed_encs'
os.makedirs(precomputed_encs_folder, exist_ok=True)
precomputed_encs_file = os.path.join(precomputed_encs_folder, f'{opt.dataset}_{opt.model_size.replace("/", "")}.pkl')

if os.path.exists(precomputed_encs_file):
    load_res = pickle.load(open(precomputed_encs_file, 'rb'))
else:
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    enc_coll = []
    label_coll = []
    with torch.no_grad():
        for batch_number, batch in enumerate(tqdm.tqdm(dataloader, desc='Precomputing image embeddings...')):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            label_coll.append(labels)

            image_encodings = F.normalize(model.encode_image(images))
            enc_coll.append(image_encodings.cpu())
    load_res = {'enc': enc_coll, 'labels': label_coll}
    pickle.dump(load_res, open(precomputed_encs_file, 'wb'))

encoding_coll = load_res['enc']
label_coll = load_res['labels']

# %% Initialize accuracy tracking for each class
class_accuracy = {class_name: {'correct': 0, 'total': 0} for class_name in dataset.classes}

# %% Generate Image Embeddings and compute scores.
encodings = []
filter_modes = ['filtered', 'filtered_dclip', 'filtered_equal_number', 'random_selection_dclip', 'random_selection_comparative']

if opt.mode in filter_modes:
    opt.reps = 5
    assert opt.reps <= 5, "If you want to iterate more than 5 times, you'll need to do some additional filtering."

origin_descriptor_fname = opt.descriptor_fname

for index in range(opt.reps):
    if opt.mode in filter_modes:
        if opt.descriptor_fname.endswith('.json'):
            opt.descriptor_fname = f"{origin_descriptor_fname[:-5]}_{index + 1}.json"
        else:
            opt.descriptor_fname = f"{origin_descriptor_fname}_{index + 1}"

    description_encodings = tools.compute_description_encodings(opt, model, mode=opt.mode)
    descr_means = torch.cat([x.mean(dim=0).reshape(1, -1) for x in description_encodings.values()])
    descr_means /= descr_means.norm(dim=-1, keepdim=True)

    for batch_number, (image_encodings, labels) in tqdm.tqdm(enumerate(zip(encoding_coll, label_coll)),
                                                             total=len(encoding_coll), desc='Classifying images...'):
        image_encodings = image_encodings.to(device)
        labels = labels.to(device)

        if opt.merge_predictions:
            image_description_similarity = image_encodings @ descr_means.T
        else:
            image_description_similarity_t = [None] * opt.n_classes
            image_description_similarity_cumulative = [None] * opt.n_classes

            for i, (k, v) in enumerate(description_encodings.items()):
                image_description_similarity_t[i] = image_encodings @ v.T
                image_description_similarity_cumulative[i] = tools.aggregate_similarity(
                    image_description_similarity_t[i], aggregation_method=opt.aggregate)

            image_description_similarity = torch.stack(image_description_similarity_cumulative, dim=1)

        # 获取每个样本的预测类
        _, predicted = torch.max(image_description_similarity, 1)

        # 更新每个类别的正确预测次数和总样本数
        for i in range(len(labels)):
            class_name = dataset.classes[labels[i].item()]
            class_accuracy[class_name]['correct'] += (predicted[i] == labels[i]).item()
            class_accuracy[class_name]['total'] += 1

# %% 计算每个类的准确率并输出结果
for class_name, stats in class_accuracy.items():
    correct = stats['correct']
    total = stats['total']
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Class: {class_name}, Accuracy: {accuracy:.2f}%")

# %% 保存每个类的准确率到 CSV 文件
os.makedirs('results', exist_ok=True)
with open(f'results_class/class_accuracy_{opt.savename}_{opt.dataset}.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Class Name", "Accuracy"])
    for class_name, stats in class_accuracy.items():
        correct = stats['correct']
        total = stats['total']
        accuracy = 100 * correct / total if total > 0 else 0
        writer.writerow([class_name, accuracy])