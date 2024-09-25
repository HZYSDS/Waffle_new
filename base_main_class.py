# base_main_class.py

import argparse
import clip
import torch
import tools
import torch.nn.functional as F
import numpy as np
import tqdm
import torchmetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['clip', 'dclip', 'filtered'], help='Mode of the experiment.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate.')
    parser.add_argument('--model_size', type=str, required=True, choices=['ViT-B/32', 'ViT-L/14', 'RN50'], help='Pretrained CLIP model to use.')
    parser.add_argument('--savename', type=str, required=True, help='Filename to save the results.')
    parser.add_argument('--shot', type=int, default=0, help='Number of images per class for few-shot learning.')
    parser.add_argument('--k', type=int, default=0, help='Number of descriptors to use.')
    parser.add_argument('--label_before_text', type=str, default='A photo of a ', help='Prefix for label text.')
    parser.add_argument('--reps', type=int, default=1, help='Number of repetitions for randomization experiments.')
    opt = parser.parse_args()

    # Seed everything for reproducibility
    tools.seed_everything(42)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(opt.model_size, device=device)
    model.eval()

    # Load the dataset
    dataset = tools.load_dataset(opt.dataset, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.shot, shuffle=True)

    # Placeholder for results
    accs1, accs5 = [], []
    num_classes = len(dataset.classes)

    accuracy_metric = torchmetrics.Accuracy(num_classes=num_classes).to(device)
    accuracy_metric_top5 = torchmetrics.Accuracy(num_classes=num_classes, top_k=5).to(device)

    for batch in tqdm.tqdm(dataloader, desc=f"Processing with mode {opt.mode}"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Compute image embeddings
        with torch.no_grad():
            image_features = model.encode_image(images)

        # Process based on the mode
        if opt.mode == 'clip':
            # Compare image features with text class embeddings (standard CLIP)
            text_features = tools.get_clip_text_features(opt, model)
            similarities = F.normalize(image_features) @ F.normalize(text_features).T

        elif opt.mode == 'dclip':
            # Compare image features with descriptors (DCLIP)
            descriptor_encodings = tools.compute_descriptor_encodings(opt, model)
            similarities = F.normalize(image_features) @ F.normalize(descriptor_encodings).T

        elif opt.mode == 'filtered':
            # Use filtered descriptors
            descriptor_encodings = tools.compute_filtered_encodings(opt, model)
            similarities = F.normalize(image_features) @ F.normalize(descriptor_encodings).T

        # Compute accuracy metrics
        accuracy_metric.update(similarities.softmax(dim=-1), labels)
        accuracy_metric_top5.update(similarities.softmax(dim=-1), labels)

        accs1.append(accuracy_metric.compute().item() * 100)
        accs5.append(accuracy_metric_top5.compute().item() * 100)
        accuracy_metric.reset()
        accuracy_metric_top5.reset()

    # Save final results
    results = {
        'acc1': np.mean(accs1),
        'acc5': np.mean(accs5),
        'std_acc1': np.std(accs1),
        'std_acc5': np.std(accs5)
    }
    tools.save_results(opt.savename, results)
    print(f"Final Results: Top-1 Accuracy: {results['acc1']}%, Top-5 Accuracy: {results['acc5']}%")

if __name__ == "__main__":
    main()