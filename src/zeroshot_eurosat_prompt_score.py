import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import clip

# Custom imports
from prompt_selection import prompt_selection_direct

# Constants for new class names
NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


def get_args():
    """
    Parses command-line arguments.
    """
    parser = ArgumentParser(description="Prompt score and zeroshot for EuroSAT dataset",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loading workers")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--train_size', type=float, default=0.95, help="Training set size ratio")
    parser.add_argument('--model_arch', type=str, default='ViT-B/32', help="Model architecture to use")
    parser.add_argument('--ckpt_path', type=str, default='./SenCLIP_AvgPool_ViTB32.ckpt', help="Path to model checkpoint")
    parser.add_argument('--root_dir', type=str, default='./', help="Root directory for datasets and outputs")
    parser.add_argument('--template_path', type=str, default='./prompts/prompt_eurosat_ground_mixed.json', help="prompt_template_path")
    parser.add_argument('--mean', type=list, default=(0.347, 0.376, 0.296), help="Normalization mean for images")
    parser.add_argument('--std', type=list, default=(0.269, 0.261, 0.276), help="Normalization std for images")

    return parser.parse_args()

def transform_img(mean, std):
    """
    Returns a torchvision transform pipeline.
    """
    return Compose([
        Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean, std)
    ])
def load_model(weight_path, model_name, device):
    """
    Loads the CLIP model with specified weights.
    """
    model, _ = clip.load(model_name, device=device)
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    return model

def zeroshot_classifier(model: nn.Module, classnames: list, templates: dict or list,
                        stack_emb: bool = True, device: torch.device = 'cpu') -> torch.Tensor:
    """
    Creates zeroshot classifier weights for the given classnames and templates.
    """
    temp_context = 'class_context' if isinstance(templates, dict) else 'unified'

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Encoding class prompts"):
            if temp_context == "class_context":
                texts = templates[classname] if isinstance(templates, dict) else [templates.format(classname)]
            else:
                texts = [template.format(classname) for template in templates]

            texts = torch.cat([clip.tokenize(text) for text in texts]).to(device=device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if stack_emb:
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings /= class_embeddings.norm()
            zeroshot_weights.append(class_embeddings)

        if stack_emb:
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device=device)
        else:
            zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device=device)

    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
 


def compute_logits(model: nn.Module, encoded_prompts: torch.Tensor, test_loader: DataLoader, device: torch.device) -> tuple:
    """
    Computes the logits for the test dataset.
    """
    logits_list = []
    labels_list = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ encoded_prompts)/2+1
            logits_list.append(logits)
            labels_list.append(labels)

    logits_stk = torch.cat(logits_list)
    labels_stk = torch.cat(labels_list)

    return logits_stk, labels_stk

def result_log(logits, labels):
    logits = logits / (logits.mean(0,keepdim=True) + 1e-6)
    
    acc1, acc5 = accuracy(logits, labels, topk=(min(logits.size(1), 1), min(logits.size(1), 5)))
    top1 = (acc1 / labels.size(0)) * 100
    top5 = (acc5 / labels.size(0)) * 100 
    return top1, top5

def main(args):
    TOP_K_IMAGES = 1000
    TOP_K_PROMPTS = [1,2,5,10,15,25]
    method = 'TF-IDF_None'#['TF-IDF','TF-IDF_None', 'TF-IDF_mul', 'TF-IDF_sum', 'Mean','Mean_mul','Mean_sum' ]#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transform_img(args.mean, args.std)
    dataset_path = os.path.join(args.root_dir, 'Datasets/EuroSAT/2750/')
    dataset = ImageFolder(dataset_path, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    class_names = [NEW_CNAMES[cls] for cls in dataset.classes]
    print(f"Class Names: {class_names}")

    # Load model
    model = load_model(args.ckpt_path, args.model_arch, device=device)

    # Load prompts
    template = json.load(open(args.template_path))
    encoded_prompts = zeroshot_classifier(model, class_names, template, stack_emb=False, device=device).T

    logits_stk, _ = compute_logits(model, encoded_prompts, test_dataloader, device=device)
    
    for k_prompts in TOP_K_PROMPTS:
            top1_acc, top5_acc = 0.0, 0.0
            prompt_sim_matrix = True if '_' in method else False
            method_name = method.split('_')[0] 
            comb = method.split('_')[1] if '_' in method else None
            selected_prompts, worst_prompts = prompt_selection_direct(logits_stk,
                                                template=template,
                                                method = method_name,
                                                topK_images = TOP_K_IMAGES, 
                                                topK_prompts = k_prompts,
                                                prompt_sim_matrix = prompt_sim_matrix,
                                                prompt_comb = comb)
            
            
            encoded_prompts = zeroshot_classifier(model,class_names, selected_prompts, stack_emb=True, device=device).T
            print(encoded_prompts.shape)
            logits, labels = compute_logits(model,encoded_prompts,test_dataloader, device=device)
            method_name = f'{method_name}-{comb}' if '_' in method else method
            top1, top5 = result_log(logits, labels)

            encoded_prompts = zeroshot_classifier(model,class_names, worst_prompts, stack_emb=True, device=device).T
            logits, labels = compute_logits(model,encoded_prompts,test_dataloader, device=device)
            w_top1, w_top5 = result_log(logits, labels)

            print(f"Results for top and worst {k_prompts} Prompts:")
            print(f"  - Top-1 Accuracy: {top1:.2f}%")
            print(f"  - Top-5 Accuracy: {top5:.2f}%")
            print(f"  - Worst Top-1 Accuracy: {w_top1:.2f}%")
            print(f"  - Worst Top-5 Accuracy: {w_top5:.2f}%")

    
if __name__ == '__main__':
    args = get_args()
    main(args)
                
        