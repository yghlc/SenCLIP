import torch, open_clip, torchvision
from tqdm import tqdm
import clip, cv2
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import logging, random, os
from argparse import ArgumentParser, RawTextHelpFormatter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchgeo import datasets
import torch.nn.functional as F
from dataset_bigearthnet import BigEarthNetDataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
def get_args():
    parser = ArgumentParser(description="Zero-shot classification with SenCLIP on BigEarthNet dataset",
                            formatter_class=RawTextHelpFormatter)
 
    parser.add_argument('--ckpt_path', type=str, default='./SenCLIP_AvgPool_ViTB32.ckpt', help="Path to model checkpoint")    
    parser.add_argument('--model_arch', type=str, default='ViT-B/32')
    parser.add_argument('--template_path', type=str, default='./prompts/prompt_ben_ground_mixed.json', help="prompt_template_path")
    parser.add_argument('--version', type=str, default='BingCLIP', help="The version of the model, e.g., 'BingCLIP' or 'RemoteCLIP'")
    parser.add_argument('--download', action='store_true',help="Set this flag to download the dataset if not present")
    parser.add_argument('--mean', type=list, default=(0.347, 0.376, 0.296), help="Normalization mean for images")
    parser.add_argument('--std', type=list, default=(0.269, 0.261, 0.276), help="Normalization std for images")


    return parser.parse_args()

def zeroshot_classifier(model, classnames, templates, device):
    temp_context = 'class_context' if isinstance(templates, dict) else 'unified'
    with torch.no_grad():
        zeroshot_weights = []
        for i, classname in tqdm(enumerate(classnames), desc="Encoding class prompts"):
            if temp_context == "class_context":
                texts = templates[classname] if isinstance(templates, dict) else [templates.format(classname)]
            else:
                texts = [template.format(classname) for template in templates] #format with class
 
            texts = torch.cat([clip.tokenize(text) for text in texts]).to(device=device)#tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device=device)
    return zeroshot_weights
def transform_img(mean, std):
   
    transform=torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean, std)
                ])

    return transform
def accuracy(output, target, topk=(1,)):
    _, pred = output.topk(max(topk), 1, True, True)
    target = target.argmax(dim=1, keepdim=True).view(-1, 1)

    correct = pred.eq(target)
    acc_list = [correct[:, :k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy() / target.size(0) for k in topk]
    
    return acc_list

def load_model(weight_path, model_name, device):
    """
    Loads the CLIP model with specified weights.
    """
    model, _ = clip.load(model_name, device=device)
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = '/home/hlc/Data/public_data_AI/BigEarthNet-v1.0'

    dataset = BigEarthNetDataset(dataset_dir, split='test',num_classes=19, norm_value = args.version, download=args.download)
    total_length = len(dataset)
    train_len = total_length-27000
    test_dataset, _ = torch.utils.data.random_split(dataset, [27000, train_len])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=True, num_workers=8,
                                            pin_memory=True, persistent_workers=True, drop_last=True)
    #convert class labels to queries/prompts
    class_names =  dataset.class_lists

    transform = transform_img(args.mean, args.std)
    # Load model
    model = load_model(args.ckpt_path, args.model_arch, device)

    templates = json.load(open(args.template_path))
    encoded_prompts = zeroshot_classifier(model, class_names, templates, device).T
    print(encoded_prompts.shape)

   #predict zero shot labels
    labels_list = []
    logits_list = []
  
    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            images= transform(images).to(device=device)
            labels = labels.to(device=device)
           
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features @ encoded_prompts)/2+1 #.softmax(dim=-1)#.to(device=device)#zeroshot_weights  #
            
            logits_list.append(logits)
            labels_list.append(labels)
           

    logits_stk = torch.cat(logits_list)
    labels_stk = torch.cat(labels_list)

    # print(logits_stk, labels_stk)
    label_count = labels_stk.sum(dim=-1)
    total_labels = label_count.sum(dim=0)
    print(label_count, total_labels)
   
    top1 = 0.
    # Measure accuracy
    for j, k in enumerate(label_count):
        _, indices = logits_stk[j].topk(k, dim=-1)
        # print(k, labels_stk[j])
        _, labels_idx = labels_stk[j].topk(k, dim=-1)
        # print(k, labels_idx)
        top_k = 0.
        for idx in labels_idx:
            if idx in indices:
                top_k += 1

        top1 += top_k   
    top1 = (top1/total_labels) * 100 #/ label_count.size(0)


    print(f"Top-1 Accuracy: {top1}")
  
if __name__=='__main__':
    set_seed(42)
    args = get_args()
    main(args)
    
    
    
   