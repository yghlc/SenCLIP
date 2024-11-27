import numpy as np
from sklearn.metrics import r2_score
import torch

def prompt_score_conc_level(similarity_scores, topK=1000, hhi_calc=True):
    """
    Calculate concentration scores or Herfindahl-Hirschman Index (HHI) for prompts.

    Args:
    - similarity_scores (torch.Tensor): Tensor of shape (num_prompts, num_images).
    - topK (int): Number of top images to consider for each prompt.
    - hhi_calc (bool): Whether to calculate HHI or concentration scores.

    Returns:
    - torch.Tensor: HHI or concentration scores, shape (num_prompts,).
    """
    print('Calculating HHI...')
    
    # Step 1: Normalize scores and compute contributions
    total_sum = similarity_scores.sum(dim=1, keepdim=True)  # Shape: (num_prompts, 1)
    img_perc_contrib = similarity_scores / (total_sum + 1e-6)  # Avoid division by zero

    # Step 2: Extract top K contributions
    topK_indices = img_perc_contrib.topk(topK, dim=1).indices  # Shape: (num_prompts, topK)
    prompt_conc_score = img_perc_contrib.gather(1, topK_indices).sum(dim=1)  # Shape: (num_prompts,)

    if hhi_calc:
        # Step 3: Compute HHI
        prompt_hhi = (img_perc_contrib.gather(1, topK_indices) ** 2).sum(dim=1)  # Shape: (num_prompts,)
        return prompt_hhi

    return prompt_conc_score


def prompt_score_tfidf(similarity_scores, topK=100):
    """
    Calculate TF-IDF-like scores for prompts based on image similarity scores.

    Args:
    - similarity_scores (torch.Tensor): Tensor of shape (num_prompts, num_images).
    - topK (int): Number of top images to consider for each prompt.

    Returns:
    - torch.Tensor: TF-IDF-like scores, shape (num_prompts,).
    """
    print('Calculating TF-IDF...')
    
    # Step 1: Identify top K image indices
    topK_indices = similarity_scores.topk(topK, dim=1).indices  # Shape: (num_prompts, topK)

    # Step 2: Compute Term Frequency (TF)
    tf = similarity_scores.gather(1, topK_indices).mean(dim=1)  # Shape: (num_prompts,)

    # Step 3: Compute Document Frequency (DF) and Inverse Document Frequency (IDF)
    df = (similarity_scores > 0).float().sum(dim=1)  # Shape: (num_prompts,)
    idf = torch.log((similarity_scores.shape[1] + 1) / (df + 1))  # Shape: (num_prompts,)

    # Step 4: Compute TF-IDF
    prompt_scores = tf * idf
    return prompt_scores


def mean_calc(similarity_scores, topK=1000):
    """
    Calculate mean scores for the top K images for each prompt.

    Args:
    - similarity_scores (torch.Tensor): Tensor of shape (num_prompts, num_images).
    - topK (int): Number of top images to consider.

    Returns:
    - torch.Tensor: Normalized mean scores, shape (num_prompts,).
    """
    print('Calculating mean scores...')
    
    # Step 1: Extract top K values
    topK_values = similarity_scores.topk(topK, dim=1).values  # Shape: (num_prompts, topK)
    mean_topK = topK_values.mean(dim=1)  # Shape: (num_prompts,)

    # Step 2: Normalize scores
    total_similarity = similarity_scores.sum(dim=1)  # Shape: (num_prompts,)
    prompt_scores = mean_topK / (total_similarity + 1e-6)
    return prompt_scores


def prompt_sim_rank(prompt_sim_matrix, prompt_scores=None, num_per_class_prompt=50, 
                    num_classes=10, mode='mean', prompt_comb=None):
    """
    Rank prompts based on similarity and other criteria.

    Args:
    - prompt_sim_matrix (torch.Tensor): Prompt similarity matrix, shape (num_prompts, num_prompts).
    - prompt_scores (torch.Tensor, optional): Additional scores to combine.
    - num_per_class_prompt (int): Number of prompts per class.
    - num_classes (int): Total number of classes.
    - mode (str): Aggregation mode (e.g., 'mean').
    - prompt_comb (str, optional): Combination method ('mul' or 'sum').

    Returns:
    - torch.Tensor: Ranked and weighted prompt scores, shape (num_classes, num_per_class_prompt).
    """
    print('Ranking prompts...')
    
    prompt_sim_matrix = prompt_sim_matrix / (prompt_sim_matrix.sum(dim=0, keepdim=True) + 1e-6)
    sim_matrix_mul = prompt_sim_matrix @ prompt_sim_matrix.T

    mean_scores_all = sim_matrix_mul.mean(dim=1)  # Global mean scores
    classes = torch.arange(num_classes).repeat_interleave(num_per_class_prompt).long()
    
    reshaped_scores = sim_matrix_mul.view(num_classes, num_per_class_prompt, -1)
    mean_scores_class = reshaped_scores.mean(dim=1)[classes, torch.arange(classes.shape[0])]

    weighted_scores = mean_scores_class / (mean_scores_all + 1e-6)
    weighted_scores = weighted_scores.view(num_classes, num_per_class_prompt)

    if prompt_scores is not None:
        prompt_scores = prompt_scores / (prompt_scores.mean(dim=0, keepdim=True) + 1e-6)
        if prompt_comb == 'mul':
            weighted_scores *= prompt_scores
        elif prompt_comb == 'sum':
            weighted_scores += prompt_scores

    return weighted_scores


def prompt_selection_direct(similarity_scores_model, template, method='Mean', topK_images=1000, 
                            topK_prompts=15, prompt_sim_matrix=False, mode='mean', prompt_comb=None):
    """
    Select prompts directly based on similarity scores and scoring methods.

    Args:
    - similarity_scores_model (torch.Tensor): Model similarity scores, shape (num_images, num_prompts).
    - template (dict): Dictionary containing prompts for each class.
    - method (str): Scoring method ('Mean', 'TF-IDF', 'HHI').
    - topK_images (int): Number of top images to consider for scoring.
    - topK_prompts (int): Number of top prompts to select.
    - prompt_sim_matrix (bool): Whether to calculate similarity rankings.
    - mode (str): Mode for ranking aggregation.
    - prompt_comb (str): Combination method for scores ('mul' or 'sum').

    Returns:
    - tuple: (Selected prompts, Worst prompts) as dictionaries by class.
    """
    cls_list = list(template.keys())
    num_classes = len(cls_list)
    num_per_class_prompt = len(template[cls_list[0]])

    similarity_scores = similarity_scores_model.transpose(-1, 0).float()

    if method == 'Mean':
        prompt_scores = mean_calc(similarity_scores, topK=topK_images)
    elif method == 'TF-IDF':
        prompt_scores = prompt_score_tfidf(similarity_scores, topK=topK_images)
    elif method == 'HHI':
        prompt_scores = prompt_score_conc_level(similarity_scores, topK=topK_images, hhi_calc=True)
    else:
        raise ValueError("Invalid method specified.")

    prompt_scores_reshaped = prompt_scores.view(num_classes, num_per_class_prompt)

    if prompt_sim_matrix:
        prompt_scores_reshaped = prompt_sim_rank(
            similarity_scores, prompt_scores if prompt_comb in ['mul', 'sum'] else None,
            num_per_class_prompt=num_per_class_prompt, num_classes=num_classes,
            mode=mode, prompt_comb=prompt_comb
        )

    top_prompts_idx = prompt_scores_reshaped.topk(topK_prompts, dim=1).indices
    worst_prompts_idx = prompt_scores_reshaped.argsort(dim=1)[:, :topK_prompts]

    top_prompts = {cls: [template[cls][idx] for idx in top_prompts_idx[i]] for i, cls in enumerate(cls_list)}
    worst_prompts = {cls: [template[cls][idx] for idx in worst_prompts_idx[i]] for i, cls in enumerate(cls_list)}

    return top_prompts, worst_prompts
