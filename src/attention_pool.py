import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal


class AttentionPoolPerImage(nn.Module):
    """
    Attention pooling mechanism that generates a single vector for each image by
    computing attention scores across embeddings.

    Args:
        input_dim (int): The dimensionality of the input embeddings.
        out (str): Aggregation method to combine weighted embeddings. 
                   Options: 'mean', 'sum', 'max'. Default: 'mean'.
    """
    def __init__(self, input_dim: int, out: Literal['mean', 'sum', 'max'] = 'mean'):
        super(AttentionPoolPerImage, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.out = out.lower()
        self._validate_out()

    def _validate_out(self):
        if self.out not in {'mean', 'sum', 'max'}:
            raise ValueError(f"Invalid value for `out`: {self.out}. Must be 'mean', 'sum', or 'max'.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AttentionPoolPerImage.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Aggregated output of shape (batch_size, input_dim).
        """
        attention_scores = self.linear(embeddings).squeeze(-1)  # Shape: (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Shape: (batch_size, seq_len)
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)  # Shape: (batch_size, seq_len, input_dim)

        return self._aggregate(weighted_embeddings)

    def _aggregate(self, weighted_embeddings: torch.Tensor) -> torch.Tensor:
        if self.out == 'max':
            return torch.max(weighted_embeddings, dim=1).values
        elif self.out == 'sum':
            return torch.sum(weighted_embeddings, dim=1)
        else:  # Default to 'mean'
            return torch.mean(weighted_embeddings, dim=1)


class AttentionPoolPerDimension(nn.Module):
    """
    Attention pooling mechanism that generates an output vector for each dimension
    by computing attention scores across sequences.

    Args:
        input_dim (int): The dimensionality of the input embeddings.
        out (str): Aggregation method to combine weighted embeddings.
                   Options: 'mean', 'sum', 'max'. Default: 'mean'.
    """
    def __init__(self, input_dim: int, out: Literal['mean', 'sum', 'max'] = 'mean'):
        super(AttentionPoolPerDimension, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
        self.out = out.lower()
        self._validate_out()

    def _validate_out(self):
        if self.out not in {'mean', 'sum', 'max'}:
            raise ValueError(f"Invalid value for `out`: {self.out}. Must be 'mean', 'sum', or 'max'.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AttentionPoolPerDimension.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Aggregated output of shape (batch_size, input_dim).
        """
        attention_scores = self.linear(embeddings)  # Shape: (batch_size, seq_len, input_dim)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_len, input_dim)
        weighted_embeddings = embeddings * attention_weights  # Element-wise multiplication

        return self._aggregate(weighted_embeddings)

    def _aggregate(self, weighted_embeddings: torch.Tensor) -> torch.Tensor:
        if self.out == 'max':
            return torch.max(weighted_embeddings, dim=1).values
        elif self.out == 'sum':
            return torch.sum(weighted_embeddings, dim=1)
        else:  # Default to 'mean'
            return torch.mean(weighted_embeddings, dim=1)
