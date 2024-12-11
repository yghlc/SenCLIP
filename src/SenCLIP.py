import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Tuple, Union, Literal
from tqdm import tqdm
import clip
from attention_pool import AttentionPoolPerDimension, AttentionPoolPerImage


class CLIPWrapper(nn.Module):
    """
    Wrapper around the CLIP model to handle vision-based feature extraction.

    Args:
        vision_layers (tuple): Layers configuration for the visual encoder.
        embed_dim (int): Embedding dimension of the output.
        vision_heads (int): Number of heads in the vision transformer.
        image_resolution (int): Input image resolution.
        vision_width (int): Width of the vision layers.
        vision_patch_size (int, optional): Patch size for vision transformer. Default: None.
        architecture (str): Architecture name (e.g., "RN50"). Default: "RN50".
        device (str): Device to load the model on. Default: "cuda".
    """
    def __init__(self, trainable_layers=["visual"], architecture="RN50", device="cuda"):
        super(CLIPWrapper, self).__init__()
        self.architecture = architecture
        self.trainable_layers = trainable_layers
        model, _ = clip.load(self.architecture, device=device)
        self.model = model.float().train()

    @property
    def dtype(self):
        """Returns the data type of the visual encoder weights."""
        return self.model.visual.conv1.weight.dtype

    def enable_gradients(self):

        """Enable gradients for specified layers."""
        for name, param in self.model.named_parameters():
            param.requires_grad = any(name.startswith(layer) for layer in self.trainable_layers)

    def get_layer_gradients(self):
        """Retrieve gradients for layers where gradients are enabled."""
        return {
            name: param.grad.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encodes an image into its corresponding feature representation."""
        return self.model.encode_image(image.type(self.dtype))


class ProjectionHead(nn.Module):
    """
    Projection head to map embeddings into a new space with residual connections.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        dropout (float): Dropout rate. Default: 0.1.
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights and biases to zero."""
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, orig: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and normalization.
        """
        projected = self.projection(orig)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected + orig
        x = self.layer_norm(x)
        return x


class SenCLIP(nn.Module):
    """
    Main class implementing the SenCLIP model with contrastive loss and pooling layers.

    Args:
        Various configurations for the model, pooling, and contrastive loss.
    """
    def __init__(
        self,
        embed_dim=1024,
        architecture="RN50",
        trainable_layers = ["visual"],
        pooling='avgpool',
        device=0,
        pool_out='mean',
        queue_size=256,
        queue_data=None,
    ):
        super().__init__()
    
        self.pooling = pooling
        self.device = f'cuda:{device}'
        self.queue_size = queue_size


        self.encoder = CLIPWrapper(trainable_layers, architecture, device=self.device)
        self.encoder.enable_gradients()

        # Pooling Layer Selection
        self.pooling_layer = self._initialize_pooling(pooling, embed_dim, pool_out)

        # Projection Head
        self.projection_head = ProjectionHead(embed_dim)

        # Queue Initialization
        self.register_buffer("queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_data = queue_data

    def _initialize_pooling(self, pooling: str, embed_dim: int, pool_out: str):
        if pooling == "attpool_perimage":
            print("Using AttentionPoolPerImage.")
            return AttentionPoolPerImage(embed_dim, pool_out)
        elif pooling == "attpool_perdim":
            print("Using AttentionPoolPerDimension.")
            return AttentionPoolPerDimension(embed_dim, pool_out)
        else:
            print("Using Adaptive Average Pool.")
            return nn.AdaptiveAvgPool1d(1)

    @torch.no_grad()
    def fill_queue(self):
        """Fill the contrastive queue with initial embeddings."""
        print("Filling queue with initial values.")
        for i, batch in tqdm(enumerate(self.queue_data)):
            y_emb = batch.to(self.device)
            frozen_ground_embedding = self.pooling_layer(y_emb).squeeze(-1)
            self._dequeue_and_enqueue(frozen_ground_embedding)
            if i >= self.queue_size:
                break
        print("Queue fill successful.")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Dequeue old keys and enqueue new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, "Batch size must divide queue size."

        # Replace the keys at the current pointer
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor]]):
        """Compute the logits and contrastive loss labels."""
        y_emb, x_img = data[:2]
        y_emb, x_img = y_emb.to(self.device), x_img.to(self.device)

        if self.pooling != 'avgpool':
            frozen_ground_embedding = self.pooling_layer(y_emb).squeeze(-1)
        else:
            with torch.no_grad():
                frozen_ground_embedding = self.pooling_layer(y_emb.transpose(1,2)).transpose(1,2).squeeze(1).clone().detach()
                
        positive_embedding = self.encoder(x_img)
        positive_embedding = self.projection_head(positive_embedding)

        # Normalize embeddings
        frozen_ground_embedding = F.normalize(frozen_ground_embedding, p=2, dim=-1)
        positive_embedding = F.normalize(positive_embedding, p=2, dim=-1)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [positive_embedding, frozen_ground_embedding]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [positive_embedding, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Enqueue embeddings
        self._dequeue_and_enqueue(frozen_ground_embedding)

        return logits, labels
