from abc import ABCMeta, abstractmethod
from typing import Optional
import torch
from torch import nn
from .attention import CrossAttention
"""
Here we define the decoder classes.
We begin with an abstract base class that is then specialised depending on the
use case.
As the Perceiver is a general use network, this is the main specialisation.
"""


class BaseDecoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract Decoder Class
    """
    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                latents: torch.Tensor):
        return NotImplementedError


class ProjectionDecoder(BaseDecoder):
    """
    This decoder doesn't use any cross attention.
    Probably not very useful for our usage.
    """
    def __init__(self,
                 latent_dim: int,
                 n_classes: int
                 ):
        super().__init__()
        self.projection = nn.Linear(latent_dim, n_classes)

    def forward(self,
                query: torch.Tensor,
                latents: torch.Tensor,
                ):
        latents = latents.mean(dim=1)
        logits = self.projection(latents)
        return logits


class PerceiverDecoder(BaseDecoder):
    """
    Basic decoder using a cross-attention layer.
    """
    def __init__(self,
                 latent_dim: int,
                 query_dim: int,
                 wide_factor: int = 1,
                 n_heads: int = 8,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 projection_dim: Optional[int] = None,
                 use_query_residual: bool = False
                 ):
        super().__init__()
        self.cross_attention = CrossAttention(
                n_heads=n_heads,
                kv_dim=latent_dim,
                q_dim=query_dim,
                wide_factor=wide_factor,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim,
                use_query_residual=use_query_residual
        )
        if projection_dim is not None:
            self.projection = nn.Linear(query_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(self,
                query: torch.Tensor,
                latents: torch.Tensor
                ):
        outputs = self.cross_attention(
            inputs_kv=latents,
            inputs_q=query
        )
        return self.projection(outputs)
