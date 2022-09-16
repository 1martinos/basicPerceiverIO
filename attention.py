from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Main attention module.
    This is pretty much the same as the GitHub repo [1].
    It is a general class which will then be specialised for the perceiver use
    cases.
    Refer to this image for a more intuitive description:
      - t.ly/D16O
    Skipping over implementing dropout and masking for now.
    """
    def __init__(self,
                 n_heads: int,      # Number of parallel attention heads
                 kv_dim: int,       # Number of input features
                 q_dim: int,        # Dimension of the keys
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 ):
        """
        Arguments:
            n_heads: Number of attention heads to use.
            kv_dim: Size of input key and value vectors.
            q_dim: Size of input query vector.
            The rest are self-explanatory optional output dimensions.
            ...
        """
        # Call nn.module init
        super().__init__()
        # Init optional values if not set
        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        self.n_heads = n_heads
        self.qk_head_dim = qk_out_dim // n_heads
        self.v_head_dim = v_out_dim // n_heads

        # Create linear layers for attention and projection
        self.k = nn.Linear(kv_dim, qk_out_dim)
        self.q = nn.Linear(q_dim, qk_out_dim)
        self.v = nn.Linear(kv_dim, v_out_dim)
        self.projection = nn.Linear(v_out_dim, output_dim)
        self.scale = math.sqrt(self.qk_head_dim)

    def forward(self,
                inputs_kv: torch.Tensor,
                inputs_q: torch.Tensor,
                ):
        """
        Arguments:
            inputs_kv: Key-Value embedding of dims (B, M, C)
            inputs_q: Query embeddings of dims (B, N, D)
        Returns:
            Tensor of shape (B, N, M)
        """
        # Embed Inputs
        k, q, v = self.k(inputs_kv), self.q(inputs_q), self.v(inputs_kv)
        k = rearrange(k, "b s (n h) -> b n s h", h=self.qk_head_dim)
        q = rearrange(q, "b s (n h) -> b n s h", h=self.qk_head_dim)
        v = rearrange(v, "b s (n h) -> b n s h", h=self.v_head_dim)

        # Calculate attention scores
        attention = (q @ k.transpose(-2, -1) * self.scale)
        # Softmax the scores
        attention = attention.softmax(dim=-1)
        # Calculate final QK @ V value
        weighted = attention @ v
        weighted = rearrange(weighted, "b n s h -> b s (n h)")
        return self.projection(weighted)


class FeedForward(nn.Module):
    """
    Transformer Feed-Forward network
    """
    def __init__(self,
                 dim: int,
                 wide_factor: int = 4,
                 ):
        """
        Arguments:
            dim: Dimension of input tensor.
            wide_factor: Widening factor, defualt = 4
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * wide_factor),
            nn.GELU(),
            nn.Linear(dim * wide_factor, dim)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class SelfAttention(nn.Module):
    """
    Implementation of SelfAttention, we use this on the latent representation
    to refine the internal model.
    This can be seen as just combining the MultiHead Attention and feedforward
    above for our uses.
    """
    def __init__(self,
                 hidden_dim: int,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 wide_factor: int = 4,
                 n_heads: int = 1,
                 ):
        """
        Arguments:
            hidden_dim: Dimension of input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(
            n_heads=n_heads,
            kv_dim=hidden_dim,
            q_dim=hidden_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=hidden_dim,
            )
        self.mlp = FeedForward(hidden_dim, wide_factor)

    def forward(self, x: torch.Tensor):
        """
        See how this layer attenuates to itself!
        Thus scales n**2 and is the usual bottleneck in standard transformer
        architectures.
        Here it will be used on the latent representation thus avoiding the
        usual size limits.
        Arguments:
            x: Input tensor of shape (B, M, C).
        """
        x_norm = self.layer_norm(x)
        attention = self.attention(
            inputs_kv=x_norm,
            inputs_q=x_norm
        )
        x = x + attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention module.
    Will be used to map input sequences into questions we ask the latent space
    to represent!
    """
    def __init__(self,
                 n_heads: int,
                 kv_dim: int,
                 q_dim: int,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 wide_factor: int = 1,
                 use_query_residual: bool = True,
                 ):
        """
        Arguments:
            kv_dim: Dimension of key/value input tensor.
            q_dim: Dimension of query input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 1.
            num_heads: Number of attention heads. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
        """
        super().__init__()
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(
            n_heads=n_heads,
            kv_dim=kv_dim,
            q_dim=q_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=q_dim
        )
        self.mlp = FeedForward(q_dim, wide_factor)
        self.use_query_residual = use_query_residual

    def forward(self,
                inputs_kv: torch.Tensor,
                inputs_q: torch.Tensor,
                ):
        """
        Arguments:
            inputs_kv: Key-Value embedding of shape (B, M, C)
            inputs_q: Query embedding of shape (B, N, D)
        """
        attention = self.attention(
                inputs_kv=self.kv_layer_norm(inputs_kv),
                inputs_q=self.q_layer_norm(inputs_q)
            )
        if self.use_query_residual:
            x = inputs_q + attention
        else:
            x = attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Implementation of scaled dot product attention.
    Scales the dot product by the dimension of the queries as per
    "Attention is All you Need" (Read this paper!)
    This has almost the same function as the above MultiHeadAttention forward
    pass.
    This is not used in the model anymore, but is included as reference
    to a simpler attention mechanism with less boilerplate code!
    """
    def forward(self,
                query,      # Queries
                key,        # Keys
                values):    # Values
        """
        Brief Explanation:
            - Attention allows a network to attenuate to different parts of an
              input depending on whats useful, and takes the form of a set of
              weights stored in a "attention matrix".

            - In self-attention, Q and K and V are all the same: it simply
              learns to attenuate to different parts of the current
              representation.

            - In cross-attention, Q and K are a different thing altogether to
              V: It learns how to attenuate the current representation based
              off another.

            - The Perceiver has a latent space representation that is formed by
              using cross-attention from the latent one to the input data.
              This is then refined using self-attention like a standard
              transformer.

            - This is useful because in a usual transformer model the attention
              requires N**2 computations (each input must attend to all other
              inputs). By only working with the latent representation for self
              attention we instead scale O(MN + LN**2), where M in the input
              data size, N is the latent dimension (N << M), and L is the
              number of transformer layers.

            - Read the original perceiver paper for more info:
                https://arxiv.org/pdf/2103.03206.pdf
        """
        scale_factor = math.sqrt(query.size()[-1])
        scores = query.matmul(key.transpose(-2, -1))
        scores = scores / scale_factor
        attention = F.softmax(scores, dim=-1)
        return attention @ values
