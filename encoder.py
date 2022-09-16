from typing import Optional
import torch
from torch import nn
from .attention import CrossAttention, SelfAttention

class PerceiverEncoder(nn.Module):
    """
    Encoder stack.
    Has a trainable latent representation tensor and refines this using cross
    attention on the input data, then further refines it using stacked
    Transformer blocks.
    """
    def __init__(
        self,
        num_latents: int,
        latent_dim: int,
        input_dim: int,
        num_self_attn_per_block: int = 4,
        num_blocks: int = 16,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        num_cross_attn_heads: int = 4,
        num_self_attn_heads: int = 16,
        cross_attn_wide_factor: int = 1,
        self_attn_wide_factor: int = 1,
        use_query_residual: bool = True,
    ):
        """
        Arguments:
            num_latents: Number of latent vectors.
            latent_dim: Dimension of latent vector.
            input_dim: Dimension of input tensor.
            num_self_attn_per_block: Number of self-attention modules per
                transformer block. Defaults to 2.
            num_blocks: Number of transformer blocks. Defaults to 4.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            num_cross_attn_heads: Number of cross-attention heads.
                Defaults to 1.
            num_self_attn_heads: Number of self-attention heads.
                Defaults to 8.
            cross_attn_widening_factor: Widening factor in cross-attention
                feed-forward layer. Defaults to 1.
            self_attn_widening_factor: Widening factor in self-attention
                feed-forward layer. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
        """
        super().__init__()
        self.num_blocks = num_blocks
        # Init the latent representation
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # Make cross attention layer
        self.cross_attn = CrossAttention(
            n_heads=num_cross_attn_heads,
            kv_dim=input_dim,
            q_dim=latent_dim,
            wide_factor=cross_attn_wide_factor,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual,
            )
        # Make successive self attention layers
        self.self_attention_block = nn.ModuleList([
                SelfAttention(
                    hidden_dim=latent_dim,
                    wide_factor=self_attn_wide_factor,
                    n_heads=num_self_attn_heads,
                    qk_out_dim=qk_out_dim,
                    v_out_dim=v_out_dim
            ) for _ in range(num_self_attn_per_block) ]
        )


    def forward(self, x: torch.Tensor):
        """
        Takes input tensor and returns a latent tensor
        """
        batch_size = x.size(0)
        # Match latent size to batch size
        queries = self.latents.repeat(batch_size,1,1)
        # Perform cross-attention with the input data
        latents = self.cross_attn(
            inputs_kv=x,
            inputs_q=queries
        )
        # Perform self-attention
        for _ in range(self.num_blocks):
            for self_attn_layer in self.self_attention_block:
                latents = self_attn_layer(latents)
        return latents





