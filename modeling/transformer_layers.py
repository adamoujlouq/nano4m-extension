# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Some functions are based on the timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m
# --------------------------------------------------------

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of hidden dimensions for rotary position embeddings."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.LongTensor,
        base: float = 10000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors of shape [B, H, L, D]."""
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}.")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.to(device=q.device, dtype=torch.float32)[..., None] * inv_freq
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=q.dtype)[:, None, :, :]
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=q.dtype)[:, None, :, :]

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class PerHeadLayerNorm(nn.Module):
    """LayerNorm over head_dim with separate affine parameters for each attention head."""
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (head_dim,)
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.register_buffer("bias", torch.zeros(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, self.normalized_shape, eps=self.eps)
        return x * self.weight[None, :, None, :] + self.bias[None, :, None, :]


class Mlp(nn.Module):
    """
    MLP module with GELU activation.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block as a drop-in replacement for Mlp.

    Uses the gating mechanism: output = W3(SiLU(W1(x)) * W2(x))

    To maintain roughly the same parameter count as Mlp with the same mlp_ratio,
    Block and DecoderBlock automatically scale hidden_features by 2/3 when
    use_swiglu=True (since SwiGLU has two input projections instead of one).

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # gate projection
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)  # value projection
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(
            self,
            dim: int,
            head_dim: int = 64,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            use_qk_norm: bool = False,
        ):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.use_qk_norm = use_qk_norm

        # Single projection for Q, K, V (3x the dimension)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        if use_qk_norm:
            self.q_norm = PerHeadLayerNorm(self.num_heads, head_dim, bias=qkv_bias)
            self.k_norm = PerHeadLayerNorm(self.num_heads, head_dim, bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            positions: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        B, L, D = x.shape # Batch size, sequence length, and dimension

        # Compute Q, K, V and reshape to [B, num_heads, L, head_dim]
        qkv = self.qkv(x)  # [B, L, 3*D]
        qkv = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, L, head_dim]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            if positions is None:
                raise ValueError("RoPE self-attention requires token positions.")
            q, k = apply_rope(q, k, positions=positions, base=self.rope_base)

        # Attention matrix: [B, num_heads, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)

        # Weight values and reshape back to [B, L, D]
        x = (attn @ v)  # [B, num_heads, L, head_dim]
        x = rearrange(x, "b h l d -> b l (h d)")

        # Output projection
        x = self.attn_out_proj(x)
        return x


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(
            self,
            dim: int,
            head_dim: int = 64,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            use_qk_norm: bool = False,
        ):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.use_qk_norm = use_qk_norm

        # Q from input x
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # K and V from context (single projection for both)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        if use_qk_norm:
            self.q_norm = PerHeadLayerNorm(self.num_heads, head_dim, bias=qkv_bias)
            self.k_norm = PerHeadLayerNorm(self.num_heads, head_dim, bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape # Batch size, x sequence length (N), and dimension
        _, M, _ = context.shape # _, context sequence length (M), _

        # Q from x: [B, num_heads, N, head_dim]
        q = self.q(x)  # [B, N, D]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)

        # K, V from context: [B, num_heads, M, head_dim]
        kv = self.kv(context)  # [B, M, 2*D]
        kv = rearrange(kv, "b m (two h d) -> two b h m d", two=2, h=self.num_heads)
        k, v = kv.unbind(0)  # Each: [B, num_heads, M, head_dim]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Attention matrix: [B, num_heads, N, M]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)

        # Weight values and reshape back to [B, N, D]
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        x = rearrange(x, "b h n d -> b n (h d)")
        
        # Output projection
        x = self.attn_out_proj(x)

        return x


class Block(nn.Module):
    """
    Basic transformer block with a multi-head self-attention mechanism and a feed-forward MLP.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        use_swiglu: If True, use SwiGLU instead of Mlp. Hidden dim is scaled by 2/3 to keep
            parameter count approximately equal to the standard Mlp.
    """
    def __init__(
            self,
            dim: int,
            head_dim: int = 64,
            mlp_ratio: float = 4.,
            use_bias: bool = False,
            use_swiglu: bool = False,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            use_qk_norm: bool = False,
        ):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=use_bias)
        self.attn = Attention(
            dim,
            head_dim=head_dim,
            qkv_bias=use_bias,
            proj_bias=use_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = LayerNorm(dim, bias=use_bias)
        # Scale hidden dim by 2/3 for SwiGLU to keep param count ~equal (two input projections vs one)
        mlp_hidden_dim = int(dim * mlp_ratio * (2 / 3 if use_swiglu else 1))
        self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, bias=use_bias) if use_swiglu \
            else Mlp(in_features=dim, hidden_features=mlp_hidden_dim, bias=use_bias)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            positions: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, positions=positions)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """
    Basic transformer decoder block with a multi-head self-attention,
    a multi-head cross-attention, and a feed-forward MLP layer.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        use_swiglu: If True, use SwiGLU instead of Mlp. Hidden dim is scaled by 2/3 to keep
            parameter count approximately equal to the standard Mlp.
    """
    def __init__(
            self,
            dim: int,
            head_dim: int = 64,
            mlp_ratio: float = 4.,
            use_bias: bool = False,
            use_swiglu: bool = False,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            use_qk_norm: bool = False,
        ):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=use_bias)
        self.query_norm = LayerNorm(dim, bias=use_bias)
        self.context_norm = LayerNorm(dim, bias=use_bias)
        self.norm2 = LayerNorm(dim, bias=use_bias)

        self.self_attn = Attention(
            dim,
            head_dim=head_dim,
            qkv_bias=use_bias,
            proj_bias=use_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
        )
        self.cross_attn = CrossAttention(
            dim,
            head_dim=head_dim,
            qkv_bias=use_bias,
            proj_bias=use_bias,
            use_qk_norm=use_qk_norm,
        )

        # Scale hidden dim by 2/3 for SwiGLU to keep param count ~equal (two input projections vs one)
        mlp_hidden_dim = int(dim * mlp_ratio * (2 / 3 if use_swiglu else 1))
        self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, bias=use_bias) if use_swiglu \
            else Mlp(in_features=dim, hidden_features=mlp_hidden_dim, bias=use_bias)

    def forward(self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
            self_positions: Optional[torch.LongTensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            context_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), mask=sa_mask, positions=self_positions)
        # Cross-attention with residual
        cross_query = x + query_pos if query_pos is not None else x
        cross_context = context + context_pos if context_pos is not None else context
        x = x + self.cross_attn(self.query_norm(cross_query), self.context_norm(cross_context), mask=xa_mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerTrunk(nn.Module):
    """Basic Transformer trunk definition that can be used for encoder-only,
    decoder-only and prefixLM models, depending on the attention mask applied.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        use_swiglu: If True, use SwiGLU instead of Mlp in each block.
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
            use_swiglu: bool = False,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            use_qk_norm: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                use_bias=use_bias,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                rope_base=rope_base,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])
    
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            positions: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask, positions=positions)
        return x


class TransformerDecoderTrunk(nn.Module):
    """Basic Transformer decoder with interleaved self- and cross-attention, that can
    be used as the decoder for encoder-decoder models.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        use_swiglu: If True, use SwiGLU instead of Mlp in each block.
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
            use_swiglu: bool = False,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            use_qk_norm: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                use_bias=use_bias,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                rope_base=rope_base,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])
    
    def forward(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
            self_positions: Optional[torch.LongTensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            context_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(
                x,
                context,
                sa_mask=sa_mask,
                xa_mask=xa_mask,
                self_positions=self_positions,
                query_pos=query_pos,
                context_pos=context_pos,
            )
        return x
