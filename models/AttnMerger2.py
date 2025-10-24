import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedMultiHeadAttention(nn.Module):
    """統一的Multi-Head Attention模組，可用於Self和Cross Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # 獨立的heads
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key_value=None):
        """
        統一的forward介面
        Args:
            query: [B, C, H, W] - Query tensor
            key_value: [B, C, H, W] or None - Key/Value tensor
                      如果是None，則為self-attention (key_value = query)
        Returns:
            out: [B, C, H, W]
        """
        # Self-attention: key_value預設為query
        if key_value is None:
            key_value = query

        B, C, H, W = query.shape
        N = H * W

        # Flatten
        query_flat = query.flatten(2).transpose(1, 2)      # [B, N, C]
        key_value_flat = key_value.flatten(2).transpose(1, 2)  # [B, N, C]

        # Multi-head attention
        head_outputs = []
        for i in range(self.num_heads):
            q = self.q_projs[i](query_flat)      # [B, N, head_dim]
            k = self.k_projs[i](key_value_flat)  # [B, N, head_dim]
            v = self.v_projs[i](key_value_flat)  # [B, N, head_dim]

            # Scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)

            head_out = attn @ v
            head_outputs.append(head_out)

        # Concatenate and project
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class UnifiedAttentionBlock(nn.Module):
    """統一的Attention Block with MLP"""
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        # Attention部分
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)  # 如果key_value不同，需要分別norm
        self.attn = UnifiedMultiHeadAttention(dim, num_heads)

        # MLP部分
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, query, key_value=None):
        """
        Args:
            query: [B, C, H, W]
            key_value: [B, C, H, W] or None
        """
        B, C, H, W = query.shape

        # === Attention Block ===
        # Flatten for norm
        query_flat = query.flatten(2).transpose(1, 2)
        query_normed = self.norm1(query_flat).transpose(1, 2).reshape(B, C, H, W)

        if key_value is None:
            # Self-attention
            attended = self.attn(query_normed)
        else:
            # Cross-attention - normalize key_value separately
            key_value_flat = key_value.flatten(2).transpose(1, 2)
            key_value_normed = self.norm2(key_value_flat).transpose(1, 2).reshape(B, C, H, W)
            attended = self.attn(query_normed, key_value_normed)

        # Residual connection
        x = query + attended

        # === MLP Block ===
        # Flatten for norm and MLP
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_normed = self.norm3(x_flat)
        x_mlp = self.mlp(x_normed)
        # Residual connection
        x_flat = x_flat + x_mlp
        # Reshape back
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        return x


class HierarchicalCrossAttentionFusion(nn.Module):
    """
    Hierarchical Cross-Attention Fusion
    Stage 1: Extract self-attention and cross-attention features separately
    Stage 2: Cross-attention between cross-features and self-features (dim*2)
    Stage 3: Multi-scale large kernel fusion
    """
    def __init__(self, dim=256, num_heads=4):  # 減少heads
        super().__init__()

        # Stage 1: 簡化 - 只做一次
        self.self_attn1 = UnifiedAttentionBlock(dim, num_heads)
        self.self_attn2 = UnifiedAttentionBlock(dim, num_heads)
        self.cross_1to2 = UnifiedAttentionBlock(dim, num_heads)
        self.cross_2to1 = UnifiedAttentionBlock(dim, num_heads)

        # Stage 2: 減少深度
        self.fusion_cross_attn = UnifiedAttentionBlock(dim * 2, num_heads)

        # Stage 3: 保持multi-scale但簡化
        self.multiscale_fusion = nn.ModuleDict({
            'branch_small': nn.Conv2d(dim * 2, dim // 2, 3, padding=1, groups=dim//2),
            'branch_large': nn.Conv2d(dim * 2, dim // 2, 7, padding=3, groups=dim//2),
            'fusion': nn.Conv2d(dim, dim, 1)
        })

    def forward(self, feat1, feat2):
        # Stage 1
        self1 = self.self_attn1(feat1)
        self2 = self.self_attn2(feat2)
        cross_1to2 = self.cross_1to2(feat1, feat2)
        cross_2to1 = self.cross_2to1(feat2, feat1)

        # Concatenate
        cross_concat = torch.cat([cross_1to2, cross_2to1], dim=1)
        self_concat = torch.cat([self1, self2], dim=1)

        # Stage 2: Single fusion attention
        fused = self.fusion_cross_attn(cross_concat, self_concat)

        # Stage 3: Simplified multi-scale
        small = self.multiscale_fusion['branch_small'](fused)
        large = self.multiscale_fusion['branch_large'](fused)
        out = self.multiscale_fusion['fusion'](torch.cat([small, large], dim=1))

        return out