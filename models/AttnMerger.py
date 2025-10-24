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
    def __init__(self, dim, num_heads=8, depth_stage1=1, depth_stage2=3):
        super().__init__()

        # Stage 1: Self and Cross attention (可以選擇重複次數)
        self.depth_stage1 = depth_stage1

        # Self-attention blocks
        self.self_attn1_blocks = nn.ModuleList([
            UnifiedAttentionBlock(dim, num_heads) for _ in range(depth_stage1)
        ])
        self.self_attn2_blocks = nn.ModuleList([
            UnifiedAttentionBlock(dim, num_heads) for _ in range(depth_stage1)
        ])

        # Cross-attention blocks
        self.cross_attn_1to2_blocks = nn.ModuleList([
            UnifiedAttentionBlock(dim, num_heads) for _ in range(depth_stage1)
        ])
        self.cross_attn_2to1_blocks = nn.ModuleList([
            UnifiedAttentionBlock(dim, num_heads) for _ in range(depth_stage1)
        ])

        # Stage 2: Cross-attention between concatenated features (dim*2)
        # 這個階段可能需要更深一點來學習複雜的融合關係
        self.fusion_cross_attn_blocks = nn.ModuleList([
            UnifiedAttentionBlock(dim * 2, num_heads) for _ in range(depth_stage2)
        ])

        # Stage 3: Multi-scale large kernel fusion (通常1層就夠)
        self.multiscale_fusion = MultiScaleLargeKernelFusion(dim * 2, dim, num_layers=1)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1, feat2: [B, C, H, W] - features from two exposure images
        Returns:
            fused: [B, C, H, W]
        """
        # Stage 1: Extract features with multiple blocks
        self1 = feat1
        self2 = feat2
        cross_1to2 = feat1
        cross_2to1 = feat2

        # Apply multiple layers in Stage 1
        for i in range(self.depth_stage1):
            self1 = self.self_attn1_blocks[i](self1)
            self2 = self.self_attn2_blocks[i](self2)
            cross_1to2 = self.cross_attn_1to2_blocks[i](cross_1to2, feat2)
            cross_2to1 = self.cross_attn_2to1_blocks[i](cross_2to1, feat1)

        # Concatenate features
        cross_concat = torch.cat([cross_1to2, cross_2to1], dim=1)  # [B, 2C, H, W]
        self_concat = torch.cat([self1, self2], dim=1)             # [B, 2C, H, W]

        # Stage 2: Multiple fusion cross-attention blocks
        fused_features = cross_concat
        for block in self.fusion_cross_attn_blocks:
            residual = fused_features
            fused_features = block(fused_features, self_concat) + residual

        # Stage 3: Multi-scale large kernel fusion
        fused = self.multiscale_fusion(fused_features)

        return fused


class MultiScaleLargeKernelFusion(nn.Module):
    """Multi-scale large kernel for final fusion"""
    def __init__(self, in_dim, out_dim, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        # 可以堆疊多層
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # 第一層做dimension reduction，後續層保持維度
            layer_in_dim = in_dim if i == 0 else out_dim

            layer = nn.ModuleDict({
                'branch1': nn.Sequential(
                    nn.Conv2d(layer_in_dim, out_dim // 4, kernel_size=3, padding=1, groups=out_dim // 4),
                    nn.BatchNorm2d(out_dim // 4),
                    nn.GELU()
                ),
                'branch2': nn.Sequential(
                    nn.Conv2d(layer_in_dim, out_dim // 4, kernel_size=7, padding=3, groups=out_dim // 4),
                    nn.BatchNorm2d(out_dim // 4),
                    nn.GELU()
                ),
                'branch3': nn.Sequential(
                    nn.Conv2d(layer_in_dim, out_dim // 4, kernel_size=15, padding=7, groups=out_dim // 4),
                    nn.BatchNorm2d(out_dim // 4),
                    nn.GELU()
                ),
                'branch4': nn.Sequential(
                    nn.Conv2d(layer_in_dim, out_dim // 4, kernel_size=31, padding=15, groups=out_dim // 4),
                    nn.BatchNorm2d(out_dim // 4),
                    nn.GELU()
                ),
                'final_conv': nn.Conv2d(out_dim, out_dim, kernel_size=1),
                'residual': nn.Conv2d(layer_in_dim, out_dim, kernel_size=1) if layer_in_dim != out_dim else nn.Identity()
            })
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Multi-scale processing
            out1 = layer['branch1'](x)
            out2 = layer['branch2'](x)
            out3 = layer['branch3'](x)
            out4 = layer['branch4'](x)

            # Concatenate multi-scale features
            multi_scale = torch.cat([out1, out2, out3, out4], dim=1)

            # Final projection
            out = layer['final_conv'](multi_scale)

            # Residual connection (除了第一層做dimension change)
            if i > 0 or isinstance(layer['residual'], nn.Identity):
                out = out + layer['residual'](x)

            x = out

        return x

# 推薦的配置
class FusionConfigs:
    """建議的不同複雜度配置"""

    @staticmethod
    def lightweight(dim=256, num_heads=8):
        """輕量配置：快速但可能效果稍差"""
        return HierarchicalCrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            depth_stage1=1,  # 單層
            depth_stage2=1   # 單層
        )

    @staticmethod
    def balanced(dim=256, num_heads=8):
        """平衡配置：推薦起始點"""
        return HierarchicalCrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            depth_stage1=1,  # Stage 1保持簡單
            depth_stage2=3   # Stage 2稍深一點學習融合
        )

    @staticmethod
    def heavy(dim=256, num_heads=8):
        """重量配置：最佳效果但計算量大"""
        return HierarchicalCrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            depth_stage1=2,  # 更深的特徵提取
            depth_stage2=3   # 更深的融合學習
        )