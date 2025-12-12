"""层次化文本编码模块

将 RoboCrystallographer 生成的文本分为全局、半全局、局部三个层次，
分别编码并使用可学习权重融合，突出全局信息的重要性。

使用方法:
1. 修改数据集，将文本分为三个部分
2. 使用 HierarchicalTextEncoder 替代原始的 BERT encoder
3. 训练时自动学习每个层次的最优权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class HierarchicalTextEncoder(nn.Module):
    """层次化文本编码器

    将文本分为全局、半全局、局部三个层次，分别编码后使用
    可学习权重融合，自动学习每个层次的重要性。
    """

    def __init__(
        self,
        model_name='m3rg-iitd/matscibert',
        embedding_dim=768,
        use_global=True,
        use_semi_global=True,
        use_local=False,  # 默认关闭局部信息
        global_weight_init=1.0,
        semi_global_weight_init=0.5,
        local_weight_init=0.1,
        learnable_weights=True,
        pooling='cls',  # 'cls', 'mean', 'attention'
    ):
        """初始化层次化文本编码器

        Args:
            model_name: BERT模型名称
            embedding_dim: 输出嵌入维度
            use_global: 是否使用全局信息
            use_semi_global: 是否使用半全局信息
            use_local: 是否使用局部信息
            global_weight_init: 全局信息初始权重
            semi_global_weight_init: 半全局信息初始权重
            local_weight_init: 局部信息初始权重
            learnable_weights: 权重是否可学习
            pooling: 池化方式 ('cls', 'mean', 'attention')
        """
        super().__init__()

        self.use_global = use_global
        self.use_semi_global = use_semi_global
        self.use_local = use_local
        self.pooling = pooling

        # 共享的 BERT 编码器
        self.bert = AutoModel.from_pretrained(model_name)

        # 层次权重（可学习）
        if learnable_weights:
            if use_global:
                self.global_weight = nn.Parameter(torch.tensor(global_weight_init))
            if use_semi_global:
                self.semi_global_weight = nn.Parameter(torch.tensor(semi_global_weight_init))
            if use_local:
                self.local_weight = nn.Parameter(torch.tensor(local_weight_init))
        else:
            if use_global:
                self.register_buffer('global_weight', torch.tensor(global_weight_init))
            if use_semi_global:
                self.register_buffer('semi_global_weight', torch.tensor(semi_global_weight_init))
            if use_local:
                self.register_buffer('local_weight', torch.tensor(local_weight_init))

        # 注意力池化（如果使用）
        if pooling == 'attention':
            self.attention_weights = nn.Linear(embedding_dim, 1)

        print(f"\n{'='*60}")
        print(f"层次化文本编码器初始化")
        print(f"{'='*60}")
        print(f"  使用全局信息: {use_global} (权重初始值: {global_weight_init if use_global else 'N/A'})")
        print(f"  使用半全局信息: {use_semi_global} (权重初始值: {semi_global_weight_init if use_semi_global else 'N/A'})")
        print(f"  使用局部信息: {use_local} (权重初始值: {local_weight_init if use_local else 'N/A'})")
        print(f"  权重可学习: {learnable_weights}")
        print(f"  池化方式: {pooling}")
        print(f"{'='*60}\n")

    def encode_text(self, text_input_ids, text_attention_mask):
        """编码单个文本

        Args:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]

        Returns:
            text_emb: [batch, 768]
        """
        outputs = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )

        if self.pooling == 'cls':
            # 使用 [CLS] token
            text_emb = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        elif self.pooling == 'mean':
            # 平均池化（忽略padding）
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
            input_mask_expanded = text_attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_emb = sum_embeddings / sum_mask  # [batch, 768]

        elif self.pooling == 'attention':
            # 注意力池化
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
            attention_scores = self.attention_weights(token_embeddings)  # [batch, seq_len, 1]
            attention_scores = attention_scores.squeeze(-1)  # [batch, seq_len]

            # Mask padding tokens
            attention_scores = attention_scores.masked_fill(
                text_attention_mask == 0, float('-inf')
            )
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # [batch, 1, seq_len]
            text_emb = torch.matmul(attention_weights, token_embeddings).squeeze(1)  # [batch, 768]

        return text_emb

    def forward(
        self,
        global_input_ids=None,
        global_attention_mask=None,
        semi_global_input_ids=None,
        semi_global_attention_mask=None,
        local_input_ids=None,
        local_attention_mask=None
    ):
        """前向传播

        Args:
            global_input_ids: 全局文本输入 [batch, seq_len]
            global_attention_mask: 全局文本mask
            semi_global_input_ids: 半全局文本输入 [batch, seq_len]
            semi_global_attention_mask: 半全局文本mask
            local_input_ids: 局部文本输入 [batch, seq_len]
            local_attention_mask: 局部文本mask

        Returns:
            fused_emb: 融合后的文本嵌入 [batch, 768]
            weights_dict: 各层次的归一化权重（用于可视化）
        """
        embeddings = []
        raw_weights = []

        # 编码全局信息
        if self.use_global and global_input_ids is not None:
            global_emb = self.encode_text(global_input_ids, global_attention_mask)
            embeddings.append(global_emb)
            raw_weights.append(self.global_weight)

        # 编码半全局信息
        if self.use_semi_global and semi_global_input_ids is not None:
            semi_global_emb = self.encode_text(semi_global_input_ids, semi_global_attention_mask)
            embeddings.append(semi_global_emb)
            raw_weights.append(self.semi_global_weight)

        # 编码局部信息
        if self.use_local and local_input_ids is not None:
            local_emb = self.encode_text(local_input_ids, local_attention_mask)
            embeddings.append(local_emb)
            raw_weights.append(self.local_weight)

        if len(embeddings) == 0:
            raise ValueError("至少需要提供一种层次的文本信息")

        # 归一化权重
        raw_weights = torch.stack(raw_weights)  # [num_levels]
        # 使用 softmax 确保权重和为1
        normalized_weights = F.softmax(raw_weights, dim=0)

        # 加权融合
        weighted_embeddings = []
        for i, emb in enumerate(embeddings):
            weighted_embeddings.append(emb * normalized_weights[i])

        fused_emb = torch.stack(weighted_embeddings, dim=0).sum(dim=0)  # [batch, 768]

        # 构建权重字典（用于监控）
        weights_dict = {}
        idx = 0
        if self.use_global:
            weights_dict['global'] = normalized_weights[idx].item()
            idx += 1
        if self.use_semi_global:
            weights_dict['semi_global'] = normalized_weights[idx].item()
            idx += 1
        if self.use_local:
            weights_dict['local'] = normalized_weights[idx].item()

        return fused_emb, weights_dict


class GlobalEnhancedTextEncoder(nn.Module):
    """增强全局信息的文本编码器

    对全局信息进行额外的注意力增强，使模型更关注全局特征。
    """

    def __init__(
        self,
        model_name='m3rg-iitd/matscibert',
        embedding_dim=768,
        global_boost_factor=2.0,  # 全局信息增强因子
    ):
        """初始化全局增强编码器

        Args:
            model_name: BERT模型名称
            embedding_dim: 嵌入维度
            global_boost_factor: 全局信息增强因子（对全局部分乘以该因子）
        """
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.global_boost_factor = global_boost_factor

        # 全局信息注意力
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        print(f"\n全局增强文本编码器 (boost_factor={global_boost_factor})")

    def forward(self, input_ids, attention_mask, global_token_mask=None):
        """前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            global_token_mask: [batch, seq_len] - 标记哪些token属于全局信息（1表示全局，0表示其他）

        Returns:
            text_emb: [batch, 768]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]

        if global_token_mask is not None:
            # 增强全局token的表示
            global_mask_expanded = global_token_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

            # 全局token自注意力增强
            global_enhanced, _ = self.global_attention(
                token_embeddings,
                token_embeddings,
                token_embeddings,
                key_padding_mask=(attention_mask == 0)
            )

            # 对全局部分进行增强
            token_embeddings = (
                token_embeddings * (1 - global_mask_expanded) +  # 非全局部分保持不变
                global_enhanced * global_mask_expanded * self.global_boost_factor  # 全局部分增强
            )

            token_embeddings = self.layer_norm(token_embeddings)

        # 使用 [CLS] token
        text_emb = token_embeddings[:, 0, :]

        return text_emb


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """使用示例"""

    # 示例1：层次化编码器（只使用全局+半全局）
    print("=" * 80)
    print("示例1：层次化编码器（推荐配置）")
    print("=" * 80)

    encoder = HierarchicalTextEncoder(
        use_global=True,
        use_semi_global=True,
        use_local=False,  # 关闭局部信息
        global_weight_init=1.0,  # 全局权重较高
        semi_global_weight_init=0.5,  # 半全局权重中等
        learnable_weights=True,
        pooling='cls'
    )

    # 模拟输入
    batch_size = 4
    seq_len = 128

    global_ids = torch.randint(0, 30000, (batch_size, seq_len))
    global_mask = torch.ones(batch_size, seq_len)

    semi_global_ids = torch.randint(0, 30000, (batch_size, seq_len))
    semi_global_mask = torch.ones(batch_size, seq_len)

    # 前向传播
    fused_emb, weights = encoder(
        global_input_ids=global_ids,
        global_attention_mask=global_mask,
        semi_global_input_ids=semi_global_ids,
        semi_global_attention_mask=semi_global_mask
    )

    print(f"\n输出形状: {fused_emb.shape}")
    print(f"权重分布: {weights}")

    # 示例2：全局增强编码器
    print("\n" + "=" * 80)
    print("示例2：全局增强编码器")
    print("=" * 80)

    enhanced_encoder = GlobalEnhancedTextEncoder(
        global_boost_factor=2.0
    )

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # 前10个token标记为全局信息
    global_token_mask = torch.zeros(batch_size, seq_len)
    global_token_mask[:, :10] = 1

    enhanced_emb = enhanced_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_token_mask=global_token_mask
    )

    print(f"\n输出形状: {enhanced_emb.shape}")

    print("\n✅ 层次化文本编码模块测试完成！")
