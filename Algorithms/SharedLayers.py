import torch
import torch.nn as nn
import torch.nn.functional as F

# 单GRU
class SharedGruBackbone(nn.Module):
    """共享的 GRU 特征提取器"""
    def __init__(self, state_dim, gru_hidden_size=128, gru_num_layers=1, output_dim=35, batch_first=True):
        super(SharedGruBackbone, self).__init__()
        
        # 使用您新版 SharedLayers 中的 GruMlp
        # 我们只使用它的 GRU 部分，所以 output_dim 可以设为 gru_hidden_size
        self.feature_extractor = GruMlp(
            input_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            output_dim=output_dim,
            batch_first=batch_first
        )

    def forward(self, x, h_0=None):
        # 如果输入维度为2 (B, D)，增加序列维度S=1
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 输入 x 的形状: (Batch, SeqLen, StateDim), e.g., (B, 10, 35)
        # h_0: (num_layers, B, gru_hidden_size) 或 None
        # GruMlp 输出形状: (B, SeqLen, gru_hidden_size)
        features_per_step, h_n = self.feature_extractor(x, h_0)
        # 返回每个时间步的特征（B, SeqLen, gru_hidden_size）和隐藏状态，由上层决定如何聚合
        return features_per_step, h_n

# 通道注意力
'''
todo
给每个通道的权重加上归一化/缩放/限幅，不过不能打破梯度传播的途径
'''
class ChannelAttention(nn.Module):
    def __init__(self, feature_dim, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio

        # 定义 Squeeze-and-Excitation (SE) 块
        # 这里的输入 feature_dim 已经是扁平的特征向量，
        # 所以直接用全连接层进行压缩和激励
        self.se_block = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // self.reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // self.reduction_ratio, self.feature_dim, bias=False),
            nn.Sigmoid() # 使用 Sigmoid 激活函数，输出权重在 (0, 1) 之间
        )

    def forward(self, x):
        # x 的维度: (Batch, Feature_Dim)

        # 1. Squeeze (挤压)
        # 在 (Batch, Feature_Dim) 的情况下，Feature_Dim 已经是“通道”维度，
        # 每个样本的 Feature_Dim 向量本身就是需要被激励的对象。
        # 因此，这里的“挤压”实际上就是输入 x 本身，或者你可以认为它已经经过了某种全局池化
        # 如果需要更强的抽象，可以在这里加一个额外的线性层，但通常不是必须的。
        squeezed_features = x 
        
        # 2. Excitation (激励)
        # 通过 MLP 生成每个通道的权重
        weights = self.se_block(squeezed_features)
        # weights 的维度: (Batch, Feature_Dim)

        # 3. Scale (缩放)
        # 将权重逐元素地乘以原始输入特征
        # output 的维度: (Batch, Feature_Dim)
        output = x * weights
        
        return output, weights

# GRU-MLP
class GruMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, gru_num_layers=1, output_dim=None, batch_first=True):
        super(GruMlp, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers  # 添加GRU层数参数
        self.output_dim = output_dim

        # GRU层处理序列数据，增加num_layers参数
        self.gru = nn.GRU(input_dim, gru_hidden_size, num_layers=gru_num_layers, batch_first=batch_first)

        # MLP输出层
        self.mlp_out = nn.Linear(gru_hidden_size, output_dim)

        # # debug: 在 GruMlp 初始化完成后立即检查 self.gru._flat_weights 类型
        # if hasattr(self, "gru") and hasattr(self.gru, "_flat_weights") and isinstance(self.gru._flat_weights, list):
        #     import traceback
        #     raise RuntimeError("DEBUG: GruMlp created with gru._flat_weights as list; "
        #                        f"element types: {[type(w) for w in self.gru._flat_weights]}\\n"
        #                        f"stack:\\n{''.join(traceback.format_stack())}")

    def forward(self, x, h_0=None):
        # x: (B, S, D)
        B, S, D = x.shape

        # 如果没有提供初始隐藏状态，则使用None，让PyTorch自动初始化
        # h_0: (num_layers, B, gru_hidden_size) 或 None

        # # 检查 GRU 内部 flat weights 类型，若为 list 则抛出错误以便定位问题源
        # if hasattr(self.gru, "_flat_weights") and isinstance(self.gru._flat_weights, list):
        #     raise TypeError(
        #         "GRU internal _flat_weights is a list (unexpected). "
        #         "This indicates GRU parameters were mutated/ wrapped incorrectly. "
        #         f"_flat_weights element types: {[type(w) for w in self.gru._flat_weights]}"
        #     )
        # # 确保 h_0 是 Tensor 或 None
        # if h_0 is not None and not isinstance(h_0, torch.Tensor):
        #     raise TypeError(f"h_0 must be a torch.Tensor or None, got {type(h_0)}")
        # # 确保 h_0 与输入在同一 device
        # if h_0 is not None and isinstance(h_0, torch.Tensor) and h_0.device != x.device:
        #     raise RuntimeError(f"h_0.device ({h_0.device}) != x.device ({x.device}); move h_0 to input device before calling forward")

        # GRU处理
        gru_output, h_n = self.gru(x, h_0)  # gru_output: (B, S, gru_hidden_size), h_n: (num_layers, B, gru_hidden_size)

        # MLP输出
        output = self.mlp_out(gru_output)  # (B, S, output_dim)

        # 返回输出和最终隐藏状态
        return output, h_n


# ATT-MLP
class AttMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, output_dim,
                 dropout=0.0, bias=True, batch_first=False):
        super(AttMlp, self).__init__()
        self.input_dim = input_dim  # D
        self.feature_embed_dim = feature_embed_dim  # n1
        self.output_dim = output_dim
        self.batch_first = batch_first

        # 特征嵌入层
        self.feature_embedding_layer = nn.Linear(1, feature_embed_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=feature_embed_dim,
                                               num_heads=attn_heads,
                                               dropout=dropout,
                                               bias=bias,
                                               batch_first=batch_first)

        # MLP输出层
        self.mlp_out = nn.Linear(feature_embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        # 为特征注意力准备 (B, S, D) -> (B*S, D, 1)
        x_reshaped = x.view(B * S, D, 1)

        # 嵌入每个1维特征到n1维 (B*S, D, n1)
        x_embedded = self.feature_embedding_layer(x_reshaped)

        # 多头自注意力
        if self.batch_first:
            # (B*S, D, n1)
            attn_output, _ = self.attention(x_embedded, x_embedded, x_embedded)
        else:
            # (D, B*S, n1)
            attn_input = x_embedded.permute(1, 0, 2)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 0, 2)  # (B*S, D, n1)

        # 池化/聚合特征注意力结果 (B*S, n1)
        pooled_features = attn_output.mean(dim=1)

        # 重塑回原始序列结构 (B, S, n1)
        final_output = pooled_features.view(B, S, self.feature_embed_dim)

        # MLP 输出 (B, S, output_dim)
        output = self.mlp_out(final_output)

        return output


# ATT-GRU-MLP
class AttGruMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, gru_hidden_size, gru_num_layers, output_dim,
                 dropout=0.0, bias=True, batch_first=True):
        super(AttGruMlp, self).__init__()
        # 初始化注意力模块，其输出维度为GRU的输入维度
        self.att = AttMlp(input_dim=input_dim,
                          feature_embed_dim=feature_embed_dim,
                          attn_heads=attn_heads,
                          output_dim=feature_embed_dim,  # ATT的输出作为GRU的输入
                          dropout=dropout,
                          bias=bias,
                          batch_first=batch_first)

        # 初始化GRU模块，其输入维度是注意力模块的输出维度
        self.gru_mlp = GruMlp(input_dim=feature_embed_dim,
                              gru_hidden_size=gru_hidden_size,
                              gru_num_layers=gru_num_layers,
                              output_dim=output_dim)

    def forward(self, x, h_0=None):
        # x: (B, S, D)

        # Step 1: Attention处理
        # att_output: (B, S, feature_embed_dim)
        att_output = self.att(x)

        # Step 2: GRU处理
        # output: (B, S, output_dim), h_n: (num_layers, B, gru_hidden_size)
        output, h_n = self.gru_mlp(att_output, h_0)

        return output, h_n


# GRU-ATT-MLP
class GruAttMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, gru_num_layers, feature_embed_dim, attn_heads, output_dim,
                 dropout=0.0, bias=True, batch_first=True):
        super(GruAttMlp, self).__init__()
        # 初始化GRU模块，其输出维度为Attention的输入维度
        self.gru_mlp = GruMlp(input_dim=input_dim,
                              gru_hidden_size=gru_hidden_size,
                              gru_num_layers=gru_num_layers,
                              output_dim=gru_hidden_size)  # GRU的输出作为ATT的输入

        # 初始化Attention模块，其输入维度是GRU模块的隐藏层大小
        self.att = AttMlp(input_dim=gru_hidden_size,
                          feature_embed_dim=feature_embed_dim,
                          attn_heads=attn_heads,
                          output_dim=output_dim,
                          dropout=dropout,
                          bias=bias,
                          batch_first=batch_first)

    def forward(self, x, h_0=None):
        # x: (B, S, D)

        # Step 1: GRU处理
        # gru_output: (B, S, gru_hidden_size)
        gru_output, h_n = self.gru_mlp(x, h_0)

        # Step 2: Attention处理
        # output: (B, S, output_dim)
        output = self.att(gru_output)

        return output, h_n
