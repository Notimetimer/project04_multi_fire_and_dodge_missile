import math
import torch
import torch.nn as nn


class ResidualSelfAttentionBlock(nn.Module):
    """
    单层带残差的缩放点积自注意力模块 (Pre-LN 结构):
        1. H_mid = H + SelfAttn(LayerNorm(H))
        2. H_out = H_mid + FFN(LayerNorm(H_mid))
    """
    def __init__(self, d: int, h: int, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d % h == 0, f"特征维度 d({d}) 必须能被头数 h({h}) 整除"

        self.d = d
        self.h = h
        self.d_k = d // h

        # --- 第一阶段：自注意力相关层 ---
        self.norm1 = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # --- 第二阶段：串联的 FFN (非线性提取层) ---
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_ratio, d),
            nn.Dropout(dropout),
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            H:    (b, N, d) 拼接后的所有实体特征
            mask: (b, N)    可选布尔掩码，True=有效单位，False=阵亡/隐身/Padding
        Returns:
            H_out: (b, N, d)
        """
        b, N, d = H.shape

        # ==========================================================
        # 1. 第一条残差分支：多头缩放点积自注意力
        # ==========================================================
        H_norm = self.norm1(H)                                           # (b, N, d)

        # 线性投影后将 d 拆分成 (h, d_k)，并把 h 换到 dim=1 作为并行批维度
        Q = self.W_q(H_norm).view(b, N, self.h, self.d_k).transpose(1, 2)  # (b, h, N, d_k)
        K = self.W_k(H_norm).view(b, N, self.h, self.d_k).transpose(1, 2)  # (b, h, N, d_k)
        V = self.W_v(H_norm).view(b, N, self.h, self.d_k).transpose(1, 2)  # (b, h, N, d_k)

        # 缩放点积打分
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (b, h, N, N)

        # 屏蔽无效单位（只屏蔽 Key 维度，防止任何单位把注意力分给死掉/不存在的单位）
        if mask is not None:
            invalid_mask = ~mask.bool()[:, None, None, :]                  # (b, 1, 1, N)
            logits = logits.masked_fill(invalid_mask, torch.finfo(logits.dtype).min)

        # Softmax 在汇聚前完成，仅用于计算注意力权重
        attn_weights = torch.softmax(logits, dim=-1)                       # (b, h, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # 加权聚合 Value，并把多头拼接回 d 维
        attn_out = torch.matmul(attn_weights, V)                           # (b, h, N, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, N, d)     # (b, N, d)
        attn_out = self.proj_drop(self.W_o(attn_out))                      # (b, N, d)

        # 残差连接 1（H 为纯直连，无权重、无激活函数）
        H_mid = H + attn_out                                               # (b, N, d)

        # ==========================================================
        # 2. 第二条残差分支：串联 FFN（引入非线性特征变换）
        # ==========================================================
        ffn_out = self.ffn(self.norm2(H_mid))                              # (b, N, d)

        # 残差连接 2
        H_out = H_mid + ffn_out                                            # (b, N, d)
        return H_out


class MobaEntityAttentionNet(nn.Module):
    """
    面向 MOBA 输入的完整封装模块：
    输入: e_ego(b, 1, d), e_enm(b, n1, d), e_ally(b, n2, d)
    输出: 各自融合全局信息后的 e_ego_out, e_enm_out, e_ally_out
    """
    def __init__(self, d: int, h: int, num_layers: int = 2, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualSelfAttentionBlock(d=d, h=h, ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

    def forward(
        self,
        e_ego: torch.Tensor,
        e_enm: torch.Tensor,
        e_ally: torch.Tensor,
        enm_mask: torch.Tensor = None,
        ally_mask: torch.Tensor = None,
    ):
        """
        Args:
            e_ego:     (b, 1, d)
            e_enm:     (b, n1, d)
            e_ally:    (b, n2, d)
            enm_mask:  (b, n1) 可选布尔掩码，True=存活且可见
            ally_mask: (b, n2) 可选布尔掩码，True=存活
        """
        b, _, _ = e_ego.shape
        n1 = e_enm.shape[1]
        n2 = e_ally.shape[1]

        # 1. 拼接为完整动态实体序列 (b, 1 + n1 + n2, d)
        H = torch.cat([e_ego, e_enm, e_ally], dim=1)

        # 2. 组装掩码 (默认本体永远有效)
        mask = None
        if enm_mask is not None or ally_mask is not None:
            ego_mask = torch.ones((b, 1), dtype=torch.bool, device=H.device)
            if enm_mask is None:
                enm_mask = torch.ones((b, n1), dtype=torch.bool, device=H.device)
            if ally_mask is None:
                ally_mask = torch.ones((b, n2), dtype=torch.bool, device=H.device)
            mask = torch.cat([ego_mask, enm_mask.bool(), ally_mask.bool()], dim=1)  # (b, N)

        # 3. 多层堆叠交互
        for layer in self.layers:
            H = layer(H, mask=mask)

        H = self.final_norm(H)

        # 4. 拆回各自维度输出
        e_ego_out = H[:, 0:1, :]               # (b, 1, d)
        e_enm_out = H[:, 1 : 1 + n1, :]        # (b, n1, d)
        e_ally_out = H[:, 1 + n1 :, :]         # (b, n2, d)

        return e_ego_out, e_enm_out, e_ally_out


# ================= 测试验证 =================
if __name__ == "__main__":
    b, n1, n2, d, h = 4, 5, 4, 128, 4

    e_ego = torch.randn(b, 1, d)
    e_enm = torch.randn(b, n1, d)
    e_ally = torch.randn(b, n2, d)

    # 假设后两个敌人不可见/阵亡
    enm_mask = torch.tensor([[True, True, True, False, False]] * b)

    net = MobaEntityAttentionNet(d=d, h=h, num_layers=2)
    ego_out, enm_out, ally_out = net(e_ego, e_enm, e_ally, enm_mask=enm_mask)

    print("ego_out :", ego_out.shape)   # torch.Size([4, 1, 128])
    print("enm_out :", enm_out.shape)   # torch.Size([4, 5, 128])
    print("ally_out:", ally_out.shape)  # torch.Size([4, 4, 128])