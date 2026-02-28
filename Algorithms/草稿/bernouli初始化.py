import os
import sys
import torch
import numpy as np

# 将工程根目录加入路径（确保能 import Algorithms.PPObernouli）
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

from Algorithms.PPObernouli import PolicyNetBernouli

def main():
    device = torch.device("cpu")
    state_dim = 10            # 根据你的模型实际 state_dim 调整
    hidden_dims = [64, 64]    # 与 PolicyNetBernouli 初始化一致
    action_dim = 1            # 二元动作时常为1

    model = PolicyNetBernouli(state_dim, hidden_dims, action_dim).to(device)
    model.eval()

    # 随机输入（多样本）
    batch_size = 5
    x = torch.randn(batch_size, state_dim, dtype=torch.float32).to(device)
    print(x)

    with torch.no_grad():
        logits = model(x)                   # 直接拿未激活的 logits
        probs = torch.sigmoid(logits)       # 对应的概率

    print("input shape:", x.shape)
    print("logits:\n", logits.cpu().numpy())
    print("probs (sigmoid(logits)):\n", probs.cpu().numpy())
    print("probs mean:", float(probs.mean().cpu().numpy()))

if __name__ == "__main__":
    main()