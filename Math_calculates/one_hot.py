import numpy as np
import torch

def labels_to_one_hot(labels, n):
    one_hot = np.zeros(n)
    idx = int(labels)
    one_hot[idx] = 1
    return one_hot

def one_hot_to_labels(one_hot):
    arr = np.asarray(one_hot)
    arr = arr.reshape(1, -1)
    labels = np.argmax(one_hot)
    return labels

def labels_to_one_hot_torch(labels, n, device=None):
    """
    将标签转换为 one-hot 编码 (支持批量操作)
    输入:
        labels: Tensor, shape (batch, 1)，标签值
        n: int, one-hot 编码的类别数
        device: torch.device, 可选，指定输出的设备
    输出:
        one_hot: Tensor, shape (batch, n)，one-hot 编码
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, n, device=device)
    one_hot.scatter_(1, labels.long(), 1)
    return one_hot

def one_hot_to_labels_torch(one_hot):
    """
    将 one-hot 编码转换为标签 (支持批量操作)
    输入:
        one_hot: Tensor, shape (batch, n)，one-hot 编码
    输出:
        labels: Tensor, shape (batch, 1)，标签值
    """
    labels = torch.argmax(one_hot, dim=1, keepdim=True)
    return labels

if __name__ == "__main__":
    # 简单示例
    n = 4
    labels = [1, 2, 3]
    for i in labels:
        oh = labels_to_one_hot(i, n)
        print("labels -> one-hot:\n", oh)
        recovered = one_hot_to_labels(oh)
        print("one-hot -> labels:\n", recovered)

    # 示例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = torch.tensor([[0], [1], [2], [3]], device=device)  # shape: (batch, 1)
    n = 4  # 类别数

    one_hot = labels_to_one_hot_torch(labels, n, device=device)
    print("Labels -> One-hot:\n", one_hot)

    recovered_labels = one_hot_to_labels_torch(one_hot)
    print("One-hot -> Labels:\n", recovered_labels)


