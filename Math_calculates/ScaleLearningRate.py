# 根据网络参数数量或动作维度缩放学习率
def scale_learning_rate(base_lr, model, ref_params=1280):
    """
    根据模型参数数量缩放学习率。
    :param base_lr: 基础学习率
    :param model: PyTorch 模型
    :return: 缩放后的学习率
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params <= 0:
        return base_lr
    return base_lr * (ref_params / total_params) ** 0.5

