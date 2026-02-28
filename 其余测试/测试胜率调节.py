class AutoCurriculum:
    def __init__(self, lr=0.01):
        self.lambda_val = 0.0  # 初始难度
        self.lr = lr           # 学习率 (对应公式中的 alpha')
        self.win_history = []  # 记录最近的胜负

    def update(self, is_win):
        # 1. 记录胜负 (0或1)
        self.win_history.append(1.0 if is_win else 0.0)
        
        # 保持滑动窗口，例如最近100场
        if len(self.win_history) > 100:
            self.win_history.pop(0)
            
        # 2. 计算当前平均胜率 W
        current_win_rate = sum(self.win_history) / len(self.win_history)
        
        # 3. 计算梯度方向并更新 (核心公式)
        # diff = W - 0.5
        # lambda = lambda + lr * diff
        diff = current_win_rate - 0.5
        self.lambda_val += self.lr * diff
        
        # 4. 边界截断 (Clip)，保证 lambda 在 0~1 之间
        self.lambda_val = max(0.0, min(1.0, self.lambda_val))
        
        return self.lambda_val

# 模拟使用
curriculum = AutoCurriculum(lr=0.05)
# 假设智能体一直在赢
for _ in range(10):
    new_lambda = curriculum.update(is_win=True)
    print(f"Win! New Lambda: {new_lambda:.3f}")