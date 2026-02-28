import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AutoCurriculumTorch:
    """
    使用 PyTorch + Adam 优化器实现自动课程学习
    目标：通过调整对手强度 lambda，使胜率保持在 0.5 附近
    """
    def __init__(self, lr=0.01):
        # lambda 作为可优化的参数（需要在 [0, 1] 范围内）
        self.lambda_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # 使用 Adam 优化器
        self.optimizer = torch.optim.Adam([self.lambda_param], lr=lr)
        
        # 记录历史数据用于可视化
        self.lambda_history = []
        self.win_rate_history = []
        self.loss_history = []
        
        # 滑动窗口记录最近的胜负
        self.recent_wins = []
        self.window_size = 100
    
    def get_lambda(self):
        """获取当前的 lambda 值（经过 sigmoid 映射到 [0, 1]）"""
        return torch.sigmoid(self.lambda_param).item()
    
    def simulate_battle(self, lambda_val):
        """
        模拟对战，返回是否获胜
        
        参数:
            lambda_val: 对手强度系数 [0, 1]
                       0 表示对手和我方水平相当
                       1 表示对手达到最高水平
        
        返回:
            is_win: bool, 是否获胜
        """
        # 模拟真实的胜率曲线：对手越强，我方胜率越低
        # 这里用一个简单的模型：base_win_rate = 0.5 - 0.4 * lambda
        base_win_rate = 0.7 - 0.4 * lambda_val
        
        # 加入随机性（模拟对战的不确定性）
        # 使用 beta 分布增加真实感
        actual_win_rate = np.clip(
            np.random.beta(base_win_rate * 10, (1 - base_win_rate) * 10),
            0.0, 1.0
        )
        
        # 根据胜率决定本局是否获胜
        is_win = np.random.random() < actual_win_rate
        
        return is_win
    
    def update(self, is_win):
        """
        根据对战结果更新 lambda
        
        参数:
            is_win: bool, 本局是否获胜
        """
        # 1. 记录胜负
        self.recent_wins.append(1.0 if is_win else 0.0)
        if len(self.recent_wins) > self.window_size:
            self.recent_wins.pop(0)
        
        # 2. 计算当前平均胜率
        if len(self.recent_wins) < 10:  # 数据不足时不更新
            return self.get_lambda()
        
        current_win_rate = sum(self.recent_wins) / len(self.recent_wins)
        
        # 3. 计算损失：(WinRate - 0.5)^2
        win_rate_tensor = torch.tensor(current_win_rate, dtype=torch.float32)
        loss = 0.5 * (win_rate_tensor - 0.5) ** 2
        
        # 4. 梯度下降（关键：手动构造梯度）
        self.optimizer.zero_grad()
        
        # 由于 win_rate 不是通过 lambda_param 计算出来的（是环境交互结果）
        # 我们需要手动指定梯度方向
        # 根据链式法则：dLoss/dλ = (WinRate - 0.5) * (dWinRate/dλ)
        # 假设 dWinRate/dλ ≈ -1（对手越强，胜率越低）
        
        gradient_direction = (current_win_rate - 0.5) * (-1.0)
        
        # 考虑 sigmoid 的导数：d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        lambda_val = torch.sigmoid(self.lambda_param)
        sigmoid_grad = lambda_val * (1 - lambda_val)
        
        # 手动设置梯度
        self.lambda_param.grad = torch.tensor(
            gradient_direction * sigmoid_grad.item(),
            dtype=torch.float32
        )
        
        # 5. 执行优化步骤
        self.optimizer.step()
        
        # 6. 记录历史
        self.lambda_history.append(self.get_lambda())
        self.win_rate_history.append(current_win_rate)
        self.loss_history.append(loss.item())
        
        return self.get_lambda()
    
    def plot_history(self):
        """可视化训练过程"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Lambda 变化曲线
        axes[0].plot(self.lambda_history, label='Lambda (Opponent Strength)', color='blue')
        axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal Lambda')
        axes[0].set_ylabel('Lambda')
        axes[0].set_title('Opponent Strength Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 胜率变化曲线
        axes[1].plot(self.win_rate_history, label='Win Rate', color='green')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target (0.5)')
        axes[1].set_ylabel('Win Rate')
        axes[1].set_title('Win Rate Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 损失曲线
        axes[2].plot(self.loss_history, label='Loss', color='orange')
        axes[2].set_xlabel('Training Episodes')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Loss (Win Rate - 0.5)^2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('d:/3_Machine_Learning_in_Python/project03_fire_and_dodge_missile/其余测试/curriculum_learning_result.png', dpi=150)
        plt.show()
        
        print(f"\n最终 Lambda: {self.lambda_history[-1]:.4f}")
        print(f"最终胜率: {self.win_rate_history[-1]:.4f}")
        print(f"最终损失: {self.loss_history[-1]:.6f}")


def main():
    """主训练循环"""
    print("=" * 60)
    print("自动课程学习 - PyTorch + Adam 实现")
    print("=" * 60)
    
    # 创建课程学习器
    curriculum = AutoCurriculumTorch(lr=0.005)
    
    # 训练参数
    num_episodes = 1000
    print_interval = 100
    
    print(f"\n开始训练，共 {num_episodes} 局对战...\n")
    
    for episode in range(num_episodes):
        # 获取当前 lambda
        current_lambda = curriculum.get_lambda()
        
        # 模拟对战
        is_win = curriculum.simulate_battle(current_lambda)
        
        # 更新 lambda
        new_lambda = curriculum.update(is_win)
        
        # 定期打印信息
        if (episode + 1) % print_interval == 0:
            recent_win_rate = sum(curriculum.recent_wins) / len(curriculum.recent_wins) if curriculum.recent_wins else 0
            print(f"Episode {episode + 1:4d} | "
                  f"Lambda: {new_lambda:.4f} | "
                  f"Win Rate: {recent_win_rate:.4f} | "
                  f"Result: {'WIN' if is_win else 'LOSS'}")
    
    print("\n训练完成！正在生成可视化图表...\n")
    
    # 绘制训练曲线
    curriculum.plot_history()


if __name__ == "__main__":
    main()
