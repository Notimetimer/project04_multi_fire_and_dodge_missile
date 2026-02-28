import numpy as np

def generate_ar1_value(previous_value: float, phi: float, error_std: float, constant: float = 0.0) -> float:
    """
    根据AR(1)模型生成下一个值。

    X_t = constant + phi * X_{t-1} + epsilon_t

    Args:
        previous_value (float): 上一步的值 (X_{t-1})。
        phi (float): 自回归系数 (phi)。应满足 -1 < phi < 1，否则序列可能发散。
        error_std (float): 误差项 epsilon_t 的标准差。
        constant (float, optional): 常数项 c。默认为 0。

    Returns:
        float: 新生成的随机数 (X_t)
    
    理论方差 = (error_std **2) / (1 - phi**2)
    """
    if not (-1 < phi < 1):
        raise ValueError("自回归系数 phi 必须在 (-1, 1) 之间以确保序列平稳。")

    # 生成白噪声误差项
    epsilon = np.random.normal(loc=0, scale=error_std)

    # 根据AR(1)模型计算当前值
    current_value = constant + phi * previous_value + epsilon

    return current_value

# --- 示例用法 ---
if __name__ == "__main__":
    # 设定参数
    initial_value = 0.0
    phi_coeff = 0.9
    std_dev = 0.1
    num_iterations = 200 # 生成200个值

    # 初始化序列
    current_series_value = initial_value
    generated_values = [current_series_value]

    print(f"初始值: {current_series_value:.2f}")

    # 循环生成后续值
    for i in range(num_iterations - 1):
        current_series_value = generate_ar1_value(current_series_value, phi_coeff, std_dev)
        generated_values.append(current_series_value)
        # print(f"第 {i+1} 步: {current_series_value:.2f}")

    # 可以用matplotlib绘制结果以验证
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(generated_values)
    plt.title(f'Generated AR(1) Series (phi={phi_coeff}, std={std_dev})')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.ylim(-1, 1)  # 设置 y 轴上下限
    plt.grid(True)
    plt.show()


    '''
    稳态方差：
    对AR(1)模型的两边同时取方差：
    Var(X_t) = Var(c + φ * X_{t-1} + ε_t)
    利用方差性质：
    常数的方差是 0：Var(c) = 0
    如果 A 和 B 是独立的随机变量，Var(A + B) = Var(A) + Var(B)
    Var(aY) = a^2 * Var(Y)
    因为 ε_t 与 X_{t-1} 独立，我们可以将方程分解：
    Var(X_t) = Var(φ * X_{t-1}) + Var(ε_t)
    Var(X_t) = φ^2 * Var(X_{t-1}) + σ_ε^2
    平稳性条件：
    对于一个平稳序列，其均值和方差不随时间变化。这意味着 Var(X_t) 应该等于 Var(X_{t-1})。
    我们设 Var(X_t) = Var(X_{t-1}) = σ_X^2。
    代入并求解 σ_X^2：
    将 σ_X^2 代入上一步的方程：
    σ_X^2 = φ^2 * σ_X^2 + σ_ε^2
    现在，我们将包含 σ_X^2 的项移到方程的左边：
    σ_X^2 - φ^2 * σ_X^2 = σ_ε^2
    σ_X^2 * (1 - φ^2) = σ_ε^2
    最后，求解 σ_X^2：
    σ_X^2 = σ_ε^2 / (1 - φ^2)
    '''