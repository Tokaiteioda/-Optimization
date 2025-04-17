import torch
from torch.autograd.functional import jacobian


def FR(gk, gk_last):
    beta = (gk @ gk) / (gk_last @ gk_last)
    return beta

def PRP(gk, gk_last):
    beta = (gk @ (gk - gk_last)) / (gk_last @ gk_last)
    return beta

def golden_ratio(f, x_left, x_right, epsilon, tau):
    """黄金分割法
    f:目标函数
    x_left:左侧初始值
    x_right:右侧初始值
    τ:0.618
    ε:终止阈值
    """
    new_left = x_left + (1 - tau) * (x_right - x_left)
    new_right = x_left + tau * (x_right - x_left)

    f_new_left, f_new_right = f(new_left), f(new_right)

    while abs(x_right - x_left) > epsilon:
        if f_new_left < f_new_right:  # 如果新的右侧函数值大于左侧
            x_right = new_right  # 则新的右侧成为下一个右侧边界

            new_right = new_left  # 这一次计算出的左侧边界为下一次的右侧边界，省去计算
            new_left = x_left + (1 - tau) * (x_right - x_left)  # 下一次的左侧边界照常计算

            f_new_right = f_new_left  # 这一次计算出的左侧函数值为下一次的右侧函数值，省去计算
            f_new_left = f(new_left)  # 下一次的左侧边界值照常计算
        else:
            x_left = new_left
            new_left = new_right
            new_right = x_left + tau * (x_right - x_left)
            f_new_left = f_new_right
            f_new_right = f(new_right)
    return (x_right + x_left) / 2


def CG_Method(f, x, epsilon):
    """
    共轭梯度法的步骤
    1. 给出 x0, ε(epsilon)>0, d0 = -g0, k := 0
    2. 若满足终止条件则停止
    3. 一维线搜索求步长 ak
    4. 计算 xk+1 = xk + ak * dk
    5. 计算 β(beta) , dk+1 = -gk+1 + βk * dk, k := k + 1,转步2
    """
    i = 0  # 计数器
    gk = jacobian(f, x)
    dk = -gk
    while True:
        i += 1

        if torch.norm(gk) < epsilon: # 终止条件
            print(f"条件满足，迭代终止\n最终结果为{x}")
            return x

        f_alpha = lambda alpha: f(x + alpha * dk)  # 黄金分割求步长
        alpha = golden_ratio(f_alpha, 0, 1, 1e-6, 0.618)

        x = x + alpha * dk  # 更新x
        print(f"第{i}次迭代,当前函数值为{f(x)},当前梯度范数‖▽f(x)‖为{torch.norm(gk)}")

        gk_last = gk  # 记录上一次的雅可比矩阵
        gk = jacobian(f, x)  # 更新雅可比矩阵

        beta = PRP(gk,gk_last)  # 计算βk
        dk = -gk + beta * dk

def f(x_init):  # x_init:初始点 n:参数
    n = x_init.shape[0]
    x = 0
    for i in range(n):
        x += ((i + 1) / 10) * (torch.exp(x_init[i]) - x_init[i])
    return x

x = torch.tensor([1.0,10.0,0.10,10.0,0.011],dtype=torch.float64)
CG_Method(f,x,1e-6)
