import numpy
import torch
from torch.autograd.functional import jacobian, hessian
import numpy as np
from typing import Callable


def newton(f, x_init: numpy.ndarray, tol=1e-6):
    """
    牛顿法原理：
        dk = -Gk ^ -1 * gk  (Gk ^ -1 :海森矩阵的逆矩阵 | gk :雅可比矩阵)
        x(k+1) := x(k) + dk
    :param tol:阈值
    :param f:目标函数
    :param x_init: 初始点
    :return:极小点
    """
    n = 0  # 计数器
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)  # 初始化x
    while True:
        n += 1

        yakebi = jacobian(f, x)  # 雅可比矩阵计算
        haisen = hessian(f, x)  # 海森矩阵计算
        haisen_inv = torch.linalg.inv(haisen)  # 海森矩阵求逆

        dk = -torch.matmul(haisen_inv, yakebi)  # 求步长(海森的逆矩阵乘以雅可比矩阵)

        x = (x + dk).detach().requires_grad_(True)  # 更新x并且保留计算图

        print(f"第{n}次迭代，当前函数值为{f(x)},x1:{x[0]},x2:{x[1]}")
        if torch.norm(yakebi) < tol:  # 收敛条件: 梯度的欧几里得长度小于阈值
            break
    return x


def damped_newton(f, x_init: numpy.ndarray, tol=1e-6):
    """
    阻尼牛顿法原理：
        dk = -Gk ^ -1 * gk  (Gk ^ -1 :海森矩阵的逆矩阵 | gk :雅可比矩阵)
        x(k+1) := x(k) + αk * dk (αk步长来自一维线搜索) (本例使用黄金分割)
    :param f:目标函数
    :param x_init:初始点
    :param tol:阈值
    :return:极小点
    """

    def golden_ratio(f_alpha, x_left, x_right, epsilon, tau):
        """黄金分割法
        f_alpha:目标函数 f (x + alpha * d)
        x_left:左侧初始值 (本例中为0)
        x_right:右侧初始值 (本例中为1)
        τ:0.618
        ε:终止阈值
        """
        new_left = x_left + (1 - tau) * (x_right - x_left)
        new_right = x_left + tau * (x_right - x_left)

        f_new_left, f_new_right = f_alpha(new_left), f_alpha(new_right)

        while abs(x_right - x_left) > epsilon:
            if f_new_left < f_new_right:  # 如果新的右侧函数值大于左侧
                x_right = new_right  # 则新的右侧成为下一个右侧边界
                new_right = new_left  # 这一次计算出的左侧边界为下一次的右侧边界，省去计算
                new_left = x_left + (1 - tau) * (x_right - x_left)  # 下一次的左侧边界照常计算

                f_new_right = f_new_left  # 这一次计算出的左侧函数值为下一次的右侧函数值，省去计算
                f_new_left = f_alpha(new_left)  # 下一次的左侧边界值照常计算
            else:
                x_left = new_left
                new_left = new_right
                new_right = x_left + tau * (x_right - x_left)
                f_new_left = f_new_right
                f_new_right = f_alpha(new_right)
        return (x_right + x_left) / 2

    n = 0
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)  # 初始化
    while True:
        n += 1

        yakebi = jacobian(f, x)  # 雅可比矩阵计算
        haisen = hessian(f, x)  # 海森矩阵计算
        haisen_inv = torch.linalg.inv(haisen)  # 海森矩阵求逆
        dk = -torch.matmul(haisen_inv, yakebi)  # 求方向(海森的逆矩阵乘以雅可比矩阵)

        f_alpha = lambda alpha: f((x + alpha * dk).detach())  # 定义一元函数f(alpha) = f(x + alpha * dk)
        alpha = golden_ratio(f_alpha, 0, 1, epsilon=1e-3, tau=0.618)

        x = (x + alpha * dk).detach().requires_grad_(True)  # 更新x并且保留计算图

        print(f"第{n}次迭代，当前函数值为{f(x)},x1:{x[0]},x2:{x[1]}")
        if torch.norm(yakebi) < tol:  # 收敛条件: 雅可比矩阵的欧几里得距离小于阈值
            break
    return x


def hybrid_newton(f, x_init: numpy.ndarray, tol=1e-6):
    """
    混合牛顿法原理: 当牛顿法可行时，使用牛顿法，不可行或不好时* 使用其他方法 (本例中使用梯度下降法)
        * 如海森矩阵 H 是奇异矩阵或病态矩阵
    :param f:目标函数
    :param x_init:初始点
    :param tol:阈值
    :return:极小值
    """
    n = 0  # 计数器
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)  # 初始化

    def golden_ratio(f_alpha, x_left, x_right, epsilon, tau):
        """黄金分割法
        f_alpha:目标函数 f (x + alpha * d)
        x_left:左侧初始值 (本例中为0)
        x_right:右侧初始值 (本例中为1)
        τ:0.618
        ε:终止阈值
        """
        new_left = x_left + (1 - tau) * (x_right - x_left)
        new_right = x_left + tau * (x_right - x_left)

        f_new_left, f_new_right = f_alpha(new_left), f_alpha(new_right)

        while abs(x_right - x_left) > epsilon:
            if f_new_left < f_new_right:  # 如果新的右侧函数值大于左侧
                x_right = new_right  # 则新的右侧成为下一个右侧边界
                new_right = new_left  # 这一次计算出的左侧边界为下一次的右侧边界，省去计算
                new_left = x_left + (1 - tau) * (x_right - x_left)  # 下一次的左侧边界照常计算

                f_new_right = f_new_left  # 这一次计算出的左侧函数值为下一次的右侧函数值，省去计算
                f_new_left = f_alpha(new_left)  # 下一次的左侧边界值照常计算
            else:
                x_left = new_left
                new_left = new_right
                new_right = x_left + tau * (x_right - x_left)
                f_new_left = f_new_right
                f_new_right = f_alpha(new_right)
        return (x_right + x_left) / 2

    while True:
        n += 1
        yakebi = jacobian(f, x)  # 求雅可比矩阵
        haisen = hessian(f, x)  # 求海森矩阵

        try:  # 尝试求海森矩阵的逆矩阵
            haisen_inv = torch.linalg.inv(haisen)
        except RuntimeError as _:  # 如果不能求逆
            dk = -yakebi  # dk = -gk
            f_alpha = lambda alpha: f((x + alpha * dk).detach())
            alpha = golden_ratio(f_alpha, 0, 1, 1e-3, 0.618)

            x = (x + alpha * dk).detach().requires_grad_(True)  # 负梯度法
            print(f"第{n}次迭代，当前函数值为{f(x)},x1:{x[0]},x2:{x[1]}")

            continue  # 直接进入下一次迭代,防止错误

        dk = -torch.matmul(haisen_inv, yakebi)  # 可以求逆
        f_alpha = lambda alpha: f((x + alpha * dk).detach())
        alpha = golden_ratio(f_alpha, 0, 1, 1e-3, 0.618)

        x = (x + alpha * dk).detach().requires_grad_(True)
        print(f"第{n}次迭代，当前函数值为{f(x)},x1:{x[0]},x2:{x[1]}")

        if torch.norm(yakebi) < tol:  # 终止条件：雅可比矩阵的欧几里得距离小于阈值
            break
    return x


def LM_newton(f, x_init: numpy.ndarray, upsilon, tol=1e-6):
    """
    LM牛顿法原理：
    (Gk + vk * I)d = -gk   (其中Gk为海森矩阵，vk为系数，I为单位矩阵,-gk为迭代方向)
    :param upsilon: 初始系数
    :param f: 目标函数
    :param x_init: 初始点
    :param tol: 阈值
    :return:
    """
    n = 0
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)

    while True:
        n += 1
        yakebi = jacobian(f, x)
        haisen = hessian(f, x)

        I = torch.eye(haisen.shape[0], haisen.shape[1])  # 单位矩阵
        A = haisen + upsilon * I
        dk = torch.linalg.solve(A,-yakebi)  # A * dk = -gk

        x = (x + dk).detach().requires_grad_(True)  # 更新x

        print(f"第{n}次迭代，当前函数值为{f(x)},x1:{x[0]},x2:{x[1]}")
        if torch.norm(yakebi) < tol:  # 收敛条件: 雅可比矩阵的欧几里得距离小于阈值
            break
    return x


def f(x):  # 目标函数
    x1, x2 = x
    return (100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2).to(x.dtype)  # 设置torch类型


x_init = np.array([-5, -5])
x = LM_newton(f, x_init, 0.01,1e-20)
