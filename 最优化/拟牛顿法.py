import torch
from torch.autograd.functional import jacobian

def SR1(H,sk,yk):  # SR1对称秩1公式
    sk = sk.reshape(-1,1)
    yk = yk.reshape(-1,1)
    H += ((sk - H @ yk) @ (sk - H @ yk).T) / ((sk - H @ yk).T @ yk)
    return H

def DFP(H,sk,yk): # DFP公式
    sk = sk.reshape(-1,1)
    yk = yk.reshape(-1,1)
    H += (sk @ sk.T) / (sk.T @ yk) - (H @ yk @ yk.T @ H) / (yk.T @ H @ yk)
    return H

def BFGS(H,sk,yk): # BFGS公式
    sk = sk.reshape(-1,1)
    yk = yk.reshape(-1,1)
    H += (1 + (yk.T @ H @ yk) / (yk.T @ sk)) * ((sk @ sk.T) / (yk.T @ sk)) - ((sk @ yk.T @ H + H @ yk @ sk.T) / (yk.T @ sk))
    return H

def Broyden(H,sk,yk,tol): # Broyden族公式
    sk = sk.reshape(-1,1)
    yk = yk.reshape(-1,1)
    H = (1 - tol) * DFP(H,sk,yk) + tol * BFGS(H,sk,yk)
    return H
def golden_ratio(f,x_left,x_right,epsilon,tau):
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
        if f_new_left < f_new_right:   # 如果新的右侧函数值大于左侧
            x_right = new_right        # 则新的右侧成为下一个右侧边界

            new_right = new_left       # 这一次计算出的左侧边界为下一次的右侧边界，省去计算
            new_left = x_left + (1 - tau) * (x_right - x_left)  # 下一次的左侧边界照常计算

            f_new_right = f_new_left   # 这一次计算出的左侧函数值为下一次的右侧函数值，省去计算
            f_new_left = f(new_left)   # 下一次的左侧边界值照常计算
        else:
            x_left = new_left
            new_left = new_right
            new_right = x_left + tau * (x_right - x_left)
            f_new_left = f_new_right
            f_new_right = f(new_right)
    return (x_right + x_left) / 2


def quasi_newton(f,x,epsilon):
    """
    拟Newton法的步骤
    1. 对于 x0,正定对称阵H0, ε(epsilon)>0, k:=0
    2. 如果满足终止条件，则停止迭代
    3. 计算 dk = -Hk @ gk
    4. 沿 dk 方向进行线搜索求 αk > 0,令 xk+1 = xk + ak * dk
    5. 修正 Hk 得 Hk+1,使得 Hk+1 满足条件式,k := k+1,转步2
    """
    i = 0 # 计数器
    n = x.shape[0] # 获取x的形状
    H = torch.eye(n,n,dtype=torch.float64) # H0初始为 n * n 的单位矩阵
    while True:
        i += 1

        gk = jacobian(f,x) # 雅可比矩阵gk计算
        dk = -H @ gk

        f_alpha = lambda alpha: f(x + alpha * dk) # f(x+1) = f(x + α * dk)
        alpha = golden_ratio(f_alpha,0,1,1e-10,0.618) # 黄金分割求步长

        x_last = x # 记录更新前的x值
        x = x + alpha * dk # x更新

        yk = jacobian(f,x) - gk # yk = gk+1 - gk(更新后的雅可比矩阵减去更新前的)
        sk = x - x_last         # sk = xk+1 - xk(更新后的x值减去更新前的)
        H = Broyden(H,sk,yk,0.5) # 拟牛顿法公式

        print(f"当前第{i}次迭代，当前函数值为{f(x)},当前梯度范数‖▽f(x)‖为{torch.norm(gk)}")
        if torch.norm(gk) < epsilon: # 终止条件:雅可比矩阵的欧氏距离小于阈值
            print(f"结果x1:{x[0]},x2:{x[1]}")
            return x

def f(x):  # 目标函数
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 - 3 * x1 - x1 * x2 + 3

x = torch.tensor([0.0,0.0],dtype=torch.float64)
quasi_newton(f,x,1e-10)