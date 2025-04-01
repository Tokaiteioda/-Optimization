import numpy as np

def Armijo(f,grad_f,x,d,rho=1e-3,alpha=1,c=0.5):
    """   Armijo准则
    f:目标函数
    grad_f:当前梯度(在当前函数值的导数)
    x:当前函数值
    d:下降方向
    rho(ρ):固定参数
    alpha(α):初始步长
    c:步长缩放因子
    要求: f(x + α * d) ≤ f(x) + ρ * grad_f(x).T.dot(d) * α
    """
    while f(x + alpha * d) > f(x) + rho * np.dot(grad_f(x),d) * alpha:
        alpha *= c  # 缩小步长
    return alpha

def Goldstein(f,grad_f,x,d,rho=1e-3,alpha=1,c=0.5):
    """ Goldstein准则
        f:目标函数
        grad_f:当前梯度(在当前函数值的导数)
        x:当前函数值
        d:下降方向
        rho(ρ):固定参数
        alpha(α):初始步长
        c:步长缩放因子
        要求: f(x + α * d) ≤ f(x) + ρ * grad_f(x).T.dot(d) * α
             且 f(x + α * d) ≥ f(x) + (1 - ρ) * grad_f(x).T.dot(d) * α
        """

    fx = f(x)
    gx = grad_f(x)
    gTd = np.dot(gx, d)
    while True:
        fx_new = f(x + alpha * d)
        if fx_new > fx + rho * gTd * alpha:
            alpha *= c  # 如果太大则缩小步长
        elif fx_new < fx + (1 - rho) * gTd * alpha:
            alpha /= c  # 如果太小则增大步长
        else:
            break  # 满足条件,结束循环
    return alpha

def Wolfe(f,grad_f,x,d,rho=1e-3,alpha=1,c=0.5,sigma=0.7):
    """ Wolfe准则
           f:目标函数
           grad_f:当前梯度(在当前函数值的导数)
           x:当前函数值
           d:下降方向
           rho(ρ):固定参数
           sigma(σ):固定参数2 (其中 1 > σ > ρ > 0)
           alpha(α):初始步长
           c:步长缩放因子
    要求: f(x + α * d) ≤ f(x) + ρ * grad_f(x).T.dot(d) * α
        且 g(x + α * d).T.dot(d) ≥ σ * grad_f(x).T.dot(d)
        强 Wolfe准则:
        | g(x + α * d).T.dot(d) | ≤ -σ * grad_f(x).T.dot(d)
    """
    fx = f(x)
    gTd = np.dot(grad_f(x),d)
    while True:
        fx_new = f(x + alpha * d)
        gx_new = grad_f(x + alpha * d)
        if fx_new >= fx + rho * gTd * alpha:
            alpha *= c  # 如果太大则缩小步长
        elif abs(np.dot(gx_new,d)) > -sigma * gTd:
            alpha *= c
        else:
            break  # 满足条件,结束循环
    return alpha