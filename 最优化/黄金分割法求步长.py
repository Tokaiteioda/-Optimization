
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