"""
对流-弥散方程（1D）
∂C/∂t + v∂C/∂x = D∂²C/∂x²
"""


def advection_dispersion_1d(C0: float, v: float, D: float, x: float, t: float) -> float:
    """
    解析解（瞬时点源）
    :param C0: 初始浓度
    :param v: 流速
    :param D: 弥散系数
    :param x: 距离
    :param t: 时间
    :return: t时x处浓度
    """
    import math

    if t == 0:
        return 0.0
    term1 = C0 / (2 * math.sqrt(math.pi * D * t))
    term2 = math.exp(-((x - v * t) ** 2) / (4 * D * t))
    return term1 * term2
