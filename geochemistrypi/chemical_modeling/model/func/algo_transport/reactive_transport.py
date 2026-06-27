"""
反应-迁移耦合模型（1D简化版）
C(x,t) = 解析解/数值解
"""
import numpy as np


def reactive_transport_1d(C0: float, v: float, D: float, k: float, x: float, t: float) -> float:
    """
    1D反应-迁移模型（一级反应+对流+弥散）
    :param C0: 初始浓度
    :param v: 流速
    :param D: 弥散系数
    :param k: 反应速率常数
    :param x: 距离
    :param t: 时间
    :return: t时x处浓度
    """
    # 解析解（无限域，瞬时点源）
    if t == 0:
        return 0.0
    term1 = C0 / (2 * np.sqrt(np.pi * D * t))
    term2 = np.exp(-((x - v * t) ** 2) / (4 * D * t) - k * t)
    return term1 * term2


if __name__ == "__main__":
    print("Reactive transport C(x,t):", reactive_transport_1d(1.0, 1e-5, 1e-9, 1e-6, 0.1, 3600))
