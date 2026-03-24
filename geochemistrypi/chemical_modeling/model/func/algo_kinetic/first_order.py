"""
一级反应动力学
C = C0 * exp(-kt)
"""
import math


def first_order_conc(c0: float, k: float, t: float) -> float:
    """
    计算一级反应在t时刻的浓度
    :param c0: 初始浓度
    :param k: 反应速率常数
    :param t: 时间
    :return: t时刻浓度
    """
    return c0 * math.exp(-k * t)
