"""
放射性衰变动力学
N = N0 * exp(-lambda * t)
"""
import math


def radioactive_decay(n0: float, decay_const: float, t: float) -> float:
    """
    计算t时刻剩余核素数量
    :param n0: 初始核素数量
    :param decay_const: 衰变常数
    :param t: 时间
    :return: t时刻剩余数量
    """
    return n0 * math.exp(-decay_const * t)
