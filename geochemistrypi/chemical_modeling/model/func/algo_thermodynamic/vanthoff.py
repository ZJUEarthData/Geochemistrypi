"""
范特霍夫方程
描述温度对平衡常数的影响
"""
import math


def vanthoff_eq(K1: float, dH: float, T1: float, T2: float, R: float = 8.314) -> float:
    """
    计算T2温度下的平衡常数
    :param K1: T1温度下的平衡常数
    :param dH: 反应焓变(J/mol)
    :param T1: 初始温度(K)
    :param T2: 目标温度(K)
    :param R: 气体常数，默认8.314 J/(mol·K)
    :return: T2下的平衡常数
    """
    return K1 * math.exp(-dH / R * (1 / T2 - 1 / T1))
