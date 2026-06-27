"""
吸附动力学（伪一级、伪二级）
"""


def pseudo_first_order(qe: float, k1: float, t: float) -> float:
    """
    伪一级吸附动力学模型
    :param qe: 平衡吸附量
    :param k1: 速率常数
    :param t: 时间
    :return: t时刻吸附量
    """
    from math import exp

    return qe * (1 - exp(-k1 * t))


def pseudo_second_order(qe: float, k2: float, t: float) -> float:
    """
    伪二级吸附动力学模型
    :param qe: 平衡吸附量
    :param k2: 速率常数
    :param t: 时间
    :return: t时刻吸附量
    """
    return (qe**2 * k2 * t) / (1 + qe * k2 * t)
