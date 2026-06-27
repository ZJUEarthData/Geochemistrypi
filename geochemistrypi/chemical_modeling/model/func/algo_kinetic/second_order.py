"""
二级反应动力学
1/C = 1/C0 + kt
"""


def second_order_conc(c0: float, k: float, t: float) -> float:
    """
    计算二级反应在t时刻的浓度
    :param c0: 初始浓度
    :param k: 反应速率常数
    :param t: 时间
    :return: t时刻浓度
    """
    denom = 1 / c0 + k * t
    return 1 / denom if denom != 0 else 0.0
