"""
溶解-沉淀平衡算法模块
计算溶液中溶解/沉淀的平衡浓度
"""


def calc_saturation_index(ion_activity_product: float, ksp: float) -> float:
    """
    计算饱和指数（SI）
    :param ion_activity_product: 离子活度积
    :param ksp: 溶度积常数
    :return: 饱和指数 SI = log10(IAP/Ksp)
    """
    import math

    return math.log10(ion_activity_product / ksp)


def is_precipitation(si: float) -> bool:
    """
    判断是否发生沉淀
    :param si: 饱和指数
    :return: True-沉淀, False-不沉淀
    """
    return si > 0
