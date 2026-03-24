"""
吉布斯自由能最小化算法
用于多组分体系的热力学平衡计算
"""
from typing import List


def gibbs_minimization(gibbs_energies: List[float], n: List[float]) -> float:
    """
    计算体系总吉布斯自由能
    :param gibbs_energies: 各组分摩尔吉布斯自由能
    :param n: 各组分摩尔数
    :return: 总吉布斯自由能
    """
    return sum([g * ni for g, ni in zip(gibbs_energies, n)])
