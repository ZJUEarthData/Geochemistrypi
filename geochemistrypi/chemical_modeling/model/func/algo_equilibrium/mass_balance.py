"""
质量平衡算法模块
提供溶液/反应体系的质量守恒计算
"""
from typing import Dict


def mass_balance(species_conc: Dict[str, float], total_mass: float) -> bool:
    """
    检查各组分浓度之和是否等于总质量
    :param species_conc: 物种浓度字典 {物种: 浓度}
    :param total_mass: 总质量
    :return: 是否平衡
    """
    return abs(sum(species_conc.values()) - total_mass) < 1e-6
