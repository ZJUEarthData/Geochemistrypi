"""
活度系数模型（以Debye-Hückel为例）
"""


def debye_huckel_ionic_strength(concs: list, charges: list) -> float:
    """
    计算离子强度
    :param concs: 离子浓度列表(mol/L)
    :param charges: 离子电荷列表
    :return: 离子强度
    """
    return 0.5 * sum([c * z**2 for c, z in zip(concs, charges)])


def debye_huckel_log_gamma(z: float, ionic_strength: float, A: float = 0.509, a: float = 0.5) -> float:
    """
    计算log10(活度系数)
    :param z: 离子电荷
    :param ionic_strength: 离子强度
    :param A: Debye-Hückel常数，25°C水中约0.509
    :param a: 离子水合半径，nm
    :return: log10(γ)
    """
    return -A * z**2 * (ionic_strength**0.5) / (1 + a * (ionic_strength**0.5))
