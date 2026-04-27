"""
Pitzer模型活度系数（适用于高离子强度）
简化版，仅示意
"""


def pitzer_log_gamma(z: float, ionic_strength: float, beta: float = 0.1, C: float = 0.03) -> float:
    """
    计算Pitzer模型下的log10(活度系数)
    :param z: 离子电荷
    :param ionic_strength: 离子强度
    :param beta: 二元相互作用参数
    :param C: 三元相互作用参数
    :return: log10(γ)
    """
    # 真实Pitzer模型更复杂，这里仅作教学演示
    return -(z**2) * (beta * ionic_strength**0.5 + C * ionic_strength)


if __name__ == "__main__":
    print("Pitzer log_gamma:", pitzer_log_gamma(2, 0.5))
