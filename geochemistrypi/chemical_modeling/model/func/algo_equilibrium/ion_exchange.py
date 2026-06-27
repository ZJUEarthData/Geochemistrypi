"""
离子交换平衡算法模块
简单的阳离子交换等温线（Gaines-Thomas模型）
"""


def gaines_thomas_exchange(eq_conc_a: float, eq_conc_b: float, selectivity: float) -> float:
    """
    计算A、B两种离子在交换体上的分布比
    :param eq_conc_a: A离子平衡浓度
    :param eq_conc_b: B离子平衡浓度
    :param selectivity: 选择性系数
    :return: A/B在交换体上的比值
    """
    return selectivity * (eq_conc_a / eq_conc_b)
