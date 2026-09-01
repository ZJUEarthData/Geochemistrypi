"""
Fick扩散定律
J = -D * (dc/dx)
"""


def fick_flux(D: float, dc_dx: float) -> float:
    """
    计算扩散通量
    :param D: 扩散系数
    :param dc_dx: 浓度梯度
    :return: 通量
    """
    return -D * dc_dx
