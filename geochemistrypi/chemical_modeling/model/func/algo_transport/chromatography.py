"""
色谱分离理论板数模型
N = (tR / sigma)^2
"""


def plate_number(tR: float, sigma: float) -> float:
    """
    计算理论板数
    :param tR: 保留时间
    :param sigma: 峰宽标准差
    :return: 理论板数N
    """
    if sigma == 0:
        return 0.0
    return (tR / sigma) ** 2
