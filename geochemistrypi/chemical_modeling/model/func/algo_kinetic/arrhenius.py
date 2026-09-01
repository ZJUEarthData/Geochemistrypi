"""
Arrhenius方程
描述温度对反应速率常数的影响
"""
import math


def arrhenius_rate_constant(A: float, Ea: float, T: float, R: float = 8.314) -> float:
    """
    计算不同温度下的速率常数
    :param A: 指前因子
    :param Ea: 活化能(J/mol)
    :param T: 温度(K)
    :param R: 气体常数，默认8.314 J/(mol·K)
    :return: 速率常数k
    """
    return A * math.exp(-Ea / (R * T))


if __name__ == "__main__":
    # 示例: A=1e13, Ea=80000 J/mol, T=298K
    k = arrhenius_rate_constant(1e13, 80000, 298)
    print(f"298K下速率常数: {k:.2e}")
