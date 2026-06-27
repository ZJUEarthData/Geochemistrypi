"""
质量作用定律通用求解器
支持多组分反应的平衡浓度计算
"""
from typing import Dict

import numpy as np
from scipy.optimize import fsolve


def law_of_mass_action(K: float, stoich: Dict[str, int], init_conc: Dict[str, float]) -> Dict[str, float]:
    """
    通用质量作用定律平衡浓度求解
    例如: aA + bB <-> cC + dD, K = ([C]^c [D]^d)/([A]^a [B]^b)
    :param K: 平衡常数
    :param stoich: 物种化学计量数, 正为生成物, 负为反应物, 如{"A":-1, "B":-1, "C":1, "D":1}
    :param init_conc: 初始浓度, {物种:浓度}
    :return: 平衡浓度, {物种:浓度}
    """
    species = list(stoich.keys())
    nu = np.array([stoich[s] for s in species])
    c0 = np.array([init_conc.get(s, 0.0) for s in species])

    def equations(x):
        c = c0 + nu * x[0]
        if np.any(c < 0):
            return 1e6  # 不允许负浓度
        prod_num = np.prod([c[i] ** max(nu[i], 0) for i in range(len(species)) if nu[i] > 0])
        prod_den = np.prod([c[i] ** abs(min(nu[i], 0)) for i in range(len(species)) if nu[i] < 0])
        Q = prod_num / prod_den if prod_den != 0 else 1e6
        return Q - K

    (x_sol,) = fsolve(equations, [0.0])
    c_eq = c0 + nu * x_sol
    return {s: max(0.0, c_eq[i]) for i, s in enumerate(species)}


if __name__ == "__main__":
    # 示例: H2 + I2 <-> 2HI, K=50, 初始各1 mol/L
    K = 50
    stoich = {"H2": -1, "I2": -1, "HI": 2}
    init_conc = {"H2": 1.0, "I2": 1.0, "HI": 0.0}
    eq = law_of_mass_action(K, stoich, init_conc)
    print("平衡浓度:", eq)
