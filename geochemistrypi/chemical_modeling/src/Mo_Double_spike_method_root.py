"""
Root Double Dilution Solver
"""
from typing import List, Tuple

from scipy.optimize import root


def equations(params: Tuple[float, float, float], R_sp: Tuple[float, float, float], R_std: Tuple[float, float, float], r_mix: Tuple[float, float, float]) -> List[float]:
    φ_ref, β_sple, β_mix = params
    R_100_sp, R_98_sp, R_97_sp = R_sp
    R_100_std, R_98_std, R_97_std = R_std
    r_100_mix, r_98_mix, r_97_mix = r_mix

    f1 = φ_ref * R_100_sp + (1 - φ_ref) * R_100_std * (95 / 100) ** β_sple - r_100_mix * (95 / 100) ** β_mix
    f2 = φ_ref * R_98_sp + (1 - φ_ref) * R_98_std * (95 / 98) ** β_sple - r_98_mix * (95 / 98) ** β_mix
    f3 = φ_ref * R_97_sp + (1 - φ_ref) * R_97_std * (95 / 97) ** β_sple - r_97_mix * (95 / 97) ** β_mix
    return [f1, f2, f3]


def solve(R_sp, R_std, r_mix, initial_guess=(0.5, 0.5, 2.0)):
    sol = root(equations, initial_guess, args=(R_sp, R_std, r_mix), method="hybr")
    return sol.x
