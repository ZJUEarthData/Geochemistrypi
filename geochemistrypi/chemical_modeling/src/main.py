# ------------------------------------------------------------------------------
# Integrated launcher for Hg fractionation and Mo double-diluent solvers
# ------------------------------------------------------------------------------
# Usage (from project root or src folder):
#     python src/main.py
#
# After launch you will be asked:
# Select method:
# 1. Internal standard method
# 2. Double spike method
# Enter 1 or 2:
# Type 1 or 2 and press <Enter>.
# ------------------------------------------------------------------------------

import os
import runpy
import sys
from typing import List, Tuple

import pandas as pd

# SciPy solver (for Mo)
try:
    from scipy.optimize import fsolve, root
except ImportError:
    print("Missing scipy. Please install it first: pip install scipy.")
    sys.exit(1)


def wait_enter() -> None:
    input("Welcome to the GeochemistryPi Chemical-Modeling section. Press Enter to continue.")


# ---------------------------
# Helper: run external Hg internal-standard script
# ---------------------------
def run_hg_internal_script() -> None:
    script_path = os.path.join(os.path.dirname(__file__), "Hg_Internal_standard_method.py")
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        print(f"Hg_Internal_standard_method.py not found: {script_path}")
        return
    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        print(f"Failed to run {script_path}: {e}")


# ---------------------------
# Mo double-spike equations and runner
# ---------------------------
def mo_equations(params: Tuple[float, float, float], R_sp: Tuple[float, float, float], R_std: Tuple[float, float, float], r_mix: Tuple[float, float, float]) -> List[float]:
    φ_ref, β_sple, β_mix = params
    R_100_sp, R_98_sp, R_97_sp = R_sp
    R_100_std, R_98_std, R_97_std = R_std
    r_100_mix, r_98_mix, r_97_mix = r_mix

    f1 = φ_ref * R_100_sp + (1 - φ_ref) * R_100_std * (95 / 100) ** β_sple - r_100_mix * (95 / 100) ** β_mix
    f2 = φ_ref * R_98_sp + (1 - φ_ref) * R_98_std * (95 / 98) ** β_sple - r_98_mix * (95 / 98) ** β_mix
    f3 = φ_ref * R_97_sp + (1 - φ_ref) * R_97_std * (95 / 97) ** β_sple - r_97_mix * (95 / 97) ** β_mix
    return [f1, f2, f3]


def solve_mo_fsolve(R_sp, R_std, r_mix, initial_guess=(0.5, 0.5, 2.0)):
    return fsolve(mo_equations, initial_guess, args=(R_sp, R_std, r_mix))


def solve_mo_root(R_sp, R_std, r_mix, initial_guess=(0.5, 0.5, 2.0)):
    sol = root(mo_equations, initial_guess, args=(R_sp, R_std, r_mix), method="hybr")
    return sol.x


def run_mo_process():
    excel_path = os.path.join(os.path.dirname(__file__), "..", "data", "Mo_data.xlsx")
    excel_path = os.path.abspath(excel_path)
    if not os.path.exists(excel_path):
        print(f"Mo data file not found: {excel_path}")
        return

    try:
        df = pd.read_excel(excel_path, sheet_name="3程序处理_输入常数")
    except Exception as e:
        print(f"Failed to read Mo data: {e}")
        return

    try:
        R_sp = (df.loc[0, "R_100_sp"], df.loc[0, "R_98_sp"], df.loc[0, "R_97_sp"])
        R_std = (df.loc[0, "R_100_std"], df.loc[0, "R_98_std"], df.loc[0, "R_97_std"])
        r_mix = (df.loc[0, "r_100_mix"], df.loc[0, "r_98_mix"], df.loc[0, "r_97_mix"])
    except Exception as e:
        print(f"Failed to extract Mo parameters: {e}")
        return

    sol_fsolve = solve_mo_fsolve(R_sp, R_std, r_mix)
    sol_root = solve_mo_root(R_sp, R_std, r_mix)

    results = []
    for method, sol in [("fsolve", sol_fsolve), ("root", sol_root)]:
        results.append({"method": method, "phi_ref_sp": float(sol[0]), "beta_sple": float(sol[1]), "beta_mix": float(sol[2])})
    result_df = pd.DataFrame(results)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "Mo_results.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Mo calculation finished. Results saved to: {os.path.abspath(csv_path)}")


# ---------------------------
# Main: method first, then element selection
# ---------------------------
def main() -> None:
    wait_enter()

    prompt_method = "Select method:\n" "1. Internal standard method\n" "2. Double spike method\n" "Enter 1 or 2: "
    method_choice = input(prompt_method).strip()
    if method_choice not in ("1", "2"):
        print("Invalid input. Please enter 1 or 2.")
        return

    if method_choice == "1":
        # Internal standard method -> list available elements (currently only Hg)
        prompt_element = "Internal standard method - select element:\n" "1. Hg\n" "Enter 1 to run Hg internal-standard method (or anything else to cancel): "
        el_choice = input(prompt_element).strip()
        if el_choice == "1":
            run_hg_internal_script()
        else:
            print("Cancelled.")
    else:
        # Double spike method -> list available elements (currently only Mo)
        prompt_element = "Double spike method - select element:\n" "1. Mo\n" "Enter 1 to run Mo double-spike calculation (or anything else to cancel): "
        el_choice = input(prompt_element).strip()
        if el_choice == "1":
            run_mo_process()
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
