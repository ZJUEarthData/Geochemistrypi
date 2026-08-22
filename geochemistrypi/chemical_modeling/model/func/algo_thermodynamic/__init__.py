def list_methods():
    """
    返回该算法下可用的方法字典，key为方法名，value为描述。
    """
    return {"gibbs_minimization": "吉布斯自由能最小化 (Gibbs minimization)", "activity_coefficient": "活度系数模型 (Activity coefficient)", "vanthoff": "范特霍夫方程 (van't Hoff equation)"}


# 地球化学热力学算法包
# Geochemical thermodynamic algorithm package


def list_elements(method: str):
    """
    返回每个方法支持的元素列表。热力学模型一般为通用（All/Any）。
    """
    if method in ("gibbs_minimization", "activity_coefficient", "vanthoff"):
        return ["Any"]
    return []


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """
    自动调度各热力学方法。参数通过kwargs传递。
    """
    import os

    import pandas as pd

    if method == "gibbs_minimization":
        from .gibbs_minimization import gibbs_minimization

        df = pd.read_excel(input_path)
        import json

        results = df.apply(
            lambda row: gibbs_minimization(
                json.loads(row["gibbs_energies"]),
                json.loads(row["stoichiometry"]),
                json.loads(row["component_totals"]),
            ),
            axis=1,
        )
        df["minimum_gibbs"] = results.map(lambda result: result["minimum_gibbs"])
        df["equilibrium_moles"] = results.map(
            lambda result: json.dumps(
                result["equilibrium_moles"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        df["max_balance_residual"] = results.map(
            lambda result: result["max_balance_residual"]
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gibbs_minimization_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "activity_coefficient":
        from .activity_coefficient import debye_huckel_log_gamma

        df = pd.read_excel(input_path)
        df["log_gamma"] = df.apply(lambda row: debye_huckel_log_gamma(row["z"], row["ionic_strength"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "activity_coefficient_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "vanthoff":
        from .vanthoff import vanthoff_eq

        df = pd.read_excel(input_path)
        df["K2"] = df.apply(lambda row: vanthoff_eq(row["K1"], row["dH"], row["T1"], row["T2"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "vanthoff_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    else:
        raise NotImplementedError(f"Method {method} not implemented in algo_thermodynamic.")
