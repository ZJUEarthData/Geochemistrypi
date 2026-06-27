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
        # 假设输入有gibbs_energies, n（都为list字符串）
        import ast

        df["Gibbs"] = df.apply(lambda row: gibbs_minimization(ast.literal_eval(row["gibbs_energies"]), ast.literal_eval(row["n"])), axis=1)
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
