def list_methods():
    """
    返回该算法下可用的方法字典，key为方法名，value为描述。
    """
    return {
        "first_order": "一级反应动力学 (First-order kinetics)",
        "second_order": "二级反应动力学 (Second-order kinetics)",
        "radioactive_decay": "放射性衰变 (Radioactive decay)",
        "adsorption_kinetics": "吸附动力学 (Adsorption kinetics)",
    }


def list_elements(method: str):
    """
    返回每个方法支持的元素列表。动力学模型一般为通用（All/Any）。
    """
    # 可根据实际需求细化
    if method in ("first_order", "second_order", "radioactive_decay", "adsorption_kinetics"):
        return ["Any"]
    return []


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """
    自动调度各动力学方法。参数通过kwargs传递。
    """
    import os

    import pandas as pd

    if method == "first_order":
        from .first_order import first_order_conc

        df = pd.read_excel(input_path)
        df["C_t"] = df.apply(lambda row: first_order_conc(row["c0"], row["k"], row["t"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "first_order_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "second_order":
        from .second_order import second_order_conc

        df = pd.read_excel(input_path)
        df["C_t"] = df.apply(lambda row: second_order_conc(row["c0"], row["k"], row["t"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "second_order_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "radioactive_decay":
        from .radioactive_decay import radioactive_decay

        df = pd.read_excel(input_path)
        df["N_t"] = df.apply(lambda row: radioactive_decay(row["n0"], row["decay_const"], row["t"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "radioactive_decay_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "adsorption_kinetics":
        from .adsorption_kinetics import pseudo_first_order, pseudo_second_order

        df = pd.read_excel(input_path)
        # 支持两种模型，优先用参数model指定，否则默认伪一级
        model = kwargs.get("model", "first")
        if model == "first":
            df["q_t"] = df.apply(lambda row: pseudo_first_order(row["qe"], row["k1"], row["t"]), axis=1)
        else:
            df["q_t"] = df.apply(lambda row: pseudo_second_order(row["qe"], row["k2"], row["t"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"adsorption_{model}_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    else:
        raise NotImplementedError(f"Method {method} not implemented in algo_kinetic.")


# 地球化学动力学算法包
# Geochemical kinetic algorithm package
