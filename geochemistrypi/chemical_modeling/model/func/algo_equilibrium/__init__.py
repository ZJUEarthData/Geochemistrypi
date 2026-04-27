def list_methods():
    """
    返回该算法下可用的方法字典，key为方法名，value为描述。
    """
    return {
        "mass_balance": "质量平衡法 (Mass balance)",
        "precipitation_dissolution": "溶解-沉淀平衡 (Precipitation/Dissolution)",
        "ion_exchange": "离子交换平衡 (Ion exchange)",
        "mass_action": "质量作用定律 (Law of mass action)",
    }


# 地球化学平衡算法包
# Geochemical equilibrium algorithm package


def list_elements(method: str):
    """
    返回每个方法支持的元素列表。平衡模型一般为通用（All/Any）。
    """
    if method in ("mass_balance", "precipitation_dissolution", "ion_exchange", "mass_action"):
        return ["Any"]
    return []


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """
    自动调度各平衡方法。参数通过kwargs传递。
    """
    import os

    import pandas as pd

    if method == "mass_balance":
        from .mass_balance import run_mass_balance

        K = kwargs.get("K")
        stoch = kwargs.get("stoich")
        result = run_mass_balance(input_path, K, stoch)
        out_path = os.path.join(out_dir, "mass_balance_results.xlsx")
        pd.DataFrame(list(result.items()), columns=["species", "equilibrium_conc"]).to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "precipitation_dissolution":
        from .precipitation_dissolution import run_precipitation_dissolution

        K = kwargs.get("K")
        stoch = kwargs.get("stoich")
        result = run_precipitation_dissolution(input_path, K, stoch)
        out_path = os.path.join(out_dir, "precipitation_dissolution_results.xlsx")
        pd.DataFrame(list(result.items()), columns=["species", "equilibrium_conc"]).to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "ion_exchange":
        from .ion_exchange import run_ion_exchange

        K = kwargs.get("K")
        stoch = kwargs.get("stoich")
        result = run_ion_exchange(input_path, K, stoch)
        out_path = os.path.join(out_dir, "ion_exchange_results.xlsx")
        pd.DataFrame(list(result.items()), columns=["species", "equilibrium_conc"]).to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "mass_action":
        from .mass_action import run_mass_action

        K = kwargs.get("K")
        stoch = kwargs.get("stoich")
        result = run_mass_action(input_path, K, stoch)
        out_path = os.path.join(out_dir, "mass_action_results.xlsx")
        pd.DataFrame(list(result.items()), columns=["species", "equilibrium_conc"]).to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    else:
        raise NotImplementedError(f"Method {method} not implemented in algo_equilibrium.")
