def list_methods():
    """
    返回该算法下可用的方法字典，key为方法名，value为描述。
    """
    return {"fick_diffusion": "Fick扩散 (Fick diffusion)", "advection_dispersion": "对流-弥散方程 (Advection-dispersion)", "chromatography": "色谱分离理论板数 (Chromatography plate number)"}


# 地球化学物质迁移算法包
# Geochemical transport algorithm package


def list_elements(method: str):
    """
    返回每个方法支持的元素列表。迁移模型一般为通用（All/Any）。
    """
    if method in ("fick_diffusion", "advection_dispersion", "chromatography"):
        return ["Any"]
    return []


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """
    自动调度各迁移方法。参数通过kwargs传递。
    """
    import os

    import pandas as pd

    if method == "fick_diffusion":
        from .fick_diffusion import fick_flux

        df = pd.read_excel(input_path)
        df["J"] = df.apply(lambda row: fick_flux(row["D"], row["dc_dx"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fick_diffusion_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "advection_dispersion":
        from .advection_dispersion import advection_dispersion_1d

        df = pd.read_excel(input_path)
        df["C_xt"] = df.apply(lambda row: advection_dispersion_1d(row["C0"], row["v"], row["D"], row["x"], row["t"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "advection_dispersion_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    elif method == "chromatography":
        from .chromatography import plate_number

        df = pd.read_excel(input_path)
        df["N"] = df.apply(lambda row: plate_number(row["tR"], row["sigma"]), axis=1)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "chromatography_results.xlsx")
        df.to_excel(out_path, index=False)
        return {"status": "success", "out_path": out_path}
    else:
        raise NotImplementedError(f"Method {method} not implemented in algo_transport.")
