"""
溶解-沉淀平衡流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_equilibrium.precipitation_dissolution import calc_saturation_index, is_precipitation


def run_precipitation_dissolution(input_path: str, ksp: float, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算饱和指数和沉淀判断
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param ksp: 溶度积常数
    :param output_path: 可选，输出结果保存路径
    :return: {行号: (SI, 是否沉淀)}
    """
    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        import json

        with open(input_path, "r") as f:
            df = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported input format!")

    results = {}
    for idx, row in df.iterrows():
        si = calc_saturation_index(row["ion_activity_product"], ksp)
        precip = is_precipitation(si)
        results[idx] = (si, precip)
        df.loc[idx, "saturation_index"] = si
        df.loc[idx, "is_precipitation"] = precip

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # ion_activity_product
    # 1e-8
    input_path = "example_precipitation.xlsx"
    print(run_precipitation_dissolution(input_path, ksp=1e-9))
