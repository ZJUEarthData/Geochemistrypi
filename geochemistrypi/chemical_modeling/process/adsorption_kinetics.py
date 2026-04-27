"""
吸附动力学流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_kinetic.adsorption_kinetics import pseudo_first_order, pseudo_second_order


def run_adsorption_kinetics(input_path: str, model: str = "first", output_path: str = None) -> dict:
    """
    读取输入数据，批量计算吸附动力学
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param model: 'first'为伪一级，'second'为伪二级
    :param output_path: 可选，输出结果保存路径
    :return: {行号: 吸附量}
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
        if model == "first":
            q = pseudo_first_order(row["qe"], row["k1"], row["t"])
        else:
            q = pseudo_second_order(row["qe"], row["k2"], row["t"])
        results[idx] = q
        df.loc[idx, "adsorption"] = q

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # qe   k1   t
    # 1.0  0.1  10
    input_path = "example_adsorption.xlsx"
    print(run_adsorption_kinetics(input_path, model="first"))
