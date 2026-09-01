"""
质量作用定律（Law of Mass Action）流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_equilibrium.mass_action import law_of_mass_action


def run_mass_action(input_path: str, K: float, stoich: dict, output_path: str = None) -> dict:
    """
    读取输入数据，调用质量作用定律算法，输出平衡浓度
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param K: 平衡常数
    :param stoich: 化学计量数dict
    :param output_path: 可选，输出结果保存路径
    :return: 平衡浓度dict
    """
    # 支持多种输入格式
    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path, index_col=0)
        init_conc = df["conc"].to_dict()
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path, index_col=0)
        init_conc = df["conc"].to_dict()
    elif input_path.endswith(".json"):
        import json

        with open(input_path, "r") as f:
            init_conc = json.load(f)
    else:
        raise ValueError("Unsupported input format!")

    result = law_of_mass_action(K, stoich, init_conc)

    if output_path:
        pd.DataFrame(list(result.items()), columns=["species", "equilibrium_conc"]).to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # index   conc
    # H2      1.0
    # I2      1.0
    # HI      0.0
    K = 50
    stoich = {"H2": -1, "I2": -1, "HI": 2}
    input_path = "example_mass_action.xlsx"
    print(run_mass_action(input_path, K, stoich))
