"""
离子交换平衡流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_equilibrium.ion_exchange import gaines_thomas_exchange


def run_ion_exchange(input_path: str, selectivity: float, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算离子交换分布比
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param selectivity: 选择性系数
    :param output_path: 可选，输出结果保存路径
    :return: {行号: 分布比}
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
        ratio = gaines_thomas_exchange(row["eq_conc_a"], row["eq_conc_b"], selectivity)
        results[idx] = ratio
        df.loc[idx, "A_B_ratio"] = ratio

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # eq_conc_a   eq_conc_b
    # 0.1         0.2
    input_path = "example_ion_exchange.xlsx"
    print(run_ion_exchange(input_path, selectivity=2.0))
