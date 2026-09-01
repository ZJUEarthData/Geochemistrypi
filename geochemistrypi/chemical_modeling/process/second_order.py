"""
二级反应动力学流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_kinetic.second_order import second_order_conc


def run_second_order(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算二级反应浓度
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: 浓度}
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
        c = second_order_conc(row["c0"], row["k"], row["t"])
        results[idx] = c
        df.loc[idx, "conc"] = c

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # c0   k    t
    # 1.0  0.1  10
    input_path = "example_second_order.xlsx"
    print(run_second_order(input_path))
