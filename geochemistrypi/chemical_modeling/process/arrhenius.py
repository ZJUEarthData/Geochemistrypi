"""
Arrhenius方程流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_kinetic.arrhenius import arrhenius_rate_constant


def run_arrhenius(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算速率常数
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: 速率常数}
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
        k = arrhenius_rate_constant(row["A"], row["Ea"], row["T"])
        results[idx] = k
        df.loc[idx, "rate_constant"] = k

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # A       Ea      T
    # 1e13    80000   298
    input_path = "example_arrhenius.xlsx"
    print(run_arrhenius(input_path))
