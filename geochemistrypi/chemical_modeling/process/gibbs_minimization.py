"""
吉布斯自由能最小化流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_thermodynamic.gibbs_minimization import gibbs_minimization


def run_gibbs_minimization(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算总吉布斯自由能
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: Gibbs}
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
        # 假设输入有gibbs_energies, n（都为list字符串）
        import ast

        gibbs_energies = ast.literal_eval(row["gibbs_energies"])
        n = ast.literal_eval(row["n"])
        G = gibbs_minimization(gibbs_energies, n)
        results[idx] = G
        df.loc[idx, "Gibbs"] = G

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # gibbs_energies      n
    # [0,-10,-20]        [1,2,3]
    input_path = "example_gibbs_minimization.xlsx"
    print(run_gibbs_minimization(input_path))
