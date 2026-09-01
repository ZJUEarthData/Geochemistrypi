"""
Pitzer活度系数流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_thermodynamic.pitzer import pitzer_log_gamma


def run_pitzer(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算log10(活度系数)
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: log_gamma}
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
        log_gamma = pitzer_log_gamma(row["z"], row["I"])
        results[idx] = log_gamma
        df.loc[idx, "log_gamma"] = log_gamma

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # z   I
    # 2   0.5
    input_path = "example_pitzer.xlsx"
    print(run_pitzer(input_path))
