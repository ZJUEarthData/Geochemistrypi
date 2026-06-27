"""
色谱理论板数流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_transport.chromatography import plate_number


def run_chromatography(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算理论板数
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: N}
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
        N = plate_number(row["tR"], row["sigma"])
        results[idx] = N
        df.loc[idx, "plate_number"] = N

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # tR   sigma
    # 10   1
    input_path = "example_chromatography.xlsx"
    print(run_chromatography(input_path))
