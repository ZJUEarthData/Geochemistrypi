"""
反应-迁移耦合流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_transport.reactive_transport import reactive_transport_1d


def run_reactive_transport(input_path: str, output_path: str = None) -> dict:
    """
    读取输入数据，批量计算反应-迁移浓度
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param output_path: 可选，输出结果保存路径
    :return: {行号: C}
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
        C = reactive_transport_1d(row["C0"], row["v"], row["D"], row["k"], row["x"], row["t"])
        results[idx] = C
        df.loc[idx, "C_xt"] = C

    if output_path:
        df.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # C0   v      D      k      x     t
    # 1.0  1e-5   1e-9   1e-6  0.1   3600
    input_path = "example_reactive_transport.xlsx"
    print(run_reactive_transport(input_path))
