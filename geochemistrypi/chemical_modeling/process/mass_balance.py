"""
质量平衡流程封装
支持Excel/CSV/Dict等多种输入
"""
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_equilibrium.mass_balance import mass_balance


def run_mass_balance(input_path: str, total_mass: float, output_path: str = None) -> dict:
    """
    读取输入数据，批量检查质量平衡
    :param input_path: 输入数据路径（支持Excel/CSV/JSON/或None）
    :param total_mass: 总质量
    :param output_path: 可选，输出结果保存路径
    :return: {行号: 是否平衡}
    """
    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path, index_col=0)
        species_conc = df["conc"].to_dict()
        result = mass_balance(species_conc, total_mass)
        df["is_balanced"] = result
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path, index_col=0)
        species_conc = df["conc"].to_dict()
        result = mass_balance(species_conc, total_mass)
        df["is_balanced"] = result
    elif input_path.endswith(".json"):
        import json

        with open(input_path, "r") as f:
            species_conc = json.load(f)
        result = mass_balance(species_conc, total_mass)
        df = pd.DataFrame(list(species_conc.items()), columns=["species", "conc"])
        df["is_balanced"] = result
    else:
        raise ValueError("Unsupported input format!")

    if output_path:
        df.to_csv(output_path, index=False)
    return {"is_balanced": result}


if __name__ == "__main__":
    # 示例：假设有Excel文件，内容如下：
    # index   conc
    # A      0.5
    # B      0.5
    input_path = "example_mass_balance.xlsx"
    print(run_mass_balance(input_path, total_mass=1.0))
