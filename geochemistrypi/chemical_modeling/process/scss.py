"""
SCSS（硫溶解度）主流程模块，集成多种算法与模型。
- 支持 Rubie、Ding、Blanchard、Hybrid Model 等主流算法
- 支持 CLI 自动调度与批量数据处理
- 依赖 model.sulfide 下的 dataset、emodels、rmodels、nn_model
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from geochemistrypi.chemical_modeling.model.func.algo_solubility.dataset import train_test_dfs
from geochemistrypi.chemical_modeling.model.func.algo_solubility.emodels import EModels

# from geochemistrypi.chemical_modeling.model.sulfide.nn_model import NN, NNP  # 如需神经网络模型可解注

# 1. 各主流SCSS算法实现


def scss_rubie(pressure, temperature):
    """Rubie et al. (2016) 算法"""
    return np.exp(14.2 - 11032 / temperature - 379 * pressure / temperature)


def scss_ding(A, B, CiXm, D, E, T, P, X_FeO, X_TiO2):
    """Ding et al. (2018) 算法"""
    return np.exp(A + B / T + CiXm + D * X_FeO * X_TiO2 + E * P / T)


def scss_blanchard(a, b, c, XmAm, T, P, X_FeS, X_FeO):
    """Blanchard et al. (2021) 算法"""
    return np.exp(a + b / T + c * P / T + XmAm / T + np.log(X_FeS) - np.log(X_FeO))


# 2. 主流程调度接口（可被CLI调用）
def run(input_path: Union[str, Path], method: str = "rubie", out_path: Union[str, Path] = None, **kwargs):
    """
    SCSS主流程调度，支持多算法选择与批量数据预测。
    method: rubie | ding | blanchard | hybrid
    input_path: 输入Excel路径，需包含标准列
    out_path: 输出结果Excel路径
    """
    df = pd.read_excel(input_path)
    if method == "rubie":
        df["SCSS_pred"] = scss_rubie(df["Pressure"], df["T"])
    elif method == "ding":
        # 需补充参数映射与特征工程
        df["SCSS_pred"] = scss_ding(...)
    elif method == "blanchard":
        df["SCSS_pred"] = scss_blanchard(...)
    elif method == "hybrid":
        # 机器学习集成预测
        x_train, x_test, y_train, y_test, scaler = train_test_dfs(df, test_size=0.30)
        emp = EModels(x_train, x_test, y_train, y_test)
        _, y_pred_emp = emp.predict_em(scaler.transform(df.drop("SCSS", axis=1)))
        df["SCSS_pred"] = y_pred_emp[0]  # 以XGB为例
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    if out_path:
        df.to_excel(out_path, index=False)
    return df


# 3. CLI/菜单集成接口
def list_methods():
    return {"rubie": "Rubie et al. (2016) 经验公式", "ding": "Ding et al. (2018) 经验公式", "blanchard": "Blanchard et al. (2021) 经验公式", "hybrid": "集成机器学习模型 (Zhang et al. 2024)"}


def list_elements(method: str):
    return ["S"]
