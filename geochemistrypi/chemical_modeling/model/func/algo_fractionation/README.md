# algo_fractionation

## 简介 | Introduction
地球化学分馏算法模块，包含常用的内标法（Hg）和双稀释法（Mo）等分馏校正方法。

This module provides geochemical isotope fractionation algorithms, including internal standard method (Hg) and double-spike method (Mo).

---

## 子模块与原理 | Submodules & Principles

### 1. 内标法 internal_standard.py
适用于汞（Hg）同位素分馏校正。通过内标样品（如3133）前后信号，计算样品分馏校正值。

For Hg isotope fractionation correction. Uses internal standard (e.g., 3133) before/after sample to compute correction.

### 2. 双稀释法 double_spike.py
适用于钼（Mo）同位素分馏校正。通过双稀释混合物的同位素比，结合非线性方程组（fsolve/root）反演分馏参数。

For Mo isotope fractionation correction. Uses double-spike mixture ratios and nonlinear equation solver (fsolve/root) to infer parameters.

---

## 参数说明 | Parameters

### internal_standard.run
- element: str，元素（目前仅支持"Hg"）/ element (currently only "Hg")
- input_path: str，输入Excel路径 / input Excel file path
- out_dir: str，输出目录 / output directory

### double_spike.run
- element: str，元素（目前仅支持"Mo"）/ element (currently only "Mo")
- input_path: str，输入Excel路径 / input Excel file path
- out_dir: str，输出目录 / output directory
- solver: str，求解器（"fsolve"或"root"，可选）/ solver ("fsolve" or "root", optional)

---

## 输入输出 | Input & Output

所有方法均以Excel为输入，输出为校正后结果的Excel或CSV文件，路径返回在out_path字段。
All methods take Excel as input, output corrected results as Excel/CSV, path returned in out_path.

---

## 用法示例 | Usage Example

```python
from algo_fractionation import run

# 内标法（Hg）
result = run("internal_standard", "Hg", "Hg_input.xlsx", "./results")
print(result["out_path"])

# 双稀释法（Mo）
result = run("double_spike", "Mo", "Mo_input.xlsx", "./results", solver="fsolve")
print(result["out_path"])
```

---

## 任务步骤 | Task Steps
1. 准备输入数据Excel（格式见样例）。
2. 选择分馏方法（internal_standard 或 double_spike）。
3. 指定元素（Hg 或 Mo）。
4. 运行run方法，获得校正结果。
5. 结果文件保存在指定输出目录。

---

## 扩展说明 | Notes
- 可扩展支持更多元素和分馏模型。
- 输入数据需包含必要的同位素比信息，具体格式见各方法实现。
- Each method can be extended for more elements and models. Input Excel must contain required isotope ratio info.
