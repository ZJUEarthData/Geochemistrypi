
# algo_thermodynamic

## 简介 | Introduction
地球化学热力学算法模块，包含吉布斯自由能最小化、活度系数、范特霍夫方程等模型。

This module provides geochemical thermodynamic algorithms, including Gibbs free energy minimization, activity coefficient models, and van't Hoff equation.

---

## 子模块与原理 | Submodules & Principles

### 1. 吉布斯自由能最小化 gibbs_minimization.py
用于多组分体系的热力学平衡计算。
Thermodynamic equilibrium for multi-component systems.

### 2. 活度系数模型 activity_coefficient.py
Debye-Hückel模型，计算离子强度和活度系数。
Debye-Hückel model for ionic strength and activity coefficient.

### 3. 范特霍夫方程 vanthoff.py
描述温度对平衡常数的影响。
Describes temperature effect on equilibrium constant.

---

## 参数说明 | Parameters

### gibbs_minimization
- gibbs_energies: list，各组分摩尔吉布斯自由能 / molar Gibbs energies
- n: list，各组分摩尔数 / molar amounts

### debye_huckel_ionic_strength
- concs: list，离子浓度 / ion concentrations
- charges: list，离子电荷 / ion charges

### debye_huckel_log_gamma
- z: float，离子电荷 / ion charge
- ionic_strength: float，离子强度 / ionic strength
- A: float，Debye-Hückel常数 / Debye-Hückel constant
- a: float，水合半径 / hydrated radius

### vanthoff_eq
- K1: float，初始温度下的平衡常数 / equilibrium constant at T1
- dH: float，反应焓变 / enthalpy change
- T1: float，初始温度 / initial temperature
- T2: float，目标温度 / target temperature
- R: float，气体常数 / gas constant

---

## 输入输出 | Input & Output

所有函数均为纯函数，输入参数见上，输出为float。
All functions are pure, see above for input/output types.

---

## 用法示例 | Usage Example

```python
from gibbs_minimization import gibbs_minimization
from activity_coefficient import debye_huckel_ionic_strength, debye_huckel_log_gamma
from vanthoff import vanthoff_eq

# 吉布斯自由能
print(gibbs_minimization([10, 20], [1, 2]))

# 活度系数
I = debye_huckel_ionic_strength([0.1, 0.2], [1, -1])
print(debye_huckel_log_gamma(1, I))

# 范特霍夫方程
print(vanthoff_eq(1.0, -40000, 298, 310))
```

---

## 扩展说明 | Notes
- 可扩展更多热力学模型。
- Extendable for more thermodynamic models.
