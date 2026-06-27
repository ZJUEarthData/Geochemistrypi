
# algo_equilibrium

## 简介 | Introduction
地球化学平衡算法模块，包含常见的质量守恒、溶解-沉淀平衡、离子交换等模型。

This module provides geochemical equilibrium algorithms, including mass balance, precipitation/dissolution equilibrium, and ion exchange.

---

## 子模块与原理 | Submodules & Principles

### 1. 质量平衡 mass_balance.py
用于检查溶液或反应体系的质量守恒。
Checks mass conservation in solution or reaction systems.

### 2. 溶解-沉淀平衡 precipitation_dissolution.py
通过饱和指数（SI）判断溶解/沉淀是否发生。
Determines precipitation/dissolution by saturation index (SI).

### 3. 离子交换 ion_exchange.py
Gaines-Thomas模型，描述两种离子在交换体上的分布。
Gaines-Thomas model for cation exchange.

---

## 参数说明 | Parameters

### mass_balance
- species_conc: dict，物种浓度字典（{物种: 浓度}）/ species concentration dict
- total_mass: float，总质量 / total mass

### calc_saturation_index
- ion_activity_product: float，离子活度积 / ion activity product
- ksp: float，溶度积常数 / solubility product

### is_precipitation
- si: float，饱和指数 / saturation index

### gaines_thomas_exchange
- eq_conc_a: float，A离子平衡浓度 / equilibrium conc. of A
- eq_conc_b: float，B离子平衡浓度 / equilibrium conc. of B
- selectivity: float，选择性系数 / selectivity coefficient

---

## 输入输出 | Input & Output

所有函数均为纯函数，输入参数见上，输出为bool/float等。
All functions are pure, see above for input/output types.

---

## 用法示例 | Usage Example

```python
from mass_balance import mass_balance
from precipitation_dissolution import calc_saturation_index, is_precipitation
from ion_exchange import gaines_thomas_exchange

# 质量平衡
species = {'Na+': 0.1, 'Cl-': 0.1}
print(mass_balance(species, 0.2))  # True

# 溶解-沉淀平衡
si = calc_saturation_index(1e-8, 1e-9)
print(is_precipitation(si))  # True

# 离子交换
ratio = gaines_thomas_exchange(0.05, 0.05, 1.2)
print(ratio)
```

---

## 扩展说明 | Notes
- 可根据实际需求扩展更多平衡模型。
- Each file can be extended for more complex equilibrium models.
