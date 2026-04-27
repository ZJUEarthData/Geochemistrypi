
# algo_kinetic

## 简介 | Introduction
地球化学动力学算法模块，包含一级/二级反应、放射性衰变、吸附动力学等模型。

This module provides geochemical kinetic algorithms, including first/second-order reactions, radioactive decay, and adsorption kinetics.

---

## 子模块与原理 | Submodules & Principles

### 1. 一级反应动力学 first_order.py
C = C0 * exp(-kt)

### 2. 二级反应动力学 second_order.py
1/C = 1/C0 + kt

### 3. 放射性衰变 radioactive_decay.py
N = N0 * exp(-λt)

### 4. 吸附动力学 adsorption_kinetics.py
伪一级、伪二级动力学模型

---

## 参数说明 | Parameters

### first_order_conc
- c0: float，初始浓度 / initial concentration
- k: float，速率常数 / rate constant
- t: float，时间 / time

### second_order_conc
- c0: float，初始浓度 / initial concentration
- k: float，速率常数 / rate constant
- t: float，时间 / time

### radioactive_decay
- n0: float，初始核素数量 / initial nuclide amount
- decay_const: float，衰变常数 / decay constant
- t: float，时间 / time

### pseudo_first_order, pseudo_second_order
- qe: float，平衡吸附量 / equilibrium adsorption
- k1/k2: float，速率常数 / rate constant
- t: float，时间 / time

---

## 输入输出 | Input & Output

所有函数均为纯函数，输入参数见上，输出为float。
All functions are pure, see above for input/output types.

---

## 用法示例 | Usage Example

```python
from first_order import first_order_conc
from second_order import second_order_conc
from radioactive_decay import radioactive_decay
from adsorption_kinetics import pseudo_first_order, pseudo_second_order

# 一级反应
print(first_order_conc(1.0, 0.1, 10))

# 二级反应
print(second_order_conc(1.0, 0.1, 10))

# 衰变
print(radioactive_decay(100, 0.01, 50))

# 吸附动力学
print(pseudo_first_order(10, 0.2, 5))
print(pseudo_second_order(10, 0.02, 5))
```

---

## 扩展说明 | Notes
- 可扩展更多复杂动力学模型。
- Extendable for more complex kinetic models.
