
# algo_transport

## 简介 | Introduction
地球化学物质迁移算法模块，包含Fick扩散、对流-弥散、色谱分离等模型。

This module provides geochemical transport algorithms, including Fick diffusion, advection-dispersion, and chromatography plate number models.

---

## 子模块与原理 | Submodules & Principles

### 1. Fick扩散 fick_diffusion.py
J = -D * (dc/dx)

### 2. 对流-弥散方程 advection_dispersion.py
一维瞬时点源解析解。
1D instantaneous point source analytical solution.

### 3. 色谱分离理论板数 chromatography.py
N = (tR / sigma)^2

---

## 参数说明 | Parameters

### fick_flux
- D: float，扩散系数 / diffusion coefficient
- dc_dx: float，浓度梯度 / concentration gradient

### advection_dispersion_1d
- C0: float，初始浓度 / initial concentration
- v: float，流速 / velocity
- D: float，弥散系数 / dispersion coefficient
- x: float，距离 / distance
- t: float，时间 / time

### plate_number
- tR: float，保留时间 / retention time
- sigma: float，峰宽标准差 / peak width std

---

## 输入输出 | Input & Output

所有函数均为纯函数，输入参数见上，输出为float。
All functions are pure, see above for input/output types.

---

## 用法示例 | Usage Example

```python
from fick_diffusion import fick_flux
from advection_dispersion import advection_dispersion_1d
from chromatography import plate_number

# Fick扩散
print(fick_flux(1e-9, 0.01))

# 对流-弥散
print(advection_dispersion_1d(1.0, 0.1, 0.01, 10, 5))

# 色谱板数
print(plate_number(10, 2))
```

---

## 扩展说明 | Notes
- 可扩展更多迁移模型。
- Extendable for more transport models.
