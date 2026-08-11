# Online v0.8.0 核心迁移基线

## 本阶段范围

- 将官方 `v0.8.0` 历史合并到 `zzx`，保留现有 Online 与 Chemical Modeling。
- 移植 Data Mining 的多输出回归/分类、OPTICS、Time Series、自动聚类选择等核心代码。
- 修复官方 v0.8.0 中 OPTICS 未注册、Time Series 菜单无法进入执行流程的问题。
- 以 `geochemistrypi/_version.py` 作为唯一版本来源；构建元数据、FastAPI、健康检查和网页显示均由它派生。

## 测试基线

现代 Online 环境：

```powershell
.\.venv-online\Scripts\python.exe -m pytest tests\test_online_api.py tests\test_v080_baseline.py -q
cd geochemistrypi\frontend
pnpm type-check
```

旧 Data Mining 单元测试（不包含需要 MLflow、Ray、FLAML 等完整科研依赖的重型训练用例）：

```powershell
.\.venv-online\Scripts\python.exe -m pytest geochemistrypi\data_mining\tests\test_data\test_data_readiness.py -q
```

## 依赖边界

当前 Online 环境继续使用 Python 3.12 与现代 FastAPI/Pydantic/scikit-learn 依赖。官方 v0.8.0 的完整 CLI 训练栈仍带有 Python 3.9 时代的固定依赖（MLflow、Ray、FLAML、Pydantic 1 等），本阶段不强行降级 Online 环境。后续应逐项升级这些依赖并为每个模型建立可复现测试，之后再让 Online API 直接调用全部 v0.8.0 训练工作流。

## 已接入 Online 的 v0.8 回归方法

Online 使用后端模型注册表动态提供以下已验证方法：

- Linear Regression
- Polynomial Regression（二阶）
- Lasso Regression
- Elastic Net
- Bayesian Ridge Regression
- Ridge Regression

请求仍使用 `/api/data-mining/regression`，新增可选表单字段 `model`。未提供时默认使用 `linear_regression`，因此旧版调用方式保持兼容。模型名称、显示名称、指标、方程、系数和预测均写入版本化 JSON 报告。

## 已接入 Online 的 v0.8 分类方法

以下方法由同一后端注册表动态提供，并已经过分层训练/测试划分、指标、混淆矩阵和下载报告验证：

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- Extra-Trees
- Multi-layer Perceptron
- Gradient Boosting
- K-Nearest Neighbors
- Stochastic Gradient Descent
- AdaBoost

请求继续使用 `/api/data-mining/classification`，新增可选表单字段 `model`；未提供时默认使用 `logistic_regression`。XGBoost 暂不标记为可用，待其现代可选依赖和模型安全边界单独完成验证后再加入注册表。

## 已接入 Online 的 v0.8 聚类方法

- K-Means
- DBSCAN
- Agglomerative Clustering
- Affinity Propagation
- Mean Shift
- OPTICS

请求继续使用 `/api/data-mining/clustering`，新增可选表单字段 `model`。K-Means 和 Agglomerative 使用 `cluster_count`；其余方法自动估计簇数。密度聚类产生的标签 `-1` 作为噪声单独统计，聚类指标和中心不包含噪声点，CSV 分配结果仍保留完整标签以便审计。
