# Geochemistryπ 本地 MCP 平台开发实施路线图

> 文档状态：实施规划稿（已审查）
> 修订日期：2026-07-25
> 适用项目：Geochemistryπ
> 当前核心版本：0.8.0
> 规划对象：Geochemistryπ 本地 MCP 平台（独立版本线）
> 当前主线：本地运行、复用现有 CLI 能力、论文级实验与可复现评测

## 1. 文档目的

本文档规划如何把 Geochemistryπ 已有且相对完整的 CLI 机器学习能力，转化为可被 AI Agent 稳定调用的本地 MCP 平台。

本路线图的近期目标不是重写全部 Geochemistryπ，也不是立即建设云端多用户产品，而是完成一条质量足够高、能够支撑论文实验的本地主线：

```text
现有 CLI 能力
  → 非交互核心与批处理 Worker
  → 本地任务管理和可复现实验产物
  → 本地 stdio MCP
  → 覆盖现有主要机器学习任务
  → Agent Benchmark 与论文级发布
```

本文档中的版本号属于 **Geochemistryπ 本地 MCP 平台**，不替代 Geochemistryπ 核心项目版本号。

## 2. 范围决策

### 2.1 当前必须完成

- 复用现有 CLI 的分类、回归、聚类、降维和异常检测能力；
- 把交互式输入转化为严格、可验证的结构化配置；
- 每次实验通过独立 Worker 进程执行；
- 提供本地 stdio MCP；
- 保存完整模型、预处理 Pipeline、指标、日志、manifest 和 provenance；
- 支持任务查询、取消、故障恢复和受控重试；
- 支持 Windows、Linux 和 macOS 的干净环境安装验证；
- 建立科学正确性、协议、安全和跨平台测试；
- 建立可复现的 Agent Benchmark，支撑论文结论；
- 提供一个薄 Python SDK，用于 Notebook、自动化脚本和论文中的接口消融实验。

### 2.2 当前明确不作为前置条件

以下内容有价值，但不阻塞本地论文版：

- 公网 REST API；
- 远程 Streamable HTTP MCP；
- 多用户认证、计费、配额和对象存储；
- 云端 CPU/GPU 调度；
- 完整 Dash 重构；
- Chemical Modeling；
- 任意 Python 或 shell 代码执行；
- Agent 自动安装系统级依赖；
- Agent 自动修改用户原始数据；
- 将完整数据集发送给 LLM。

### 2.3 后续扩展轨道

本地论文主线完成后，可独立推进：

```text
产品化轨道：REST / 远程 MCP / 多用户 / 云端任务队列
科学扩展轨道：Chemical Modeling / 单位系统 / 物理约束 / 联合 Benchmark
```

这两个轨道不应反向扩大当前 MCP 主线的范围。

## 3. 核心架构结论

采用“一套核心任务语义，多个轻量适配器”的架构：

```text
                         ┌── CLI
用户或 AI ──结构化请求──> ├── Python SDK
                         └── MCP
                              │
                              ▼
                    Geochemistryπ Run Manager
                              │
                              ▼
                     独立 Geochemistryπ Worker
                              │
                              ▼
                Application Service / ML Engine
                              │
                              ▼
             Runs / Artifacts / Manifest / Provenance
```

必须坚持：

1. CLI 是功能来源，但不是 MCP 的交互协议。
2. MCP 不模拟键盘输入，不解析 CLI 自然语言输出。
3. MCP 不暴露任意命令、模块路径或 Python 代码执行能力。
4. 机器学习请求使用严格 Schema，结果使用结构化 JSON。
5. 一个实验对应一个独立 Worker 进程和一个独立运行目录。
6. MCP 进程 stdout 只包含协议消息；Worker 输出重定向到日志。
7. 第一条纵向链路先完成分类，再按同一模式迁移其他现有 CLI 任务。
8. 科学正确性优先于对旧 CLI 指标的机械复制。
9. 本地论文版不依赖远程多用户基础设施。
10. MCP、Python SDK 和 CLI 最终共享相同的任务契约和 Worker。

## 4. 当前项目基线

### 4.1 已有能力

现有完整能力主要集中在：

- `geochemistrypi/data_mining/cli_pipeline.py`
- `geochemistrypi/data_mining/process/`
- `geochemistrypi/data_mining/model/`
- `geochemistrypi/data_mining/data/`
- `geochemistrypi/data_mining/utils/`

CLI 已覆盖：

- CSV、XLSX 等数据读取；
- 数据选择、统计和缺失值处理；
- 特征工程、特征缩放和特征选择；
- 分类、回归、聚类、降维和异常检测；
- 手动调参和 AutoML；
- 模型评价、图表和数据产物；
- 模型推理和预处理重放；
- MLflow 实验记录。

### 4.2 已确认的工程阻塞

截至 2026-07-25 的静态检查显示：

- 项目运行环境为 Python 3.9、Pydantic 1；
- MCP Python SDK 稳定 v1 需要 Python 3.10+、Pydantic 2；
- 分类模型主文件约 3942 行；
- 项目中约有 936 处 `print()`；
- 约有 352 处 `GEOPI_OUTPUT...` 引用；
- 约有 322 处疑似交互输入调用；
- `ClassificationModelSelection.activate()` 仍调用交互式 `manual_hyper_parameters()`；
- `WorkflowBase` 使用类级共享数据状态；
- MLflow active run、matplotlib 和随机数状态也具有进程级影响；
- 自由特征表达式使用 `eval()`；
- 当前 FastAPI 分类入口仍同步执行完整训练，只返回简单成功消息。

因此，第一版应使用双环境、双进程隔离：

```text
Agent
  │ stdio
  ▼
geochemistrypi-mcp（Python 3.10+ / Pydantic 2）
  │ JSON 文件 + subprocess
  ▼
Geochemistryπ Worker（当前 Engine 环境）
```

### 4.3 当前测试状态

截至 2026-07-25 的本地检查：

- 主测试文件 33 个测试通过、2 个失败；
- `test_data_readiness.py` 因错误导入路径在收集阶段失败；
- 部分 API 测试在导入路由时隐式依赖数据库环境变量。

因此在建立 Golden 基线前，必须先完成“测试套件可完整收集、测试环境无隐式外部依赖”的修复。

### 4.4 已确认的科学风险

当前 CLI 在训练测试划分之前，对完整 `X` 执行了部分缩放和特征选择。这可能让测试集信息参与预处理拟合，产生数据泄漏和过于乐观的指标。

因此必须区分：

- **Characterization Tests**：记录旧 CLI 当前行为；
- **Scientific Correctness Tests**：验证新的无泄漏实现；
- **Adapter Equivalence Tests**：验证 CLI、Python SDK 和 MCP 对同一新任务契约产生等价结果。

不能把“与旧 CLI 指标完全相同”作为科学正确性的必要条件。若修复数据泄漏导致指标变化，必须记录变更原因、旧结果和新结果。

## 5. 目标代码结构

### 5.1 结构设计原则

代码结构永久保持简单。工程质量通过清晰职责、单向依赖、严格契约和自动化测试保证，不通过增加 domain、infrastructure、ports 等多层目录实现。

必须同时满足：

1. 现有 `geochemistrypi/data_mining/` 永久保留在当前位置；
2. 不把整个项目重构为 Clean Architecture；
3. 不要求最终版本迁移全部旧模型类；
4. 新增 `application`、`legacy_adapters` 和 `worker` 作为稳定边界；
5. Contracts、Runtime 和 MCP 使用独立轻量包；
6. 新机器学习任务只增加同级 Service 和 Adapter；
7. 现有 CLI 继续可用；
8. 新代码不能继续扩大旧代码中的全局状态、环境变量和交互式输入；
9. 依赖方向必须通过 CI 自动检查；
10. 不为形式上的“架构完整”创建没有实际价值的抽象层。

### 5.2 分类阶段和最终版本共同使用的稳定结构

以下结构不是临时过渡结构，而是分类 Alpha、本地稳定版和最终论文版本共同使用的基础结构：

```text
Geochemistrypi/
├── geochemistrypi/
│   ├── data_mining/                  # 完全保留当前位置
│   ├── application/
│   │   └── classification_service.py
│   ├── legacy_adapters/
│   │   └── classification_adapter.py
│   └── worker/
│       └── main.py
│
├── packages/
│   ├── geochemistrypi-contracts/
│   ├── geochemistrypi-runtime/
│   └── geochemistrypi-mcp/
│
└── tests/
    ├── characterization/
    ├── scientific/
    ├── integration/
    └── mcp/
```

稳定调用关系：

```text
MCP Tool
  → LocalRunManager
  → Worker
  → ClassificationService
  → LegacyClassificationAdapter
  → 现有 ClassificationModelSelection
```

`legacy_adapters` 是稳定的工程边界，不要求在后续版本删除。它负责把新的结构化任务请求转换为现有 `data_mining` 能理解的参数和调用。

整个项目不移动、不重命名现有 `data_mining` 主体文件。只对现有实现增加必要的非交互参数入口、输出上下文和结果返回能力。

现有 CLI 使用：

```python
interactive = True
```

Worker 使用：

```python
interactive = False
```

### 5.3 扩展其他现有 CLI 能力

迁移回归、聚类、降维和异常检测时，只增加同级文件，不改变总体目录：

```text
geochemistrypi/
├── data_mining/                      # 保持不变
├── application/
│   ├── dataset_service.py
│   ├── validation_service.py
│   ├── classification_service.py
│   ├── regression_service.py
│   ├── clustering_service.py
│   ├── decomposition_service.py
│   ├── anomaly_service.py
│   ├── prediction_service.py
│   └── client.py                     # 薄 Python API，可选
├── legacy_adapters/
│   ├── classification_adapter.py
│   ├── regression_adapter.py
│   ├── clustering_adapter.py
│   ├── decomposition_adapter.py
│   └── anomaly_adapter.py
└── worker/
    └── main.py
```

如果某个 Service 或 Adapter 在后续确实变得过大，可以只针对该任务拆分子目录，但这不是版本目标或发布前置条件。

### 5.4 目录职责

#### `data_mining`

- 保存当前 CLI 和机器学习实现；
- 可以在原位置修复数据泄漏、全局状态和非交互参数；
- 不要求为 MCP 重新实现所有算法；
- 不要求最终迁移到新的 domain/infrastructure 目录。

#### `application`

- 提供结构化、非交互的完整任务；
- 负责任务级验证和步骤编排；
- 返回结构化结果；
- 不读取 MCP 请求对象；
- 不直接包含 MCP 协议代码；
- 不直接导入旧 `data_mining`，而是通过相应 Adapter。

#### `legacy_adapters`

- 是新任务系统调用现有 CLI 实现的唯一边界；
- 转换模型名称、参数、DataFrame 和旧方法签名；
- 隔离旧代码中的 `print()`、环境变量和类级状态；
- 可以作为最终版本中的永久适配层；
- 不承担 RunManager、MCP Schema 或论文评分逻辑。

#### `worker`

- 读取跨进程 request；
- 创建 Engine 侧运行上下文；
- 调用 Application Service；
- 捕获 stdout/stderr；
- 写入 result、错误和 Worker 状态；
- 一个 Worker 一次只执行一个实验。

#### `geochemistrypi-contracts`

- 保存版本化 JSON Schema、稳定枚举和错误码；
- 不依赖 pandas、sklearn、MCP 或完整 Engine；
- 同时安装到 Engine 和 MCP 环境；
- 跨进程契约不出现 DataFrame、sklearn 对象、Python 类路径或 callable。

#### `geochemistrypi-runtime`

负责：

- `run_id`；
- 运行目录；
- 状态机；
- 队列和并发；
- Worker 启动和取消；
- 幂等；
- artifact、manifest 和 provenance 存储。

Runtime 不实现机器学习，不导入 `geochemistrypi` 或 `data_mining`。

#### `geochemistrypi-mcp`

- 实现 MCP tools/resources 和协议模型；
- 调用 Runtime 创建和管理 Worker；
- 不导入完整 Geochemistryπ Engine；
- 不复制 Application Service 中的科学验证或机器学习逻辑。

### 5.5 强制依赖方向

```text
geochemistrypi-contracts
       ↑
geochemistrypi-runtime ← geochemistrypi-mcp
       ↑
     Worker
       ↓
   Application Service
       ↓
   Legacy Adapter
       ↓
现有 geochemistrypi.data_mining
```

强制规则：

1. Application Service 不直接导入旧 `data_mining`；
2. 只有 `legacy_adapters` 可以从新代码导入旧 `data_mining`；
3. Runtime 不导入 Engine；
4. MCP 不导入完整 Engine；
5. Worker 是 Runtime 与 Application 的进程边界；
6. MCP、Worker 和 Engine 使用同一 contract version；
7. 不能形成 MCP → Engine 的进程内调用；
8. 不允许 `data_mining` 反向依赖 MCP、Runtime 或 Application；
9. Python API 只调用 Application/Runtime，不复制训练逻辑；
10. 所有跨进程输入在 Worker 内再次验证。

### 5.6 不进行大规模重构

Adapter 调用关系：

```text
ClassificationService
  → LegacyClassificationAdapter
  → geochemistrypi.data_mining.process.classify
```

要求：

- 所有版本均不要求移动现有 `data_mining` 主体；
- 每次只接入一个纵向任务；
- 迁移前建立 Characterization Tests；
- 不进行一次性全库目录搬迁；
- 不创建 `domain/`、`infrastructure/`、`ports/` 等强制分层；
- 不为每个模型建立重复的 Service；
- 不为尚无第二个实现的逻辑提前创建抽象基类；
- 可以直接在 `data_mining` 原位置修复科学或工程问题；
- Adapter 可以长期存在，不设删除期限；
- 只有某个文件已经明显难以维护时，才对该文件做局部拆分。

### 5.7 工程级代码规则

- 新模块不得使用 `eval()`；
- Application Service 不使用 `input()`；
- Application Service 不通过普通 `print()` 返回结果；
- MCP 和 Application 不直接读取旧 `GEOPI_OUTPUT...` 环境变量；
- 配置只在入口层读取一次，再通过配置对象注入；
- 使用结构化 logging，不用输出文本承担返回值职责；
- 不使用进程级全局变量保存运行数据；
- 不创建无明确职责的 `utils/`、`helpers/`、`misc.py`；
- 通用模块按行为命名，例如 `atomic_json.py`、`path_policy.py`；
- 每个公开包明确 `__all__` 和稳定 API；
- 包之间禁止循环依赖；
- 同步机器学习逻辑保留在 Worker 内，异步只用于接口和任务管理；
- 公开契约、内部任务对象和 MCP Pydantic 模型不得混为同一类型。

### 5.8 架构自动检查

CI 至少加入：

- `ruff`：格式、导入和静态规则；
- `pyright --strict`：至少覆盖所有新增模块；
- `import-linter` 或等价 architecture tests；
- JSON Schema round-trip tests；
- wheel 安装和跨环境导入测试；
- 依赖漏洞检查；
- 禁止新增 `eval()` 和任意命令执行的安全检查。

以下情况必须使 CI 失败：

```text
application 直接导入旧 data_mining
runtime 导入 Engine
MCP 导入完整 geochemistrypi
跨进程 Schema 缺少版本
新代码绕过 Legacy Adapter 调用旧实现
Application 新增交互式 input()
MCP stdout 混入普通 print()
```

### 5.9 契约包要求

跨进程 JSON Schema 必须随 wheel 安装，不能只放在仓库根目录。轻量 `geochemistrypi-contracts` 包应兼容 Engine 和 MCP 的 Python 环境。

公开 JSON Schema 是跨进程线协议的规范来源。内部 dataclass、MCP Pydantic 模型和 Python SDK 类型可以分别实现，但必须通过 round-trip contract tests 证明一致。

## 6. 统一数据契约

### 6.1 通用规则

所有公开 Schema：

- 使用 JSON Schema Draft 2020-12；
- 包含稳定 `$id`；
- `additionalProperties: false`；
- 明确单位、范围、枚举和默认值；
- 区分缺失、`null` 和默认值；
- Schema 文件随 wheel 发布；
- 每次运行记录 contract version 和 Schema SHA-256；
- 不兼容时明确失败，不静默猜测。

### 6.2 DatasetRef

```json
{
  "kind": "local_file",
  "path": "D:/data/geochemistry.csv",
  "format": "csv",
  "id_column": "Sample_ID",
  "read_options": {
    "encoding": "utf-8",
    "delimiter": ",",
    "sheet_name": null
  },
  "expected_sha256": null,
  "snapshot_policy": "copy"
}
```

要求：

- `format` 与扩展名必须一致；
- CSV 记录编码、分隔符、表头和 NA 规则；
- XLSX 记录 sheet；
- 限制压缩前后大小、行数、列数和单元格数量；
- `inspect` 不默认返回全部数据；
- `validate` 返回数据指纹；
- `start` 必须重新验证；
- 论文运行默认把输入快照复制到运行目录；
- 若使用 `reference` 模式，运行前后必须比较 SHA-256，并在 manifest 中记录可复现性警告。

### 6.3 ClassificationExperimentSpec

```json
{
  "schema_version": "1.0",
  "client_request_id": "optional-client-id",
  "dataset": {
    "kind": "local_file",
    "path": "D:/data/geochemistry.csv",
    "format": "csv",
    "id_column": "Sample_ID",
    "snapshot_policy": "copy"
  },
  "target_column": "Deposit_Type",
  "feature_columns": null,
  "group_column": null,
  "preprocessing": {
    "missing_values": "median",
    "scaling": "standard",
    "class_balance": "none"
  },
  "split": {
    "strategy": "stratified_random",
    "test_size": 0.2,
    "group_column": null,
    "random_seed": 42
  },
  "model": {
    "name": "random_forest",
    "mode": "manual",
    "parameters": {
      "n_estimators": 300,
      "max_depth": 10
    }
  },
  "evaluation": {
    "primary_metric": "macro_f1",
    "positive_label": null,
    "cross_validation_folds": 5
  }
}
```

`parameters` 在早期可作为受控字典，但必须由模型注册表对应的严格 Schema 二次校验。稳定版应使用按模型区分的 discriminated union。

### 6.4 数据划分策略

不能只支持随机划分。地球化学数据可能存在矿区、钻孔、空间位置或采样批次相关性，应逐步支持：

- `stratified_random`
- `group`
- `stratified_group`
- `spatial_block`
- `temporal`
- `predefined_holdout`

第一版分类至少支持 `stratified_random` 和 `group`。论文 Benchmark 中必须根据数据生成过程选择划分方式，不能默认随机拆分相邻或同源样本。

### 6.5 ExperimentResult

```json
{
  "schema_version": "1.0",
  "run_id": "20260725-143011-a82c4f",
  "request_hash": "...",
  "status": "completed",
  "metrics": {
    "accuracy": 0.91,
    "balanced_accuracy": 0.88,
    "macro_f1": 0.87,
    "weighted_f1": 0.89
  },
  "artifacts": [
    {
      "artifact_id": "trained-pipeline",
      "role": "trained_pipeline",
      "media_type": "application/x-joblib",
      "relative_path": "artifacts/model/pipeline.joblib",
      "size_bytes": 123456,
      "sha256": "..."
    }
  ],
  "warnings": [],
  "manifest_path": "manifest.json",
  "provenance_path": "provenance.json"
}
```

Agent 通过 `run_id` 和 `artifact_id` 引用产物，不允许在请求中指定任意模型文件路径。

### 6.6 错误结构

```json
{
  "error": {
    "code": "INVALID_TARGET_COLUMN",
    "message": "Target column was not found.",
    "stage": "validation",
    "run_id": null,
    "retryable": false,
    "details": {}
  }
}
```

错误必须适合 Agent 修复，但不得返回完整 traceback、敏感环境变量或无限长度的数据列清单。

## 7. 模型注册表与任务能力

模型注册表不只映射类名，还必须保存：

- 稳定模型 ID；
- 显示名称；
- 参数 Schema；
- 支持的任务；
- 是否支持多分类；
- 是否支持概率预测；
- 是否支持新数据预测；
- 是否支持 `partial_fit`；
- 是否支持特征重要性；
- 是否支持 AutoML；
- 是否确定性；
- 所需可选依赖；
- 默认 CPU 线程限制。

示意：

```python
ModelCapability(
    id="random_forest",
    task="classification",
    parameter_schema="random-forest-classification.schema.json",
    supports_predict=True,
    supports_predict_proba=True,
    supports_feature_importance=True,
    supports_automl=True,
)
```

不同算法的能力不能被强行统一：

- KMeans 有聚类中心，DBSCAN 和 OPTICS 没有同等含义的中心；
- PCA 支持对新数据 `transform`；
- 当前 T-SNE 和 MDS 主要是 fit-only embedding，不能承诺通用的新数据变换；
- LOF 是否支持新数据检测取决于 novelty 模式；
- 不支持概率输出的分类器不能生成虚假概率指标。

接口和结果 Schema 应根据 capability 返回“不支持”，而不是产生空文件或伪造结果。

## 8. 科学正确性要求

### 8.1 无泄漏 Pipeline

监督学习必须遵循：

```text
读取原始数据
  → 确定 X / y / group
  → 划分训练集和最终测试集
  → 仅在训练集拟合预处理和特征选择
  → 在训练集内部交叉验证或调参
  → 最终模型只评估一次保留测试集
```

缺失值填补、缩放、特征选择和重采样必须进入 sklearn 或 imbalanced-learn Pipeline。重采样只能发生在训练折内部。

### 8.2 分类质量要求

至少保存：

- accuracy；
- balanced accuracy；
- macro、weighted 和 per-class precision/recall/F1；
- confusion matrix；
- 类别支持数；
- 若支持概率：ROC-AUC、PR-AUC 和适用的校准信息；
- 划分策略和每个划分中的类别分布。

类别不平衡时不能只报告 accuracy 或 weighted F1。

### 8.3 回归质量要求

- MAE、MSE、RMSE、R²；
- 目标转换和反转换；
- 残差诊断；
- 重复样本、极端值和小样本警告；
- 划分和交叉验证可复现。

### 8.4 聚类质量要求

- 样本簇标签；
- silhouette 等适用指标；
- 噪声点数量；
- 缩放方式；
- 算法特定产物；
- 不把监督准确率当作默认聚类指标；
- 只有算法真正定义聚类中心时才保存中心。

### 8.5 降维质量要求

- 保存低维表示；
- 保存原始特征顺序；
- PCA 保存组件、载荷和解释方差；
- 明确区分 transformable decomposition 与 fit-only embedding；
- 对随机嵌入保存初始化和随机种子；
- 不把可视化嵌入直接解释为稳定预测空间。

### 8.6 异常检测质量要求

- 异常分数、阈值和标签；
- contamination 或等价参数；
- 区分训练异常与应用数据异常；
- 记录是否支持新数据检测；
- 不默认把异常解释为错误数据。

### 8.7 AutoML

- 最终测试集不得参与模型选择；
- 搜索空间和时间预算必须记录；
- 保存最佳配置和搜索摘要；
- 固定所有可控随机种子；
- AutoML 失败不能覆盖手动模式产物；
- 第一版稳定分类 MCP 可以暂不启用 AutoML，待手动流程稳定后接入。

## 9. 安全特征工程

第一版 MCP 完全禁用任意表达式和 `eval()`。

后续采用结构化操作，而不是让 Agent 提交 Python 表达式：

```json
{
  "output_column": "Fe_Mg_ratio",
  "operation": "divide",
  "inputs": ["FeO", "MgO"],
  "on_zero": "null"
}
```

允许的操作使用白名单：

```text
add
subtract
multiply
divide
ratio
log
log10
sqrt
clip
standardize
```

必须限制：

- 操作数量；
- 依赖深度；
- 输出列数量；
- 除零行为；
- log 和 sqrt 的定义域；
- NaN 和无穷值；
- 输出数值范围；
- 重复列名；
- 对目标列和 ID 列的误用。

## 10. 运行目录、状态机与任务语义

### 10.1 运行目录

```text
runs/<run_id>/
├── request.json
├── request.sha256
├── status.json
├── control.json
├── result.json
├── manifest.json
├── provenance.json
├── worker.json
├── worker.stdout.log
├── worker.stderr.log
├── inputs/
├── artifacts/
└── errors/
```

### 10.2 状态

```text
queued
  ├── validating
  │     ├── running
  │     │     ├── completed
  │     │     ├── cancel_requested → cancelled
  │     │     └── failed
  │     ├── cancel_requested → cancelled
  │     └── failed
  └── cancel_requested → cancelled

非正常恢复状态：
orphaned
corrupted
```

`orphaned` 表示状态可以被识别但原 Worker 已丢失；它不是“训练可以从中间继续”。除非算法明确支持 checkpoint，否则恢复指状态协调或创建一个带 `retry_of` 的新运行。

### 10.3 状态写入规则

临时文件加原子替换只能防止半写 JSON，不能防止多个进程互相覆盖。

必须规定状态所有权：

- RunManager 创建 `queued` 状态；
- Worker 启动后成为运行状态的主要写入者；
- 取消请求写入独立 `control.json`，不直接覆盖 Worker 状态；
- Worker 退出后，RunManager 才能执行孤儿或损坏状态修复；
- `status.json` 包含 `revision`、`updated_at` 和 `owner`；
- 恢复逻辑检查 PID、进程启动时间和随机 worker nonce，不能只检查 PID。

### 10.4 幂等

`start_*` 从本地版本开始就支持 `client_request_id` 或 `idempotency_key`：

- 相同 key 和相同请求哈希返回已有 `run_id`；
- 相同 key 和不同请求拒绝；
- 不带 key 时创建新运行；
- 创建运行目录和登记任务必须是原子的。

这能避免 Agent、MCP 客户端或 HTTP 重试造成重复训练。

### 10.5 取消

取消顺序：

1. 写入 `cancel_requested` 控制信息；
2. 发送温和终止信号；
3. 等待受限宽限期；
4. 终止进程树；
5. Worker 或恢复器写入最终状态；
6. 保存已经生成且完整的日志，不把部分模型标记为有效产物。

Windows 使用 Job Object 或等价进程树控制；POSIX 使用独立 process group/session。仅保存 PID 不足以可靠取消 Ray、AutoML 或子进程。

## 11. Manifest 与 Provenance

每次运行至少记录：

- 请求和请求哈希；
- 数据文件 SHA-256、大小、行列数和读取配置；
- 输入快照策略；
- 特征、目标、ID、group 和划分索引；
- 预处理、特征工程和重采样；
- 模型、超参数和搜索空间；
- 指标定义和计算版本；
- Python、操作系统和 Geochemistryπ 版本；
- 关键依赖版本；
- Git commit 或 wheel 版本；
- 随机种子和可控确定性设置；
- CPU/GPU、线程和运行时间；
- 产物的 media type、大小和 SHA-256；
- 警告、失败阶段和错误码；
- Engine、MCP 平台和 contract 版本。

固定随机种子不等于完全可复现。跨平台 BLAS、并行算法、GPU 和依赖版本可能导致浮点差异，必须记录这些条件。

MLflow 可以作为可选观察和展示层，但本地文件 manifest/result 必须是任务系统的事实来源。关闭 MLflow 后，核心 Worker 仍应正常完成。

## 12. MCP 接口

### 12.1 分类 Alpha 工具

```text
get_capabilities
inspect_dataset
validate_classification
start_classification
get_run_status
get_run_result
cancel_run
```

`start_classification` 必须自行完成最终验证。`validate_classification` 只用于 Agent 提前发现和修正问题，不能成为安全前置假设。

### 12.2 稳定版通用工具

```text
list_runs
list_artifacts
read_artifact_text
get_run_log_summary
predict_application_data
```

限制：

- 文本和 JSON 产物可受限读取；
- 二进制模型不返回到 LLM 上下文；
- 大表格只返回摘要、分页或产物引用；
- 产物使用 `artifact_id`，不接受任意路径；
- 日志摘要需要脱敏和长度限制。

可选提供 MCP Resources：

```text
geopi://runs/<run_id>/manifest
geopi://runs/<run_id>/result
geopi://runs/<run_id>/artifacts/<artifact_id>
```

Tools 用于产生副作用的操作，Resources 用于只读结果浏览。首版可只实现 Tools，Resources 不作为 Alpha 阻塞项。

### 12.3 完整 ML 工具

```text
validate_regression
start_regression
validate_clustering
start_clustering
validate_decomposition
start_decomposition
validate_anomaly_detection
start_anomaly_detection
```

不为每个底层算法或绘图函数创建 MCP 工具。

### 12.4 MCP 实现规则

- Pydantic 模型 `extra="forbid"`；
- 工具描述说明用途、使用时机、限制和失败条件；
- 错误结构稳定；
- MCP stdout 只能输出协议数据；
- 服务器日志写 stderr 或 MCP logging notification；
- 不返回完整 traceback、模型二进制或完整训练数据；
- 工具调用不会阻塞 MCP 消息循环；
- 长任务立即返回 `run_id`；
- 协议层和 Worker 层都校验 contract version。

### 12.5 MCP 包依赖

```toml
[project]
name = "geochemistrypi-mcp"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.27,<2",
    "pydantic>=2.11,<3",
    "platformdirs>=4",
    "psutil>=6",
    "jsonschema>=4.20",
    "geochemistrypi-contracts==<matching-version>"
]
```

开发和 CI 使用 lock/constraints 固定完整依赖树。发布 wheel 的直接依赖仍需合理上下界，因为消费者安装时不会自动使用仓库 lock 文件。

## 13. 本地安全模型

### 13.1 威胁模型

第一版主要防御：

- Agent 误读、误写或误删用户文件；
- 恶意或错误生成的模型参数和特征操作；
- 路径穿越和链接逃逸；
- 不可信模型文件加载；
- 资源耗尽；
- stdout 协议污染；
- 日志泄露。

第一版本地 MCP 与用户运行在同一操作系统账户下，不宣称能隔离同账户恶意本地进程。远程多租户安全属于后续独立轨道。

### 13.2 路径规则

- 允许根目录来自用户配置，不来自 Agent 请求；
- 输入路径必须绝对化、规范化并位于允许根目录；
- 输出目录完全由系统生成；
- 检查符号链接、Windows junction/reparse point、UNC 和设备路径；
- 拒绝 NTFS alternate data stream 等非普通文件；
- 检查完成后尽快打开或快照文件，降低 TOCTOU 风险；
- 原始数据只读；
- 无人工确认不删除运行目录。

建议使用 `platformdirs` 下的 TOML 配置文件作为主配置，环境变量只用于显式覆盖。跨平台路径列表不应只依赖一个未经转义的字符串环境变量。

### 13.3 模型产物

joblib/pickle 可以在加载时执行代码，因此：

- 只允许通过可信 `run_id + artifact_id` 加载本系统生成的模型；
- 加载前校验 manifest、SHA-256、producer 和兼容版本；
- 不接受用户上传的任意 pickle/joblib 作为预测输入；
- 版本不兼容时明确失败；
- 论文归档同时保存环境 lock 和训练代码版本。

### 13.4 资源限制

- 最大数据大小、行数、列数和单元格数；
- 最大并发和排队长度；
- 最大运行时间；
- 最大 CPU 线程；
- 最大内存；
- 最大产物大小；
- 最大日志大小；
- 最大特征操作数量。

本地 `psutil` 轮询只能提供部分限制。文档必须区分“best effort 本地限制”和“由容器、Job Object 或 cgroup 强制的限制”，不能宣称所有平台都能用纯 Python 实现硬隔离。

## 14. Python SDK

Python SDK 是 MCP 的同语义适配器，不重新实现机器学习逻辑：

```python
from geochemistrypi import GeoPiClient

client = GeoPiClient.local(
    output_root="D:/geopi-runs",
    allowed_roots=["D:/geochemical-data"],
)

summary = client.inspect_dataset(
    "D:/geochemical-data/samples.csv"
)

run = client.start_classification(
    spec=classification_spec,
)

result = client.wait(run.run_id)
```

第一版 SDK 作为 `geochemistrypi/application/client.py` 的薄门面，并通过 `geochemistrypi.__init__` 导出，不单独建立 `geochemistrypi-sdk` 包。只有未来确实需要独立发布时，才重新评估拆包。

用途：

- Notebook 和科研脚本；
- 无 MCP 客户端的本地程序；
- 测试核心任务系统；
- 论文中比较 Python function calling 与 MCP 的接口消融实验。

SDK 至少提供：

- 同步客户端；
- 异步客户端；
- 不自动等待的任务接口；
- 结构化异常；
- 完整类型提示。

第一版不要求启动 HTTP 服务。

## 15. 版本路线

### 15.1 v0.0：基线修复与架构决策

目标：建立可信的实施起点。

交付：

- 修复当前测试收集错误和环境耦合；
- 建立 Characterization 与 Scientific Tests 的目录和命名；
- 记录已知数据泄漏和非确定性；
- 确定跨进程契约包；
- 确定 Worker、状态所有权和运行目录 ADR；
- 确定本地安全威胁模型；
- 建立最小 CI。

退出条件：

- 当前测试可完整收集；
- 基线测试结果稳定；
- 数据泄漏修复策略已决策；
- JSON Schema 能在 Engine 和 MCP 两个环境中加载；
- 关键 ADR 已合并。

预计：1 至 2 周。

### 15.2 v0.1：非交互分类核心

目标：将分类流程转化为可配置、可测试、可返回结果的 Worker。

包含：

- DatasetService；
- ValidationService；
- ClassificationExperimentSpec；
- 模型注册表；
- 参数直接注入，禁止交互式超参数询问；
- 训练集先划分、无泄漏 Pipeline；
- RunContext；
- Worker；
- result、manifest 和 provenance；
- 分类训练和预测；
- 日志隔离；
- 固定种子和容差 Golden Tests。

暂不包含：

- MCP；
- AutoML；
- 任意表达式；
- 其他 ML 任务。

退出条件：

- 分类 Worker 不依赖 CLI 输入；
- 相同配置重复运行满足确定性容差；
- 预处理只在训练集拟合；
- 结果和产物可校验；
- 失败不会长期停留在 running；
- 不可信输入不能触发任意代码执行；
- Characterization Tests 与 Scientific Tests 分别通过。

预计：3 至 5 周。

### 15.3 v0.2：本地 MCP 分类 Alpha

目标：Agent 能完成完整的本地分类实验。

交付：

- 独立 MCP 包和环境；
- RunManager；
- durable local queue，默认并发 1；
- 分类 Alpha 七个工具；
- 幂等启动；
- 取消；
- 路径白名单；
- MCP Inspector 测试；
- stdout 污染测试；
- Engine/MCP 契约握手。

退出条件：

- Agent 可以完成“检查数据 → 验证 → 启动 → 查询 → 读取结果”；
- 训练不阻塞 MCP 消息循环；
- MCP 重启后可读取已完成运行；
- 同一幂等 key 不会重复训练；
- MCP 无网络监听；
- 数据和产物保留在本地。

预计：2 至 3 周。

### 15.4 v0.3：本地分类稳定版

目标：形成科研用户可长期使用的分类训练与预测闭环。

交付：

- 完整 Pipeline 产物；
- `predict_application_data`；
- 任务取消和孤儿状态协调；
- `list_runs`、`list_artifacts`、日志摘要；
- 模型兼容性和可信产物检查；
- PowerShell 和 Bash 安装脚本；
- `geochemistrypi-mcp doctor`；
- Windows、Ubuntu、macOS wheel 安装测试；
- CPU、内存、日志和数据大小限制；
- 用户文档和最小示例。

注意：“任务恢复”在此阶段主要指状态恢复和安全重试，不承诺从训练中间 checkpoint 继续。

退出条件：

- 干净环境可安装并运行；
- 分类训练和预测闭环；
- 无孤立子进程；
- 三个平台通过核心测试；
- 所有模型只通过可信运行产物加载；
- 连续运行、取消和 MCP 异常退出测试通过。

预计：3 至 4 周。

### 15.5 v0.4：覆盖现有 CLI 的主要 ML 能力

迁移顺序：

1. regression；
2. clustering；
3. PCA 和其他 decomposition/embedding；
4. anomaly detection；
5. 安全结构化 feature engineering；
6. AutoML；
7. 完整 application-data inference；
8. 薄 Python SDK。

每类任务：

- 独立 Spec 和 Result Schema；
- 独立科学验证；
- 独立模型 capability；
- 至少 3 个代表性测试数据集；
- 适用的训练和应用数据闭环；
- MCP 与 Python SDK 适配器一致性测试。

退出条件：

- 当前 CLI 的主要 ML 任务具有非交互 Service；
- MCP 和 Python SDK 可调用；
- 不支持的算法能力明确返回；
- 所有监督任务无预处理泄漏；
- 每类任务具有科学 Golden Tests；
- AutoML 不接触最终测试集。

预计：5 至 8 周。

### 15.6 v0.5：论文实验版

目标：冻结论文实验所需接口、任务、数据和评分系统。

交付：

- 稳定的本地 MCP；
- 稳定的 Python SDK；
- 版本化公开契约；
- Benchmark 数据集卡片和任务集；
- Agent 执行 harness；
- 自动评分与盲法人工复核；
- 失败分类法；
- 统计分析脚本；
- 论文表格和图形自动生成；
- 完整复现实验说明；
- 可归档 release candidate。

退出条件：

- Benchmark 能从干净环境独立重跑；
- 主指标、样本量和分析计划已预注册或冻结；
- 所有实验保存 prompt、模型版本、工具版本、配置和产物；
- 论文结果可由原始运行记录自动生成；
- 至少完成一次内部 replication。

预计：5 至 8 周，不包含数据授权等待和论文写作。

### 15.7 v1.0：论文级本地发布

发布条件：

- v0.5 Benchmark 完成；
- 公开 Schema 在 v1 内稳定；
- 安全和兼容性审查完成；
- Windows、Linux、macOS wheel 可安装；
- 文档示例全部自动运行；
- 代码、数据、任务和结果可归档；
- release、数据和 Benchmark 获得持久标识；
- 论文主张与公开证据一致。

## 16. 测试体系

### 16.1 测试层级

```text
Unit Tests
  → Contract Round-trip Tests
  → Scientific Correctness Tests
  → Worker Integration Tests
  → MCP Protocol Tests
  → Security / Property Tests
  → Cross-platform Wheel Install Tests
  → Agent Benchmark
```

### 16.2 Golden 测试规则

- 不把当前 CLI 输出自动视为科学真值；
- 确定性算法使用精确或严格容差；
- 浮点和跨平台输出使用显式容差；
- 非确定性算法使用不变量、分布或范围检查；
- Golden 文件记录依赖版本和生成方式；
- 旧行为 Characterization 与新科学结果分目录；
- 变更 Golden 必须说明原因。

### 16.3 必测失败场景

- 非法和越界路径；
- 文件在验证后被修改；
- CSV 编码和 XLSX sheet 错误；
- 空列、重复列、重复 ID；
- 类别样本过少；
- group 泄漏；
- 无效模型参数；
- Worker 崩溃；
- 取消和完成竞争；
- MCP 重启；
- PID 重用模拟；
- 状态 JSON 损坏；
- 磁盘满；
- 日志过大；
- 模型哈希不匹配；
- 不兼容模型版本；
- stdio 污染；
- 重复幂等请求；
- 恶意特征操作。

### 16.4 覆盖率

覆盖率用于发现盲区，不作为唯一质量目标：

- 核心契约、安全边界和状态转换要求高分支覆盖；
- 关键状态机使用属性测试；
- 安全解析器考虑 mutation testing；
- 科学正确性以独立参考实现、不变量和领域审查为主；
- 不为了追求单一百分比编写无价值测试。

### 16.5 CI

最低矩阵：

```text
Windows
Ubuntu
macOS
```

CI 必须：

1. 构建所有 wheel；
2. 创建全新 Engine 和 MCP 环境；
3. 安装 wheel，而不是 editable；
4. 运行 contract handshake；
5. 运行 doctor；
6. 启动 MCP；
7. 列出工具；
8. 调用 `inspect_dataset`；
9. 运行小型分类任务；
10. 读取结果和产物；
11. 取消一个长任务；
12. 检查无残留进程。

完整科学矩阵可在 nightly 或 release CI 运行，避免每个 PR 都执行所有重型 AutoML 测试。

## 17. 论文 Benchmark

### 17.1 研究问题

主问题：

> 领域专用、具有严格契约、科学约束和 provenance 的工具接口，能否使通用 AI Agent 更可靠、高效和可复现地完成地球化学机器学习任务？

MCP 是标准化接口之一，不应被表述为唯一科学贡献。论文贡献应定位为：

- 领域工作流结构化；
- 科学验证和泄漏防护；
- 可复现实验机制；
- 受控 Agent 工具接口；
- 系统性的 Agent 评估。

### 17.2 对照条件

至少比较：

```text
A. Agent + 原始 Python / scikit-learn
B. Agent + Geochemistryπ Python SDK
C. Agent + Geochemistryπ MCP
```

解释：

- A 与 B/C 比较领域任务系统的价值；
- B 与 C 使用相同核心和契约，用于隔离 Python function calling 与 MCP transport 的影响；
- 必要时增加有限消融：移除科学验证、移除 provenance 或简化工具描述；
- 消融只能在离线沙箱 Benchmark 中运行，不能成为面向用户的不安全模式。

不再使用“Agent + 交互式 CLI”作为唯一基线。若保留 CLI 条件，必须明确人工输入或脚本输入规则，否则不同条件的人力支持不公平。

### 17.3 任务

- 数据集检查；
- 目标和特征选择；
- 分类和回归；
- 缺失值处理；
- group 或空间泄漏识别；
- 类别不平衡；
- 模型与参数选择；
- 聚类；
- 降维；
- 异常检测；
- 应用数据预测；
- 错误恢复；
- 结果解释；
- 产物和 provenance 检查。

### 17.4 指标

预先指定三个主指标：

1. 任务完成率；
2. 科学有效率；
3. 可复现实验率。

次要指标：

- 数据泄漏率；
- 无效参数率；
- 失败恢复率；
- 模型性能；
- 产物完整性；
- 人工干预次数；
- 工具调用次数；
- Agent 轮次；
- 总时间和计算成本；
- Token 数量。

不同模型供应商的 Token 不能直接视为完全等价成本，必须同时报告实际费用或标准化成本。

### 17.5 最低实验规模

论文正式实验建议：

- 5 至 8 个具有明确许可和数据卡片的地球化学数据集；
- 30 至 50 个任务；
- 至少 3 种 Agent 或模型系列；
- 每个关键条件至少 5 次重复；
- 固定预算、工具版本和硬件类别；
- 任务顺序随机化；
- 自动评分与盲法人工复核结合；
- 报告方差、置信区间和失败类型。

最终样本量应通过 pilot 结果和功效分析确定，不能只根据实施方便决定。

### 17.6 统计分析

- 任务和数据集是主要分析层级；
- 同一任务的多次重复不是完全独立样本；
- 使用分层 bootstrap、混合效应模型或等价的重复测量方法；
- 预先冻结主指标和多重比较策略；
- 报告效应量和置信区间，不只报告 p 值；
- 对无法自动评分的结果进行双人复核并报告一致性。

### 17.7 污染和公平控制

- 记录 Agent、模型、日期和确切版本；
- 固定系统提示和最大轮次；
- 控制计算时间和资源预算；
- 原始 Python 条件运行在沙箱内；
- 不允许某个条件获得额外人工帮助；
- 公共数据集搭配未公开变体或受控合成扰动，降低训练数据记忆影响；
- 保存全部工具调用和失败轨迹；
- 数据、任务和评分脚本版本化。

## 18. 版本与兼容策略

分别记录：

```json
{
  "engine_version": "0.9.0",
  "agent_interface_version": "0.5.0",
  "contract_version": "1.0"
}
```

规则：

- 新增可选字段通常为兼容小版本；
- 修改字段含义、类型或默认行为需要主版本；
- 删除字段需要主版本；
- 工具名称、错误码和 artifact role 也是公开契约；
- MCP、SDK 和 Worker 启动时协商支持的 contract 范围；
- 不兼容时明确失败；
- 至少保留一个稳定版本的迁移或读取能力；
- 官方参考链接之外，发布记录还应保存确切 SDK 版本和 lock。

## 19. 推荐 Pull Request 顺序

### PR 0：测试基线修复

- 修复测试导入；
- 移除测试收集阶段数据库依赖；
- 建立最小 CI；
- 记录当前通过、失败和非确定性。

### PR 1：科学基线与数据泄漏决策

- Characterization 数据集；
- Scientific reference 数据集；
- 先划分后预处理；
- Golden 容差规则；
- 变更说明。

### PR 2：Contracts 包

- JSON Schema v1；
- Schema `$id` 和版本；
- Engine dataclass；
- round-trip tests；
- wheel package data 测试。

### PR 3：Runtime 包、RunContext 与原子运行目录

- `geochemistrypi-runtime` 包骨架；
- `RunContext`；
- request/result/status；
- artifact 引用；
- manifest；
- provenance；
- 状态所有权和修复规则；
- architecture tests，确保 Runtime 不导入 Engine。

### PR 4：ClassificationService

- 参数直接注入；
- 模型注册表；
- 无交互分类；
- `LegacyClassificationAdapter`；
- 无泄漏 Pipeline；
- 结构化指标；
- 与 reference tests 对照；
- 新代码除 Legacy Adapter 外不得导入旧 `data_mining`。

### PR 5：Worker

- 命令入口；
- 在 `worker/main.py` 中完成最小依赖组装；
- stdout/stderr 隔离；
- 退出码；
- 信号处理；
- Worker identity；
- 失败和取消测试。

### PR 6：MCP 骨架与 RunManager 集成

- 独立环境；
- 复用 `geochemistrypi-runtime`，不导入完整 Engine；
- contract handshake；
- `get_capabilities`；
- `inspect_dataset`；
- 队列；
- 幂等；
- Inspector 测试。

### PR 7：MCP 分类闭环

- validate；
- start；
- status；
- result；
- cancel；
- artifact；
- E2E tests。

### PR 8：预测、安装与 doctor

- 完整 Pipeline；
- 可信模型加载；
- application data；
- PowerShell/Bash；
- wheel 安装；
- 跨平台 CI。

### PR 9：其他 ML 任务

- regression；
- clustering；
- decomposition；
- anomaly detection；
- capability matrix。

### PR 10：Python SDK 与 Benchmark Harness

- 同步/异步 SDK；
- Adapter equivalence；
- Agent harness；
- 评分和统计脚本；
- 论文产物生成。

## 20. 工作量与里程碑

| 阶段 | 单人高质量估算 | 可并行部分 |
|---|---:|---|
| v0.0 基线修复 | 1–2 周 | ADR、测试数据 |
| v0.1 非交互分类核心 | 3–5 周 | 契约、科学测试 |
| v0.2 MCP Alpha | 2–3 周 | Inspector、文档 |
| v0.3 分类稳定版 | 3–4 周 | 安装和跨平台测试 |
| v0.4 完整主要 ML | 5–8 周 | 各任务 Service |
| v0.5 论文实验版 | 5–8 周 | Benchmark 数据和评分 |
| v1.0 论文级发布 | 取决于实验复核 | 文档、归档、论文写作 |

合理预期：

- 可演示的本地分类 MCP：约 6 至 10 周；
- 高质量分类训练与预测闭环：约 9 至 14 周；
- 覆盖现有主要 CLI 能力并具备论文实验基础：约 4 至 7 个月；
- Benchmark 执行、复核和论文写作另计。

如果有 2 至 3 名熟悉项目的开发者，测试、契约、适配器和不同任务 Service 可以并行，但 ClassificationService、RunManager 和公开契约需要明确的单一设计负责人。

## 21. 近期最优先行动

```text
1. 修复当前测试收集和数据库环境耦合
2. 冻结旧 CLI Characterization 行为
3. 为数据泄漏修复建立独立 Scientific Tests
4. 定义跨进程 JSON Schema 和 contracts 包
5. 实现 RunContext 与运行目录
6. 让分类模型参数可直接注入，完全绕过交互输入
7. 提取 ClassificationService
8. 实现独立 Worker
9. 接入 RunManager
10. 最后接入本地 MCP
```

完成分类纵向链路后，再迁移现有 CLI 的其他机器学习任务。不要先为所有算法设计一套过度抽象的万能接口。

## 22. 本地主线完成定义

论文级本地主线完成时，应满足：

1. CLI、Python SDK 和 MCP 共享同一任务系统。
2. 分类、回归、聚类、降维和异常检测可结构化调用。
3. 每次实验具有稳定 `run_id`、状态、日志、结果、manifest 和 provenance。
4. 所有监督任务通过无泄漏 Pipeline 执行。
5. 模型预测复用训练时保存的完整预处理。
6. Agent 不能提交任意代码、命令或不可信模型。
7. 路径、资源和产物访问受到限制。
8. 任务具备幂等、取消、故障协调和受控重试。
9. Windows、Linux 和 macOS 的 wheel 安装与核心流程通过。
10. Benchmark 能比较原始 Python、Python SDK 和 MCP。
11. 论文结果可从公开或受控归档的运行记录重建。
12. 论文贡献被表述为领域工作流、科学约束、可复现机制和 Agent 评估，而不是单纯“使用 MCP”。

## 23. 后续产品化与科学扩展

### 23.1 远程产品化

在本地 v1.0 后再单独规划：

- REST/OpenAPI；
- Streamable HTTP MCP；
- 用户、项目和身份认证；
- 队列和 Worker 心跳；
- CPU/GPU 资源调度；
- 数据和产物对象存储；
- PostgreSQL 元数据；
- 多租户隔离；
- 审计、配额和保留策略。

远程请求使用 dataset ID 或受控上传，禁止客户端提交服务器文件路径。

### 23.2 Chemical Modeling

Chemical Modeling 必须拥有独立科学问题、验证数据、单位系统、守恒约束和求解器语义。科学范围未确定前，不把它加入当前 MCP 契约，也不用于扩大当前论文标题。

未来可形成：

```text
Application Service
├── Machine Learning Engine
└── Chemical Modeling Engine
```

只有完成独立验证和联合 Benchmark 后，才考虑 “Machine Learning and Chemical Modeling for Scientists and AI Agents” 的平台定位。

## 24. 官方技术参考

- MCP Python SDK 稳定 v1：<https://github.com/modelcontextprotocol/python-sdk/tree/v1.x>
- MCP Python SDK 依赖：<https://github.com/modelcontextprotocol/python-sdk/blob/v1.x/pyproject.toml>
- MCP Inspector：<https://modelcontextprotocol.io/docs/tools/inspector>
- MCP Debugging：<https://modelcontextprotocol.io/docs/tools/debugging>
- scikit-learn Common Pitfalls：<https://scikit-learn.org/stable/common_pitfalls.html>
- Geochemistryπ：<https://github.com/ZJUEarthData/Geochemistrypi>
