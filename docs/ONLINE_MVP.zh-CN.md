# Geochemistryπ Online MVP 中文运行说明

本文档用于说明 Geochemistryπ 第一个可运行的 Online 版本。当前目标是先完成“选择算法 → 上传数据文件 → 执行计算 → 下载结果”的完整流程，暂不追求正式上线和最终页面设计。

## 一、当前已实现的功能

- 提供 Vue Online 操作页面；
- 提供独立的 FastAPI 后端服务；
- 从后端动态读取化学建模算法目录；
- 显示方法级“已验证”和“测试中”状态；
- 显示公式、输入列、含义、类型、单位、示例和注意事项；
- 支持上传 `.xlsx` 和 UTF-8 `.csv` 文件，单个文件最大 10 MB；
- 每个计算任务最长运行 30 分钟，超时后强制终止；
- 全站同时只允许执行 1 个计算任务，其他计算请求按顺序排队；
- 每次计算使用独立任务目录，避免不同任务的文件互相覆盖；
- 计算完成后可以下载结果文件；
- 回归与分类使用统一的监督学习 Pipeline；
- 训练完成后可下载 `trained_pipeline.joblib` 模型包，其中保存特征、目标列、模型、软件版本和训练元数据；
- 可通过 `POST /api/data-mining/inference` 引用训练任务 Job ID，上传不含目标列的独立 `.xlsx` 或 `.csv` Application Data 进行推理；
- 输入不正确时返回可读的错误信息；
- 已建立后端接口自动化测试。

目前已经完整验证的方法为：

```text
任务：algo_kinetic
方法：first_order、second_order、radioactive_decay
输入列：根据方法使用 c0/k/t 或 n0/decay_const/t

任务：algo_transport
方法：fick_diffusion、chromatography
输入列：根据方法使用 D/dc_dx 或 tR/sigma

任务：algo_thermodynamic
方法：vanthoff、activity_coefficient
输入列：根据方法使用 K1/dH/T1/T2 或 z/ionic_strength
```

对于 `second_order`，`c0` 必须大于 0，`k` 和 `t` 必须大于或等于 0。Online 会在计算前拒绝空值、非数值和超出范围的输入。

页面上能够查看其他算法的状态；测试中方法只能查看说明，在完成输入格式、科学正确性和结果文件验证前不能执行计算。

### 已保存 Pipeline 与 Application Data 推理

每个成功完成的回归或分类任务都会保存一个完整的 scikit-learn Pipeline。该 Pipeline 包含使用训练集学习的中位数填补器，以及所选模型和模型内部的标准化等变换。Application Data 必须包含与训练特征同名的全部列，不需要包含目标列。只要一行至少有一个可用数值特征就会执行预测，缺失特征使用训练集的中位数填补；如果某行所有必需特征都为空或非数值，该行仍保留在下载结果中，但会标记为已排除。

推理接口只接受服务端自己产生的训练 Job ID，不接受用户上传任意 `.joblib` 文件，因为加载不可信的 pickle 兼容文件可能执行任意代码。下载的模型包只能在可信、且 Geochemistry Pi 与 scikit-learn 版本兼容的 Python 环境中加载。

## 资源限制

以下限制同时适用于 Chemical Modeling 和 Data Mining，并由后端强制执行。前端检查只是为了提前提示用户，不能替代后端限制。

- 上传数据集最大为 `20 MiB`（`20,971,520` 字节）。
- 单个计算任务从真正开始执行后最长运行 `30 分钟`，排队时间不计入。到达时限后，独立计算进程会被终止，API 返回 HTTP `504`。
- 整个 Online 实例同时最多执行 `1` 个计算任务。后续请求进入串行队列，当前任务完成或超时后自动逐个开始。
- 页面通过 `GET /api/tasks/{task_id}` 实时显示排队位置、执行状态、已运行时间和阶段进度。
- `POST /api/tasks/{task_id}/cancel` 可以将排队任务移出队列，或终止正在运行的计算进程；随后自动启动下一个任务。
- 进度显示来自真实任务阶段（`queued` → `running` → 最终状态）。目前没有迭代回调的算法使用动态运行条，不显示虚假的完成百分比。
- 算法目录、健康检查和结果下载不占用计算任务名额。

## 二、一键启动

### 1. 启动系统

进入项目根目录，双击：

```text
start-online.cmd
```

启动程序会自动完成以下工作：

1. 检测 Python 3.11 或更高版本，已安装则直接使用；
2. 检测 Node.js 20 或更高版本，已安装则直接使用；
3. 若缺少可用版本，通过 Windows WinGet 安装 Python 3.12 或 Node.js LTS；
4. 查找或创建 `.venv-online` Python 虚拟环境；
5. 检查并按需安装 Online 后端和前端依赖；
6. 核对前端、后端是否来自当前项目目录和同一次源码构建；
7. 若端口上运行的是旧版 Geochemistry Pi，则安全替换为当前版本，再打开 Online 页面。

如果 `5173` 或 `8000` 端口被其他软件占用，启动程序会显示该进程的 PID 并停止启动，不会结束无关进程。

第一次启动可能需要下载系统软件和项目依赖，因此耗时会比以后启动更长。Windows 仍可能显示管理员权限确认，请允许安装。如果电脑没有 WinGet，启动窗口会说明需要手动安装的软件。窗口显示 `Geochemistry Pi Online is ready.` 表示启动成功。

如需严格禁止自动安装，可以在 PowerShell 中运行：

```powershell
.\start-online.cmd -SkipInstall
```

### 2. 停止系统

使用完毕后，双击：

```text
stop-online.cmd
```

该脚本只会停止一键启动程序记录的前端和后端进程，不会随意结束其他 Python 或 Node.js 程序。

## 三、访问地址

| 用途 | 地址 |
|---|---|
| Online 操作页面 | <http://127.0.0.1:5173/online> |
| API 接口文档 | <http://127.0.0.1:8000/docs> |
| 后端健康检查 | <http://127.0.0.1:8000/api/health> |

这些是本机地址，只能在运行服务的这台电脑上直接访问。目前尚未部署成公网网站。

## 四、完成一次测试计算

1. 双击 `start-online.cmd`。
2. 打开 Online 操作页面。
3. 任务选择 `kinetic`。
4. 方法选择 `First-order kinetics`。
5. 元素选择 `Any`。
6. 上传包含 `c0`、`k`、`t` 三列的 `.xlsx` 或 UTF-8 `.csv` 文件。
7. 单击“开始计算”。
8. 计算成功后，下载 `first_order_results.xlsx`。

测试文件位于工作区的 `outputs/kinetic_input_example.xlsx`。

## 五、日志与故障排查

运行日志位于：

```text
runtime/logs/backend.out.log
runtime/logs/backend.err.log
runtime/logs/frontend.out.log
runtime/logs/frontend.err.log
```

常见问题：

### 网页无法打开

1. 确认启动窗口中已经显示启动成功；
2. 手动打开 <http://127.0.0.1:5173/online>；
3. 检查 `runtime/logs/frontend.err.log`；
4. 双击 `stop-online.cmd`，然后重新运行 `start-online.cmd`。

### 后端接口不可用

打开 <http://127.0.0.1:8000/api/health>。正常情况下应看到状态为 `ok`。如果无法访问，请检查 `runtime/logs/backend.err.log`。

### 数据文件上传后提示缺少列

确认数据文件第一行包含正确的列名。已验证的一阶动力学方法必须包含：

```text
c0, k, t
```

列名不能用中文替代，也不要在列名前后添加空格。

### 端口被占用

前端默认使用 `5173` 端口，后端默认使用 `8000` 端口。启动程序会自动替换其他目录中的旧版 Geochemistry Pi；如果占用端口的是无关软件，启动程序会保留该进程并显示 PID，请先关闭相应软件后再重新运行 `start-online.cmd`。

## 六、手动启动方法

一般情况下直接使用一键启动即可。需要调试时，可以按照以下步骤手动启动。

在项目根目录创建 Python 环境并安装依赖：

```powershell
python -m venv .venv-online
.\.venv-online\Scripts\python.exe -m pip install -r requirements-online.txt
```

安装前端依赖：

```powershell
cd geochemistrypi\frontend
pnpm install
cd ..\..
```

在第一个终端启动后端：

```powershell
.\.venv-online\Scripts\python.exe -m uvicorn geochemistrypi.online.app:app --host 127.0.0.1 --port 8000
```

在第二个终端启动前端：

```powershell
cd geochemistrypi\frontend
pnpm start -- --host 127.0.0.1 --port 5173
```

## 七、验证命令

运行后端测试：

```powershell
.\.venv-online\Scripts\python.exe -m pip install -r requirements-online-dev.txt
.\.venv-online\Scripts\python.exe -m pytest tests\test_online_api.py -q
```

检查前端类型并构建生产版本：

```powershell
cd geochemistrypi\frontend
pnpm run build
```

当前验收结果：

```text
后端：Online API 与任务执行器测试全部通过
前端：生产构建成功
```

## 八、当前版本的限制

- 计算期间仍会保持 HTTP 请求，最长不超过已强制设置的 30 分钟；
- 任务信息只保存在文件系统中，尚未使用数据库；
- 轻量 API 尚无登录、权限和用户隔离；
- 上传文件及结果尚无自动清理策略；
- Chemical Modeling 支持 `.xlsx` 和以逗号分隔的 UTF-8 `.csv` 数据文件；
- 当前只有一阶动力学、二阶动力学、放射性衰变、Fick 扩散、色谱理论板数、范特霍夫方程和当前简化活度系数模型完成了端到端验证；
- 除四个动力学方法外，其他方法的详细输入说明仍在整理；
- 算法目录尚未描述每种方法的可选参数；
- 未安装 `scikit-learn` 时，`algo_solubility` 会显示为不可用；
- 页面目前以功能可用为主，尚未完成最终视觉设计。

## 九、后续改进清单

### P0：多人使用或公开部署前必须完成

1. 定义并验证所有公开算法的输入列、参数、单位和输出；
2. 将长时间计算改为后台任务，支持状态、进度、超时、取消和重试；
3. 使用持久化存储记录数据集、任务状态和结果文件；
4. 增加用户登录、权限控制和用户数据隔离；
5. 制定上传文件和结果文件的保留、清理、配额及删除策略；
6. 增加结构化日志、错误编号、安全检查和生产环境配置；
7. 验证相同输入在 CLI 与 Online 中得到科学意义一致的结果。

### P1：下一轮工程开发

1. 使用统一的数据集、任务和结果模型接入 Data Mining；
2. 为算法目录增加参数类型、参数说明和输入模板下载；
3. 增加历史任务、进度显示、取消和重新运行功能；
4. 增加 Docker、数据库迁移、持续集成测试和部署环境；
5. 在外部使用前将公开接口升级为 `/api/v1` 版本。

### P2：页面与产品完善

1. 完成视觉设计和响应式页面；
2. 增加结果表格、图形预览和科学元数据；
3. 增加中英文切换和操作引导；
4. 增加项目分享和可复现的计算报告。

## 十、当前阶段结论

当前版本已经完成本地 Online MVP 的最小闭环，适合项目内部演示和继续开发，但还不应直接作为无访问限制的公开生产网站。
