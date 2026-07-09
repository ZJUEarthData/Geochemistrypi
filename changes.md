# Classification 多分类改造说明

这次改动主要是围绕 classification 模块完成“用户可以自己定义多分类”的功能。原来的流程更偏向二分类，尤其是在 CLI 里，目标列基本默认是已经准备好的二分类标签。现在我把 classification 的标签处理提前到了模型训练之前，让用户可以在命令行里自己决定最终要分成几类，以及每一类叫什么名字。

## 1. 调整 CLI 里的 classification 流程

修改文件：`geochemistrypi/data_mining/cli_pipeline.py`

原来 classification 的标签处理是在后面的模型训练阶段才发生的，这样会有几个问题：

- feature selection 用到的还是原始 Y；
- train/test split 不能按照用户最终定义的类别来分层；
- 如果选择 all models，每个模型都有可能重复处理标签；
- XGBoost 这类模型在训练前需要连续整数标签，太晚处理容易出错。

所以我把 classification 的标签自定义提前到了用户选择完 Y 之后，也就是在 feature scaling、feature selection 和 train/test split 之前。

现在 CLI 的 classification 流程大致变成：

选择 X
选择原始 Y
如果是 classification，则进入标签自定义流程
生成最终分类标签和整数编码
再进行 feature scaling / feature selection
再进行 train/test split
再训练模型


在代码里，我新增了两个变量：

label_config = None
metric_average = None

`label_config` 用来保存标签转换的完整配置，比如用了什么策略、分了几类、每个类别对应什么编码。`metric_average` 用来保存多分类指标的平均方式，比如 micro、macro 或 weighted。

当 `mode_num == 2`，也就是当前任务是 classification 时，会调用：

label_customizer = ClassificationModelSelection("__label_customizer__")
y, label_config = label_customizer.clf_workflow.customize_label(...)

这里的 `"__label_customizer__"` 只是为了复用 classification workflow 里的标签处理逻辑，并不是一个真正要训练的模型。

如果最终类别数大于 2，CLI 会让用户选择多分类指标的平均方式：

Micro
Macro
Weighted

我默认建议使用 weighted，因为地球化学数据里不同类别的样本数量经常不均衡，weighted 更适合这种情况。

---

## 2. 新增四种标签自定义方式

修改文件：`geochemistrypi/data_mining/constants.py`

原来标签自定义选项比较粗略，不能很好地表达用户自定义多分类的场景。我把 `CUSTOMIZE_LABEL_STRATEGY` 改成了四种更明确的方式：

CUSTOMIZE_LABEL_STRATEGY = [
    "Keep Original Labels and Encode",
    "Map Existing Labels to Custom Labels",
    "Numeric Interval Bins",
    "Quantile Bins",
]

这四种方式分别对应：

1. **Keep Original Labels and Encode**
   保留原始标签含义，只是把它们编码成模型可训练的整数。

2. **Map Existing Labels to Custom Labels**
   用户手动把已有标签映射成新标签，比如把多个岩性类别合并成更大的类别。

3. **Numeric Interval Bins**
   对数值型 Y 按区间分箱，例如把 SiO2 按边界分成 low、middle、high。

4. **Quantile Bins**
   对数值型 Y 按分位数自动分箱，例如用户输入 3，就自动分成 3 类。

这样 CLI 里用户可以很清楚地知道自己是在用哪种方式定义分类标签。


## 3. 完善 classification 的标签转换入口

修改文件：`geochemistrypi/data_mining/model/classification.py`

我主要改的是 `ClassificationWorkflowBase.customize_label(...)`。

这个函数现在变成 classification 标签转换的统一入口。它既支持 CLI 交互式调用，也支持以后 API 或非交互方式传入 `label_mapping`。

现在它会先做一些必要检查：

if not isinstance(y, pd.DataFrame) or y.shape[1] != 1:
    raise ValueError("Classification target Y must be exactly one column.")

if y.isnull().any().any():
    raise ValueError("Classification target Y contains missing values. Please handle missing labels before training.")

这么做是因为分类任务的目标列必须明确，而且目标标签缺失会直接影响训练结果，不能在这里静默处理。


## 4. 统一把最终标签编码成连续整数

修改文件：`geochemistrypi/data_mining/model/classification.py`

多分类里一个非常重要的点是：用户看到的类别名和模型实际训练用的标签不一定一样。

比如用户定义的是：

low
middle
high

模型实际训练时应该使用：

0
1
2

所以我在 `customize_label(...)` 里加入了统一编码逻辑：

custom_label_to_code = {label: idx for idx, label in enumerate(custom_labels)}

然后把最终标签映射成整数：

encoded[target_column] = encoded[target_column].map(custom_label_to_code)
encoded[target_column] = encoded[target_column].astype(int)

这样做的好处是：

- 所有分类模型拿到的标签格式一致；
- XGBoost 不会因为标签不是 `0..K-1` 而报错；
- 后面保存预测结果时，可以再根据映射关系解码成人类可读标签。


## 5. 保存标签转换配置 label_config

修改文件：`geochemistrypi/data_mining/model/classification.py`

为了保证实验结果可以复现，我给每一次标签转换都生成了一个 `label_config`。


```python
{
    "target_transform_version": 1,
    "strategy": strategy,
    "target_column": target_column,
    "num_classes": len(custom_labels),
    "classes": [...],
    "custom_label_to_code": {...},
    "code_to_custom_label": {...},
    "class_counts": {...},
}
```

如果是区间分箱或分位数分箱，还会保存 bins 信息。这样以后看实验结果时，可以知道当时到底是怎么把原始 Y 转成最终分类标签的。

同时我也保存了这些文件：

Y Raw Before Customizing Label
Y Human-Readable After Customizing Label
Y Encoded After Customizing Label
Target Label Mapping
Target Class Counts
Target Transform Configuration
```

这样做是为了让输出结果不仅有模型指标，还能看到分类标签是怎么来的。


## 6. 支持字典映射、区间分箱和分位数分箱

修改文件：`geochemistrypi/data_mining/model/classification.py`

### 6.1 字典映射

用户可以把已有标签映射成自定义标签。例如：

A -> acid
B -> basic
C -> neutral

我在代码里检查了映射是否完整：

missing = [label for label in observed if label not in mapping_dict and str(label) not in mapping_dict]
if missing:
    raise ValueError(...)

这样可以避免有些标签没有被映射，最后悄悄变成空值。

### 6.2 数值区间分箱

如果 Y 是数值列，用户可以自己输入切分边界，比如：  45; 60

然后给三个类别命名：  low; middle; high

代码会用 `pd.cut(...)` 生成最终分类标签。

这里我检查了：

- Y 必须是数值列；
- 边界数量必须正确；
- 边界必须严格递增；
- 分类数必须至少为 2。

### 6.3 分位数分箱

用户也可以选择 Quantile Bins，然后输入想要的分类数，比如 3 或 4。

代码会用 pd.qcut(...) 根据数据分布自动计算分位数边界，再转成用户定义的类别名。

我也检查了分类数不能超过 Y 的唯一值数量，否则分位数分箱没有意义。

## 7. 修改 train/test split，classification 使用分层拆分

修改文件：`geochemistrypi/data_mining/data/data_readiness.py`

原来的 `data_split(...)` 不支持 stratify，而且 X/y 和 name 是分两次拆分的。现在我给它加了 `stratify` 参数：

def data_split(X, y, names, test_size=0.2, stratify=None):

并且改成一次性拆分：

X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
    X,
    y,
    names,
    test_size=test_size,
    random_state=42,
    stratify=stratify_values,
)

这样可以保证：

- X、Y 和样本名不会错位；
- classification 任务可以按最终类别做分层拆分；
- train/test 里的类别比例更稳定。

在 CLI 里，如果是 classification，会先检查每个类别至少有 2 个样本：

if (class_counts < 2).any():
    raise ValueError(...)

如果某个类别只有 1 个样本，就不继续训练，因为这种情况下分层拆分不可靠。


## 8. ClassificationModelSelection 不再重复转换标签

修改文件：`geochemistrypi/data_mining/process/classify.py`

原来的 `ClassificationModelSelection.activate(...)` 里面会自己调用 `customize_label(...)`。但是现在 CLI 已经在前面处理过标签了，如果这里再处理一次，就会重复编码。

所以我给 `ClassificationModelSelection` 增加了几个参数：

label_config: dict | None = None
labels_already_customized: bool = False
metric_average: str | None = None


如果 `labels_already_customized=True`，就不会再调用 `customize_label(...)`。

同时我把 `label_config` 和 `metric_average` 挂到具体 workflow 上：

self.clf_workflow.label_config = self.label_config
self.clf_workflow.metric_average = self.metric_average

这样后面的评分、输出、预测解码都可以拿到同一套标签配置。


## 9. 多分类指标 average 只选择一次

修改文件：`geochemistrypi/data_mining/model/func/algo_classification/_common.py`

原来的 `score(...)` 在多分类时会直接在函数内部询问用户选择 micro、macro 或 weighted。这样如果训练多个模型，就会反复询问。

我把函数改成：

def score(y_true, y_predict, average=None, interactive=True):

如果外面已经传入 `average`，就直接使用，不再询问。

如果没有传入，并且是多分类，则默认使用 weighted。这样在非交互情况下也不会卡住。

同时我给 precision、recall、F1 加了：  zero_division=0

这样某个类别没有被预测出来时，不会因为除零问题直接报错。

## 10. 交叉验证适配小样本多分类

修改文件：`geochemistrypi/data_mining/model/func/algo_classification/_common.py`

原来 cross validation 默认是 10 折，但多分类时有些类别样本可能不足 10 个。这样会导致交叉验证报错。

我加了一个最小类别样本数检查：

min_class_count = int(y_train_series.value_counts().min())
cv_num = min(cv_num, min_class_count)

如果某个类别训练集中少于 2 个样本，就跳过交叉验证并保存 warning。

这样不会因为 CV 折数太大导致整个训练流程失败。


## 11. XGBoost 多分类兼容

修改文件：`geochemistrypi/data_mining/model/classification.py`

XGBoost 对多分类标签要求比较严格，标签必须是连续整数：  0, 1, 2, ..., K-1

所以我在 `XGBoostClassification` 里增加了专门的 `fit(...)` 逻辑。

训练前会检查：

classes = sorted(y_series.dropna().unique().tolist())
expected_classes = list(range(len(classes)))
if classes != expected_classes:
    raise ValueError(...)

如果是多分类，会自动设置：

objective="multi:softprob"
num_class=类别数
eval_metric="mlogloss"

如果是二分类，则使用：

eval_metric="logloss"

这样 XGBoost 可以直接训练用户自定义出来的三分类、四分类或更多分类。

## 12. 多分类时不再静默跳过二分类图表

修改文件：`geochemistrypi/data_mining/model/classification.py`

原来的 ROC、Precision-Recall、Precision-Recall Threshold 图只适合二分类。多分类时如果继续画，逻辑不完整；但如果直接不画，又会让用户以为是程序漏输出。

所以我保留了原来的判断：

if class_count == 2:
    画 ROC / PR / Threshold
else:
    保存跳过说明

多分类时会保存一个说明文件，告诉用户这些图为什么没有生成。

## 13. 新增可追溯输出文件 `_traceability.py`

新增文件：`geochemistrypi/data_mining/model/func/algo_classification/_traceability.py`

我没有把所有保存逻辑继续塞进 `classification.py` 或 `classify.py`，而是单独新建了一个 `_traceability.py` 文件。

这里面主要负责保存和标签可追溯有关的内容，包括：

save_target_transform_configuration(...)
save_class_counts(...)
decode_predictions(...)
save_decoded_predictions(...)
save_metric_configuration(...)
save_skipped_binary_plot_notice(...)

这样主流程只需要调用这些函数，不用把保存逻辑写得很杂。

## 14. 保存 decoded prediction

修改文件：`geochemistrypi/data_mining/process/classify.py`
新增文件：`geochemistrypi/data_mining/model/func/algo_classification/_traceability.py`

模型训练时实际输出的是整数编码，比如：

0
1
2

但是用户真正关心的是：

low
middle
high

所以我增加了预测结果解码函数：

def decode_predictions(predictions, label_config):
    ...

训练结束后会额外保存：

Y Train Predict Decoded
Y Test Predict Decoded

这样输出文件里既有模型实际预测的编码，也有人能看懂的分类标签。


## 15. 保存 train/test 类别数量和指标配置

修改文件：`geochemistrypi/data_mining/process/classify.py`
新增文件：`geochemistrypi/data_mining/model/func/algo_classification/_traceability.py`

训练结束后会保存：

Y Train Class Counts
Y Test Class Counts
Classification Target Traceability
Metric Configuration - <Algorithm>
```

这样可以知道：

- 训练集里每一类有多少样本；
- 测试集里每一类有多少样本；
- 本次分类标签是怎么生成的；
- 多分类指标用的是 micro、macro 还是 weighted。

这些信息对科研平台很重要，因为如果分类边界和类别分布不清楚，模型指标本身就没有办法解释。

---

## 16. 这次改完之后的 CLI classification 流程

现在用户在 CLI 里做 classification 时，流程变成：

选择 X
选择 Y
进入 Classification Label Customization
选择标签定义方式
输入分类数和类别名
生成 label_config
把最终标签编码成 0..K-1
选择多分类指标 average
做 feature scaling / feature selection
做 stratified train/test split
训练模型
保存模型指标
保存标签映射和类别数量
保存 decoded prediction
多分类时保存二分类图表跳过说明

## 17. 主要改动文件汇总

这次和多分类功能直接相关的文件有：

geochemistrypi/data_mining/cli_pipeline.py
geochemistrypi/data_mining/constants.py
geochemistrypi/data_mining/data/data_readiness.py
geochemistrypi/data_mining/model/classification.py
geochemistrypi/data_mining/model/func/algo_classification/_common.py
geochemistrypi/data_mining/model/func/algo_classification/_traceability.py
geochemistrypi/data_mining/process/classify.py

其中 `_traceability.py` 是新增加的文件。

---

## 18. 后面 CLI 实测时又补掉的几个坑

后面我又在当前环境里真正跑了一轮 CLI，不只是看代码。这里又发现几个会影响用户实际使用的问题，所以顺手一起修了。

第一个问题是 `geochemistrypi --help` 一开始会直接崩。原因不是多分类逻辑本身，而是当前环境里的 Typer 0.7.0 和 Click 8.4.1 有兼容问题。Typer 旧代码里有些地方还是按旧版 Click 的写法调用，比如 `make_metavar()` 不传 `ctx`，但新版 Click 已经要求传 `ctx` 了。另外 Click 8.4 里 `flag_value=None` 的语义也变了，导致一些普通字符串参数会被误判成 flag。这个会影响得很直接，比如 `data-mining --data xxx.csv` 可能解析不对。

我没有去大改 CLI 框架，也没有强行要求用户降级依赖，而是在 `geochemistrypi/cli.py` 里加了很小的兼容处理：

- 如果当前 Click 使用新的 `UNSET` sentinel，就把 Typer 0.7 传进来的旧式 `flag_value=None` 转回新版 Click 能理解的状态；
- 如果当前 Click 的 `make_metavar` 需要 `ctx`，就让 Typer 的 help 相关方法兼容这个参数；
- 同时关闭 Typer 0.7 那条不兼容的 rich help 路径，让 help 先稳定可用。

这样处理以后，下面这些命令都能正常返回：

```powershell
.\.conda\Scripts\geochemistrypi.exe --help
.\.conda\Scripts\geochemistrypi.exe --version
.\.conda\Scripts\geochemistrypi.exe data-mining --help
```

第二个问题是 CLI 入口一导入就会把整条 data mining pipeline、classification、mlflow 都加载进来。这样即使用户只是看 `--help` 或 `--version`，也会出现一些和训练无关的输出，甚至还有之前留下的调试打印。这个体验很别扭，也容易让人以为程序已经开始跑训练了。

所以我把 pipeline 改成延迟导入：只有真正进入 `data-mining` 命令、准备跑训练流程时，才导入 `cli_pipeline` 和 `DataSource`。顺便也删掉了 `classification.py` 里那个 import 时打印当前文件路径的调试输出。现在 help 和 version 都是干净的，不会夹杂模型模块或 mlflow 的信息。

第三个问题是 Windows 控制台编码。之前入口文案里有 `π`，CLI 启动提示里还有 sparkle 符号。它们在某些 PowerShell/GBK 环境里会变成乱码，甚至影响输出观感。我把 CLI 面向用户的这些文案都改成了 ASCII 版本，比如统一写成 `Geochemistry Pi`，启动提示里的特殊符号也换成普通短横线。这个改动不影响项目名字和功能，只是让命令行更稳。

第四个问题是用户如果想把一个字符串标签列当作 classification 的 Y，原来的列选择逻辑还是偏向数值列，容易把这类真实场景挡掉。现在我把列选择加了一个 `require_numeric` 参数：

- 选 X 的时候还是要求数值列；
- 选 classification 的 Y 时允许字符串标签列；
- 如果用户选的是 numeric interval bins 或 quantile bins，再在标签转换那里检查 Y 必须是数值列。

这样既没有放松特征矩阵 X 的要求，也不会阻止用户用已经存在的类别标签做多分类。

第五个问题是标签编码顺序。多分类里不能只按数据里第一次出现的顺序去编码，因为用户自己输入的顺序才是更重要的语义。比如用户定义的是 `low; middle; high`，那就应该稳定编码成：

```text
low -> 0
middle -> 1
high -> 2
```

所以我让 interval、quantile 和 dict mapping 都保存 `label_order`，最后编码时优先按用户定义的顺序来，而不是按数据出现顺序来。这个对 XGBoost 也很重要，因为它要求标签是连续的 `0..K-1`。

第六个问题是 Decision Tree 的树图在当前 sklearn 版本里会因为 `node_ids=None` 报错。这个不是多分类算法本身的问题，但会让训练已经完成后卡在画图阶段。我把默认的 `None` 转成 sklearn 能接受的 `False`，避免模型训练成功以后被一个附属图表打断。

最后我补了一组回归测试，主要覆盖这些地方：

- CLI app 能正常构建；
- `--help`、`--version`、`data-mining --help` 在当前 Click/Typer 组合下能正常渲染；
- `--version` 不会误抢 `data-mining --data ...` 这种真实训练入口；
- help/version 不会提前导入 pipeline，也不会混入调试输出；
- CLI 可见文案不再包含容易乱码的特殊字符；
- classification 的 Y 可以选择字符串标签列；
- 非交互标签转换能返回 train/test 的编码结果和配置；
- interval 标签编码按用户输入顺序走；
- dict mapping 标签编码也按用户定义顺序走；
- Decision Tree 默认 `node_ids=None` 不再报错。
