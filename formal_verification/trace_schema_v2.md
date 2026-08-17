# GeoPi Lean 桥接轨迹规范第二版

## 一 轨迹分层

counterexample_trace.json 保存一份正向基线和二十份人工单点反例。每个反例只对应一个 publicCheckId，任何同时触发两个检查项的反例都不满足流程自检要求。

production_trace.json 只保存真实业务审计事实。该轨迹读取项目内置分类训练数据与应用数据，并调用当前业务函数。生产案例不读取 expectedConformant 作为放行条件，也不携带 targetCheckId。

## 二 顶层字段

| 字段 | 类型 | 约束 |
|---|---|---|
| schemaVersion | 自然数 | 固定为 2 |
| sourceCommit | 字符串 | 非空，工作树含已跟踪变更时追加 dirty |
| generatedAt | 字符串 | 使用 UTC 时间 |
| cases | CaseTrace 列表 | 非空，caseId 全部唯一 |

## 三 案例类别

baseline 表示反例套件的正向基线。expectedConformant 固定为 true，targetCheckId 为空。

counterexample 表示人工单点反例。expectedConformant 固定为 false，targetCheckId 必须属于 Checker.lean 定义的二十个公开检查项。

production 表示真实业务审计。targetCheckId 为空，expectedConformant 不参与接受判定。

## 四 事实域

DatasetTrace 保存业务样本的稳定自然数身份键、原始身份非空掩码、过滤前后行集合、训练测试划分、监督学习视图行序、列角色验证记录和派生特征来源。运行探针在一个案例内为相同原始身份分配相同键，重复身份仍产生重复键，原始身份序列摘要保存在生产观测文件中。

PipelineTrace 保存有效训练与推理 schema、状态拟合行、拟合次数、训练态与推理态摘要、模型训练输入摘要、阶段顺序、输出标量总数和非有限标量数量。

LabelTrace 保存运行时 codec、完整数据映射、训练映射、测试映射、持久化映射、拟合次数和预测解码事实。编码值只要求单射与完整覆盖，不要求从零开始连续。

PredictionTrace 保存预测范围、来源行、样本行、内存预测、导出预测、索引不匹配策略和运行标识。

ExecutionTrace 保存可用模型、已选模型、实际训练模型、共享注册表前后状态、注册表变更操作和活动运行标识。

## 五 严格解码

Python 与 Lean 解码器都拒绝未知字段、缺失字段、错误类型、重复 JSON 键、非法 caseKind、重复 caseId 和无效 targetCheckId。合法但违反业务合同的轨迹返回退出码 1。格式或编排错误返回退出码 2。

## 六 判定边界

Lean 证明对象是轨迹中已经捕获的有限事实。Python 负责从真实对象与源码结构提取事实，Lean 负责按固定命题判定事实，Python 镜像检查器负责逐字段交叉比对。第三方机器学习算法本身的数学正确性不属于该轨迹的证明范围。
