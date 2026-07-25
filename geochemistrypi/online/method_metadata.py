"""Online verification status and input documentation for scientific methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MethodStatus = Literal["verified", "testing"]


@dataclass(frozen=True)
class InputColumnMetadata:
    """Human-readable definition of one workbook input column."""

    name: str
    label: str
    description: str
    unit: str
    example: float | int | str
    data_type: str = "number"
    required: bool = True
    minimum: float | None = None
    exclusive_minimum: bool = False


@dataclass(frozen=True)
class MethodMetadata:
    """Online-specific readiness information for a discovered method."""

    status: MethodStatus
    status_message: str
    formula: str | None = None
    input_columns: tuple[InputColumnMetadata, ...] = ()
    input_notes: tuple[str, ...] = ()


TESTING_MESSAGE = "该方法尚未完成 Online 输入格式和科学结果验证，暂不允许计算。"


METHOD_METADATA: dict[tuple[str, str], MethodMetadata] = {
    (
        "algo_kinetic",
        "first_order",
    ): MethodMetadata(
        status="verified",
        status_message="已完成 Excel 上传、计算、结果下载及数值回归测试。",
        formula="C_t = c0 × exp(-k × t)",
        input_columns=(
            InputColumnMetadata(
                name="c0",
                label="初始浓度",
                description="反应开始时的物质浓度。",
                unit="自定义，须与结果浓度一致",
                example=100.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="k",
                label="一级反应速率常数",
                description="控制浓度随时间衰减速度的一级速率常数。",
                unit="时间⁻¹",
                example=0.1,
                minimum=0,
            ),
            InputColumnMetadata(
                name="t",
                label="反应时间",
                description="从反应开始到计算时刻所经过的时间。",
                unit="须与 k 的时间单位一致",
                example=5.0,
                minimum=0,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 c0、k、t。",
            "每一行代表一次独立计算，所有单元格均应为数值。",
            "k 与 t 必须使用相互匹配的时间单位。",
        ),
    ),
    (
        "algo_kinetic",
        "second_order",
    ): MethodMetadata(
        status="verified",
        status_message="已完成公式核对、Excel 输入校验、独立参考数值、结果下载及回归测试。",
        formula="1 / C_t = 1 / c0 + k × t",
        input_columns=(
            InputColumnMetadata(
                name="c0",
                label="初始浓度",
                description="反应开始时的物质浓度。",
                unit="自定义，须与结果浓度一致",
                example=100.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="k",
                label="二级反应速率常数",
                description="二级反应的速率常数。",
                unit="浓度⁻¹·时间⁻¹",
                example=0.01,
                minimum=0,
            ),
            InputColumnMetadata(
                name="t",
                label="反应时间",
                description="从反应开始到计算时刻所经过的时间。",
                unit="须与 k 的时间单位一致",
                example=5.0,
                minimum=0,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 c0、k、t。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "c0 必须大于 0；k 和 t 必须大于或等于 0。",
            "k 与 t 必须使用相互匹配的时间单位。",
        ),
    ),
    (
        "algo_kinetic",
        "radioactive_decay",
    ): MethodMetadata(
        status="verified",
        status_message="已完成公式核对、Excel 输入校验、独立参考数值、结果下载及回归测试。",
        formula="N_t = n0 × exp(-decay_const × t)",
        input_columns=(
            InputColumnMetadata(
                name="n0",
                label="初始核素数量",
                description="计算开始时的核素数量或活度基准值。",
                unit="自定义，须与结果数量一致",
                example=1000.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="decay_const",
                label="衰变常数",
                description="描述核素单位时间衰变概率的常数。",
                unit="时间⁻¹",
                example=0.05,
                minimum=0,
            ),
            InputColumnMetadata(
                name="t",
                label="衰变时间",
                description="从初始时刻到计算时刻所经过的时间。",
                unit="须与 decay_const 的时间单位一致",
                example=10.0,
                minimum=0,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 n0、decay_const、t。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "n0、decay_const 和 t 必须大于或等于 0。",
            "decay_const 与 t 必须使用相互匹配的时间单位。",
        ),
    ),
    (
        "algo_kinetic",
        "adsorption_kinetics",
    ): MethodMetadata(
        status="testing",
        status_message=TESTING_MESSAGE,
        formula="q_t = qe × (1 - exp(-k1 × t))",
        input_columns=(
            InputColumnMetadata(
                name="qe",
                label="平衡吸附量",
                description="达到平衡时单位吸附剂对应的吸附量。",
                unit="按数据集约定",
                example=50.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="k1",
                label="伪一级速率常数",
                description="伪一级吸附动力学速率常数。",
                unit="时间⁻¹",
                example=0.2,
                minimum=0,
            ),
            InputColumnMetadata(
                name="t",
                label="吸附时间",
                description="从吸附开始到计算时刻所经过的时间。",
                unit="须与 k1 的时间单位一致",
                example=5.0,
                minimum=0,
            ),
        ),
        input_notes=(
            "当前调度器默认使用伪一级模型。",
            "该方法需要补充模型选择参数并完成验证后才会开放计算。",
        ),
    ),
    (
        "algo_transport",
        "fick_diffusion",
    ): MethodMetadata(
        status="verified",
        status_message="已完成公式核对、Excel 输入校验、独立参考数值、结果下载及回归测试。",
        formula="J = -D × dc_dx",
        input_columns=(
            InputColumnMetadata(
                name="D",
                label="扩散系数",
                description="描述物质在介质中扩散能力的系数。",
                unit="长度²/时间",
                example=1e-9,
                minimum=0,
            ),
            InputColumnMetadata(
                name="dc_dx",
                label="浓度梯度",
                description="浓度沿空间方向的变化率，方向由正负号表示。",
                unit="浓度/长度",
                example=-1000.0,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 D、dc_dx。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "D 必须大于或等于 0；dc_dx 可以为正数、负数或 0。",
            "通量 J 的单位由 D 与 dc_dx 的单位共同决定，负号表示通量沿浓度降低方向。",
        ),
    ),
    (
        "algo_transport",
        "chromatography",
    ): MethodMetadata(
        status="verified",
        status_message="已完成公式核对、Excel 输入校验、独立参考数值、结果下载及回归测试。",
        formula="N = (tR / sigma)²",
        input_columns=(
            InputColumnMetadata(
                name="tR",
                label="保留时间",
                description="组分从进样到检测器峰值出现所经过的时间。",
                unit="时间",
                example=10.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="sigma",
                label="峰宽标准差",
                description="色谱峰时间分布的标准差。",
                unit="须与 tR 使用相同时间单位",
                example=0.5,
                minimum=0,
                exclusive_minimum=True,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 tR、sigma。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "tR 必须大于或等于 0；sigma 必须大于 0。",
            "tR 与 sigma 必须使用相同的时间单位，计算得到的理论板数 N 无量纲。",
        ),
    ),
    (
        "algo_thermodynamic",
        "vanthoff",
    ): MethodMetadata(
        status="verified",
        status_message="已完成公式和单位核对、Excel 输入校验、独立参考数值、结果下载及回归测试。",
        formula="K2 = K1 × exp[-dH / R × (1 / T2 - 1 / T1)]",
        input_columns=(
            InputColumnMetadata(
                name="K1",
                label="初始平衡常数",
                description="温度 T1 下的平衡常数。",
                unit="按模型定义，通常无量纲",
                example=10.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="dH",
                label="反应焓变",
                description="反应焓变；吸热反应为正，放热反应为负。",
                unit="J/mol",
                example=50000.0,
            ),
            InputColumnMetadata(
                name="T1",
                label="初始温度",
                description="已知平衡常数 K1 对应的绝对温度。",
                unit="K",
                example=298.15,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="T2",
                label="目标温度",
                description="需要计算平衡常数 K2 的绝对温度。",
                unit="K",
                example=350.0,
                minimum=0,
                exclusive_minimum=True,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 K1、dH、T1、T2。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "K1、T1 和 T2 必须大于 0；dH 可以为正数、负数或 0。",
            "dH 必须使用 J/mol，T1 和 T2 必须使用 K；当前气体常数固定为 8.314 J/(mol·K)。",
        ),
    ),
    (
        "algo_thermodynamic",
        "activity_coefficient",
    ): MethodMetadata(
        status="verified",
        status_message="已完成当前简化 Debye-Hückel 公式、固定参数、输入校验、参考数值和结果下载验证。",
        formula="log10(γ) = -0.509 × z² × √I / (1 + 0.5 × √I)",
        input_columns=(
            InputColumnMetadata(
                name="z",
                label="离子电荷数",
                description="离子的带电数，阳离子为正、阴离子为负、中性物种为 0。",
                unit="无量纲整数",
                example=1,
                data_type="integer",
            ),
            InputColumnMetadata(
                name="ionic_strength",
                label="离子强度",
                description="溶液的离子强度 I。",
                unit="mol/L",
                example=0.1,
                minimum=0,
            ),
        ),
        input_notes=(
            "Excel 第一行必须使用列名 z、ionic_strength。",
            "每一行代表一次独立计算，所有单元格均应为数值且不能留空。",
            "z 必须为整数，可以为正数、负数或 0；ionic_strength 必须大于或等于 0。",
            "当前实现固定使用 A=0.509、a=0.5，输出为 log10(γ)。",
            "该简化模型不应直接用于未验证的温度、溶剂或高离子强度条件。",
        ),
    ),
}


def get_method_metadata(task: str, method: str) -> MethodMetadata:
    """Return registered metadata or a safe testing-only default."""

    return METHOD_METADATA.get(
        (task, method),
        MethodMetadata(
            status="testing",
            status_message=TESTING_MESSAGE,
            input_notes=("该方法的 Online 输入列、单位和参数说明正在整理。",),
        ),
    )
