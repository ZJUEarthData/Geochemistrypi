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
    (
        "algo_equilibrium",
        "mass_balance",
    ): MethodMetadata(
        status="verified",
        status_message="已完成质量守恒公式、动态物种列、容差判定、结果下载和错误边界验证。",
        formula="Σ c_i = total_mass（绝对误差 < 1×10⁻⁶）",
        input_columns=(
            InputColumnMetadata(
                name="total_mass",
                label="总质量或总浓度",
                description="需要与同一行所有物种浓度之和进行比较的总量。",
                unit="自定义，必须与各物种列一致",
                example=0.2,
                minimum=0,
            ),
        ),
        input_notes=(
            "除 total_mass 外，每个其他列都被视为一个物种的浓度列。",
            "至少需要一个物种列；物种浓度必须为非负有限数值且不能留空。",
            "输出包含 species_sum、mass_difference 和 is_balanced。",
        ),
    ),
    (
        "algo_equilibrium",
        "precipitation_dissolution",
    ): MethodMetadata(
        status="verified",
        status_message="已完成饱和指数公式、沉淀/平衡/溶解分类、结果下载和错误边界验证。",
        formula="SI = log10(ion_activity_product / ksp)",
        input_columns=(
            InputColumnMetadata(
                name="ion_activity_product",
                label="离子活度积",
                description="参与溶解-沉淀平衡的离子活度乘积。",
                unit="按所用热力学定义",
                example=1e-8,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="ksp",
                label="溶度积常数",
                description="目标矿物或固相的溶度积常数。",
                unit="必须与离子活度积定义一致",
                example=1e-9,
                minimum=0,
                exclusive_minimum=True,
            ),
        ),
        input_notes=(
            "SI > 0 输出 precipitation；SI = 0 输出 equilibrium；SI < 0 输出 dissolution。",
            "ion_activity_product 和 ksp 都必须大于 0。",
        ),
    ),
    (
        "algo_equilibrium",
        "ion_exchange",
    ): MethodMetadata(
        status="verified",
        status_message="已完成 Gaines-Thomas 比值公式、零分母防护、结果下载和错误边界验证。",
        formula="exchange_ratio = selectivity × eq_conc_a / eq_conc_b",
        input_columns=(
            InputColumnMetadata(
                name="eq_conc_a",
                label="A离子平衡浓度",
                description="离子 A 的平衡浓度。",
                unit="自定义浓度单位",
                example=0.05,
                minimum=0,
            ),
            InputColumnMetadata(
                name="eq_conc_b",
                label="B离子平衡浓度",
                description="离子 B 的平衡浓度，作为分母必须大于 0。",
                unit="必须与 A 相同",
                example=0.05,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="selectivity",
                label="选择性系数",
                description="Gaines-Thomas 选择性系数。",
                unit="无量纲",
                example=1.2,
                minimum=0,
            ),
        ),
        input_notes=("每行输出对应的 exchange_ratio。",),
    ),
    (
        "algo_equilibrium",
        "mass_action",
    ): MethodMetadata(
        status="verified",
        status_message="已完成单反应反应进度求解、独立解析参考值、JSON输入、收敛和错误边界验证。",
        formula="K = Π(c_product^ν) / Π(c_reactant^|ν|)",
        input_columns=(
            InputColumnMetadata(
                name="K",
                label="平衡常数",
                description="所定义反应的正平衡常数。",
                unit="按反应活度定义",
                example=4.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="stoich",
                label="化学计量数",
                description='JSON对象；反应物为负、生成物为正，例如 {"A":-1,"B":1}。',
                unit="无量纲",
                example='{"A":-1,"B":1}',
                data_type="string",
            ),
            InputColumnMetadata(
                name="initial_concentrations",
                label="初始浓度",
                description='与 stoich 含有相同物种的 JSON 对象，例如 {"A":1.0,"B":0.0}。',
                unit="自定义一致浓度单位",
                example='{"A":1.0,"B":0.0}',
                data_type="string",
            ),
        ),
        input_notes=(
            "当前求解器适用于单个理想反应；每行定义一个反应。",
            "stoich 与 initial_concentrations 必须包含完全相同的物种。",
            "求解采用反应进度上的对数反应商二分法，不依赖 SciPy。",
        ),
    ),
    (
        "algo_kinetic",
        "adsorption_kinetics",
    ): MethodMetadata(
        status="verified",
        status_message="已完成伪一级和伪二级吸附模型、模型选择、结果下载和错误边界验证。",
        formula="first: q_t=qe(1-exp(-kt)); second: q_t=qe²kt/(1+qekt)",
        input_columns=(
            InputColumnMetadata(
                name="model",
                label="吸附模型",
                description="使用 first（伪一级）或 second（伪二级）。",
                unit="文本",
                example="first",
                data_type="string",
            ),
            InputColumnMetadata(
                name="qe",
                label="平衡吸附量",
                description="平衡时的单位吸附剂吸附量。",
                unit="按数据集约定",
                example=50.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="k",
                label="速率常数",
                description="与所选模型对应的吸附速率常数。",
                unit="随模型与时间单位确定",
                example=0.2,
                minimum=0,
            ),
            InputColumnMetadata(
                name="t",
                label="吸附时间",
                description="从吸附开始到计算时刻的时间。",
                unit="必须与 k 匹配",
                example=5.0,
                minimum=0,
            ),
        ),
        input_notes=("model 仅接受 first 或 second，不区分大小写。",),
    ),
    (
        "algo_transport",
        "advection_dispersion",
    ): MethodMetadata(
        status="verified",
        status_message="已完成一维瞬时点源解析式、独立参考值、结果下载和错误边界验证。",
        formula="C(x,t)=C0/[2√(πDt)] × exp[-(x-vt)²/(4Dt)]",
        input_columns=(
            InputColumnMetadata(
                name="C0",
                label="源强度",
                description="一维瞬时点源解析解中的初始源强度。",
                unit="按模型定义",
                example=100.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="v",
                label="平流速度",
                description="沿坐标正方向为正的平均平流速度。",
                unit="长度/时间",
                example=1.0,
            ),
            InputColumnMetadata(
                name="D",
                label="弥散系数",
                description="一维水动力弥散系数。",
                unit="长度²/时间",
                example=0.5,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="x",
                label="位置",
                description="相对于点源的空间位置。",
                unit="长度",
                example=2.0,
            ),
            InputColumnMetadata(
                name="t",
                label="时间",
                description="释放后的时间；t=0 按当前实现约定输出 0。",
                unit="时间",
                example=1.0,
                minimum=0,
            ),
        ),
        input_notes=(
            "D 必须大于 0，t 必须大于或等于 0。",
            "当前公式是无限一维域中的瞬时点源解析解。",
        ),
    ),
    (
        "algo_fractionation",
        "internal_standard",
    ): MethodMetadata(
        status="verified",
        status_message="已完成 Hg 3133 前后夹标计算、独立参考值、结果下载和错误边界验证。",
        formula="δ = [2R_sample/(R_std,previous + R_std,next) - 1] × 1000‰",
        input_columns=(
            InputColumnMetadata(
                name="Label",
                label="样品标识",
                description="标准物质行必须填写 3133；其他文本表示待计算样品。",
                unit="文本",
                example="3133",
                data_type="string",
            ),
            InputColumnMetadata(
                name="202Hg",
                label="202Hg 信号",
                description="用于计算样品相对前后 3133 标准的总汞偏差。",
                unit="仪器信号单位",
                example=100.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="202Hg/198Hg",
                label="202Hg/198Hg",
                description="202Hg 与 198Hg 的同位素比值。",
                unit="比值",
                example=1.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="201Hg/198Hg",
                label="201Hg/198Hg",
                description="201Hg 与 198Hg 的同位素比值。",
                unit="比值",
                example=1.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="200Hg/198Hg",
                label="200Hg/198Hg",
                description="200Hg 与 198Hg 的同位素比值。",
                unit="比值",
                example=1.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="199Hg/198Hg",
                label="199Hg/198Hg",
                description="199Hg 与 198Hg 的同位素比值。",
                unit="比值",
                example=1.0,
                minimum=0,
                exclusive_minimum=True,
            ),
        ),
        input_notes=(
            "每个样品行的前方和后方都必须各有一行 Label=3133 的标准物质。",
            "标准物质行本身不计算分馏值，因此其结果单元格为空是正常现象。",
            "输出包括 THg(%)、d199/d200/d201/d202(‰) 以及 D199、D200、D201。",
        ),
    ),
    (
        "algo_fractionation",
        "double_spike",
    ): MethodMetadata(
        status="verified",
        status_message="已用已知参数正向生成混合比值，并验证反求收敛、残差、结果下载和错误边界。",
        formula="rᵢ,mix = [φRᵢ,sp + (1-φ)Rᵢ,std(95/mᵢ)^βsample] / (95/mᵢ)^βmix",
        input_columns=tuple(
            InputColumnMetadata(
                name=name,
                label=name,
                description=description,
                unit="同位素比值",
                example=example,
                minimum=0,
                exclusive_minimum=True,
            )
            for name, description, example in (
                ("R_100_sp", "双稀释剂的 100Mo/95Mo 比值。", 0.5),
                ("R_98_sp", "双稀释剂的 98Mo/95Mo 比值。", 2.0),
                ("R_97_sp", "双稀释剂的 97Mo/95Mo 比值。", 0.7),
                ("R_100_std", "标准样品的 100Mo/95Mo 比值。", 0.1),
                ("R_98_std", "标准样品的 98Mo/95Mo 比值。", 1.5),
                ("R_97_std", "标准样品的 97Mo/95Mo 比值。", 0.6),
                ("r_100_mix", "测得混合物的 100Mo/95Mo 比值。", 0.237502),
                ("r_98_mix", "测得混合物的 98Mo/95Mo 比值。", 1.661191),
                ("r_97_mix", "测得混合物的 97Mo/95Mo 比值。", 0.631402),
            )
        ),
        input_notes=(
            "Excel 必须包含名为“3程序处理_输入常数”的工作表。",
            "九个常数读取自该工作表的第一行，所有比值都必须大于 0。",
            "Online 当前固定使用 SciPy fsolve，并检查收敛状态、残差与 0≤phi_ref≤1。",
            "输出 CSV 包含 phi_ref、beta_sple 和 beta_mix。",
        ),
    ),
    (
        "algo_solubility",
        "rubie",
    ): MethodMetadata(
        status="verified",
        status_message="已核对原公式、单位、适用范围，并完成独立参考数值、结果下载和输入边界验证。",
        formula="ln(SCSS ppm) = 14.2 - 11032/T - 379P/T",
        input_columns=(
            InputColumnMetadata(
                name="Pressure",
                label="压力",
                description="硫化物饱和硫含量经验式使用的压力。",
                unit="GPa",
                example=10.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="T",
                label="温度",
                description="硫化物饱和硫含量经验式使用的绝对温度。",
                unit="K",
                example=2500.0,
                minimum=0,
                exclusive_minimum=True,
            ),
        ),
        input_notes=(
            "SCSS_pred 的输出单位为 ppm。",
            "该公式实际来源为 Laurenz et al. (2016)；为兼容原项目仍保留方法键 rubie。",
            "实验标定大致覆盖 7–21 GPa 和 2373–2673 K；范围外结果属于外推。",
        ),
    ),
    (
        "algo_thermodynamic",
        "gibbs_minimization",
    ): MethodMetadata(
        status="verified",
        status_message="已完成非负物种量、组分守恒、HiGHS 线性最小化、独立解析参考值和不可行边界验证。",
        formula="min G = Σ(gᵢnᵢ), subject to A·n=b and nᵢ≥0",
        input_columns=(
            InputColumnMetadata(
                name="gibbs_energies",
                label="物种摩尔 Gibbs 能",
                description='JSON 对象，例如 {"A":0,"B":-10}。',
                unit="任意一致的能量/摩尔单位",
                example='{"A":0,"B":-10}',
                data_type="string",
            ),
            InputColumnMetadata(
                name="stoichiometry",
                label="物种组分化学计量",
                description='JSON 嵌套对象，例如 {"A":{"X":1},"B":{"X":1}}。',
                unit="组分摩尔数/物种摩尔数",
                example='{"A":{"X":1},"B":{"X":1}}',
                data_type="string",
            ),
            InputColumnMetadata(
                name="component_totals",
                label="体系组分总量",
                description='JSON 对象，例如 {"X":1}。',
                unit="摩尔或任意一致的物质量单位",
                example='{"X":1}',
                data_type="string",
            ),
        ),
        input_notes=(
            "每一行定义一个独立的封闭体系。",
            "gibbs_energies 与 stoichiometry 必须包含相同物种。",
            "stoichiometry 与 component_totals 必须包含相同组分；系数和总量不能为负。",
            "当前模型把输入 Gibbs 能视为固定温压下的常数，适用于理想纯物种/相；不包含活度、混合熵或非理想相互作用。",
        ),
    ),
    (
        "algo_solubility",
        "ding",
    ): MethodMetadata(
        status="verified",
        status_message="已核对论文公式、回归系数、阳离子摩尔分数、Ni 修正和适用域，并完成独立数值及错误边界验证。",
        formula=(
            "ln(SCSS_Ni-free)=A+B/T+ΣCᵢXᵢ+D·XFe·XTi+E·P/T；"
            "SCSS=SCSS_Ni-free/(0.0013Ni²-0.0109Ni+1)"
        ),
        input_columns=tuple(
            InputColumnMetadata(
                name=name,
                label=label,
                description=description,
                unit=unit,
                example=example,
                minimum=0,
                exclusive_minimum=name in {"Pressure", "T"},
            )
            for name, label, description, unit, example in (
                ("Pressure", "压力", "模型计算压力。", "GPa", 1.5),
                ("T", "温度", "模型计算绝对温度。", "K", 1873.15),
                ("SiO2", "SiO₂", "硅酸盐熔体 SiO₂ 质量分数。", "wt.%", 43.8),
                ("TiO2", "TiO₂", "硅酸盐熔体 TiO₂ 质量分数。", "wt.%", 5.0),
                ("Al2O3", "Al₂O₃", "硅酸盐熔体 Al₂O₃ 质量分数。", "wt.%", 10.0),
                ("FeO", "FeO*", "以 FeO* 表示的总铁质量分数。", "wt.%", 18.7),
                ("MgO", "MgO", "硅酸盐熔体 MgO 质量分数。", "wt.%", 8.0),
                ("CaO", "CaO", "硅酸盐熔体 CaO 质量分数。", "wt.%", 11.0),
                ("Na2O", "Na₂O", "硅酸盐熔体 Na₂O 质量分数。", "wt.%", 2.0),
                ("K2O", "K₂O", "硅酸盐熔体 K₂O 质量分数。", "wt.%", 0.5),
                (
                    "sulfide_Ni",
                    "硫化物 Ni",
                    "平衡硫化物熔体中的 Ni；≤8.5 wt.% 时按纯 FeS 基线处理。",
                    "wt.%",
                    30.0,
                ),
            )
        ),
        input_notes=(
            "SCSS_Ni_free 和 SCSS_pred 的单位均为 ppm。",
            "Ni-free 标定范围：1200–1800 °C、0.0001–5.5 GPa；输入温度必须换算为 K。",
            "主要组成标定范围：SiO₂ 33–55、TiO₂ 0.01–15、FeO* 5–30、Al₂O₃ 5–20、CaO 5–19、MgO 6–23 wt.%。",
            "Ni 修正用于硫化物 Ni 约 10–50 wt.%；0–8.5 wt.% 输出 Ni-free 基线。",
            "八种氧化物应输入 wt.% 而不是 0–1 小数，合计须在 90–105 wt.% 之间。",
        ),
    ),
    (
        "algo_solubility",
        "blanchard",
    ): MethodMetadata(
        status="verified",
        status_message="已核对 Blanchard et al. (2021) 的 Eq.11/Eq.12、含水阳离子分数、硫化物 Fe/(Fe+Ni+Cu) 原子比和标定域，并完成数值与错误边界验证。",
        formula=(
            "Eq.11: ln(SCSS)=a+b/T+cP/T+ΣAᵢXᵢ+ln(XFe,sulf)-ln(XFeO,melt)；"
            "Eq.12 将 ΣAᵢXᵢ 改为 ΣAᵢXᵢ/T"
        ),
        input_columns=(
            InputColumnMetadata(
                name="Pressure",
                label="压力",
                description="模型计算压力。",
                unit="GPa",
                example=10.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="T",
                label="温度",
                description="模型计算绝对温度。",
                unit="K",
                example=2473.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="SiO2",
                label="SiO₂",
                description="硅酸盐熔体 SiO₂ 含量。",
                unit="wt.%",
                example=45.8,
                minimum=0,
            ),
            InputColumnMetadata(
                name="TiO2",
                label="TiO₂",
                description="硅酸盐熔体 TiO₂ 含量。",
                unit="wt.%",
                example=0.21,
                minimum=0,
            ),
            InputColumnMetadata(
                name="Al2O3",
                label="Al₂O₃",
                description="硅酸盐熔体 Al₂O₃ 含量。",
                unit="wt.%",
                example=4.53,
                minimum=0,
            ),
            InputColumnMetadata(
                name="FeO",
                label="FeO*",
                description="以 FeO* 表示的硅酸盐熔体总铁；模型将其作为 FeO 处理。",
                unit="wt.%",
                example=8.17,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="MgO",
                label="MgO",
                description="硅酸盐熔体 MgO 含量。",
                unit="wt.%",
                example=37.09,
                minimum=0,
            ),
            InputColumnMetadata(
                name="CaO",
                label="CaO",
                description="硅酸盐熔体 CaO 含量。",
                unit="wt.%",
                example=3.68,
                minimum=0,
            ),
            InputColumnMetadata(
                name="Na2O",
                label="Na₂O",
                description="硅酸盐熔体 Na₂O 含量。",
                unit="wt.%",
                example=0.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="K2O",
                label="K₂O",
                description="硅酸盐熔体 K₂O 含量。",
                unit="wt.%",
                example=0.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="H2O",
                label="H₂O",
                description="硅酸盐熔体 H₂O 含量；参与含水阳离子分数归一化。",
                unit="wt.%",
                example=0.01,
                minimum=0,
            ),
            InputColumnMetadata(
                name="Fe",
                label="硫化物 Fe",
                description="平衡硫化物中的元素 Fe 含量，用于计算原子比 Fe/(Fe+Ni+Cu)。",
                unit="wt.%",
                example=60.0,
                minimum=0,
                exclusive_minimum=True,
            ),
            InputColumnMetadata(
                name="Ni",
                label="硫化物 Ni",
                description="平衡硫化物中的元素 Ni 含量。",
                unit="wt.%",
                example=0.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="Cu",
                label="硫化物 Cu",
                description="平衡硫化物中的元素 Cu 含量。",
                unit="wt.%",
                example=0.0,
                minimum=0,
            ),
            InputColumnMetadata(
                name="MnO",
                label="MnO（可选）",
                description="可选归一化组分；缺失时按 0 处理。",
                unit="wt.%",
                example=0.14,
                required=False,
                minimum=0,
            ),
            InputColumnMetadata(
                name="P2O5",
                label="P₂O₅（可选）",
                description="可选归一化组分；缺失时按 0 处理。",
                unit="wt.%",
                example=0.0,
                required=False,
                minimum=0,
            ),
            InputColumnMetadata(
                name="Cr2O3",
                label="Cr₂O₃（可选）",
                description="可选归一化组分；缺失时按 0 处理。",
                unit="wt.%",
                example=0.37,
                required=False,
                minimum=0,
            ),
        ),
        input_notes=(
            "SCSS_eq11_ppm 与 SCSS_eq12_ppm 同时输出；SCSS_pred 采用论文后续讨论使用的 Eq.11（Model 1）。",
            "Pressure 必须以 GPa、T 必须以 K 输入；合并标定数据覆盖约 0.0001–24 GPa 和 1423–2623 K。",
            "论文排除了 FeO* <0.5 wt.% 的高度还原实验；Online 因此要求 FeO* 为 0.5–40.1 wt.%。",
            "所有氧化物均输入 wt.% 而不是 0–1 小数；含可选组分在内的氧化物合计须在 90–110 wt.% 之间。",
            "Fe、Ni、Cu 是硫化物中的元素质量百分数；公式只使用原子比 Fe/(Fe+Ni+Cu)，不需要输入 S 或 O。",
        ),
    ),
    (
        "algo_solubility",
        "hybrid",
    ): MethodMetadata(
        status="verified",
        status_message="已按 ZhangZhou et al. (2024) 的公开数据与代码固定随机森林、训练划分、随机种子和温压校正系数，并完成批量推理及错误边界验证。",
        formula=(
            "SCSS_hybrid = RF(16项温压与组成特征) × "
            "exp(551.22/T - 121.83×Pressure/T)"
        ),
        input_columns=tuple(
            InputColumnMetadata(
                name=name,
                label=label,
                description=description,
                unit=unit,
                example=example,
                minimum=0,
                exclusive_minimum=name in {"Pressure", "T", "Fe", "S"},
            )
            for name, label, description, unit, example in (
                ("Pressure", "压力", "模型计算压力。", "GPa", 2.5),
                ("T", "温度", "模型计算绝对温度。", "K", 1773.0),
                ("SiO2", "SiO₂", "硅酸盐熔体 SiO₂ 含量。", "wt.%", 36.83),
                ("TiO2", "TiO₂", "硅酸盐熔体 TiO₂ 含量。", "wt.%", 10.56),
                ("Al2O3", "Al₂O₃", "硅酸盐熔体 Al₂O₃ 含量。", "wt.%", 9.94),
                ("FeO", "FeO", "硅酸盐熔体 FeO 含量。", "wt.%", 18.57),
                ("MgO", "MgO", "硅酸盐熔体 MgO 含量。", "wt.%", 9.24),
                ("CaO", "CaO", "硅酸盐熔体 CaO 含量。", "wt.%", 12.89),
                ("NiO", "NiO", "硅酸盐熔体 NiO 含量。", "wt.%", 0.43),
                ("Na2O", "Na₂O", "硅酸盐熔体 Na₂O 含量。", "wt.%", 0.34),
                ("K2O", "K₂O", "硅酸盐熔体 K₂O 含量。", "wt.%", 0.03),
                ("H2O", "H₂O", "硅酸盐熔体 H₂O 含量；干体系填写 0。", "wt.%", 0.0),
                ("Fe", "硫化物 Fe", "硫化物熔体中的 Fe 含量。", "wt.%", 64.06),
                (
                    "Ni+Cu+Co",
                    "硫化物 Ni+Cu+Co",
                    "硫化物熔体中 Ni、Cu、Co 的合计含量。",
                    "wt.%",
                    0.0,
                ),
                ("S", "硫化物 S", "硫化物熔体中的 S 含量。", "wt.%", 34.19),
                ("O", "硫化物 O", "硫化物熔体中的 O 含量。", "wt.%", 1.64),
            )
        ),
        input_notes=(
            "输入只需要 16 项预测特征，不再需要 SCSS 实测值；每一行会得到独立预测。",
            "输出包含 RF_base_pred_ppm、PT_correction_factor 和最终 SCSS_pred（ppm）。",
            "模型版本为 zhangzhou2024-hybrid-rf-v1：使用公开 Dataset #4 的 542 条数据、固定 70/30 划分和可复现随机种子构建。",
            "标定域为 0.0001–24 GPa、1423–2623 K；所有组成列必须使用 wt.%，Online 不允许超出训练数据范围外推。",
            "该模型是经验/机器学习预测，不替代实验测量；使用结果时应同时报告模型版本和适用域。",
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
