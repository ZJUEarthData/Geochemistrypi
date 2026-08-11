import { ref } from 'vue'

export type Locale = 'en' | 'zh'

const STORAGE_KEY = 'geochemistrypi-online-locale'

function initialLocale(): Locale {
  if (typeof window === 'undefined') return 'en'
  return window.localStorage.getItem(STORAGE_KEY) === 'zh' ? 'zh' : 'en'
}

export const locale = ref<Locale>(initialLocale())

export function setLocale(nextLocale: Locale) {
  locale.value = nextLocale
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, nextLocale)
    document.documentElement.lang = nextLocale === 'zh' ? 'zh-CN' : 'en'
  }
}

setLocale(locale.value)

export function t(english: string, chinese: string) {
  return locale.value === 'zh' ? chinese : english
}

const methodDescriptions: Record<string, string> = {
  mass_balance: 'Mass balance',
  precipitation_dissolution: 'Precipitation/Dissolution equilibrium',
  ion_exchange: 'Ion exchange equilibrium',
  mass_action: 'Law of mass action',
  internal_standard: 'Internal standard method',
  double_spike: 'Double-spike (double-diluent) method',
  first_order: 'First-order kinetics',
  second_order: 'Second-order kinetics',
  radioactive_decay: 'Radioactive decay',
  adsorption_kinetics: 'Adsorption kinetics',
  rubie: 'Laurenz et al. (2016) empirical equation (legacy key: rubie)',
  ding: 'Ding et al. (2018) empirical equation',
  blanchard: 'Blanchard et al. (2021) empirical equation',
  hybrid: 'Hybrid machine-learning model (Zhang et al., 2024)',
  gibbs_minimization: 'Gibbs free-energy minimization',
  activity_coefficient: 'Activity coefficient model',
  vanthoff: "van't Hoff equation",
  fick_diffusion: 'Fick diffusion',
  advection_dispersion: 'Advection–dispersion equation',
  chromatography: 'Chromatographic plate number'
}

const methodDescriptionsZh: Record<string, string> = {
  mass_balance: '质量平衡检查',
  precipitation_dissolution: '溶解-沉淀平衡',
  ion_exchange: '离子交换平衡',
  mass_action: '质量作用定律',
  internal_standard: '内标法',
  double_spike: '双稀释剂法',
  first_order: '一级反应动力学',
  second_order: '二级反应动力学',
  radioactive_decay: '放射性衰变',
  adsorption_kinetics: '吸附动力学',
  rubie: 'Laurenz 等（2016）经验公式（旧方法键：rubie）',
  ding: 'Ding 等（2018）经验公式',
  blanchard: 'Blanchard 等（2021）经验公式',
  hybrid: '集成机器学习模型（Zhang 等，2024）',
  gibbs_minimization: '吉布斯自由能最小化',
  activity_coefficient: '活度系数模型',
  vanthoff: '范特霍夫方程',
  fick_diffusion: '菲克扩散',
  advection_dispersion: '对流-弥散方程',
  chromatography: '色谱分离理论板数'
}

const methodStatuses: Record<string, string> = {
  mass_balance:
    'Verified the mass-conservation formula, dynamic species columns, tolerance classification, result downloads, and error boundaries.',
  precipitation_dissolution:
    'Verified the saturation-index formula, precipitation/equilibrium/dissolution classification, result downloads, and error boundaries.',
  ion_exchange:
    'Verified the Gaines–Thomas ratio, zero-denominator protection, result downloads, and error boundaries.',
  mass_action:
    'Verified single-reaction extent solving, independent reference values, JSON input, convergence, and error boundaries.',
  internal_standard:
    'Verified Hg 3133 before-and-after bracketing, independent reference values, result downloads, and error boundaries.',
  double_spike:
    'Generated mixture ratios from known parameters and verified inverse convergence, residuals, result downloads, and error boundaries.',
  first_order:
    'Verified Excel upload, calculation, result downloads, and numerical regression tests.',
  second_order:
    'Verified the formula, Excel input validation, independent reference values, result downloads, and regression tests.',
  radioactive_decay:
    'Verified the formula, Excel input validation, independent reference values, result downloads, and regression tests.',
  adsorption_kinetics:
    'Verified pseudo-first-order and pseudo-second-order adsorption models, model selection, result downloads, and error boundaries.',
  rubie:
    'Checked the original equation, units, and applicability range, and verified independent reference values, result downloads, and input boundaries.',
  ding: 'Checked the published equation, regression coefficients, cation mole fractions, Ni correction, and applicability domain, and verified independent values and error boundaries.',
  blanchard:
    'Checked Eq. 11/12 from Blanchard et al. (2021), hydrous cation fractions, sulfide Fe/(Fe+Ni+Cu) atomic ratio, and calibration domain, and verified numerical values and error boundaries.',
  hybrid:
    'Fixed the random forest, training split, random seed, and pressure–temperature correction coefficients using the public data and code from ZhangZhou et al. (2024), and verified batch inference and error boundaries.',
  gibbs_minimization:
    'Verified non-negative species amounts, component conservation, HiGHS linear minimization, independent analytical reference values, and infeasible boundaries.',
  activity_coefficient:
    'Verified the current simplified Debye–Hückel equation, fixed parameters, input validation, reference values, and result downloads.',
  vanthoff:
    'Verified the equation and units, Excel input validation, independent reference values, result downloads, and regression tests.',
  fick_diffusion:
    'Verified the formula, Excel input validation, independent reference values, result downloads, and regression tests.',
  advection_dispersion:
    'Verified the one-dimensional instantaneous point-source solution, independent reference values, result downloads, and error boundaries.',
  chromatography:
    'Verified the formula, Excel input validation, independent reference values, result downloads, and regression tests.'
}

const methodFormulas: Record<string, string> = {
  mass_balance: 'Σ c_i = total_mass (absolute error < 1×10⁻⁶)',
  blanchard:
    'Eq. 11: ln(SCSS)=a+b/T+cP/T+ΣAᵢXᵢ+ln(XFe,sulf)-ln(XFeO,melt); Eq. 12 replaces ΣAᵢXᵢ with ΣAᵢXᵢ/T',
  hybrid:
    'SCSS_hybrid = RF(16 pressure, temperature, and composition features) × exp(551.22/T - 121.83×Pressure/T)'
}

export function chemicalMethodDescription(name: string, source: string) {
  if (locale.value === 'zh') return methodDescriptionsZh[name] || source
  return methodDescriptions[name] || formatMachineName(name)
}

const dataMiningDescriptions: Record<string, [string, string]> = {
  dataset_profile: ['Dataset overview and quality check', '数据集概览与质量检查'],
  data_preprocessing: ['Data preprocessing', '数据预处理'],
  regression: ['Regression', '回归'],
  classification: ['Classification', '分类'],
  clustering: ['Clustering', '聚类']
}

export function dataMiningFeatureDescription(name: string, source: string) {
  const description = dataMiningDescriptions[name]
  return description ? t(description[0], description[1]) : apiText(source)
}

export function chemicalMethodStatus(name: string, source: string) {
  return locale.value === 'zh' ? source : methodStatuses[name] || englishFallback(source)
}

export function chemicalMethodFormula(name: string, source: string) {
  return locale.value === 'zh' ? source : methodFormulas[name] || apiText(source)
}

function formatMachineName(value: string) {
  return value.replace(/^algo_/, '').replace(/_/g, ' ')
}

const exactEnglish: Record<string, string> = {
  页面质量概览: 'Quality overview',
  逐列统计: 'Column statistics',
  数据预览: 'Data preview',
  'JSON 报告': 'JSON report',
  处理结果预览: 'Processed-data preview',
  'CSV 处理数据': 'Processed CSV data',
  'JSON 处理记录': 'JSON processing record',
  回归指标: 'Regression metrics',
  模型系数: 'Model coefficients',
  '预测结果 CSV': 'Prediction CSV',
  'JSON 模型报告': 'JSON model report',
  分类指标: 'Classification metrics',
  混淆矩阵: 'Confusion matrix',
  聚类指标: 'Clustering metrics',
  簇大小与中心: 'Cluster sizes and centers',
  '聚类结果 CSV': 'Cluster-assignment CSV',
  '已完成 Excel/CSV 上传、数据类型识别、缺失值、重复行、唯一值、数值统计、预览和 JSON 报告下载验证。':
    'Verified Excel/CSV upload, data-type detection, missing values, duplicate rows, unique values, numerical statistics, preview, and JSON report download.',
  '已完成列选择、缺失值处理、结果预览、CSV 数据下载和 JSON 处理记录验证。':
    'Verified column selection, missing-value handling, result preview, CSV download, and JSON processing record.',
  '已接入 v0.8 线性、二阶多项式、Lasso、Elastic Net、Bayesian Ridge 和 Ridge 回归，并完成固定随机种子训练测试划分、R²/MAE/RMSE、系数、预测结果和报告下载验证。':
    'Verified v0.8 Linear, second-order Polynomial, Lasso, Elastic Net, Bayesian Ridge, and Ridge regression with a fixed-seed train/test split, R²/MAE/RMSE, coefficients, predictions, and report downloads.',
  '已完成数值特征标准化、逻辑回归、分层训练测试划分、Accuracy/Precision/Recall/F1、混淆矩阵和结果下载验证。':
    'Verified numeric-feature standardization, logistic regression, stratified train/test splitting, Accuracy/Precision/Recall/F1, a confusion matrix, and result downloads.',
  '已完成数值特征标准化、K-means 聚类、簇数设置、三项聚类评价指标、聚类中心和结果下载验证。':
    'Verified numeric-feature standardization, K-means clustering, cluster-count configuration, three evaluation metrics, cluster centers, and result downloads.',
  总质量或总浓度: 'Total mass or concentration',
  '需要与同一行所有物种浓度之和进行比较的总量。':
    'Total to compare with the sum of all species concentrations in the same row.',
  '自定义，必须与各物种列一致': 'User-defined; must match all species columns',
  离子活度积: 'Ion activity product',
  '参与溶解-沉淀平衡的离子活度乘积。':
    'Product of the ion activities involved in the dissolution–precipitation equilibrium.',
  按所用热力学定义: 'As defined by the thermodynamic model',
  溶度积常数: 'Solubility product constant',
  '目标矿物或固相的溶度积常数。': 'Solubility product of the target mineral or solid phase.',
  必须与离子活度积定义一致: 'Must use the same definition as the ion activity product',
  A离子平衡浓度: 'Ion A equilibrium concentration',
  '离子 A 的平衡浓度。': 'Equilibrium concentration of ion A.',
  B离子平衡浓度: 'Ion B equilibrium concentration',
  '离子 B 的平衡浓度，作为分母必须大于 0。':
    'Equilibrium concentration of ion B; it is the denominator and must be greater than 0.',
  自定义浓度单位: 'User-defined concentration unit',
  '必须与 A 相同': 'Must match A',
  选择性系数: 'Selectivity coefficient',
  'Gaines-Thomas 选择性系数。': 'Gaines–Thomas selectivity coefficient.',
  无量纲: 'Dimensionless',
  平衡常数: 'Equilibrium constant',
  '所定义反应的正平衡常数。': 'Positive equilibrium constant for the defined reaction.',
  按反应活度定义: 'As defined by the reaction activities',
  化学计量数: 'Stoichiometric coefficients',
  'JSON对象；反应物为负、生成物为正，例如 {"A":-1,"B":1}。':
    'JSON object; reactants are negative and products are positive, for example {"A":-1,"B":1}.',
  初始浓度: 'Initial concentrations',
  '与 stoich 含有相同物种的 JSON 对象，例如 {"A":1.0,"B":0.0}。':
    'JSON object containing the same species as stoich, for example {"A":1.0,"B":0.0}.',
  自定义一致浓度单位: 'Any consistent concentration unit',
  样品标识: 'Sample identifier',
  '标准物质行必须填写 3133；其他文本表示待计算样品。':
    'Standard rows must contain 3133; other text identifies samples to calculate.',
  文本: 'Text',
  '202Hg 信号': '202Hg signal',
  '用于计算样品相对前后 3133 标准的总汞偏差。':
    'Used to calculate total-Hg deviation relative to the preceding and following 3133 standards.',
  仪器信号单位: 'Instrument signal unit',
  比值: 'Ratio',
  压力: 'Pressure',
  '模型计算压力。': 'Pressure used by the model.',
  温度: 'Temperature',
  '模型计算绝对温度。': 'Absolute temperature used by the model.',
  反应时间: 'Reaction time',
  '从反应开始到计算时刻所经过的时间。':
    'Elapsed time from reaction start to the calculation point.',
  '须与 k 的时间单位一致': "Must match k's time unit",
  初始核素数量: 'Initial nuclide amount',
  '计算开始时的核素数量或活度基准值。':
    'Nuclide amount or activity baseline at the start of the calculation.',
  衰变常数: 'Decay constant',
  '描述核素单位时间衰变概率的常数。': 'Constant describing the probability of decay per unit time.',
  衰变时间: 'Decay time',
  '从初始时刻到计算时刻所经过的时间。':
    'Elapsed time from the initial point to the calculation point.',
  '须与 decay_const 的时间单位一致': "Must match decay_const's time unit",
  吸附模型: 'Adsorption model',
  '使用 first（伪一级）或 second（伪二级）。':
    'Use first (pseudo-first-order) or second (pseudo-second-order).',
  平衡吸附量: 'Equilibrium adsorption capacity',
  '平衡时的单位吸附剂吸附量。': 'Amount adsorbed per unit adsorbent at equilibrium.',
  按数据集约定: 'As defined by the dataset',
  速率常数: 'Rate constant',
  '与所选模型对应的吸附速率常数。': 'Adsorption rate constant for the selected model.',
  随模型与时间单位确定: 'Defined by the model and time unit',
  吸附时间: 'Adsorption time',
  '从吸附开始到计算时刻的时间。': 'Elapsed time from adsorption start to the calculation point.',
  '必须与 k 匹配': 'Must match k',
  '自定义，须与结果浓度一致': 'User-defined; must match the result concentration',
  '时间⁻¹': 'time⁻¹',
  '浓度⁻¹·时间⁻¹': 'concentration⁻¹·time⁻¹',
  '必须与 k 的时间单位一致': "Must match k's time unit",
  '硅酸盐熔体 H₂O 含量；干体系填写 0。':
    'H₂O content of the silicate melt; enter 0 for anhydrous systems.',
  '硫化物 Fe': 'Sulfide Fe',
  '硫化物熔体中的 Fe 含量。': 'Fe content of the sulfide melt.',
  '硫化物 Ni': 'Sulfide Ni',
  '硫化物熔体中 Ni、Cu、Co 的合计含量。': 'Combined Ni, Cu, and Co content of the sulfide melt.',
  '硫化物 Ni+Cu+Co': 'Sulfide Ni+Cu+Co',
  '硫化物 S': 'Sulfide S',
  '硫化物熔体中的 S 含量。': 'S content of the sulfide melt.',
  '硫化物 O': 'Sulfide O',
  '硫化物熔体中的 O 含量。': 'O content of the sulfide melt.',
  '种摩尔 Gibbs 能': 'Species molar Gibbs energy',
  'JSON 对象，例如 {"A":0,"B":-10}。': 'JSON object, for example {"A":0,"B":-10}.',
  '任意一致的能量/摩尔单位': 'Any consistent energy-per-mole unit',
  物种组分化学计量: 'Species component stoichiometry',
  'JSON 嵌套对象，例如 {"A":{"X":1},"B":{"X":1}}。':
    'Nested JSON object, for example {"A":{"X":1},"B":{"X":1}}.',
  '组分摩尔数/物种摩尔数': 'component moles/species moles',
  体系组分总量: 'System component totals',
  'JSON 对象，例如 {"X":1}。': 'JSON object, for example {"X":1}.',
  摩尔或任意一致的物质量单位: 'Moles or any consistent amount-of-substance unit',
  离子电荷数: 'Ion charge',
  '离子的带电数，阳离子为正、阴离子为负、中性物种为 0。':
    'Ion charge number: positive for cations, negative for anions, and 0 for neutral species.',
  无量纲整数: 'Dimensionless integer',
  离子强度: 'Ionic strength',
  '溶液的离子强度 I。': 'Ionic strength I of the solution.',
  初始平衡常数: 'Initial equilibrium constant',
  '温度 T1 下的平衡常数。': 'Equilibrium constant at temperature T1.',
  '按模型定义，通常无量纲': 'As defined by the model; usually dimensionless',
  反应焓变: 'Reaction enthalpy',
  '反应焓变；吸热反应为正，放热反应为负。':
    'Reaction enthalpy; positive for endothermic and negative for exothermic reactions.',
  初始温度: 'Initial temperature',
  '已知平衡常数 K1 对应的绝对温度。':
    'Absolute temperature corresponding to known equilibrium constant K1.',
  目标温度: 'Target temperature',
  '需要计算平衡常数 K2 的绝对温度。':
    'Absolute temperature for calculating equilibrium constant K2.',
  扩散系数: 'Diffusion coefficient',
  '描述物质在介质中扩散能力的系数。': 'Coefficient describing diffusion through the medium.',
  '长度²/时间': 'length²/time',
  浓度梯度: 'Concentration gradient',
  '浓度沿空间方向的变化率，方向由正负号表示。':
    'Rate of concentration change in space; the sign indicates direction.',
  '浓度/长度': 'concentration/length',
  源强度: 'Source strength',
  '一维瞬时点源解析解中的初始源强度。':
    'Initial source strength in the one-dimensional instantaneous point-source solution.',
  按模型定义: 'As defined by the model',
  平流速度: 'Advection velocity',
  '沿坐标正方向为正的平均平流速度。':
    'Mean advection velocity, positive in the positive coordinate direction.',
  '长度/时间': 'length/time',
  弥散系数: 'Dispersion coefficient',
  '一维水动力弥散系数。': 'One-dimensional hydrodynamic dispersion coefficient.',
  位置: 'Position',
  '相对于点源的空间位置。': 'Spatial position relative to the point source.',
  长度: 'Length',
  时间: 'Time',
  '释放后的时间；t=0 按当前实现约定输出 0。':
    'Time after release; the current implementation returns 0 when t=0.',
  保留时间: 'Retention time',
  '组分从进样到检测器峰值出现所经过的时间。':
    'Time from injection until the component peak reaches the detector.',
  峰宽标准差: 'Peak-width standard deviation',
  '色谱峰时间分布的标准差。': 'Standard deviation of the chromatographic peak-time distribution.',
  '须与 tR 使用相同时间单位': 'Must use the same time unit as tR',
  同位素比值: 'Isotope ratio',
  '反应开始时的物质浓度。': 'Concentration of the substance at the start of the reaction.',
  一级反应速率常数: 'First-order rate constant',
  '控制浓度随时间衰减速度的一级速率常数。':
    'First-order rate constant controlling concentration decay over time.',
  二级反应速率常数: 'Second-order rate constant',
  '二级反应的速率常数。': 'Rate constant of the second-order reaction.',
  '自定义，须与结果数量一致': 'User-defined; must match the result quantity',
  '硫化物饱和硫含量经验式使用的压力。':
    'Pressure used by the empirical sulfide-saturation sulfur-content equation.',
  '硫化物饱和硫含量经验式使用的绝对温度。':
    'Absolute temperature used by the empirical sulfide-saturation sulfur-content equation.',
  '以 FeO* 表示的总铁质量分数。': 'Total iron mass fraction expressed as FeO*.',
  '平衡硫化物熔体中的 Ni；≤8.5 wt.% 时按纯 FeS 基线处理。':
    'Ni in the equilibrium sulfide melt; values ≤8.5 wt.% use the pure-FeS baseline.',
  '以 FeO* 表示的硅酸盐熔体总铁；模型将其作为 FeO 处理。':
    'Total iron in the silicate melt expressed as FeO*; the model treats it as FeO.',
  '硅酸盐熔体 H₂O 含量；参与含水阳离子分数归一化。':
    'H₂O content of the silicate melt, included in hydrous cation-fraction normalization.',
  '平衡硫化物中的元素 Ni 含量。': 'Elemental Ni content of the equilibrium sulfide.',
  '硫化物 Cu': 'Sulfide Cu',
  '平衡硫化物中的元素 Cu 含量。': 'Elemental Cu content of the equilibrium sulfide.',
  'MnO（可选）': 'MnO (optional)',
  'P₂O₅（可选）': 'P₂O₅ (optional)',
  'Cr₂O₃（可选）': 'Cr₂O₃ (optional)',
  '可选归一化组分；缺失时按 0 处理。':
    'Optional normalization component; treated as 0 when absent.',
  '物种摩尔 Gibbs 能': 'Species molar Gibbs energy'
}

const noteEnglish: Record<string, string> = {
  '除 total_mass 外，每个其他列都被视为一个物种的浓度列。':
    'Every column except total_mass is treated as a species-concentration column.',
  '至少需要一个物种列；物种浓度必须为非负有限数值且不能留空。':
    'At least one species column is required; concentrations must be non-negative, finite, and non-empty.',
  '输出包含 species_sum、mass_difference 和 is_balanced。':
    'Output includes species_sum, mass_difference, and is_balanced.',
  'SI > 0 输出 precipitation；SI = 0 输出 equilibrium；SI < 0 输出 dissolution。':
    'SI > 0 returns precipitation; SI = 0 returns equilibrium; SI < 0 returns dissolution.',
  'ion_activity_product 和 ksp 都必须大于 0。':
    'ion_activity_product and ksp must both be greater than 0.',
  '每行输出对应的 exchange_ratio。': 'Each row returns its exchange_ratio.',
  '当前求解器适用于单个理想反应；每行定义一个反应。':
    'The current solver handles one ideal reaction; each row defines one reaction.',
  'stoich 与 initial_concentrations 必须包含完全相同的物种。':
    'stoich and initial_concentrations must contain exactly the same species.',
  '求解采用反应进度上的对数反应商二分法，不依赖 SciPy。':
    'The solver bisects the logarithmic reaction quotient over reaction extent and does not require SciPy.',
  '每个样品行的前方和后方都必须各有一行 Label=3133 的标准物质。':
    'Every sample row must have one Label=3133 standard row before it and one after it.',
  '标准物质行本身不计算分馏值，因此其结果单元格为空是正常现象。':
    'Fractionation is not calculated for standard rows, so blank result cells are expected.',
  '输出包括 THg(%)、d199/d200/d201/d202(‰) 以及 D199、D200、D201。':
    'Output includes THg (%), d199/d200/d201/d202 (‰), and D199/D200/D201.',
  'Excel 必须包含名为“3程序处理_输入常数”的工作表。':
    'The Excel workbook must include the legacy constants-input worksheet.',
  '九个常数读取自该工作表的第一行，所有比值都必须大于 0。':
    'The nine constants are read from the first row of that worksheet; all ratios must be greater than 0.',
  'Online 当前固定使用 SciPy fsolve，并检查收敛状态、残差与 0≤phi_ref≤1。':
    'Online currently uses SciPy fsolve and checks convergence status, residuals, and 0≤phi_ref≤1.',
  '输出 CSV 包含 phi_ref、beta_sple 和 beta_mix。':
    'The output CSV contains phi_ref, beta_sple, and beta_mix.',
  'model 仅接受 first 或 second，不区分大小写。':
    'model accepts only first or second, case-insensitively.',
  'SCSS_pred 的输出单位为 ppm。': 'SCSS_pred is returned in ppm.',
  '该公式实际来源为 Laurenz et al. (2016)；为兼容原项目仍保留方法键 rubie。':
    'The equation is from Laurenz et al. (2016); the rubie method key is retained for compatibility.',
  '实验标定大致覆盖 7–21 GPa 和 2373–2673 K；范围外结果属于外推。':
    'Experimental calibration covers approximately 7–21 GPa and 2373–2673 K; results outside this range are extrapolations.',
  'SCSS_Ni_free 和 SCSS_pred 的单位均为 ppm。':
    'SCSS_Ni_free and SCSS_pred are both returned in ppm.',
  'Ni-free 标定范围：1200–1800 °C、0.0001–5.5 GPa；输入温度必须换算为 K。':
    'Ni-free calibration range: 1200–1800 °C and 0.0001–5.5 GPa; input temperature must be converted to K.',
  '主要组成标定范围：SiO₂ 33–55、TiO₂ 0.01–15、FeO* 5–30、Al₂O₃ 5–20、CaO 5–19、MgO 6–23 wt.%。':
    'Main calibration ranges (wt.%): SiO₂ 33–55, TiO₂ 0.01–15, FeO* 5–30, Al₂O₃ 5–20, CaO 5–19, and MgO 6–23.',
  'Ni 修正用于硫化物 Ni 约 10–50 wt.%；0–8.5 wt.% 输出 Ni-free 基线。':
    'The Ni correction applies at about 10–50 wt.% sulfide Ni; 0–8.5 wt.% returns the Ni-free baseline.',
  '八种氧化物应输入 wt.% 而不是 0–1 小数，合计须在 90–105 wt.% 之间。':
    'Enter the eight oxides as wt.%, not 0–1 fractions; their total must be 90–105 wt.%.',
  '每一行定义一个独立的封闭体系。': 'Each row defines an independent closed system.',
  'gibbs_energies 与 stoichiometry 必须包含相同物种。':
    'gibbs_energies and stoichiometry must contain the same species.',
  'stoichiometry 与 component_totals 必须包含相同组分；系数和总量不能为负。':
    'stoichiometry and component_totals must contain the same components; coefficients and totals cannot be negative.',
  '当前模型把输入 Gibbs 能视为固定温压下的常数，适用于理想纯物种/相；不包含活度、混合熵或非理想相互作用。':
    'Input Gibbs energies are treated as constants at fixed pressure and temperature for ideal pure species/phases; activities, mixing entropy, and non-ideal interactions are not included.',
  '当前实现固定使用 A=0.509、a=0.5，输出为 log10(γ)。':
    'The current implementation fixes A=0.509 and a=0.5 and returns log10(γ).',
  '该简化模型不应直接用于未验证的温度、溶剂或高离子强度条件。':
    'Do not apply this simplified model directly to unverified temperatures, solvents, or high ionic strengths.',
  '当前公式是无限一维域中的瞬时点源解析解。':
    'The current equation is the instantaneous point-source solution in an infinite one-dimensional domain.',
  'k 与 t 必须使用相互匹配的时间单位。': 'k and t must use matching time units.',
  'c0 必须大于 0；k 和 t 必须大于或等于 0。':
    'c0 must be greater than 0; k and t must be greater than or equal to 0.',
  'n0、decay_const 和 t 必须大于或等于 0。':
    'n0, decay_const, and t must be greater than or equal to 0.',
  'decay_const 与 t 必须使用相互匹配的时间单位。':
    'decay_const and t must use matching time units.',
  'SCSS_eq11_ppm 与 SCSS_eq12_ppm 同时输出；SCSS_pred 采用论文后续讨论使用的 Eq.11（Model 1）。':
    "Both SCSS_eq11_ppm and SCSS_eq12_ppm are returned; SCSS_pred uses Eq. 11 (Model 1), as in the paper's subsequent discussion.",
  'Pressure 必须以 GPa、T 必须以 K 输入；合并标定数据覆盖约 0.0001–24 GPa 和 1423–2623 K。':
    'Enter Pressure in GPa and T in K; the combined calibration dataset covers about 0.0001–24 GPa and 1423–2623 K.',
  '论文排除了 FeO* <0.5 wt.% 的高度还原实验；Online 因此要求 FeO* 为 0.5–40.1 wt.%。':
    'The paper excluded highly reducing experiments with FeO* <0.5 wt.%; Online therefore requires FeO* between 0.5 and 40.1 wt.%.',
  '所有氧化物均输入 wt.% 而不是 0–1 小数；含可选组分在内的氧化物合计须在 90–110 wt.% 之间。':
    'Enter all oxides as wt.%, not 0–1 fractions; the oxide total, including optional components, must be 90–110 wt.%.',
  'Fe、Ni、Cu 是硫化物中的元素质量百分数；公式只使用原子比 Fe/(Fe+Ni+Cu)，不需要输入 S 或 O。':
    'Fe, Ni, and Cu are elemental weight percentages in the sulfide; the equation uses only the Fe/(Fe+Ni+Cu) atomic ratio, so S and O are not required.',
  '输入只需要 16 项预测特征，不再需要 SCSS 实测值；每一行会得到独立预测。':
    'Only the 16 prediction features are required; measured SCSS values are not needed, and each row receives an independent prediction.',
  '输出包含 RF_base_pred_ppm、PT_correction_factor 和最终 SCSS_pred（ppm）。':
    'Output includes RF_base_pred_ppm, PT_correction_factor, and final SCSS_pred (ppm).',
  '模型版本为 zhangzhou2024-hybrid-rf-v1：使用公开 Dataset #4 的 542 条数据、固定 70/30 划分和可复现随机种子构建。':
    'Model version zhangzhou2024-hybrid-rf-v1 was built from 542 public Dataset #4 records using a fixed 70/30 split and reproducible random seed.',
  '标定域为 0.0001–24 GPa、1423–2623 K；所有组成列必须使用 wt.%，Online 不允许超出训练数据范围外推。':
    'The calibration domain is 0.0001–24 GPa and 1423–2623 K; all composition columns must use wt.%, and Online does not permit extrapolation beyond the training range.',
  '该模型是经验/机器学习预测，不替代实验测量；使用结果时应同时报告模型版本和适用域。':
    'This empirical/machine-learning prediction does not replace experimental measurement; report the model version and applicability domain with results.',
  'z 必须为整数，可以为正数、负数或 0；ionic_strength 必须大于或等于 0。':
    'z must be an integer and may be positive, negative, or 0; ionic_strength must be greater than or equal to 0.',
  'K1、T1 和 T2 必须大于 0；dH 可以为正数、负数或 0。':
    'K1, T1, and T2 must be greater than 0; dH may be positive, negative, or 0.',
  'dH 必须使用 J/mol，T1 和 T2 必须使用 K；当前气体常数固定为 8.314 J/(mol·K)。':
    'dH must use J/mol and T1/T2 must use K; the gas constant is fixed at 8.314 J/(mol·K).',
  'D 必须大于或等于 0；dc_dx 可以为正数、负数或 0。':
    'D must be greater than or equal to 0; dc_dx may be positive, negative, or 0.',
  '通量 J 的单位由 D 与 dc_dx 的单位共同决定，负号表示通量沿浓度降低方向。':
    'Flux J has units determined jointly by D and dc_dx; the negative sign indicates flux toward lower concentration.',
  'D 必须大于 0，t 必须大于或等于 0。':
    'D must be greater than 0 and t must be greater than or equal to 0.',
  'tR 必须大于或等于 0；sigma 必须大于 0。':
    'tR must be greater than or equal to 0; sigma must be greater than 0.',
  'tR 与 sigma 必须使用相同的时间单位，计算得到的理论板数 N 无量纲。':
    'tR and sigma must use the same time unit; the calculated theoretical plate number N is dimensionless.'
}

export function apiText(source: string | null | undefined) {
  if (!source) return ''
  if (locale.value === 'zh') return source
  if (exactEnglish[source]) return exactEnglish[source]
  if (noteEnglish[source]) return noteEnglish[source]

  const isotopeRatio = source.match(/^([0-9]+Hg) 与 ([0-9]+Hg) 的同位素比值。$/)
  if (isotopeRatio) return `Isotope ratio of ${isotopeRatio[1]} to ${isotopeRatio[2]}.`
  const molybdenumRatio = source.match(/^(.+)的 ([0-9]+Mo\/[0-9]+Mo) 比值。$/)
  if (molybdenumRatio) {
    const sourceLabels: Record<string, string> = {
      双稀释剂: 'double spike',
      标准样品: 'standard sample',
      测得混合物: 'measured mixture'
    }
    return `${molybdenumRatio[2]} ratio of the ${sourceLabels[molybdenumRatio[1]] || 'input material'}.`
  }
  const silicateContent = source.match(/^硅酸盐熔体 (.+) (质量分数|含量)。$/)
  if (silicateContent) return `${silicateContent[1]} content of the silicate melt.`
  const firstRow = source.match(/^Excel 第一行必须使用列名 (.+)。$/)
  if (firstRow) return `The first Excel row must use the column names ${firstRow[1]}.`
  if (source === '每一行代表一次独立计算，所有单元格均应为数值。')
    return 'Each row is an independent calculation; every cell must be numeric.'
  if (source === '每一行代表一次独立计算，所有单元格均应为数值且不能留空。')
    return 'Each row is an independent calculation; every cell must be numeric and non-empty.'
  if (source.includes('不能留空') && source.includes('必须'))
    return (
      source.replace(/[一-鿿，。；]/g, '').trim() ||
      'All required values must be valid and non-empty.'
    )

  const dynamicTranslations: Array<[RegExp, (...values: string[]) => string]> = [
    [
      /^(\d+) 个非数值缺失单元格未被均值\/中位数规则修改。$/,
      (count) => `${count} non-numeric missing cells were not changed by the mean/median rule.`
    ],
    [
      /^处理结果仍有 (\d+) 个缺失单元格。$/,
      (count) => `${count} missing cells remain after processing.`
    ],
    [
      /^训练前删除了 (\d+) 行含缺失值或无穷值的记录。$/,
      (count) => `${count} rows containing missing or infinite values were removed before training.`
    ],
    [
      /^聚类前删除了 (\d+) 行含缺失值或无穷值的记录。$/,
      (count) =>
        `${count} rows containing missing or infinite values were removed before clustering.`
    ],
    [/^发现 (\d+) 行完全重复记录。$/, (count) => `Found ${count} completely duplicate rows.`],
    [
      /^数值列中发现 (\d+) 个无穷大值。$/,
      (count) => `Found ${count} infinite values in numeric columns.`
    ],
    [/^全部为空的列：(.+)$/, (columns) => `Entirely empty columns: ${columns}`],
    [
      /^缺失率不低于 30% 的列：(.+)$/,
      (columns) => `Columns with at least 30% missing values: ${columns}`
    ],
    [/^仅含单一非空值的列：(.+)$/, (columns) => `Columns with only one non-empty value: ${columns}`]
  ]
  for (const [pattern, translate] of dynamicTranslations) {
    const match = source.match(pattern)
    if (match) return translate(...match.slice(1))
  }

  const fixedWarnings: Record<string, string> = {
    '处理结果中没有缺失单元格。': 'No missing cells remain after processing.',
    '测试集目标值缺少变化，无法计算有效的 R²。':
      'The test target has no variation, so a valid R² cannot be calculated.',
    '所有数据行均可用于回归。': 'All data rows can be used for regression.',
    '所有数据行均可用于分类。': 'All data rows can be used for classification.',
    '所有数据行均可用于聚类。': 'All data rows can be used for clustering.',
    'Silhouette 指标使用固定随机种子抽样 10,000 行计算。':
      'The silhouette score was calculated from a fixed-seed sample of 10,000 rows.',
    '未发现高缺失率、完全重复、常量列或无穷大值问题。':
      'No high missing-rate, duplicate-row, constant-column, or infinite-value issues were found.'
  }
  if (fixedWarnings[source]) return fixedWarnings[source]

  return englishFallback(source)
}

function englishFallback(source: string) {
  if (!/[\u3400-\u9fff]/.test(source)) return source
  const parentheticalEnglish = source.match(/\(([A-Za-z][^()]*)\)/)
  if (parentheticalEnglish) return parentheticalEnglish[1]
  return 'Method guidance is available for this input.'
}

export function warningIsSuccess(source: string) {
  return [
    '处理结果中没有缺失单元格',
    '均可用于回归',
    '均可用于分类',
    '均可用于聚类',
    '未发现'
  ].some((message) => source.includes(message))
}
