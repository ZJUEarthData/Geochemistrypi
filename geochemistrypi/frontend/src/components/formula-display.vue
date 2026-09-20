<script setup lang="ts">
import { computed } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps<{
  method: string
  fallback: string
}>()

const formulas: Record<string, string> = {
  mass_balance: String.raw`\begin{aligned}\sum_{i=1}^{n}c_i&=m_{\mathrm{total}}\\[2pt]\left|\sum_{i=1}^{n}c_i-m_{\mathrm{total}}\right|&<1\times10^{-6}\end{aligned}`,
  precipitation_dissolution: String.raw`\mathrm{SI}=\log_{10}\!\left(\frac{\mathrm{IAP}}{K_{\mathrm{sp}}}\right)`,
  ion_exchange: String.raw`R_{\mathrm{ex}}=K_{\mathrm{sel}}\frac{C_{A,\mathrm{eq}}}{C_{B,\mathrm{eq}}}`,
  mass_action: String.raw`K=\frac{\prod_j c_{j,\mathrm{product}}^{\nu_j}}{\prod_k c_{k,\mathrm{reactant}}^{\lvert\nu_k\rvert}}`,
  internal_standard: String.raw`\delta=\left[\frac{2R_{\mathrm{sample}}}{R_{\mathrm{std,prev}}+R_{\mathrm{std,next}}}-1\right]\times10^3\,\permil`,
  double_spike: String.raw`R_{i,\mathrm{mix}}=\frac{\phi R_{i,\mathrm{sp}}+(1-\phi)R_{i,\mathrm{std}}(95/m_i)^{\beta_{\mathrm{sample}}}}{(95/m_i)^{\beta_{\mathrm{mix}}}}`,
  first_order: String.raw`C_t=C_0e^{-kt}`,
  second_order: String.raw`\frac{1}{C_t}=\frac{1}{C_0}+kt`,
  radioactive_decay: String.raw`N_t=N_0e^{-\lambda t}`,
  adsorption_kinetics: String.raw`\begin{aligned}\mathrm{PFO}:\quad q_t&=q_e\!\left(1-e^{-kt}\right)\\[3pt]\mathrm{PSO}:\quad q_t&=\frac{q_e^2kt}{1+q_ekt}\end{aligned}`,
  rubie: String.raw`\ln\!\left(\mathrm{SCSS}_{\mathrm{ppm}}\right)=14.2-\frac{11032}{T}-\frac{379P}{T}`,
  ding: String.raw`\begin{aligned}\ln\!\left(\mathrm{SCSS}_{\mathrm{Ni\text{-}free}}\right)&=A+\frac{B}{T}+\sum_i C_iX_i+D X_{\mathrm{Fe}}X_{\mathrm{Ti}}+\frac{EP}{T}\\[3pt]\mathrm{SCSS}&=\frac{\mathrm{SCSS}_{\mathrm{Ni\text{-}free}}}{0.0013\,\mathrm{Ni}^2-0.0109\,\mathrm{Ni}+1}\end{aligned}`,
  blanchard: String.raw`\begin{aligned}\ln(\mathrm{SCSS})&=a+\frac{b}{T}+\frac{cP}{T}+\sum_i A_iX_i+\ln X_{\mathrm{Fe,sulf}}-\ln X_{\mathrm{FeO,melt}}&&\text{(Eq. 11)}\\[3pt]\ln(\mathrm{SCSS})&=a+\frac{b}{T}+\frac{cP}{T}+\frac{\sum_i A_iX_i}{T}+\ln X_{\mathrm{Fe,sulf}}-\ln X_{\mathrm{FeO,melt}}&&\text{(Eq. 12)}\end{aligned}`,
  hybrid: String.raw`\mathrm{SCSS}_{\mathrm{hybrid}}=\mathrm{RF}(\mathbf{x}_{16})\exp\!\left(\frac{551.22}{T}-\frac{121.83P}{T}\right)`,
  gibbs_minimization: String.raw`\min_{\mathbf n\geq0}G=\sum_i g_i n_i,\qquad A\mathbf n=\mathbf b`,
  activity_coefficient: String.raw`\log_{10}\gamma=-0.509\,z^2\frac{\sqrt I}{1+0.5\sqrt I}`,
  vanthoff: String.raw`K_2=K_1\exp\!\left[-\frac{\Delta H}{R}\left(\frac{1}{T_2}-\frac{1}{T_1}\right)\right]`,
  fick_diffusion: String.raw`J=-D\frac{\mathrm dc}{\mathrm dx}`,
  advection_dispersion: String.raw`C(x,t)=\frac{C_0}{2\sqrt{\pi Dt}}\exp\!\left[-\frac{(x-vt)^2}{4Dt}\right]`,
  chromatography: String.raw`N=\left(\frac{t_R}{\sigma}\right)^2`
}

const renderedFormula = computed(() => {
  const expression = formulas[props.method]
  if (!expression) return ''

  return katex.renderToString(expression, {
    displayMode: true,
    throwOnError: false,
    strict: false,
    output: 'htmlAndMathml'
  })
})
</script>

<template>
  <div class="formula-display" :title="fallback">
    <div v-if="renderedFormula" class="formula-math" v-html="renderedFormula"></div>
    <code v-else class="formula-fallback">{{ fallback }}</code>
  </div>
</template>

<style scoped lang="scss">
.formula-display {
  container-type: inline-size;
  min-width: 0;
  max-width: 100%;
  overflow-x: auto;
  padding: 10px 14px;
  border: 1px solid #c8e2df;
  border-radius: 7px;
  color: #145f65;
  background: #f0f8f7;
}

.formula-math {
  width: max-content;
  min-width: 0;
  max-width: none;
}

:deep(.katex-display) {
  margin: 0;
  text-align: left;
}

:deep(.katex) {
  font-family: 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif;
  font-size: clamp(0.78em, 2.2cqi, 1.08em);
}

.formula-fallback {
  color: inherit;
  font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', serif;
  white-space: normal;
}

@media (max-width: 560px) {
  .formula-display {
    padding: 9px 10px;
  }

  :deep(.katex) {
    font-size: clamp(0.72em, 3.4vw, 0.92em);
  }
}
</style>
