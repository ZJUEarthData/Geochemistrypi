<script lang="ts" setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

const canvas = ref<HTMLCanvasElement | null>(null)
let resizeObserver: ResizeObserver | null = null

const elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
const normalizedValues = [118, 76, 45, 19, 7.8, 6.1, 2.2, 2.0, 1.15, 0.72, 0.43, 0.25, 0.15, 0.09]

function renderChart() {
  const element = canvas.value
  if (!element) return

  const box = element.getBoundingClientRect()
  const width = Math.max(260, box.width)
  const height = Math.max(250, box.height)
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  element.width = Math.round(width * dpr)
  element.height = Math.round(height * dpr)

  const context = element.getContext('2d')
  if (!context) return
  context.scale(dpr, dpr)
  context.clearRect(0, 0, width, height)

  const padding = { top: 18, right: 14, bottom: 36, left: 48 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const xStep = plotWidth / (elements.length - 1)
  const logMin = -1.2
  const logMax = 2.2
  const toY = (value: number) =>
    padding.top + ((logMax - Math.log10(value)) / (logMax - logMin)) * plotHeight

  context.lineWidth = 1
  context.font = '9.5px Inter, system-ui, sans-serif'
  context.textAlign = 'right'
  context.textBaseline = 'middle'

  const yTicks = [100, 10, 1, 0.1]
  for (const tick of yTicks) {
    const y = toY(tick)
    context.beginPath()
    context.setLineDash([2, 4])
    context.strokeStyle = 'rgba(64, 139, 143, 0.2)'
    context.moveTo(padding.left, y)
    context.lineTo(width - padding.right, y)
    context.stroke()
    context.setLineDash([])
    context.fillStyle = '#658185'
    context.fillText(tick >= 1 ? String(tick) : '0.1', padding.left - 10, y)
  }

  elements.forEach((label, index) => {
    const x = padding.left + xStep * index
    context.beginPath()
    context.setLineDash([2, 4])
    context.strokeStyle = 'rgba(64, 139, 143, 0.13)'
    context.moveTo(x, padding.top)
    context.lineTo(x, padding.top + plotHeight)
    context.stroke()
    context.setLineDash([])
    context.fillStyle = '#58777b'
    context.textAlign = 'center'
    context.textBaseline = 'top'
    context.fillText(label, x, padding.top + plotHeight + 12)
  })

  context.beginPath()
  normalizedValues.forEach((value, index) => {
    const x = padding.left + xStep * index
    const y = toY(value)
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.strokeStyle = '#dc6a52'
  context.lineWidth = 2
  context.lineJoin = 'round'
  context.stroke()

  normalizedValues.forEach((value, index) => {
    const x = padding.left + xStep * index
    const y = toY(value)
    context.beginPath()
    context.arc(x, y, 3.4, 0, Math.PI * 2)
    context.fillStyle = '#f7fbfa'
    context.fill()
    context.lineWidth = 2
    context.strokeStyle = '#dc6a52'
    context.stroke()
  })
}

onMounted(() => {
  renderChart()
  if (canvas.value) {
    resizeObserver = new ResizeObserver(renderChart)
    resizeObserver.observe(canvas.value)
  }
})

onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<template>
  <div class="trace-chart">
    <canvas
      ref="canvas"
      role="img"
      aria-label="Illustrative trace-element pattern normalized to PAAS"
    ></canvas>
  </div>
</template>

<style lang="scss" scoped>
.trace-chart {
  width: 100%;
  min-width: 0;
  height: 286px;
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
