<script setup lang="ts">
import { computed } from 'vue'

import { t } from '@/i18n'

type StatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger'

const props = defineProps<{
  fileName?: string
  rows?: number | null
  columns?: number | null
  missingCells?: number | null
  method?: string
  parameters?: string[]
  status: string
  statusTone?: StatusTone
  jobId?: string
}>()

const dimensions = computed(() => {
  if (props.rows == null || props.columns == null) return '—'
  return `${props.rows.toLocaleString()} × ${props.columns.toLocaleString()}`
})

const missingValue = computed(() =>
  props.missingCells == null ? '—' : props.missingCells.toLocaleString()
)
</script>

<template>
  <section class="run-summary" :aria-label="t('Run summary', '运行摘要')">
    <header>
      <div>
        <p>{{ t('RUN SUMMARY', '运行摘要') }}</p>
        <h2>{{ t('Current analysis', '当前分析') }}</h2>
      </div>
      <span class="status-pill" :class="statusTone || 'neutral'">{{ status }}</span>
    </header>

    <dl class="summary-grid">
      <div class="summary-item wide">
        <dt>{{ t('Dataset', '数据集') }}</dt>
        <dd class="mono" :title="fileName">{{ fileName || '—' }}</dd>
      </div>
      <div class="summary-item">
        <dt>{{ t('Rows × columns', '行数 × 列数') }}</dt>
        <dd class="mono">{{ dimensions }}</dd>
      </div>
      <div class="summary-item">
        <dt>{{ t('Missing values', '缺失值') }}</dt>
        <dd class="mono">{{ missingValue }}</dd>
      </div>
      <div class="summary-item wide">
        <dt>{{ t('Selected method', '已选方法') }}</dt>
        <dd>{{ method || '—' }}</dd>
      </div>
      <div class="summary-item wide">
        <dt>{{ t('Parameters', '参数') }}</dt>
        <dd v-if="parameters?.length" class="parameter-list">
          <code v-for="parameter in parameters" :key="parameter">{{ parameter }}</code>
        </dd>
        <dd v-else>—</dd>
      </div>
      <div class="summary-item wide">
        <dt>{{ t('Job ID', '任务 ID') }}</dt>
        <dd class="mono">{{ jobId || t('Assigned after run', '运行后生成') }}</dd>
      </div>
      <div class="summary-item wide">
        <dt>{{ t('Software version', '软件版本') }}</dt>
        <dd class="mono">Geochemistry π 0.7.0</dd>
      </div>
    </dl>
  </section>
</template>

<style scoped lang="scss">
.run-summary {
  padding: 28px 24px;
  color: #263f46;
  background: #fff;
}

header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 18px;
  border-bottom: 1px solid #e1e7e8;

  p {
    margin: 0;
    color: #6a7f84;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.12em;
  }

  h2 {
    margin: 5px 0 0;
    color: #173f47;
    font-size: 18px;
    font-weight: 680;
  }
}

.status-pill {
  flex: 0 0 auto;
  padding: 5px 9px;
  border: 1px solid #d5dee0;
  border-radius: 999px;
  color: #536970;
  background: #f5f7f8;
  font-size: 12px;
  font-weight: 650;

  &.info {
    border-color: #b9d8dc;
    color: #236a75;
    background: #eef7f8;
  }

  &.success {
    border-color: #aad8bf;
    color: #26704f;
    background: #edf8f2;
  }

  &.warning {
    border-color: #e7cb91;
    color: #865c0d;
    background: #fff8e8;
  }

  &.danger {
    border-color: #efb6ad;
    color: #a03c30;
    background: #fff1ef;
  }
}

.summary-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  margin: 0;
}

.summary-item {
  min-width: 0;
  padding: 15px 0;
  border-bottom: 1px solid #edf0f1;

  &:nth-child(2n):not(.wide) {
    padding-left: 14px;
    border-left: 1px solid #edf0f1;
  }

  &.wide {
    grid-column: 1 / -1;
  }

  dt {
    margin-bottom: 6px;
    color: #708287;
    font-size: 13px;
    font-weight: 620;
  }

  dd {
    min-width: 0;
    margin: 0;
    overflow-wrap: anywhere;
    color: #203f46;
    font-size: 14px;
    line-height: 1.5;
  }
}

.mono,
code {
  font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-variant-numeric: tabular-nums;
}

.parameter-list {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;

  code {
    padding: 4px 7px;
    border: 1px solid #d6e2e3;
    border-radius: 4px;
    color: #27666e;
    background: #f5f8f8;
    font-size: 12px;
  }
}

@media (max-width: 1180px) {
  .run-summary {
    border-top: 1px solid #e1e7e8;
  }
}

@media (max-width: 560px) {
  .run-summary {
    padding: 22px 20px;
  }
}
</style>
