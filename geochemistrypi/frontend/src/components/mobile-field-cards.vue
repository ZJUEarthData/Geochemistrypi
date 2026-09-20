<script lang="ts" setup>
import { t } from '@/i18n'

defineProps<{
  rows: readonly object[]
}>()

type Field = {
  key: string
  label: string
  value: unknown
}

function fieldLabel(key: string) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

function fieldsFor(row: object): Field[] {
  return Object.entries(row as Record<string, unknown>).flatMap(([key, value]) => {
    if (key === 'values' && value && typeof value === 'object' && !Array.isArray(value)) {
      return Object.entries(value as Record<string, unknown>).map(([nestedKey, nestedValue]) => ({
        key: `${key}.${nestedKey}`,
        label: nestedKey,
        value: nestedValue
      }))
    }

    return [{ key, label: fieldLabel(key), value }]
  })
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(' · ') : '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<template>
  <div class="mobile-field-cards" aria-label="Mobile data records">
    <article v-for="(row, rowIndex) in rows" :key="rowIndex" class="field-card">
      <header>{{ t('Record', '记录') }} {{ rowIndex + 1 }}</header>
      <dl>
        <div v-for="field in fieldsFor(row)" :key="field.key">
          <dt>{{ field.label }}</dt>
          <dd>{{ formatValue(field.value) }}</dd>
        </div>
      </dl>
    </article>
  </div>
</template>

<style lang="scss" scoped>
.mobile-field-cards {
  display: none;
}

@media (max-width: 760px) {
  .mobile-field-cards {
    display: grid;
    width: 100%;
    min-width: 0;
    gap: 12px;
  }

  .field-card {
    min-width: 0;
    padding: 14px;
    border: 1px solid #d8e3e2;
    border-radius: 8px;
    background: #fff;
  }

  header {
    padding-bottom: 10px;
    border-bottom: 1px solid #e7ecec;
    color: #173f47;
    font-size: 14px;
    font-weight: 700;
  }

  dl {
    display: grid;
    gap: 10px;
    margin: 12px 0 0;
  }

  dl > div {
    display: grid;
    grid-template-columns: minmax(90px, 0.42fr) minmax(0, 1fr);
    gap: 10px;
    min-width: 0;
  }

  dt {
    color: #6a7f84;
    font-size: 13px;
    font-weight: 620;
    overflow-wrap: anywhere;
  }

  dd {
    min-width: 0;
    margin: 0;
    color: #294f56;
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
}
</style>
