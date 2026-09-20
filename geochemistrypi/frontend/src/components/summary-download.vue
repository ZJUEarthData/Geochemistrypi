<script setup lang="ts">
import { onUnmounted, ref } from 'vue'

import { generateSummary, type SummaryResponse } from '@/api/data-mining'
import { artifactUrl } from '@/api/online'
import { t } from '@/i18n'

const props = defineProps<{ jobId: string }>()
const loading = ref(false)
const error = ref('')
const summary = ref<SummaryResponse | null>(null)
let mounted = true
onUnmounted(() => {
  mounted = false
})

async function downloadSummary() {
  if (loading.value) return
  if (!summary.value) {
    loading.value = true
    error.value = ''
    try {
      const result = await generateSummary(props.jobId)
      if (!mounted) return
      summary.value = result
    } catch (cause) {
      if (!mounted) return
      error.value =
        cause instanceof Error ? cause.message : t('Summary download failed.', 'Summary 下载失败。')
    } finally {
      loading.value = false
    }
  }
  if (!mounted || !summary.value) return
  // Keep a normal link below in case the browser blocks the automatic download.
  const link = document.createElement('a')
  link.href = artifactUrl(summary.value.artifact.download_url)
  link.download = summary.value.artifact.name
  document.body.appendChild(link)
  link.click()
  link.remove()
}
</script>

<template>
  <section class="summary-download" aria-label="Summary download" :aria-busy="loading">
    <div class="summary-download__heading">
      <div>
        <strong>Summary (.zip)</strong>
        <p>
          {{
            t(
              'All existing result files in one summary/ folder. No analysis is rerun.',
              '将现有结果文件打包到 summary/ 文件夹，不重新运行分析。'
            )
          }}
        </p>
      </div>
      <el-button type="success" :loading="loading" :disabled="loading" @click="downloadSummary">
        {{ t('Download Summary (.zip)', '下载 Summary (.zip)') }}
      </el-button>
    </div>
    <p v-if="error" class="summary-download__error" role="alert">{{ error }}</p>
    <p v-if="summary" aria-live="polite">
      <a :href="artifactUrl(summary.artifact.download_url)" :download="summary.artifact.name">
        {{ t('Download again', '重新下载') }}: {{ summary.artifact.name }}
      </a>
      ({{ summary.files.length }} {{ t('files', '个文件') }})
    </p>
  </section>
</template>

<style scoped>
.summary-download {
  margin-top: 20px;
  padding: 16px;
  border: 1px solid #cfe0e2;
  border-radius: 8px;
  background: #f7fafb;
  overflow-wrap: anywhere;
}

.summary-download__heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.summary-download p {
  margin: 6px 0 0;
  font-size: 13px;
  line-height: 1.5;
  color: #64777c;
}

.summary-download .summary-download__error {
  color: #b42318;
}

.summary-download a {
  color: #176b74;
}

@media (max-width: 680px) {
  .summary-download__heading {
    align-items: flex-start;
    flex-direction: column;
  }
  .summary-download__heading > .el-button {
    max-width: 100%;
    white-space: normal;
    height: auto;
    min-height: 32px;
  }
}
</style>
