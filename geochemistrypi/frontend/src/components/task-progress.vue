<script setup lang="ts">
import type { OnlineTaskStatus } from '@/api/online'
import { t } from '@/i18n'

defineProps<{
  task: OnlineTaskStatus
  cancelling?: boolean
}>()

defineEmits<{ cancel: [] }>()
</script>

<template>
  <section class="task-progress" aria-live="polite">
    <div class="task-progress__heading">
      <div>
        <strong>{{
          task.status === 'queued'
            ? t('Waiting in queue', '正在排队')
            : task.status === 'cancelling'
              ? t('Cancelling', '正在取消')
              : t('Calculation progress', '计算进度')
        }}</strong>
        <span v-if="task.queue_position">
          {{ t('Queue position', '排队位置') }}: {{ task.queue_position }}
        </span>
        <span v-else>{{ Math.round(task.elapsed_seconds) }} s</span>
      </div>
      <el-button
        v-if="task.cancellable"
        type="danger"
        plain
        :loading="cancelling"
        @click="$emit('cancel')"
      >
        {{ t('Cancel task', '取消任务') }}
      </el-button>
    </div>
    <el-progress
      :percentage="task.progress"
      :status="task.status === 'completed' ? 'success' : undefined"
      :stroke-width="12"
      :indeterminate="task.status === 'running' || task.status === 'cancelling'"
      :duration="3"
      striped
      :striped-flow="task.status === 'running'"
    />
    <p>
      {{
        task.status === 'queued'
          ? t('Your task will start automatically when earlier tasks finish.', '前面的任务完成后，本任务将自动开始。')
          : task.status === 'running'
            ? t('The calculation is running. You may cancel it at any time.', '任务正在运行，可随时取消。')
            : task.status === 'cancelling'
              ? t('Stopping the calculation process…', '正在停止计算进程…')
              : task.status === 'cancelled'
                ? t('The task was cancelled.', '任务已取消。')
                : task.status === 'timed_out'
                  ? t('The 30-minute runtime limit was reached.', '任务已达到 30 分钟运行时限。')
                  : task.message
      }}
    </p>
  </section>
</template>

<style scoped>
.task-progress {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid #cfe0e2;
  border-radius: 8px;
  background: #f7fafb;
}

.task-progress__heading,
.task-progress__heading > div {
  display: flex;
  align-items: center;
  gap: 12px;
}

.task-progress__heading {
  justify-content: space-between;
  margin-bottom: 14px;
}

.task-progress__heading span,
.task-progress p {
  color: #64777c;
  font-size: 13px;
}

.task-progress p {
  margin: 10px 0 0;
}

@media (max-width: 560px) {
  .task-progress__heading,
  .task-progress__heading > div {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
