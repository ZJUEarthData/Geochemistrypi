import { onUnmounted, ref } from 'vue'

import {
  cancelTask,
  createTaskId,
  getTaskStatus,
  type OnlineTaskStatus
} from '@/api/online'

export function useTaskTracking() {
  const taskId = ref('')
  const taskStatus = ref<OnlineTaskStatus | null>(null)
  const cancellingTask = ref(false)
  const cancelledByUser = ref(false)
  let timer: ReturnType<typeof setTimeout> | undefined
  let polling = false
  let registered = false

  function beginTask(label: string) {
    stopPolling()
    taskId.value = createTaskId()
    cancelledByUser.value = false
    registered = false
    taskStatus.value = {
      task_id: taskId.value,
      label,
      status: 'queued',
      progress: 0,
      queue_position: 1,
      submitted_at: new Date().toISOString(),
      started_at: null,
      finished_at: null,
      elapsed_seconds: 0,
      timeout_seconds: 1800,
      cancellable: true,
      message: 'Submitting calculation…',
      error: null
    }
    schedulePoll(150)
    return taskId.value
  }

  async function pollTask() {
    if (!taskId.value || polling) return
    polling = true
    try {
      taskStatus.value = await getTaskStatus(taskId.value)
      registered = true
    } catch {
      // The upload may still be reaching the backend before task registration.
    } finally {
      polling = false
    }
    if (taskStatus.value?.cancellable) schedulePoll(700)
  }

  async function cancelCurrentTask() {
    if (!taskId.value || !taskStatus.value?.cancellable) return
    cancellingTask.value = true
    try {
      taskStatus.value = await cancelTask(taskId.value)
      cancelledByUser.value = true
      schedulePoll(150)
    } finally {
      cancellingTask.value = false
    }
  }

  async function finishTask() {
    await pollTask()
    if (!registered && taskStatus.value) {
      taskStatus.value = {
        ...taskStatus.value,
        status: cancelledByUser.value ? 'cancelled' : 'failed',
        progress: 100,
        queue_position: null,
        finished_at: new Date().toISOString(),
        cancellable: false,
        message: cancelledByUser.value
          ? 'The task was cancelled.'
          : 'The request failed before the calculation entered the queue.'
      }
    }
    if (!taskStatus.value?.cancellable) stopPolling()
  }

  function schedulePoll(delay: number) {
    if (timer) clearTimeout(timer)
    timer = setTimeout(pollTask, delay)
  }

  function stopPolling() {
    if (timer) clearTimeout(timer)
    timer = undefined
  }

  onUnmounted(stopPolling)

  return {
    taskId,
    taskStatus,
    cancellingTask,
    cancelledByUser,
    beginTask,
    finishTask,
    cancelCurrentTask
  }
}
