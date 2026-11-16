<template>
  <div class="stack">
    <div class="row" v-if="showHead">
      <JobState :state="state" />
      <span class="small muted">{{ progressText }}</span>
      <div class="bar grow" style="min-width: 120px"><i :style="{ width: pct + '%' }" /></div>
      <span class="small faint">{{ pct }}%</span>
      <button v-if="canCancel" class="tiny danger" @click="cancel">取消</button>
    </div>
    <div class="log" :class="{ small }" ref="box">{{ text || '（等待日志…）' }}</div>
    <div class="small faint" v-if="state === 'queued'">作业排队中，调度器会自动用对应 conda 环境执行。</div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import JobState from './JobState.vue'

const props = defineProps({
  project: { type: String, required: true },
  jobId: { type: String, required: true },
  showHead: { type: Boolean, default: true },
  small: { type: Boolean, default: false },
})
const emit = defineEmits(['update', 'done'])

const text = ref('')
const state = ref('queued')
const progress = ref({ percent: 0, message: '', done: 0, total: 0 })
const box = ref(null)
let timer = null
let offset = 0

const pct = computed(() => Math.round(Number(progress.value.percent || 0)))
const progressText = computed(() => progress.value.message || '')
const canCancel = computed(() => ['queued', 'running', 'canceling'].includes(state.value))

async function poll() {
  if (!props.jobId) return
  try {
    const res = await api.jobLog(props.project, props.jobId, offset)
    if (res.text) {
      text.value += res.text
      offset = res.next_offset
      if (box.value) {
        const nearBottom = box.value.scrollHeight - box.value.scrollTop - box.value.clientHeight < 120
        if (nearBottom) requestAnimationFrame(() => { if (box.value) box.value.scrollTop = box.value.scrollHeight })
      }
    }
    if (res.state) state.value = res.state
    if (res.progress) progress.value = res.progress
    emit('update', { state: state.value, progress: progress.value })
    if (['succeeded', 'failed', 'cancelled'].includes(state.value)) {
      emit('done', { state: state.value, progress: progress.value })
      stop()
    }
  } catch (e) {
    /* 轮询失败不弹错，下次重试 */
  }
}

function stop() { if (timer) { clearInterval(timer); timer = null } }
function start() {
  stop()
  offset = 0
  text.value = ''
  state.value = 'queued'
  progress.value = { percent: 0, message: '' }
  poll()
  timer = setInterval(poll, 1500)
}

async function cancel() {
  try { await api.cancelJob(props.project, props.jobId) } catch (e) { /* ignore */ }
  poll()
}

watch(() => props.jobId, () => { if (props.jobId) start() })
onMounted(() => { if (props.jobId) start() })
onUnmounted(stop)
</script>
