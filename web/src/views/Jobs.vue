<template>
  <div class="stack">
    <section class="panel">
      <div class="panel-head"><h2>作业</h2><span class="spacer" />
        <select v-model="filters.kind" style="width: 130px">
          <option value="">全部类型</option>
          <option value="label">自动标注</option>
          <option value="tile">裁减</option>
          <option value="relation">关系规则</option>
          <option value="train">训练</option>
          <option value="evaluate">评估</option>
        </select>
        <select v-model="filters.state" style="width: 130px">
          <option value="">全部状态</option>
          <option value="queued">排队中</option>
          <option value="running">运行中</option>
          <option value="succeeded">成功</option>
          <option value="failed">失败</option>
          <option value="cancelled">已取消</option>
        </select>
        <button class="tiny ghost" @click="load">刷新</button>
      </div>
      <div class="small faint" style="margin-bottom: 8px">
        作业由项目调度器按类型送入对应 conda 环境执行；关掉 Web 页面不会中断作业。
      </div>
      <div v-if="!jobs.length" class="empty">没有作业</div>
      <div class="scroll-x" v-else>
        <table>
          <thead><tr><th>作业</th><th>类型</th><th>状态</th><th style="width: 220px">进度</th>
            <th>环境</th><th>耗时</th><th>创建</th><th></th></tr></thead>
          <tbody>
            <tr v-for="j in jobs" :key="j.id" :class="{ selected: j.id === activeId }"
                @click="open(j)" style="cursor: pointer">
              <td>{{ j.title }}<div class="small faint mono">{{ j.id }}</div></td>
              <td class="mono small">{{ j.kind }}</td>
              <td><JobState :state="j.state" /></td>
              <td>
                <div class="bar"><i :style="{ width: (j.progress?.percent || 0) + '%' }" /></div>
                <span class="small faint">{{ j.progress?.message || '' }}</span>
              </td>
              <td class="mono small">{{ j.env }}</td>
              <td class="small muted">{{ duration(j) }}</td>
              <td class="small muted">{{ timeAgo(j.created_at) }}</td>
              <td @click.stop>
                <button class="tiny danger" v-if="['queued','running','canceling'].includes(j.state)"
                        @click="cancel(j)">取消</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel" v-if="active">
      <div class="panel-head">
        <h2>{{ active.title }}</h2>
        <span class="spacer" />
        <JobState :state="active.state" />
        <button class="tiny ghost" @click="active = null; activeId = ''">收起</button>
      </div>
      <div class="kv" style="margin-bottom: 10px">
        <dt>作业</dt><dd class="mono">{{ active.id }}</dd>
        <dt>环境</dt><dd class="mono">{{ active.env }}</dd>
        <dt>参数</dt><dd class="mono small">{{ JSON.stringify(active.spec || {}).slice(0, 400) }}</dd>
        <dt v-if="active.error">错误</dt><dd v-if="active.error" style="color: var(--err)">{{ active.error }}</dd>
      </div>
      <LogPane :project="projectId" :job-id="active.id" @done="load" />
    </section>

    <section class="panel" v-if="active?.result">
      <div class="panel-head"><h2>结果</h2></div>
      <pre class="log small" style="height: auto; max-height: 320px">{{ JSON.stringify(active.result, null, 2) }}</pre>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { fmtDuration, timeAgo, toast } from '../state'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const jobs = ref([])
const filters = ref({ kind: '', state: '' })
const active = ref(null)
const activeId = ref('')
let timer = null

const durationOf = (j) => {
  if (!j.started_at) return '—'
  const end = j.finished_at ? new Date(j.finished_at) : new Date()
  return fmtDuration((end - new Date(j.started_at)) / 1000)
}
function duration(j) { return durationOf(j) }

async function load() {
  if (!props.projectId) return
  try {
    jobs.value = await api.jobs(props.projectId, {
      limit: 100, kind: filters.kind || '', state: filters.state || '',
    })
    if (activeId.value) {
      const found = jobs.value.find(j => j.id === activeId.value)
      if (found) active.value = found
    }
  } catch (e) { /* 轮询失败忽略 */ }
}

function open(j) {
  activeId.value = j.id
  active.value = j
}

async function cancel(j) {
  if (!confirm(`取消作业「${j.title}」？`)) return
  try { await api.cancelJob(props.projectId, j.id); await load(); toast('已请求取消', 'ok') }
  catch (e) { toast('取消失败：' + e.message, 'err') }
}

watch(() => [props.projectId, filters.value.kind, filters.value.state], load)
watch(() => props.routeId, (v) => { if (v) { activeId.value = v; api.job(props.projectId, v).then(j => { active.value = j }).catch(() => {}) } })
onMounted(() => { load(); timer = setInterval(() => { if (!document.hidden) load() }, 3000) })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>
