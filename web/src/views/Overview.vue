<template>
  <div v-if="!project" class="stack">
    <div class="empty">
      <p style="font-size: 15px; color: var(--text)">还没有项目</p>
      <p>项目是平台的隔离单位：每个项目有自己的数据集、模型、训练记录、评估记录和启停脚本。</p>
      <button class="primary" @click="$emit('create-project')">创建第一个项目</button>
    </div>
    <section class="panel">
      <div class="panel-head"><h2>平台已经准备好了</h2></div>
      <div class="cards">
        <div class="stat" v-for="e in envCards" :key="e.name">
          <div class="label">{{ e.roleLabel }}</div>
          <div class="value" style="font-size: 16px">{{ e.name }}</div>
          <div class="hint">{{ e.cuda ? 'CUDA 可用' : '仅 CPU' }} · {{ e.torch || '—' }}</div>
        </div>
      </div>
      <div class="small faint mt">
        四个环境都在 <span class="mono">envs/prefixes/</span> 下，互相独立；<span class="mono">main_system</span>
        启动系统，其余三个分别被自动分配给标注 / 训练 / 评估作业。
      </div>
    </section>
  </div>

  <template v-else>
    <div class="cards">
      <div class="stat">
        <div class="label">数据集</div>
        <div class="value">{{ stats.datasets }}</div>
        <div class="hint">共 {{ stats.datasets_images }} 张图片</div>
      </div>
      <div class="stat">
        <div class="label">模型权重</div>
        <div class="value">{{ stats.models }}</div>
        <div class="hint">训练产物 + 上传权重</div>
      </div>
      <div class="stat">
        <div class="label">训练记录</div>
        <div class="value">{{ stats.trainings }}</div>
        <div class="hint">最近一次 {{ lastTraining ? timeAgo(lastTraining.created_at) : '—' }}</div>
      </div>
      <div class="stat">
        <div class="label">评估记录</div>
        <div class="value">{{ stats.evaluations }}</div>
        <div class="hint">最近一次 {{ lastEval ? timeAgo(lastEval.created_at) : '—' }}</div>
      </div>
      <div class="stat">
        <div class="label">运行中作业</div>
        <div class="value">{{ stats.jobs_running }}</div>
        <div class="hint">调度器 {{ project.runner?.running ? '运行中' : '未运行' }}</div>
      </div>
      <div class="stat">
        <div class="label">项目占用</div>
        <div class="value">{{ fmtBytes(stats.disk_bytes) }}</div>
        <div class="hint mono">projects/{{ project.id }}</div>
      </div>
    </div>

    <div class="grid-2">
      <section class="panel">
        <div class="panel-head"><h2>四步跑通</h2><span class="spacer" />
          <span class="panel-note">上传 → 自动标注 → 训练 → 评估</span></div>
        <div class="stack" style="gap: 8px">
          <button class="row" style="justify-content: flex-start" @click="navigate('datasets')">
            <span style="font-size: 18px">①</span><span class="grow" style="text-align: left">上传图片 / 创建数据集</span>
            <span class="faint small">多格式统一转码 →</span>
          </button>
          <button class="row" style="justify-content: flex-start" @click="navigate('label')">
            <span style="font-size: 18px">②</span><span class="grow" style="text-align: left">自动标注（detect / SAM / pose / VLM）</span>
            <span class="faint small">按 imagz 裁减 →</span>
          </button>
          <button class="row" style="justify-content: flex-start" @click="navigate('train')">
            <span style="font-size: 18px">③</span><span class="grow" style="text-align: left">选择数据集与模型开始训练</span>
            <span class="faint small">全选 / 手动选图 →</span>
          </button>
          <button class="row" style="justify-content: flex-start" @click="navigate('eval')">
            <span style="font-size: 18px">④</span><span class="grow" style="text-align: left">选数据集 + 权重做评估测试</span>
            <span class="faint small">mAP / PR / 预测图 →</span>
          </button>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>运行环境</h2><span class="spacer" />
          <button class="tiny ghost" @click="navigate('envs')">详情</button></div>
        <div class="stack" style="gap: 8px">
          <div v-for="e in envCards" :key="e.name" class="row">
            <span class="chip" :class="e.role === 'main' ? 'info' : 'muted'">{{ e.roleLabel }}</span>
            <span class="mono small">{{ e.name }}</span>
            <span class="spacer" />
            <span class="chip" :class="e.cuda ? 'ok' : 'warn'">{{ e.cuda ? 'CUDA' : 'CPU' }}</span>
            <span class="small faint">{{ e.gpu || '' }}</span>
          </div>
        </div>
        <div class="small faint mt">
          作业按类型自动选环境：标注 → label_system，训练 → train_system，评估 → test_system。
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>最近作业</h2><span class="spacer" />
        <button class="tiny ghost" @click="navigate('jobs')">全部作业</button></div>
      <div v-if="!jobs.length" class="empty">还没有作业。上传图片后从「自动标注」开始。</div>
      <div class="scroll-x" v-else>
        <table>
          <thead>
            <tr><th>作业</th><th>类型</th><th>状态</th><th>进度</th><th>环境</th><th>创建</th></tr>
          </thead>
          <tbody>
            <tr v-for="job in jobs" :key="job.id" @click="openJob(job)" style="cursor: pointer">
              <td>{{ job.title }}</td>
              <td class="mono">{{ job.kind }}</td>
              <td><JobState :state="job.state" /></td>
              <td style="min-width: 160px">
                <div class="bar"><i :style="{ width: (job.progress?.percent || 0) + '%' }" /></div>
                <span class="small faint">{{ job.progress?.message || '' }}</span>
              </td>
              <td class="mono small">{{ job.env }}</td>
              <td class="small muted">{{ timeAgo(job.created_at) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>项目结构</h2><span class="spacer" />
        <span class="panel-note">整个目录可拷走 / 备份</span></div>
      <div class="kv">
        <dt>项目目录</dt><dd class="mono">projects/{{ project.id }}/</dd>
        <dt>数据集</dt><dd class="mono">datasets/（图片 + 标签 + 数据卡）</dd>
        <dt>模型</dt><dd class="mono">models/（训练产物 / 上传权重）</dd>
        <dt>训练记录</dt><dd class="mono">trainings/（每轮指标 / 曲线 / 权重）</dd>
        <dt>评估记录</dt><dd class="mono">evaluations/（mAP / PR 曲线 / 预测图）</dd>
        <dt>作业</dt><dd class="mono">jobs/（排队 / 进度 / 日志）</dd>
        <dt>启停脚本</dt><dd class="mono">start.sh / stop.sh</dd>
      </div>
    </section>
  </template>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { currentProject, fmtBytes, navigate, state, timeAgo } from '../state'
import JobState from '../components/JobState.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const detail = ref(null)
const jobs = ref([])
const project = computed(() => {
  const base = currentProject()
  return detail.value && base ? { ...base, ...detail.value, stats: detail.value.stats || base.stats, runner: base.runner } : base
})

const stats = computed(() => project.value?.stats || { datasets: 0, datasets_images: 0, models: 0,
  trainings: 0, evaluations: 0, jobs_running: 0, disk_bytes: 0 })
const lastTraining = computed(() => (project.value?.trainings || [])[0])
const lastEval = computed(() => (project.value?.evaluations || [])[0])

const ROLE_LABEL = { main: '启动系统', label: '自动标注', train: '模型训练', test: '模型评估' }
const envCards = computed(() => (props.meta?.envs || []).map(e => ({
  name: e.name, role: e.role || (e.tags || [])[0] || '', roleLabel: ROLE_LABEL[e.role] || e.role || '环境',
  cuda: !!e.cuda, torch: e.torch, gpu: e.gpu,
})))

async function loadJobs() {
  if (!props.projectId) return
  try { jobs.value = await api.jobs(props.projectId, { limit: 8 }) } catch (e) { jobs.value = [] }
}
async function loadDetail() {
  if (!props.projectId) return
  try { detail.value = await api.project(props.projectId) } catch (e) { detail.value = null }
}
function openJob(job) { navigate(`jobs/${job.id}`) }

watch(() => props.projectId, () => { loadJobs(); loadDetail() })
watch(() => props.meta, loadDetail)
onMounted(() => { loadJobs(); loadDetail() })
</script>
