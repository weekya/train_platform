<template>
  <div v-if="state.ready" class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">ML</div>
        <div>
          <div class="brand-name">mlplatform</div>
          <div class="brand-sub">标注 · 训练 · 评估</div>
        </div>
      </div>

      <div class="stack" style="gap: 6px">
        <select :value="state.projectId" @change="onProjectChange($event.target.value)"
                v-if="state.projects.length" style="font-size: 13px">
          <option v-for="p in state.projects" :key="p.id" :value="p.id">{{ p.name }}（{{ p.id }}）</option>
        </select>
        <button class="tiny primary" @click="showCreate = true">＋ 新建项目</button>
      </div>

      <nav class="nav">
        <div v-for="item in navItems" :key="item.name" class="nav-item"
             :class="{ active: state.route.name === item.name }" @click="navigate(item.name)">
          <span class="nav-icon">{{ item.icon }}</span>{{ item.label }}
          <span v-if="item.name === 'jobs' && runningJobs" class="chip info right">{{ runningJobs }}</span>
        </div>
      </nav>

      <div class="sidebar-foot">
        v{{ state.meta?.version }} · {{ state.projects.length }} 个项目<br />
        <span v-if="state.meta?.vlm?.configured" :title="state.meta.vlm.base_url">
          线上 VLM: {{ state.meta.vlm.provider }} / {{ state.meta.vlm.model }}
        </span>
        <span v-else>线上 VLM 未配置（自动标注 → VLM）</span><br />
        <a href="/api/docs" target="_blank">接口文档</a>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <h1>{{ pageTitle }}</h1>
        <span class="sub" v-if="project">{{ project.name }} · {{ project.task }}</span>
        <span class="spacer"></span>
        <template v-if="project">
          <span class="chip" :class="runnerRunning ? 'ok' : 'warn'">
            调度器 {{ runnerRunning ? '运行中' : '未运行' }}
          </span>
          <button class="tiny" v-if="!runnerRunning" :disabled="busy" @click="startRunner">启动调度器</button>
          <button class="tiny" v-else :disabled="busy" @click="stopRunner">停止调度器</button>
        </template>
        <button class="tiny ghost" :disabled="state.loading" @click="refresh">刷新</button>
      </header>

      <main class="content">
        <div class="content-inner">
          <component v-if="state.route.name === 'labelvlm'" :is="Labeling" :key="viewKey"
                     :project-id="state.projectId" :route-id="state.route.id" :meta="state.meta"
                     preset-model="vlm-remote" @refresh="refresh" @create-project="showCreate = true" />
          <component v-else :is="viewComponent" :key="viewKey" :project-id="state.projectId"
                     :route-id="state.route.id" :meta="state.meta"
                     @refresh="refresh" @create-project="showCreate = true" />
        </div>
      </main>
    </div>

    <Modal v-if="showCreate" title="新建项目" @close="showCreate = false">
      <div class="stack">
        <div class="form-grid">
          <label class="field">项目名称
            <input v-model="form.name" placeholder="例如：安全帽检测" />
          </label>
          <label class="field">项目 id（可留空）
            <input v-model="form.id" placeholder="例如：helmet" />
          </label>
          <label class="field">任务类型
            <select v-model="form.task">
              <option v-for="t in TASKS" :key="t.id" :value="t.id">{{ t.label }}（{{ t.id }}）</option>
            </select>
          </label>
          <label class="field">类别（逗号分隔）
            <input v-model="form.classes" placeholder="person, helmet, vest" />
          </label>
        </div>
        <label class="field">描述
          <textarea v-model="form.description" placeholder="这个项目要解决什么问题、数据来自哪里"></textarea>
        </label>
        <div class="small faint">
          项目目录会创建在 <span class="mono">projects/&lt;id&gt;/</span>，里面自带独立的数据集、模型、
          训练记录、评估记录和 start.sh / stop.sh。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="showCreate = false">取消</button>
        <button class="primary" :disabled="busy || !form.name.trim()" @click="createProject">创建项目</button>
      </template>
    </Modal>

    <div class="toasts">
      <div v-for="t in state.toasts" :key="t.id" class="toast" :class="t.kind">{{ t.message }}</div>
    </div>
  </div>
  <div v-else class="loading">{{ state.error || '正在连接平台…' }}</div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { api } from './api'
import {
  JOB_STATES, TASKS, currentProject, navigate, refreshMeta, selectProject, state, syncRoute, toast,
} from './state'
import Modal from './components/Modal.vue'
import Overview from './views/Overview.vue'
import Datasets from './views/Datasets.vue'
import DatasetDetail from './views/DatasetDetail.vue'
import Labeling from './views/Labeling.vue'
import Training from './views/Training.vue'
import TrainingDetail from './views/TrainingDetail.vue'
import Evaluation from './views/Evaluation.vue'
import EvaluationDetail from './views/EvaluationDetail.vue'
import Environments from './views/Environments.vue'
import Jobs from './views/Jobs.vue'
import Models from './views/Models.vue'

const VIEWS = {
  datasets: { list: Datasets, detail: DatasetDetail },
  train: { list: Training, detail: TrainingDetail },
  eval: { list: Evaluation, detail: EvaluationDetail },
  models: { list: Models },
  label: { list: Labeling },
  labelvlm: { list: Labeling },
  overview: { list: Overview },
  envs: { list: Environments },
  jobs: { list: Jobs },
}

const navItems = [
  { name: 'overview', label: '项目概览', icon: '◉' },
  { name: 'datasets', label: '数据集', icon: '▦' },
  { name: 'label', label: '自动标注', icon: '✎' },
  { name: 'labelvlm', label: 'VLM 标注', icon: '✧' },
  { name: 'train', label: '模型训练', icon: '⚙' },
  { name: 'models', label: '模型', icon: '◈' },
  { name: 'eval', label: '评估测试', icon: '◎' },
  { name: 'envs', label: '运行环境', icon: '⛭' },
  { name: 'jobs', label: '作业日志', icon: '≡' },
]

const showCreate = ref(false)
const busy = ref(false)
const form = reactive({ name: '', id: '', task: 'detect', classes: '', description: '' })

const project = computed(() => currentProject())
const runnerRunning = computed(() => !!project.value?.runner?.running)
const runningJobs = computed(() => Number(project.value?.stats?.jobs_running || 0))

const pageTitle = computed(() => {
  const item = navItems.find(n => n.name === state.route.name)
  return item ? item.label : '项目概览'
})

const viewComponent = computed(() => {
  const group = VIEWS[state.route.name] || VIEWS.overview
  return group.detail && state.route.id ? group.detail : group.list
})
const viewKey = computed(() => `${state.route.name}:${state.projectId}:${state.route.id}`)

function onProjectChange(pid) {
  selectProject(pid)
  navigate('overview')
}

async function refresh() {
  try {
    await refreshMeta()
    state.catalog = null
  } catch (e) {
    toast('刷新失败：' + e.message, 'err')
  }
}

async function createProject() {
  busy.value = true
  try {
    const classes = form.classes.split(',').map(s => s.trim()).filter(Boolean)
    const body = {
      name: form.name.trim(), task: form.task, classes,
      description: form.description.trim(),
    }
    if (form.id.trim()) body.id = form.id.trim()
    const res = await api.createProject(body)
    selectProject(res.id)
    showCreate.value = false
    form.name = ''
    form.id = ''
    form.classes = ''
    form.description = ''
    await refreshMeta()
    navigate('overview')
    toast(`项目已创建：${res.id}`, 'ok')
  } catch (e) {
    toast('创建失败：' + e.message, 'err')
  } finally {
    busy.value = false
  }
}

async function startRunner() {
  busy.value = true
  try {
    await api.runnerStart(state.projectId)
    await refreshMeta()
    toast('调度器已启动', 'ok')
  } catch (e) {
    toast('启动失败：' + e.message, 'err')
  } finally {
    busy.value = false
  }
}

async function stopRunner() {
  if (!confirm('停止调度器？默认会同时取消正在运行的作业。')) return
  busy.value = true
  try {
    await api.runnerStop(state.projectId, { cancel_jobs: true })
    await refreshMeta()
    toast('调度器已停止', 'ok')
  } catch (e) {
    toast('停止失败：' + e.message, 'err')
  } finally {
    busy.value = false
  }
}

let timer = null
onMounted(async () => {
  syncRoute()
  window.addEventListener('hashchange', syncRoute)
  try {
    await refresh()
  } catch (e) {
    state.error = '连接平台失败：' + e.message
  }
  timer = setInterval(() => {
    if (!document.hidden) refresh().catch(() => {})
  }, 8000)
})
onUnmounted(() => {
  window.removeEventListener('hashchange', syncRoute)
  if (timer) clearInterval(timer)
})
</script>
