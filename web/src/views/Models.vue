<template>
  <div class="stack">
    <section class="panel">
      <div class="panel-head"><h2>模型</h2><span class="spacer" />
        <span class="panel-note">{{ models.length }} 个项目模型 + {{ trainableCount }} 个可训练架构</span>
        <button class="tiny ghost" @click="load">刷新</button>
      </div>
      <div class="small faint" style="margin-bottom: 8px">
        训练产物和上传的权重都在项目里（<span class="mono">models/&lt;模型 id&gt;/</span>），
        可下载、改名、设为默认（标注/训练时的默认权重），也可以直接上传现成权重。
      </div>

      <div class="row" style="margin-bottom: 10px">
        <input type="file" ref="fileInput" style="display: none" @change="onPick" />
        <button class="tiny primary" :disabled="uploading" @click="fileInput?.click()">
          {{ uploading ? '上传中…' : '上传权重（.pt/.onnx/.engine）' }}
        </button>
        <select v-model="upload.task" style="width: 130px" aria-label="上传任务">
          <option value="detect">detect</option>
          <option value="segment">segment</option>
          <option value="pose">pose</option>
          <option value="classify">classify</option>
        </select>
        <input v-model="upload.classes" placeholder="类别（逗号分隔，可空）" style="flex: 1" />
      </div>

      <div v-if="!models.length" class="empty">
        还没有模型。训练一次（「模型训练」页）或上传一个权重，就会出现在这里。
      </div>
      <div class="scroll-x" v-else>
        <table>
          <thead><tr>
            <th>模型</th><th>任务</th><th>实现</th><th>类别</th><th>指标</th>
            <th>大小</th><th>来源</th><th>时间</th><th style="width: 200px"></th>
          </tr></thead>
          <tbody>
            <tr v-for="m in models" :key="m.id">
              <td>
                <b>{{ m.name }}</b>
                <div class="small faint mono">{{ m.id }}</div>
              </td>
              <td class="mono small">{{ m.task }}</td>
              <td class="mono small">
                <span v-if="m.family">{{ m.family }}</span>
                <span v-else class="faint">（未登记实现）</span>
              </td>
              <td class="small">{{ (m.classes || []).join(', ') || '—' }}</td>
              <td class="small mono">
                <span v-if="m.metrics && m.metrics.map != null">
                  mAP50-95 {{ fmtNum(m.metrics.map) }}<br />
                  <span class="faint">mAP50 {{ fmtNum(m.metrics.map50) }}</span>
                </span>
                <span v-else class="faint">—</span>
              </td>
              <td class="small muted">{{ size(m.size) }}</td>
              <td class="small muted">
                <span v-if="m.source?.kind === 'training'">训练 {{ short(m.source.training) }}</span>
                <span v-else>{{ m.kind || 'uploaded' }}</span>
              </td>
              <td class="small muted">{{ timeAgo(m.created_at) }}</td>
              <td>
                <div class="row tight">
                  <a class="tiny" :href="downloadUrl(m)" download>下载</a>
                  <button class="tiny ghost" @click="rename(m)">改名</button>
                  <button class="tiny ghost" :disabled="isDefault(m)" @click="setDefault(m)">
                    {{ isDefault(m) ? '默认' : '设为默认' }}
                  </button>
                  <button class="tiny danger" @click="remove(m)">删除</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>可训练架构</h2><span class="spacer" />
        <span class="panel-note">来自 configs/models/ + configs/models/_impls.yaml</span></div>
      <div class="small faint" style="margin-bottom: 8px">
        这里的"架构"是训练入口（预训练权重 / 从零训练的 yaml），训练产物会出现在上面的模型列表里。
      </div>
      <table>
        <thead><tr><th>架构</th><th>任务</th><th>实现</th><th>能力</th><th>权重</th><th>本地</th></tr></thead>
        <tbody>
          <tr v-for="f in families" :key="f.id">
            <td><b>{{ f.label }}</b><div class="small faint mono">{{ f.id }}</div></td>
            <td class="mono small">{{ (f.tasks || []).join(' / ') }}</td>
            <td class="mono small">{{ f.backend || '—' }}<div class="faint small">{{ f.framework }}</div></td>
            <td class="small">{{ (f.capabilities || []).join(' / ') }}</td>
            <td class="small mono">{{ weightNames(f) }}</td>
            <td class="small">{{ localCount(f) }} / {{ weightCount(f) }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api, urls } from '../api'
import { currentProject, loadCatalog, timeAgo, toast } from '../state'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const emit = defineEmits(['refresh'])

const models = ref([])
const catalog = ref(null)
const uploading = ref(false)
const fileInput = ref(null)
const upload = ref({ task: 'detect', classes: '' })

const families = computed(() => (catalog.value?.families || []).filter(f => (f.capabilities || []).includes('train')))
const trainableCount = computed(() => families.value.length)
const project = computed(() => currentProject())
const defaultModel = computed(() => (project.value?.defaults || {}).model || '')

function fmtNum(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(4) : '—'
}
function short(id) {
  return String(id || '').replace(/^job-/, '').slice(0, 13)
}
function size(bytes) {
  const n = Number(bytes || 0)
  if (n > 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  if (n > 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${n} B`
}
function downloadUrl(m) {
  return urls.modelFile(props.projectId, m.id)
}
function isDefault(m) {
  return defaultModel.value === m.id
}
function weightCount(f) {
  let n = 0
  for (const task of Object.keys(f.weights || {})) n += Object.keys(f.weights[task] || {}).length
  return n
}
function localCount(f) {
  let n = 0
  for (const task of Object.keys(f.weights || {})) {
    for (const scale of Object.keys(f.weights[task] || {})) {
      if (f.weights[task][scale]?.local) n += 1
    }
  }
  return n
}
function weightNames(f) {
  const names = []
  for (const task of Object.keys(f.weights || {})) {
    for (const scale of Object.keys(f.weights[task] || {})) {
      const w = f.weights[task][scale] || {}
      names.push(`${scale}:${w.name}${w.local ? '（本地）' : ''}`)
    }
  }
  return names.join(' · ')
}

async function load() {
  if (!props.projectId) return
  try {
    models.value = await api.models(props.projectId)
    catalog.value = await loadCatalog()
  } catch (e) {
    toast('加载失败：' + e.message, 'err')
  }
}

async function onPick(event) {
  const file = (event.target.files || [])[0]
  if (!file) return
  uploading.value = true
  try {
    const entry = await api.uploadModel(props.projectId, file, {
      name: file.name.replace(/\.[^.]+$/, ''),
      task: upload.value.task,
      classes: upload.value.classes,
    })
    toast(`已上传：${entry.name}`, 'ok')
    await load()
  } catch (e) {
    toast('上传失败：' + e.message, 'err')
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

async function rename(m) {
  const name = prompt('模型显示名', m.name)
  if (!name || name === m.name) return
  try {
    await api.renameModel(props.projectId, m.id, { name })
    toast('已改名', 'ok')
    await load()
  } catch (e) {
    toast('改名失败：' + e.message, 'err')
  }
}

async function setDefault(m) {
  const current = (project.value?.defaults || {})
  try {
    await api.updateProject(props.projectId, { defaults: { ...current, model: m.id } })
    toast(`默认权重已设为 ${m.name}`, 'ok')
    // 让 App 重新拉项目元数据（默认值存在 project.yaml 里）
    emit('refresh')
    await load()
  } catch (e) {
    toast('设置失败：' + e.message, 'err')
  }
}

async function remove(m) {
  if (!confirm(`删除模型「${m.name}」？权重文件会一起删掉（训练记录保留）。`)) return
  try {
    await api.deleteModel(props.projectId, m.id)
    toast('已删除', 'ok')
    await load()
  } catch (e) {
    toast('删除失败：' + e.message, 'err')
  }
}

watch(() => props.projectId, load)
onMounted(load)
</script>
