<template>
  <div class="stack">
    <section class="panel">
      <div class="panel-head"><h2>新建训练</h2><span class="spacer" />
        <span class="panel-note">选择数据集图片 → 选模型 → 配置参数 → 训练</span></div>

      <div class="section-title">1. 选择数据（全选或手动挑选图片）</div>
      <DatasetPicker v-model="selections" :project="projectId" :datasets="compatibleDatasets"
                     :task-hint="task" />
      <div class="row mt" v-if="selectionInfo">
        <span class="chip info">将使用 {{ selectionInfo.images }} 张图片</span>
        <span class="chip muted" v-if="selectionInfo.task">{{ selectionInfo.task }}</span>
        <span class="small faint">train {{ selectionInfo.splits?.train }} / val {{ selectionInfo.splits?.val }} / test {{ selectionInfo.splits?.test }}</span>
      </div>

      <div class="divider" />
      <div class="section-title">2. 模型</div>
      <div class="grid-2">
        <ModelSelect v-model="model" :catalog="catalog" :task="task" :project-models="models" />
        <div class="stack" style="gap: 10px">
          <div class="form-grid">
            <label class="field">训练集/验证集划分
              <select v-model="splitMode">
                <option value="dataset">沿用数据集自带划分</option>
                <option value="ratio">按比例重新划分</option>
                <option value="all_train">全部作为训练集</option>
              </select>
            </label>
            <label class="field" v-if="splitMode === 'ratio'">验证集比例
              <input type="number" v-model.number="valRatio" min="0.05" max="0.5" step="0.05" />
            </label>
            <label class="field">实验名称
              <input v-model="name" placeholder="例如：baseline-640" />
            </label>
          </div>
          <div class="small faint">
            「从零训练（架构 yaml）」不需要下载任何预训练权重；「本地」标记的权重已在
            <span class="mono">weights/pretrained/</span> 里。
          </div>
        </div>
      </div>

      <div class="divider" />
      <div v-if="modelFamily" class="small muted" style="margin-bottom: 6px">
        当前模型：<b>{{ modelFamily.label }}</b>
        <span v-if="modelFamily.notes"> · {{ modelFamily.notes }}</span>
        <span v-if="modelFamily.unsupported_params?.length" class="chip warn" style="margin-left: 6px">
          不支持 {{ modelFamily.unsupported_params.join(' / ') }}
        </span>
        <span v-if="!modelFamilyIsMatched" class="chip" style="margin-left: 6px">
          项目权重：沿用 {{ modelFamily.label }} 的参数表
        </span>
      </div>
      <div class="row">
        <div class="section-title" style="margin: 0">3. 训练参数（自由配置）</div>
        <span class="panel-note" style="font-size: 12px">
          参数与说明来自当前模型目录
          <span class="mono" v-if="modelFamily">configs/models/{{ modelFamily.id }}/params.yaml</span>
          （通用参数在 <span class="mono">configs/models/_common/params.yaml</span>）。
          标注了适用任务的参数在其它任务下会自动隐藏；鼠标移到参数旁的 <b>?</b> 图标上看说明。
        </span>
        <span class="spacer" />
        <button class="tiny ghost" @click="showAll = !showAll">{{ showAll ? '收起参数' : '展开全部参数' }}</button>
        <button class="tiny ghost" @click="resetParams">恢复默认</button>
      </div>
      <div class="form-grid mt">
        <label class="field" v-for="p in basicParams" :key="p.key">
          <span class="row tight" style="gap: 4px">
            {{ p.label }} <span class="faint mono">{{ p.key }}</span>
            <HelpDot :text="paramHelp(p)" />
          </span>
          <select v-if="p.key === 'device'" v-model="params.device">
            <option value="auto">auto</option>
            <option value="0">0（第一张 GPU）</option>
            <option value="0,1">0,1</option>
            <option value="cpu">cpu</option>
          </select>
          <input v-else-if="p.type === 'int' || p.type === 'float'" type="number" v-model.number="params[p.key]"
                 :min="p.min" :max="p.max" :step="p.step || (p.type === 'int' ? 1 : 0.001)" />
          <select v-else-if="p.type === 'bool'" v-model="params[p.key]">
            <option :value="true">是</option><option :value="false">否</option>
          </select>
          <input v-else v-model="params[p.key]" />
        </label>
      </div>
      <div v-if="hiddenParams.length" class="small faint mt">
        已隐藏 {{ hiddenParams.length }} 个仅适用于其它任务的参数：
        {{ hiddenParams.map(p => `${p.key}（${(p.tasks || []).join('/')}）`).join('、') }}
      </div>
      <div v-if="showAll" class="stack mt">
        <div v-for="group in advancedGroups" :key="group.name">
          <div class="section-title">{{ group.label }}</div>
          <div class="form-grid">
            <label class="field" v-for="p in group.items" :key="p.key">
              <span class="row tight" style="gap: 4px">
                {{ p.label }} <span class="faint mono">{{ p.key }}</span>
                <HelpDot :text="paramHelp(p)" />
              </span>
              <input v-if="p.type === 'int' || p.type === 'float'" type="number" v-model.number="params[p.key]"
                     :min="p.min" :max="p.max" :step="p.step || (p.type === 'int' ? 1 : 0.001)" />
              <select v-else-if="p.type === 'bool'" v-model="params[p.key]">
                <option :value="true">是</option><option :value="false">否</option>
              </select>
              <input v-else v-model="params[p.key]" />
            </label>
          </div>
        </div>
      </div>

      <div class="row mt">
        <button class="primary" :disabled="busy || !selections.length || !model" @click="start">
          {{ busy ? '创建中…' : '开始训练' }}
        </button>
        <span class="small faint" v-if="task">
          作业将在 <span class="mono">train_system</span> 环境执行；训练记录、曲线和权重会自动归档到本项目。
        </span>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>训练记录</h2><span class="spacer" />
        <span class="panel-note">{{ trainings.length }} 条</span></div>
      <div v-if="!trainings.length" class="empty">还没有训练记录</div>
      <div class="scroll-x" v-else>
        <table>
          <thead>
            <tr><th>实验</th><th>状态</th><th class="num">mAP50-95</th><th class="num">mAP50</th>
              <th class="num">P</th><th class="num">R</th><th>模型</th><th>轮数</th><th>创建</th><th></th></tr>
          </thead>
          <tbody>
            <tr v-for="t in trainings" :key="t.id" @click="navigate(`train/${t.id}`)" style="cursor: pointer">
              <td>{{ t.name }}<div class="small faint mono">{{ t.id }}</div></td>
              <td><JobState :state="t.state" /></td>
              <td class="num">{{ fmtNum(t.metrics?.map) }}</td>
              <td class="num">{{ fmtNum(t.metrics?.map50) }}</td>
              <td class="num">{{ fmtNum(t.metrics?.precision) }}</td>
              <td class="num">{{ fmtNum(t.metrics?.recall) }}</td>
              <td class="mono small">{{ t.model?.ref }}</td>
              <td class="num">{{ t.metrics?.epochs_run || t.params?.epochs || '—' }}</td>
              <td class="small muted">{{ timeAgo(t.created_at) }}</td>
              <td><button class="tiny ghost" @click.stop="toggleJob(t)">{{ t.job ? '日志' : '' }}</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <Modal v-if="logFor" :title="`训练日志 · ${logFor.name}`" wide @close="logFor = null">
      <LogPane :project="projectId" :job-id="logFor.job" />
    </Modal>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import { fmtNum, loadCatalog, navigate, timeAgo, toast } from '../state'
import DatasetPicker from '../components/DatasetPicker.vue'
import HelpDot from '../components/HelpDot.vue'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'
import Modal from '../components/Modal.vue'
import ModelSelect from '../components/ModelSelect.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })

const datasets = ref([])
const models = ref([])
const trainings = ref([])
const catalog = ref(null)
const selections = ref([])
const model = ref('')
const name = ref('')
const splitMode = ref('dataset')
const valRatio = ref(0.2)
const busy = ref(false)
const showAll = ref(false)
const logFor = ref(null)
const params = reactive({})
const schemaKeys = ref(new Set())

const compatibleDatasets = computed(() => {
  const current = task.value
  return datasets.value.filter(d => !current || d.task === current)
})
const task = computed(() => {
  const first = selections.value[0]
  if (!first) return ''
  const ds = datasets.value.find(d => d.id === first.dataset)
  return ds?.task || ''
})
const selectionInfo = computed(() => {
  if (!selections.value.length) return null
  let images = 0
  const splits = { train: 0, val: 0, test: 0 }
  for (const sel of selections.value) {
    const ds = datasets.value.find(d => d.id === sel.dataset)
    if (!ds) continue
    const count = sel.images === 'all' ? ds.stats.images : sel.images.length
    images += count
    if (sel.images === 'all') {
      splits.train += ds.stats.splits.train
      splits.val += ds.stats.splits.val
      splits.test += ds.stats.splits.test
    }
  }
  return { images, splits, task: task.value }
})
const families = computed(() => catalog.value?.families || [])
const modelFamily = computed(() => {
  const ref = model.value
  // 只考虑"当前任务可用"的模型（一个模型 = 架构族 + 检测头）
  const list = families.value.filter(f => f.role !== 'segmenter'
    && (!task.value || (f.tasks || []).includes(task.value)))
  const fallback = list.find(f => f.family === 'yolo11') || list[0] || null
  if (!ref) return fallback
  const match = list.find(f => Object.values(f.weights || {}).some(
    mapping => Object.values(mapping || {}).some(w => w.name === ref)))
  return match || fallback
})
const modelFamilyIsMatched = computed(() => {
  const ref = model.value
  return !!families.value.find(f => Object.values(f.weights || {}).some(
    mapping => Object.values(mapping || {}).some(w => w.name === ref)))
})
const familySchema = computed(() => modelFamily.value?.params || catalog.value?.training_params || [])
const visibleParams = computed(() => familySchema.value.filter(
  p => !p.tasks || !task.value || p.tasks.includes(task.value)))
const hiddenParams = computed(() => familySchema.value.filter(
  p => p.tasks && task.value && !p.tasks.includes(task.value)))
const basicParams = computed(() => visibleParams.value.filter(p => !p.group).slice(0, 12))
const advancedGroups = computed(() => {
  const groups = {}
  for (const p of visibleParams.value.filter(p => !p.group)) {
    if (basicParams.value.includes(p)) continue
    ;(groups['更多'] = groups['更多'] || []).push(p)
  }
  for (const p of visibleParams.value.filter(p => p.group)) {
    ;(groups[p.group] = groups[p.group] || []).push(p)
  }
  const labels = { '更多': '更多训练参数', augment: '数据增强' }
  return Object.entries(groups).map(([name, items]) => ({ name, label: labels[name] || name, items }))
})

function paramHelp(p) {
  const parts = []
  if (p.help) parts.push(p.help)
  const meta = []
  if (p.default !== undefined && p.default !== null) meta.push(`默认 ${p.default}`)
  if (p.min !== undefined || p.max !== undefined) {
    meta.push(`范围 ${p.min ?? '−∞'} ~ ${p.max ?? '+∞'}`)
  }
  if (p.tasks && p.tasks.length) meta.push(`仅 ${p.tasks.join(' / ')} 任务`)
  if (meta.length) parts.push(meta.join(' · '))
  return parts.join('\n')
}

function defaultOf(p) {
  if (p.type === 'bool') return !!p.default
  if (p.type === 'int') return Number(p.default)
  if (p.type === 'float') return Number(p.default)
  return String(p.default ?? '')
}

function resetParams() {
  for (const p of familySchema.value) params[p.key] = defaultOf(p)
  schemaKeys.value = new Set(familySchema.value.map(p => p.key))
}

// 参数表随模型切换（每个模型一套默认值）：切模型就把参数重置为它自己的默认值
watch(familySchema, (next) => {
  const nextKeys = new Set((next || []).map(p => p.key))
  for (const p of next || []) params[p.key] = defaultOf(p)
  for (const key of Object.keys(params)) if (!nextKeys.has(key)) delete params[key]
  schemaKeys.value = nextKeys
}, { immediate: true })

async function load() {
  if (!props.projectId) return
  datasets.value = await api.datasets(props.projectId)
  models.value = await api.models(props.projectId)
  trainings.value = await api.trainings(props.projectId)
  catalog.value = await loadCatalog()
  if (!model.value) {
    const first = selections.value[0]?.dataset && datasets.value.find(d => d.id === selections.value[0].dataset)
    const t = first?.task || 'detect'
    const family = (catalog.value?.families || []).find(f => f.family === 'yolo11' && f.head === t)
      || (catalog.value?.families || []).find(f => f.head === t)
    model.value = family?.weights?.[t]?.n?.name || 'yolo11n.pt'
  }
}

async function start() {
  if (!selections.value.length) { toast('先选择数据集', 'warn'); return }
  if (!model.value) { toast('先选择模型', 'warn'); return }
  busy.value = true
  try {
    const payload = { ...params }
    const res = await api.startTraining(props.projectId, {
      datasets: selections.value,
      model: model.value,
      params: payload,
      name: name.value.trim() || undefined,
      split_mode: splitMode.value,
      val_ratio: splitMode.value === 'ratio' ? valRatio.value : undefined,
    })
    toast(`训练作业已创建：${res.job.id}`, 'ok')
    trainings.value = await api.trainings(props.projectId)
    navigate(`train/${res.training.id}`)
  } catch (e) {
    toast('创建失败：' + e.message, 'err')
  } finally { busy.value = false }
}

function toggleJob(t) { logFor.value = t }

// 切数据集/任务后，如果当前模型不属于新任务的检测头，自动换成该任务的默认模型
watch(task, (newTask) => {
  if (!newTask) return
  const supported = families.value.filter(f => f.role !== 'segmenter' && (f.tasks || []).includes(newTask))
  if (!supported.length) return
  const stillValid = supported.some(f => Object.values(f.weights || {}).some(
    mapping => Object.values(mapping || {}).some(w => w.name === model.value)))
  if (stillValid) return
  const family = supported.find(f => f.family === 'yolo11') || supported[0]
  const name = (family.weights || {})[newTask]?.n?.name
  if (name) model.value = name
})

watch(() => props.projectId, load)
onMounted(load)
</script>
