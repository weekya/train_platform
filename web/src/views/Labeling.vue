<template>
  <div class="stack">
    <div class="grid-2">
      <section class="panel">
        <div class="panel-head"><h2>自动标注</h2><span class="spacer" />
          <span class="panel-note">用模型给选中的图片打标签</span></div>

        <div class="form-grid">
          <label class="field">数据集
            <select v-model="form.dataset" aria-label="数据集" @change="onDataset">
              <option value="">选择数据集…</option>
              <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
                {{ ds.name }}（{{ ds.stats.images }} 张 / 已标注 {{ ds.stats.labeled }}）
              </option>
            </select>
          </label>
          <label class="field">模型（标注器）
            <select v-model="form.model" aria-label="标注模型">
              <option v-for="l in labelers" :key="l.id" :value="l.id">
                {{ l.label }}{{ l.virtual ? '（线上）' : '' }}
              </option>
            </select>
            <span class="hint">
              能力：{{ (modelInfo.capabilities || []).join(' / ') || '—' }}；
              实现 <span class="mono">{{ form.model || '（未选择）' }}</span>
            </span>
          </label>
          <label class="field">推理分辨率 imagz
            <input type="number" v-model.number="form.imgsz" min="64" step="32" />
          </label>
          <label class="field">置信度阈值
            <input type="number" v-model.number="form.conf" min="0" max="1" step="0.01" />
          </label>
          <label class="field">NMS IoU
            <input type="number" v-model.number="form.iou" min="0" max="1" step="0.01" />
          </label>
          <label class="field">设备
            <input v-model="form.device" placeholder="auto / 0 / cpu" />
          </label>
        </div>

        <div class="divider" />
        <div class="form-grid">
          <label class="field">本次标注类别（必填，逗号分隔）
            <input v-model="form.classes" aria-label="标注类别" :disabled="autoClasses"
                   :placeholder="classPlaceholder" />
            <span class="hint">
              当前数据集类别：{{ (selectedDataset?.classes || []).join(', ') || '（空）' }}；
              填写后本次标注/提示词只用这些类别，提交时会把数据集类别改成这里填的
            </span>
          </label>
          <label class="field">模型类别不在数据集里时
            <select v-model="form.unknown" aria-label="模型类别不在数据集里时" :disabled="autoClasses">
              <option value="skip">跳过</option>
              <option value="add">自动新增类别</option>
            </select>
          </label>
          <label class="field">批大小
            <input type="number" v-model.number="form.batch" min="1" step="1" />
          </label>
          <div class="field">
            <span>类别快捷操作</span>
            <div class="row tight">
              <button class="tiny" :disabled="busy || autoClasses" @click="fillDatasetClasses">沿用数据集类别</button>
              <button class="tiny ghost" @click="gotoDatasetDetail">类别说明</button>
            </div>
            <label v-if="supportsAutoClasses" class="check" style="margin-top: 4px">
              <input type="checkbox" v-model="autoClasses" /> 自动发现类别（不填类别，模型返回什么就加什么）
            </label>
          </div>
        </div>

        <template v-if="usesLocalWeights">
          <div class="divider" />
          <div class="small muted" style="margin-bottom: 6px">
            权重（本地已有权重会标「本地」，缺失的会在作业里下载）
            <span v-if="isSam">· 这里的权重是给 SAM 提供框和类别的检测器</span>
          </div>
          <ModelSelect v-model="form.weights" :catalog="catalog" :task="modelTask"
                       :project-models="models" />
        </template>

        <template v-if="isSam">
          <div class="divider" />
          <div class="form-grid">
            <label class="field">SAM 权重
              <select v-model="form.samWeights">
                <option value="FastSAM-s.pt">FastSAM-s.pt（本地，全景分割）</option>
                <option value="sam2.1_t.pt">sam2.1_t.pt（需下载）</option>
                <option value="sam2_t.pt">sam2_t.pt（需下载）</option>
                <option value="sam_b.pt">sam_b.pt（需下载）</option>
              </select>
            </label>
            <label class="field">SAM 推理尺寸
              <input type="number" v-model.number="form.samImgsz" min="256" step="64" />
            </label>
          </div>
          <label class="check" style="margin-top: 8px">
            <input type="checkbox" v-model="form.everything" /> 全景分割（不需要检测框，所有实例归类别 0）
          </label>
        </template>

        <template v-if="isRemoteVlm">
          <div class="divider" />
          <div class="row" style="margin-bottom: 8px">
            <span class="small muted">线上接口（模型在服务商侧推理，不在本机跑）</span>
            <span class="spacer" />
            <span v-if="vlmKeyReady" class="chip ok">Key 就绪 · {{ vlmKeySource }}</span>
            <span v-else class="chip warn">未配置 Key</span>
            <button class="tiny" :disabled="vlmTesting" @click="testVlm">
              {{ vlmTesting ? '测试中…' : '测试连接' }}
            </button>
          </div>
          <div class="form-grid">
            <label class="field">VLM 模型（一个模型一个目录）
              <select v-model="form.vlmModelId" aria-label="VLM 模型" @change="applyVlmModel">
                <option v-for="m in vlmModels" :key="m.id" :value="m.id">
                  {{ m.id }} · {{ m.label }}{{ m.vision ? '' : '（仅文本，不能标注）' }}
                </option>
              </select>
              <span class="hint mono">{{ selectedVlmModel?.params_dir || '' }}</span>
            </label>
            <label class="field">接口地址（OpenAI 兼容）
              <input v-model="form.vlmUrl" :placeholder="vlmHint" />
            </label>
            <label class="field">模型名
              <input v-model="form.vlmModel" placeholder="服务商文档里的视觉模型名" />
              <span v-if="vlmNotes" class="hint">{{ vlmNotes }}</span>
            </label>
            <label class="field">API Key（可空，推荐用环境变量）
              <input v-model="form.vlmKey" type="password"
                     :placeholder="vlmState.has_api_key ? '已从环境变量读取，留空即沿用' : '在此填写或 export 环境变量'" />
              <span class="hint">环境变量：{{ keyEnvHint }}</span>
            </label>
            <label class="field">并发数
              <input type="number" v-model.number="form.vlmConcurrency" min="1" max="16" />
            </label>
            <label class="field">每分钟请求上限（0=不限）
              <input type="number" v-model.number="form.vlmRate" min="0" step="10" />
            </label>
          </div>
          <div class="small faint mt">
            提示词里的 <span class="mono">{classes}</span> 本次为：
            <b>{{ promptClasses }}</b>
            <span class="faint">（每次自动标注都必须填写；提交时数据集类别会被本次填写的接管）</span>
          </div>
          <div v-if="classMetaLines.length" class="small faint mt">
            类别说明（数据集详情 → 类别说明）：
            <div v-for="line in classMetaLines" :key="line.name" class="mono">
              {{ line.name }}：{{ line.description }}
            </div>
          </div>
          <div v-if="selectedVlmModel && !selectedVlmModel.vision" class="small mt" style="color: var(--warn)">
            ⚠ 这个模型不接受图片输入（vision=false），不能用于自动标注。
          </div>

          <div class="divider" />
          <div class="row" style="margin-bottom: 6px">
            <b style="font-size: 13px">提示词（{{ taskLabel }} 任务）</b>
            <span class="small faint" v-if="vlmPromptSource">· {{ vlmPromptSource }}</span>
            <span class="spacer" />
            <span v-if="vlmPromptDirty" class="chip warn">已修改，仅本次作业生效</span>
            <button class="tiny ghost" @click="resetVlmPrompt">恢复默认提示词</button>
          </div>
          <textarea v-model="form.vlmPrompt" rows="9" @input="vlmPromptDirty = true"
                    placeholder="留空使用模型目录里的默认提示词"></textarea>
          <div class="small faint">
            占位符：<span class="mono">{width}</span> <span class="mono">{height}</span>
            <span class="mono">{classes}</span> <span class="mono">{task}</span>
            <span class="mono">{kpt_count}</span> <span class="mono">{kpt_names}</span>；
            标注任务由数据集决定，模型输出会按该任务过滤（detect 只留框、segment 要多边形、pose 要关键点）。
          </div>
          <div v-if="vlmTest" class="small mt" :style="{ color: vlmTest.ok ? 'var(--ok)' : 'var(--err)' }">
            <template v-if="vlmTest.ok">
              ✔ 连接成功：{{ vlmTest.model }}，延迟 {{ vlmTest.latency_ms }}ms
              <span v-if="vlmTest.usage && vlmTest.usage.total_tokens">· {{ vlmTest.usage.total_tokens }} tokens</span>
            </template>
            <template v-else>✘ 连接失败：{{ vlmTest.error }}</template>
          </div>
        </template>

        <div class="row mt">
          <label class="check"><input type="checkbox" v-model="form.overwrite" /> 覆盖已有标注
            <span v-if="isRemoteVlm" class="faint">（会重新调用线上模型，按 token 计费）</span>
          </label>
          <label class="check"><input type="checkbox" v-model="form.refine" /> 用 SAM 细化轮廓</label>
        </div>

        <div class="divider" />
        <div class="row">
          <b style="font-size: 13px">标注后按 imagz 裁减生成新数据集</b>
          <label class="check right"><input type="checkbox" v-model="form.tile" /> 启用</label>
        </div>
        <div v-if="form.tile" class="form-grid mt">
          <label class="field">裁减 imagz
            <input type="number" v-model.number="form.tileImgsz" min="64" step="32" />
          </label>
          <label class="field">重叠
            <input type="number" v-model.number="form.tileOverlap" min="0" max="0.9" step="0.05" />
          </label>
          <label class="field">新数据集名
            <input v-model="form.tileName" />
          </label>
        </div>

        <div class="row mt">
          <button class="primary" :disabled="!form.dataset || busy" @click="start">
            {{ busy ? '创建中…' : '开始标注' }}
          </button>
          <span class="small faint" v-if="selectedDataset">
            将标注 {{ selectedDataset.stats.images }} 张图片；作业会自动分配到
            <span class="mono">label_system</span> 环境执行
          </span>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>作业进度</h2><span class="spacer" />
          <span class="panel-note" v-if="jobId"><span class="mono">{{ jobId }}</span></span></div>
        <div v-if="!jobId" class="empty">
          还没有发起作业。左侧填好参数后点「开始标注」，进度和日志会实时出现在这里。
        </div>
        <LogPane v-else :project="projectId" :job-id="jobId" @done="onDone" />
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>最近的标注 / 裁减作业</h2><span class="spacer" />
        <button class="tiny ghost" @click="navigate('jobs')">全部作业</button></div>
      <div v-if="!jobs.length" class="empty">还没有作业</div>
      <table v-else>
        <thead><tr><th>作业</th><th>类型</th><th>状态</th><th>进度</th><th>创建</th></tr></thead>
        <tbody>
          <tr v-for="j in jobs" :key="j.id">
            <td>{{ j.title }}</td>
            <td class="mono">{{ j.kind }}</td>
            <td><JobState :state="j.state" /></td>
            <td class="small muted">{{ j.progress?.message }}（{{ Math.round(j.progress?.percent || 0) }}%）</td>
            <td class="small muted">{{ timeAgo(j.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import { loadCatalog, navigate, timeAgo, toast } from '../state'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'
import ModelSelect from '../components/ModelSelect.vue'

const props = defineProps({
  projectId: String,
  routeId: String,
  meta: Object,
  presetModel: { type: String, default: '' },
})

const datasets = ref([])
const models = ref([])
const jobs = ref([])
const catalog = ref(null)
const vlmInfo = ref(null)
const jobId = ref('')
const busy = ref(false)
const autoClasses = ref(false)

const form = reactive({
  dataset: '', model: '', task: 'detect', weights: '', imgsz: 640, conf: 0.25, iou: 0.7,
  device: 'auto', batch: 16, classes: '', unknown: 'skip', overwrite: false, refine: false,
  samWeights: 'FastSAM-s.pt', samImgsz: 1024, everything: false,
  vlmModelId: '', vlmUrl: '', vlmModel: '', vlmKey: '', vlmConcurrency: 4, vlmRate: 0, vlmPrompt: '',
  tile: false, tileImgsz: 640, tileOverlap: 0.2, tileName: '',
})
const vlmModels = ref([])
const vlmState = ref({ configured: false, api_key_source: '', base_url: '', model: '', model_id: '' })
const vlmTest = ref(null)
const vlmTesting = ref(false)
const vlmPromptDirty = ref(false)
const selectedVlmModel = computed(() => vlmModels.value.find(m => m.id === form.vlmModelId) || null)
const taskLabel = computed(() => ({ detect: '目标检测', segment: '实例分割', pose: '姿态关键点', classify: '图像分类' }[form.task] || form.task || 'detect'))
const vlmPromptSource = computed(() => {
  if (vlmPromptDirty.value) return '本次作业自定义'
  const src = (selectedVlmModel.value?.prompt_sources || {})[form.task]
  return src || '内置模板'
})
const defaultVlmPrompt = computed(() => {
  const model = selectedVlmModel.value
  if (!model) return ''
  return (model.prompts || {})[form.task] || (model.prompts || {}).detect || ''
})

// 可标注的模型（来自 /api/catalog 的 labelers：声明了 predict 能力的实现）
const labelers = computed(() => (catalog.value?.labelers || []).filter(l => l.id))
const modelInfo = computed(() => labelers.value.find(l => l.id === form.model) || {})
const modelFramework = computed(() => String(modelInfo.value.framework || ''))
const isRemoteVlm = computed(() => modelFramework.value === 'remote-vlm')
const isSam = computed(() => modelFramework.value.endsWith('-sam'))
const usesLocalWeights = computed(() => !isRemoteVlm.value)
const supportsAutoClasses = computed(() => (modelInfo.value.capabilities || []).includes('auto_classes'))
const modelTask = computed(() => {
  const tasks = modelInfo.value.tasks || []
  if (tasks.includes('pose')) return 'pose'
  if (tasks.includes('segment')) return 'segment'
  if (tasks.includes('classify')) return 'classify'
  return 'detect'
})

const selectedDataset = computed(() => datasets.value.find(d => d.id === form.dataset) || null)
const classPlaceholder = computed(() => {
  const classes = selectedDataset.value?.classes || []
  const hint = classes.length ? `当前数据集：${classes.join(', ')}` : '例如 person,bus'
  return `${hint}；不知道时勾选 VLM「自动发现类别」`
})
const classMetaLines = computed(() => {
  const meta = selectedDataset.value?.class_meta || {}
  return Object.entries(meta)
    .map(([name, node]) => ({ name, description: (node || {}).description || '' }))
    .filter(line => line.description)
})
const promptClasses = computed(() => {
  if (autoClasses.value) return '（自动发现：由 VLM 返回决定）'
  const typed = form.classes.split(',').map(s => s.trim()).filter(Boolean)
  return typed.length ? typed.join(', ') : '（请填写）'
})
const vlmHint = computed(() => selectedVlmModel.value?.base_url || 'https://...（服务商的 OpenAI 兼容地址）')
const keyEnvHint = computed(() => {
  const model = selectedVlmModel.value
  return (model?.key_envs || [])[0] || 'MLPLATFORM_VLM_API_KEY'
})
const vlmNotes = computed(() => selectedVlmModel.value?.notes || '')
const selectedProvider = computed(() => selectedVlmModel.value)
const vlmKeyReady = computed(() => {
  if (form.vlmKey) return true
  if (selectedVlmModel.value?.key_available_from_env) return true
  return !!(vlmState.value.has_api_key && vlmState.value.model_id === form.vlmModelId)
})
const vlmKeySource = computed(() => {
  if (form.vlmKey) return '本次作业填写'
  if (selectedVlmModel.value?.key_available_from_env) return selectedVlmModel.value.key_env_used
  return vlmState.value.api_key_source || 'config'
})

function applyVlmModel(keepPrompt = false) {
  const model = selectedVlmModel.value
  if (!model) return
  form.vlmUrl = model.base_url || ''
  form.vlmModel = model.model || ''
  const defaults = model.defaults || {}
  if (defaults.concurrency) form.vlmConcurrency = defaults.concurrency
  if (defaults.rate_limit_per_min !== undefined) form.vlmRate = defaults.rate_limit_per_min
  if (!keepPrompt) {
    form.vlmPrompt = defaultVlmPrompt.value
    vlmPromptDirty.value = false
  }
  vlmTest.value = null
}

function resetVlmPrompt() {
  form.vlmPrompt = defaultVlmPrompt.value
  vlmPromptDirty.value = false
}

// 切到 VLM 时确保模型目录/提示词已加载
watch(() => form.model, () => {
  if (isRemoteVlm.value) {
    if (!form.vlmModelId) loadVlm()
    else applyVlmModel()
  }
})

// 任务/模型变化时，提示词自动切到对应模板（用户手动改过则保留）
watch(defaultVlmPrompt, (text) => {
  if (!vlmPromptDirty.value) form.vlmPrompt = text
})

async function testVlm() {
  vlmTesting.value = true
  vlmTest.value = null
  try {
    vlmTest.value = await api.vlmTest({
      model_id: form.vlmModelId || undefined,
      base_url: form.vlmUrl || undefined,
      model: form.vlmModel || undefined,
      api_key: form.vlmKey || undefined,
    })
  } catch (e) {
    vlmTest.value = { ok: false, error: e.message }
  } finally {
    vlmTesting.value = false
  }
}

async function loadVlm() {
  try {
    const info = await api.vlm()
    vlmInfo.value = info
    vlmModels.value = info.models || []
    vlmState.value = info.config || {}
    form.vlmModelId = vlmState.value.model_id || info.default || vlmModels.value[0]?.id || ''
    applyVlmModel()
    if (selectedVlmModel.value && !selectedVlmModel.value.vision) {
      form.vlmModelId = vlmState.value.model_id || info.default || ''
      applyVlmModel()
    }
  } catch (e) {
    /* 未配置线上接口时也能正常打开页面 */
  }
}

function onDataset() {
  const ds = selectedDataset.value
  if (!ds) return
  form.task = ds.task
  form.classes = ''
  // 数据集没有类别（用 VLM 自动发现时）：默认允许新增类别，否则模型返回的类别会被全部丢弃
  if (!(ds.classes || []).length) form.unknown = 'add'
  if (!form.tileName) form.tileName = `${ds.name} @ ${form.tileImgsz}px`
}

async function load() {
  if (!props.projectId) return
  datasets.value = await api.datasets(props.projectId)
  models.value = await api.models(props.projectId)
  jobs.value = (await api.jobs(props.projectId, { limit: 10 })).filter(j => j.kind === 'label' || j.kind === 'tile')
  catalog.value = await loadCatalog()
  const preferred = props.routeId || datasets.value[0]?.id || ''
  if (!form.dataset && preferred) { form.dataset = preferred; onDataset() }
  await loadVlm()
  if (!form.model) form.model = defaultModelId()
  if (props.presetModel) {
    form.model = props.presetModel
    if (isRemoteVlm.value) applyVlmModel()
  }
}

/** 默认模型：优先第一个本地实现（有 train/evaluate/predict 的那个），否则第一个可标注实现。 */
function defaultModelId() {
  const list = labelers.value
  const local = list.find(l => !l.virtual && (l.capabilities || []).includes('predict'))
  return (local || list[0] || {}).id || ''
}

function fillDatasetClasses() {
  form.classes = (selectedDataset.value?.classes || []).join(', ')
}

watch(autoClasses, (on) => {
  if (on) form.unknown = 'add'
})

function gotoDatasetDetail() {
  if (!form.dataset) { toast('先选择数据集', 'warn'); return }
  navigate(`datasets/${form.dataset}`)
}

async function start() {
  if (!form.dataset) { toast('请选择数据集', 'warn'); return }
  const typed = form.classes.split(',').map(c => c.trim()).filter(Boolean)
  const dsClasses = selectedDataset.value?.classes || []
  if (!autoClasses.value && !typed.length) {
    toast('请填写本次标注类别（或勾选 VLM 自动发现类别）', 'warn')
    return
  }
  if (!autoClasses.value && typed.join('|') !== dsClasses.join('|')) {
    const removed = dsClasses.filter(c => !typed.includes(c))
    const warn = removed.length
      ? `旧类别中这些会被删除（对应标注一并删除）：${removed.join(', ')}`
      : '旧标注会按类别名重映射到新列表'
    if (!confirm(`本次标注类别 ${typed.join(', ')} 与数据集当前类别不同。\n提交后数据集类别会改成：${typed.join(', ')}；${warn}。继续？`)) {
      return
    }
  }
  busy.value = true
  try {
    const labeler = {
      model: form.model,
      weights: isRemoteVlm.value ? undefined : form.weights || undefined,
      imgsz: form.imgsz, conf: form.conf, iou: form.iou, device: form.device,
      classes: autoClasses.value ? [] : typed,
      auto_classes: autoClasses.value,
      unknown: form.unknown,
    }
    if (isSam.value) {
      labeler.sam = { weights: form.samWeights, imgsz: form.samImgsz, everything: form.everything }
    }
    if (isRemoteVlm.value) {
      labeler.vlm = {
        model_id: form.vlmModelId,
        base_url: form.vlmUrl,
        model: form.vlmModel,
        api_key: form.vlmKey,
        prompt: vlmPromptDirty.value ? form.vlmPrompt : '',
        concurrency: form.vlmConcurrency,
        rate_limit_per_min: form.vlmRate,
      }
    }
    const body = {
      labeler, overwrite: form.overwrite, batch: form.batch,
      refine: form.refine ? { weights: form.samWeights, imgsz: form.samImgsz } : null,
    }
    if (form.tile) {
      body.tile = {
        imgsz: form.tileImgsz, overlap: form.tileOverlap, name: form.tileName,
      }
    }
    const res = await api.runLabel(props.projectId, form.dataset, body)
    jobId.value = res.job.id
    toast(`作业已创建：${res.job.id}`, 'ok')
    jobs.value = (await api.jobs(props.projectId, { limit: 10 })).filter(j => j.kind === 'label' || j.kind === 'tile')
  } catch (e) {
    toast('创建失败：' + e.message, 'err')
  } finally { busy.value = false }
}

function onDone(info) {
  jobs.value = []
  api.jobs(props.projectId, { limit: 10 }).then(list => {
    jobs.value = list.filter(j => j.kind === 'label' || j.kind === 'tile')
  }).catch(() => {})
}

watch(() => props.projectId, load)
watch(() => props.routeId, (v) => { if (v) { form.dataset = v; onDataset() } })
watch(() => props.presetModel, (m) => {
  if (m) {
    form.model = m
    if (isRemoteVlm.value) applyVlmModel()
  }
})
onMounted(load)
</script>
