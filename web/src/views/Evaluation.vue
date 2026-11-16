<template>
  <div class="stack">
    <div class="grid-2">
      <section class="panel">
        <div class="panel-head"><h2>评估 / 检测测试</h2><span class="spacer" />
          <span class="panel-note">选数据集 + 权重跑标准指标</span></div>
        <div class="form-grid">
          <label class="field">评估方法（项目预设）
            <div class="row tight">
              <select :value="presetId" @change="applyPreset($event.target.value)" style="flex: 1">
                <option value="">自定义参数</option>
                <option v-for="p in presets" :key="p.id" :value="p.id">{{ p.name }}（{{ p.split }}）</option>
              </select>
              <button class="tiny ghost" title="把当前参数保存为评估方法" @click="savePreset">存为方法</button>
              <button class="tiny danger" v-if="presetId" title="删除该评估方法" @click="deletePreset">删</button>
            </div>
          </label>
          <label class="field">数据集
            <select v-model="form.dataset">
              <option value="">选择数据集…</option>
              <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
                {{ ds.name }}（{{ ds.stats.images }} 张）
              </option>
            </select>
          </label>
          <label class="field">使用划分
            <select v-model="form.split">
              <option value="all">全部图片</option>
              <option value="val">仅 val</option>
              <option value="test">仅 test</option>
              <option value="train">仅 train</option>
            </select>
          </label>
          <label class="field">推理尺寸
            <input type="number" v-model.number="form.imgsz" min="32" step="32" />
          </label>
          <label class="field">可视化阈值 conf
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
        <div class="small muted" style="margin-bottom: 6px">权重</div>
        <ModelSelect v-model="form.weights" :catalog="catalog" :task="form.task || 'detect'"
                     :project-models="models" :allow-custom="true" />
        <label class="field mt">评估名称（可空）
          <input v-model="form.name" placeholder="例如：baseline 在 val 上" />
        </label>
        <div class="row mt">
          <button class="primary" :disabled="!form.dataset || !form.weights || busy" @click="start">
            {{ busy ? '创建中…' : '开始评估' }}
          </button>
          <span class="small faint">作业在 <span class="mono">test_system</span> 环境执行</span>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>作业进度</h2><span class="spacer" />
          <span class="panel-note mono" v-if="jobId">{{ jobId }}</span></div>
        <div v-if="!jobId" class="empty">还没有发起评估。填写左侧参数后点「开始评估」。</div>
        <LogPane v-else :project="projectId" :job-id="jobId" @done="onDone" />
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>评估记录</h2><span class="spacer" />
        <span class="panel-note">{{ records.length }} 条</span></div>
      <div v-if="!records.length" class="empty">还没有评估记录</div>
      <table v-else>
        <thead>
          <tr><th>评估</th><th>状态</th><th>数据集 / 划分</th><th class="num">mAP50-95</th>
            <th class="num">mAP50</th><th class="num">P</th><th class="num">R</th><th>权重</th><th>创建</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in records" :key="r.id" @click="navigate(`eval/${r.id}`)" style="cursor: pointer">
            <td>{{ r.name }}<div class="small faint mono">{{ r.id }}</div></td>
            <td><JobState :state="r.state" /></td>
            <td class="small mono">{{ datasetName(r.dataset) }} · {{ r.split }}</td>
            <td class="num">{{ fmtNum(r.metrics?.map) }}</td>
            <td class="num">{{ fmtNum(r.metrics?.map50) }}</td>
            <td class="num">{{ fmtNum(r.metrics?.precision) }}</td>
            <td class="num">{{ fmtNum(r.metrics?.recall) }}</td>
            <td class="small mono">{{ r.weights?.ref }}</td>
            <td class="small muted">{{ timeAgo(r.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import { fmtNum, loadCatalog, navigate, timeAgo, toast } from '../state'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'
import ModelSelect from '../components/ModelSelect.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const datasets = ref([])
const models = ref([])
const records = ref([])
const catalog = ref(null)
const jobId = ref('')
const busy = ref(false)

const form = reactive({
  dataset: '', split: 'all', weights: '', imgsz: 640, conf: 0.25, iou: 0.7, device: 'auto', name: '',
})
const presets = ref([])
const presetId = ref('')

function applyPreset(id) {
  presetId.value = id
  const preset = presets.value.find(p => p.id === id)
  if (!preset) return
  form.split = preset.split || 'all'
  Object.assign(form, preset.params || {})
}

async function savePreset() {
  const name = (window.prompt('评估方法名称（例如：标准 val / 低阈值高召回）') || '').trim()
  if (!name) return
  try {
    const preset = await api.saveEvalPreset(props.projectId, {
      name, split: form.split, note: '',
      params: { imgsz: form.imgsz, conf: form.conf, iou: form.iou, device: form.device },
    })
    presets.value = await api.evalPresets(props.projectId)
    presetId.value = preset.id
    toast('评估方法已保存', 'ok')
  } catch (e) { toast('保存失败：' + e.message, 'err') }
}

async function deletePreset() {
  if (!presetId.value) return
  try {
    await api.deleteEvalPreset(props.projectId, presetId.value)
    presets.value = await api.evalPresets(props.projectId)
    presetId.value = ''
    toast('已删除评估方法', 'ok')
  } catch (e) { toast('删除失败：' + e.message, 'err') }
}

function datasetName(id) { return datasets.value.find(d => d.id === id)?.name || id }

async function load() {
  if (!props.projectId) return
  datasets.value = await api.datasets(props.projectId)
  models.value = await api.models(props.projectId)
  records.value = await api.evaluations(props.projectId)
  presets.value = await api.evalPresets(props.projectId)
  if (!presetId.value && presets.value.length) applyPreset(presets.value[0].id)
  catalog.value = await loadCatalog()
  if (!form.dataset && datasets.value[0]) form.dataset = datasets.value[0].id
  if (!form.weights) {
    const preferred = props.routeId && models.value.some(m => m.id === props.routeId) ? props.routeId : ''
    form.weights = preferred || models.value[0]?.id || ''
  }
}

async function start() {
  busy.value = true
  try {
    const res = await api.startEvaluation(props.projectId, {
      dataset: form.dataset,
      weights: form.weights,
      split: form.split,
      name: form.name.trim() || undefined,
      params: { imgsz: form.imgsz, conf: form.conf, iou: form.iou, device: form.device },
    })
    jobId.value = res.job.id
    toast(`评估作业已创建：${res.job.id}`, 'ok')
  } catch (e) { toast('创建失败：' + e.message, 'err') }
  finally { busy.value = false }
}

function onDone() { api.evaluations(props.projectId).then(list => { records.value = list }).catch(() => {}) }

watch(() => props.projectId, load)
watch(() => props.routeId, (v) => { if (v && models.value.some(m => m.id === v)) form.weights = v })
onMounted(load)
</script>
