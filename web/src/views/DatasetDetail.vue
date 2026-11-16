<template>
  <div v-if="dataset" class="stack">
    <div class="row">
      <button class="ghost" @click="navigate('datasets')">← 数据集</button>
      <b style="font-size: 15px">{{ dataset.name }}</b>
      <span class="chip muted">{{ dataset.task }}</span>
      <span class="chip" v-if="dataset.source?.kind === 'tile'">裁减产物</span>
      <span class="spacer" />
      <button class="tiny" @click="navigate(`label/${dataset.id}`)">自动标注</button>
      <button class="tiny" @click="openTile">按 imagz 裁减</button>
      <button class="tiny ghost" @click="openRelations">关系规则</button>
      <button class="tiny ghost" @click="openClassMeta">类别说明</button>
      <button class="tiny ghost" @click="openClasses">编辑类别</button>
      <button class="tiny ghost" @click="recompute">重新统计</button>
      <button class="tiny danger" @click="removeDataset">删除数据集</button>
    </div>

    <div class="cards">
      <div class="stat"><div class="label">图片</div><div class="value">{{ stats.images }}</div>
        <div class="hint">已标注 {{ stats.labeled }} · 实例 {{ stats.instances }}</div></div>
      <div class="stat"><div class="label">划分</div>
        <div class="value" style="font-size: 17px">{{ stats.splits.train }} / {{ stats.splits.val }} / {{ stats.splits.test }}</div>
        <div class="hint">train / val / test</div></div>
      <div class="stat"><div class="label">尺寸范围</div>
        <div class="value" style="font-size: 15px">{{ stats.width_range[0] }}–{{ stats.width_range[1] }} × {{ stats.height_range[0] }}–{{ stats.height_range[1] }}</div>
        <div class="hint">宽 × 高（像素）</div></div>
      <div class="stat"><div class="label">占用</div><div class="value">{{ fmtBytes(stats.bytes) }}</div>
        <div class="hint mono">{{ dataset.id }}</div></div>
    </div>

    <section class="panel">
      <div class="panel-head">
        <h2>类别分布</h2><span class="spacer" />
        <span class="panel-note">类别：{{ (dataset.classes || []).join(', ') || '未设置' }}</span>
      </div>
      <div class="row tight">
        <span v-for="(n, c) in (stats.per_class || {})" :key="c" class="chip info">{{ c }} {{ n }}</span>
        <span v-if="!(stats.per_class && Object.keys(stats.per_class).length)" class="small faint">还没有标注</span>
      </div>
      <div class="small faint mt" v-if="dataset.tiling">
        裁减来源：{{ dataset.tiling.source_dataset }} · imagz {{ dataset.tiling.imagz }}px ·
        窗口 {{ dataset.tiling.summary?.windows || 0 }} 个 · 模式 {{ JSON.stringify(dataset.tiling.summary?.modes || {}) }}
        <div v-if="dataset.tiling.summary?.reason_sample" class="mono">{{ dataset.tiling.summary.reason_sample }}</div>
      </div>
      <div class="small faint mt" v-if="dataset.source?.path">来源：<span class="mono">{{ dataset.source.path }}</span></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>图片</h2><span class="spacer" />
        <select v-model="filters.split" style="width: 130px" @change="loadImages(0)">
          <option value="">全部划分</option>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
        <input v-model="filters.q" placeholder="搜索文件名…" style="width: 190px"
               @keyup.enter="loadImages(0)" />
        <select v-model.number="filters.limit" aria-label="每页张数" style="width: 110px" @change="loadImages(0)">
          <option :value="20">每页 20 张</option>
          <option :value="40">每页 40 张</option>
          <option :value="100">每页 100 张</option>
        </select>
        <span class="small faint">{{ total }} 张</span>
      </div>
      <div v-if="loading" class="loading">加载中…</div>
      <div v-else-if="!images.length" class="empty">没有图片。回到数据集列表页上传，或换个筛选条件。</div>
      <div v-else class="scroll-x">
        <table class="image-list">
          <thead>
            <tr>
              <th style="width: 40px">#</th>
              <th>文件名</th>
              <th>划分</th>
              <th class="num">尺寸</th>
              <th class="num">标注</th>
              <th>标注来源</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(img, idx) in images" :key="img.name" @click="openImage(img)" style="cursor: pointer"
                :class="{ selected: editing && editing.name === img.name }">
              <td class="small faint">{{ filters.offset + idx + 1 }}</td>
              <td>
                <b class="mono" style="font-weight: 500">{{ img.name }}</b>
                <div class="small faint" v-if="img.origin">
                  <template v-if="img.origin.source_image">来自 {{ img.origin.source_image }}</template>
                  <template v-if="img.origin.crop"> · 裁减窗口 {{ img.origin.crop.join(',') }}</template>
                </div>
              </td>
              <td><span class="chip" :class="img.labels ? 'ok' : 'warn'">{{ splitLabel(img.split) }}</span></td>
              <td class="num small">{{ img.width }}×{{ img.height }}</td>
              <td class="num">{{ img.labels || 0 }}</td>
              <td class="small muted">{{ img.labeled_by || '—' }}</td>
              <td @click.stop>
                <div class="row tight" style="flex-wrap: nowrap">
                  <button class="tiny primary" @click="openImage(img)">打开图片</button>
                  <button class="tiny" @click="changeSplit(img, 'train')" title="设为 train">train</button>
                  <button class="tiny" @click="changeSplit(img, 'val')" title="设为 val">val</button>
                  <button class="tiny" @click="changeSplit(img, 'test')" title="设为 test">test</button>
                  <button class="tiny danger" @click="removeImage(img)">删除</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="row mt" v-if="total > filters.limit">
        <button class="tiny" :disabled="filters.offset <= 0" @click="loadImages(filters.offset - filters.limit)">上一页</button>
        <span class="small muted">{{ filters.offset + 1 }} – {{ Math.min(filters.offset + filters.limit, total) }} / {{ total }}</span>
        <button class="tiny" :disabled="filters.offset + filters.limit >= total" @click="loadImages(filters.offset + filters.limit)">下一页</button>
      </div>
    </section>

    <Modal v-if="editing" :title="`标注 · ${editing.name}`" xwide @close="editing = null">
      <LabelEditor :project="projectId" :dataset="dataset.id" :image="editing"
                   :task="dataset.task" :classes="dataset.classes" :kpt-shape="dataset.kpt_shape"
                   :initial-items="editingItems" @saved="onSaved" @navigate="navigateImage" />
    </Modal>

    <Modal v-if="classMetaOpen" title="类别说明 / 别名（给 VLM 提示词用）" wide @close="classMetaOpen = false">
      <div class="stack">
        <div class="small faint">
          类别说明会写进 VLM 提示词的类别清单，关系/组合类类别（如"在桌上的瓶子"）直接在这里用自然语言描述，
          由模型自己判断；别名用于把模型返回的 label 映射回类别（大小写、下划线、空格不敏感）。
        </div>
        <div v-for="row in classMetaForm" :key="row.name" class="panel" style="padding: 10px">
          <div class="form-grid">
            <label class="field">类别
              <input :value="row.name" disabled />
            </label>
            <label class="field">说明（自然语言）
              <input v-model="row.description" placeholder="例如：放在桌面上的瓶子（不含地上的瓶子）" />
            </label>
            <label class="field">别名（逗号分隔）
              <input v-model="row.aliases" placeholder="bottle on table, 桌上的瓶子" />
            </label>
          </div>
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="classMetaOpen = false">取消</button>
        <button class="primary" :disabled="busy" @click="saveClassMeta">保存</button>
      </template>
    </Modal>

    <Modal v-if="relationsOpen" title="位置关系规则（例如：在桌上的瓶子）" wide @close="relationsOpen = false">
      <div class="stack">
        <div class="small faint">
          每条规则：把满足「关系」的 <b>主体</b> 框改判成 <b>输出类别</b>（如 bottle → bottle_on_table）。
          关系类别需要先出现在数据集类别里；基础类别（bottle、table）先用模型标好。
          <br />当前类别：<span class="mono">{{ (dataset?.classes || []).join(', ') }}</span>
        </div>
        <div v-for="(rule, idx) in relationRules" :key="idx" class="panel" style="padding: 10px">
          <div class="form-grid">
            <label class="field">输出类别
              <select v-model="rule.output">
                <option value="">选择…</option>
                <option v-for="c in dataset?.classes || []" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <label class="field">主体类别
              <select v-model="rule.subject">
                <option value="">选择…</option>
                <option v-for="c in dataset?.classes || []" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <label class="field">关系
              <select v-model="rule.relation">
                <option value="on">在…上（底边中点在参照物框内）</option>
                <option value="inside">在…内部</option>
                <option value="overlap">与…重叠</option>
                <option value="above">在…上方</option>
                <option value="below">在…下方</option>
                <option value="left_of">在…左侧</option>
                <option value="right_of">在…右侧</option>
              </select>
            </label>
            <label class="field">参照物类别
              <select v-model="rule.object">
                <option value="">选择…</option>
                <option v-for="c in dataset?.classes || []" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <label class="field">容差（按参照物尺寸比例）
              <input type="number" v-model.number="rule.tolerance" min="0" max="1" step="0.01" />
            </label>
            <label class="check" style="margin-top: 18px">
              <input type="checkbox" v-model="rule.drop_object" /> 删除参照物（如 table）
            </label>
            <label class="check" style="margin-top: 18px">
              <input type="checkbox" v-model="rule.drop_unmatched_subjects" /> 删除不满足的主体
            </label>
          </div>
          <div class="row mt">
            <button class="tiny danger" @click="relationRules.splice(idx, 1)">删除这条规则</button>
          </div>
        </div>
        <button class="tiny" @click="addRule">＋ 添加规则</button>
        <div class="small faint">
          例：输出 bottle_on_table、主体 bottle、关系 on、参照物 table → 只有"底部落在桌子框内"的瓶子会变成
          bottle_on_table；勾「删除参照物」后 table 框会被删掉。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="relationsOpen = false">取消</button>
        <button class="primary" :disabled="busy" @click="submitRelations">应用规则</button>
      </template>
    </Modal>

    <Modal v-if="classesOpen" title="编辑数据集类别" @close="classesOpen = false">
      <div class="stack">
        <label class="field">类别（逗号分隔，顺序即类别 id）
          <input v-model="classForm" placeholder="person, helmet, vest" />
        </label>
        <div class="small faint">
          这里定义的类别会用于：自动标注的提示词类别、训练/评估的类别列表、标签文件的类别 id。
          <br />修改后已有标签按新列表解释，若删除了原有类别请确认标签仍然正确。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="classesOpen = false">取消</button>
        <button class="primary" :disabled="!classForm.trim()" @click="saveClasses">保存</button>
      </template>
    </Modal>

    <Modal v-if="tileOpen" :title="`按 imagz 裁减 · ${dataset.name}`" @close="tileOpen = false">
      <div class="stack">
        <div class="form-grid">
          <label class="field">imagz
            <input type="number" v-model.number="tileForm.imgsz" min="64" step="32" />
          </label>
          <label class="field">重叠
            <input type="number" v-model.number="tileForm.overlap" min="0" max="0.9" step="0.05" />
          </label>
          <label class="field">最小保留比例
            <input type="number" v-model.number="tileForm.min_contain" min="0" max="1" step="0.05" />
          </label>
          <label class="field">新数据集名
            <input v-model="tileForm.name" />
          </label>
        </div>
        <div class="small faint">
          目标比 imagz 大 → 窗口放大装下目标；目标比 imagz 小 → 切窗并平移避开目标；只裁不缩。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="tileOpen = false">取消</button>
        <button class="primary" :disabled="busy" @click="startTile">开始裁减</button>
      </template>
    </Modal>
  </div>
  <div v-else class="loading">加载数据集…</div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import { fmtBytes, navigate, toast } from '../state'
import Modal from '../components/Modal.vue'
import LabelEditor from '../components/LabelEditor.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const dataset = ref(null)
const images = ref([])
const total = ref(0)
const loading = ref(true)
const busy = ref(false)
const editing = ref(null)
const editingItems = ref([])
const tileOpen = ref(false)
const classesOpen = ref(false)
const classForm = ref('')
const relationsOpen = ref(false)
const relationRules = ref([])
const classMetaOpen = ref(false)
const classMetaForm = ref([])
const tileForm = reactive({ imgsz: 640, overlap: 0.2, min_contain: 0.6, name: '' })
const filters = reactive({ offset: 0, limit: 20, split: '', q: '' })

const stats = computed(() => dataset.value?.stats || {
  images: 0, labeled: 0, instances: 0, per_class: {}, splits: { train: 0, val: 0, test: 0 },
  bytes: 0, width_range: [0, 0], height_range: [0, 0],
})

const splitLabel = (s) => (s === 'train' ? 'train' : s === 'val' ? 'val' : s === 'test' ? 'test' : s)

async function loadInfo() {
  if (!props.projectId || !props.routeId) return
  const res = await api.dataset(props.projectId, props.routeId, { offset: 0, limit: 1 })
  dataset.value = res.dataset
  total.value = res.total
}
async function loadImages(offset = 0) {
  loading.value = true
  try {
    const res = await api.dataset(props.projectId, props.routeId, {
      offset, limit: filters.limit, split: filters.split, q: filters.q,
    })
    images.value = res.images || []
    total.value = res.total || 0
    filters.offset = offset
  } catch (e) {
    toast('加载失败：' + e.message, 'err')
  } finally { loading.value = false }
}

async function openImage(img) {
  editing.value = img
  editingItems.value = []
  try {
    const res = await api.labels(props.projectId, props.routeId, img.name)
    editingItems.value = res.items || []
  } catch (e) { toast('读取标注失败：' + e.message, 'err') }
}

function navigateImage(delta) {
  const idx = images.value.findIndex(i => i.name === editing.value?.name)
  const next = images.value[idx + delta]
  if (next) openImage(next)
  else toast('已经是这一页的边界了', 'warn')
}

async function onSaved(res) {
  const idx = images.value.findIndex(i => i.name === editing.value?.name)
  if (idx >= 0) images.value[idx] = res.image
  if (res.stats) dataset.value.stats = res.stats
  editing.value = null
}

async function changeSplit(img, split) {
  try {
    const res = await api.setSplit(props.projectId, props.routeId, img.name, split)
    const idx = images.value.findIndex(i => i.name === img.name)
    if (idx >= 0) images.value[idx] = res.image
    if (dataset.value) dataset.value.stats = res.stats
    toast(`已设为 ${split}`, 'ok')
  } catch (e) { toast('修改失败：' + e.message, 'err') }
}

async function removeImage(img) {
  if (!confirm(`删除图片「${img.name}」及其标注？`)) return
  try {
    const res = await api.deleteImage(props.projectId, props.routeId, img.name)
    if (dataset.value) dataset.value.stats = res.stats
    await loadImages(filters.offset)
    toast('已删除', 'ok')
  } catch (e) { toast('删除失败：' + e.message, 'err') }
}

async function recompute() {
  try {
    const res = await api.recomputeDataset(props.projectId, props.routeId)
    if (dataset.value) dataset.value.stats = res.stats
    await loadImages(filters.offset)
    toast('统计已更新', 'ok')
  } catch (e) { toast('失败：' + e.message, 'err') }
}

function openClassMeta() {
  const meta = dataset.value?.class_meta || {}
  classMetaForm.value = (dataset.value?.classes || []).map(name => ({
    name,
    description: (meta[name] || {}).description || '',
    aliases: ((meta[name] || {}).aliases || []).join(', '),
  }))
  classMetaOpen.value = true
}

async function saveClassMeta() {
  const class_meta = {}
  for (const row of classMetaForm.value) {
    const description = (row.description || '').trim()
    const aliases = (row.aliases || '').split(',').map(a => a.trim()).filter(Boolean)
    if (description || aliases.length) class_meta[row.name] = { description, aliases }
  }
  busy.value = true
  try {
    const res = await api.updateDataset(props.projectId, props.routeId, { class_meta })
    dataset.value = { ...dataset.value, ...res }
    classMetaOpen.value = false
    toast('类别说明已保存', 'ok')
  } catch (e) { toast('保存失败：' + e.message, 'err') }
  finally { busy.value = false }
}

function newRule() {
  const classes = dataset.value?.classes || []
  return {
    output: classes[2] || classes[0] || '',
    subject: classes[1] || classes[0] || '',
    object: classes[0] || '',
    relation: 'on',
    tolerance: 0.05,
    drop_object: true,
    drop_unmatched_subjects: false,
  }
}

function openRelations() {
  if (!relationRules.value.length) relationRules.value = [newRule()]
  relationsOpen.value = true
}

function addRule() {
  relationRules.value.push(newRule())
}

async function submitRelations() {
  const rules = relationRules.value
    .filter(r => r.output && r.subject && r.object)
    .map(r => ({ ...r, tolerance: Number(r.tolerance) || 0 }))
  if (!rules.length) { toast('请至少填写一条完整规则', 'warn'); return }
  busy.value = true
  try {
    const res = await api.runRelation(props.projectId, props.routeId, { rules })
    relationsOpen.value = false
    toast(`关系规则作业已创建：${res.job.id}`, 'ok')
    navigate('jobs')
  } catch (e) { toast('创建失败：' + e.message, 'err') }
  finally { busy.value = false }
}

function openClasses() {
  classForm.value = (dataset.value?.classes || []).join(', ')
  classesOpen.value = true
}

async function saveClasses() {
  const classes = classForm.value.split(',').map(c => c.trim()).filter(Boolean)
  if (!classes.length) { toast('类别不能为空', 'warn'); return }
  try {
    const res = await api.updateDataset(props.projectId, props.routeId, { classes })
    dataset.value = { ...dataset.value, ...res }
    classesOpen.value = false
    await loadImages(filters.offset)
    toast('类别已更新', 'ok')
  } catch (e) { toast('保存失败：' + e.message, 'err') }
}

function openTile() {
  tileOpen.value = true
  tileForm.name = `${dataset.value.name} @ ${tileForm.imgsz}px`
}
async function startTile() {
  busy.value = true
  try {
    const job = await api.runTile(props.projectId, props.routeId, { ...tileForm })
    tileOpen.value = false
    toast(`裁减作业已创建：${job.job.id}`, 'ok')
    navigate('jobs')
  } catch (e) { toast('创建失败：' + e.message, 'err') }
  finally { busy.value = false }
}

async function removeDataset() {
  if (!confirm(`删除数据集「${dataset.value.name}」？`)) return
  try {
    await api.deleteDataset(props.projectId, props.routeId)
    toast('已删除', 'ok')
    navigate('datasets')
  } catch (e) { toast('失败：' + e.message, 'err') }
}

watch(() => props.routeId, async () => { await loadInfo(); await loadImages(0) })
onMounted(async () => { await loadInfo(); await loadImages(0) })
</script>
