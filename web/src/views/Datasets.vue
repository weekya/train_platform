<template>
  <div class="stack">
    <div class="row">
      <button class="primary" @click="showCreate = true">＋ 新建数据集</button>
      <button class="ghost" @click="showImport = true">导入已有 YOLO 目录</button>
      <span class="right small faint">
        图片统一转成 <span class="mono">{{ imageFormat }}</span> 格式（EXIF 转正 / 透明铺白 / 不拉伸）
      </span>
    </div>

    <section class="panel" v-if="datasets.length">
      <div class="panel-head">
        <h2>数据集列表</h2>
        <span class="spacer" />
        <span class="panel-note">{{ datasets.length }} 个 · 点击名称进入图片列表</span>
        <input v-model="filter" placeholder="搜索名称 / id" style="width: 200px" />
      </div>
      <div class="scroll-x">
        <table>
          <thead>
            <tr>
              <th>名称 / ID</th><th>任务</th><th class="num">图片</th><th class="num">已标注</th>
              <th class="num">实例</th><th>划分 train/val/test</th><th>类别分布</th><th>大小</th>
              <th>来源</th><th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="ds in filteredDatasets" :key="ds.id">
              <td>
                <a href="javascript:void(0)" @click="open(ds)"><b>{{ ds.name }}</b></a>
                <div class="small faint mono">{{ ds.id }}</div>
              </td>
              <td><span class="chip muted">{{ ds.task }}</span></td>
              <td class="num">{{ ds.stats.images }}</td>
              <td class="num">{{ ds.stats.labeled }}</td>
              <td class="num">{{ ds.stats.instances }}</td>
              <td class="small muted">{{ ds.stats.splits.train }} / {{ ds.stats.splits.val }} / {{ ds.stats.splits.test }}</td>
              <td>
                <span v-for="(n, c) in (ds.stats.per_class || {})" :key="c" class="chip" style="margin-right: 4px">{{ c }} {{ n }}</span>
                <span v-if="!(ds.stats.per_class && Object.keys(ds.stats.per_class).length)" class="small faint">未标注</span>
              </td>
              <td class="small muted">{{ fmtBytes(ds.stats.bytes) }}</td>
              <td class="small muted">{{ sourceLabel(ds) }}</td>
              <td @click.stop>
                <div class="row tight" style="flex-wrap: nowrap">
                  <button class="tiny" @click="open(ds)">图片列表</button>
                  <button class="tiny" @click="navigate(`label/${ds.id}`)">自动标注</button>
                  <button class="tiny" @click="openTile(ds)">裁减</button>
                  <button class="tiny danger" @click="remove(ds)">删除</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="!filteredDatasets.length" class="empty">没有匹配的数据集</div>
    </section>
    <div v-else class="empty">还没有数据集。先上传一批图片（支持 jpg/png/webp/bmp/tif/gif 等）。</div>

    <!-- 上传 -->
    <section class="panel">
      <div class="panel-head"><h2>上传图片</h2><span class="spacer" />
        <span class="panel-note">支持多选 / 拖拽；相同内容图片会自动去重</span></div>
      <div class="row" style="align-items: flex-end">
        <label class="field" style="min-width: 240px">目标数据集
          <select v-model="uploadDataset">
            <option value="">选择数据集…</option>
            <option v-for="ds in datasets" :key="ds.id" :value="ds.id">{{ ds.name }}（{{ ds.id }}）</option>
          </select>
        </label>
        <label class="field">划分
          <select v-model="uploadSplit" style="width: 150px">
            <option value="auto">自动按比例</option>
            <option value="train">全部 train</option>
            <option value="val">全部 val</option>
            <option value="test">全部 test</option>
          </select>
        </label>
        <label class="check" style="margin-bottom: 8px">
          <input type="checkbox" v-model="keepRaw" /> 保留原图（便于换 imagz 重新裁减）
        </label>
      </div>
      <div class="dropzone" :class="{ over: dragOver }" @dragover.prevent="dragOver = true"
           @dragleave.prevent="dragOver = false" @drop.prevent="onDrop">
        <input ref="fileInput" type="file" multiple accept="image/*" style="display: none"
               @change="onFilePick" />
        <div v-if="uploading" class="loading">正在上传并转码 {{ uploadProgress }}…</div>
        <template v-else>
          <p style="margin: 0 0 4px">拖拽图片到这里，或 <button class="link" @click="$refs.fileInput?.click()">选择文件</button></p>
          <p class="small faint" style="margin: 0">单文件上限 {{ maxUploadMb }}MB</p>
        </template>
      </div>
      <div v-if="uploadResult" class="small mt" :class="uploadResult.errors?.length ? '' : 'muted'">
        新增 {{ uploadResult.added }} 张，重复跳过 {{ uploadResult.skipped }} 张
        <span v-if="uploadResult.errors?.length" style="color: var(--err)">，失败 {{ uploadResult.errors.length }} 张</span>
        <div v-for="e in (uploadResult.errors || []).slice(0, 5)" :key="e.file" class="small" style="color: var(--err)">
          {{ e.file }}：{{ e.error }}
        </div>
      </div>
    </section>

    <!-- 新建 -->
    <Modal v-if="showCreate" title="新建数据集" @close="showCreate = false">
      <div class="stack">
        <div class="form-grid">
          <label class="field">名称
            <input v-model="createForm.name" placeholder="例如：底座数据 v1" />
          </label>
          <label class="field">任务类型
            <select v-model="createForm.task">
              <option v-for="t in TASKS" :key="t.id" :value="t.id">{{ t.label }}（{{ t.id }}）</option>
            </select>
          </label>
          <label class="field">类别（逗号分隔）
            <input v-model="createForm.classes" :disabled="createForm.autoDiscover"
                   placeholder="person, car（不知道就先勾下面自动发现）" />
          </label>
        </div>
        <label class="field">描述
          <textarea v-model="createForm.description" placeholder="数据来源、采集条件、标注规范…"></textarea>
        </label>
        <label class="check">
          <input type="checkbox" v-model="createForm.autoDiscover" />
          暂不指定类别（用 VLM 标注时自动发现类别）
        </label>
        <div class="small faint" v-if="createForm.autoDiscover">
          数据集类别留空：VLM 提示词会变成 "any object of interest"，模型返回的类别会在标注时
          自动加入数据集（需要把「模型类别不在数据集里时」设为「自动新增类别」）。
          也可以之后在数据集详情页「编辑类别」补充/修正。
        </div>
        <div class="small faint" v-else>
          数据集任务决定标签格式：detect = 框、segment = 多边形、pose = 框 + 关键点。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="showCreate = false">取消</button>
        <button class="primary" :disabled="!createForm.name.trim()" @click="create">创建</button>
      </template>
    </Modal>

    <!-- 导入 -->
    <Modal v-if="showImport" title="导入已有 YOLO 目录" @close="showImport = false">
      <div class="stack">
        <label class="field">服务器上的目录路径
          <input v-model="importForm.path" placeholder="/data/my_dataset（包含 images/ 与 labels/）" />
        </label>
        <div class="form-grid">
          <label class="field">数据集名称（可空）
            <input v-model="importForm.name" placeholder="默认用目录名" />
          </label>
          <label class="field">类别（可空，默认读 data.yaml）
            <input v-model="importForm.classes" placeholder="person, car" />
          </label>
        </div>
        <div class="small faint">
          支持 <span class="mono">images/{train,val,test}</span> + <span class="mono">labels/{train,val,test}</span> 结构；
          图片会统一转码，标签原样保留。data.yaml 里带 kpt_shape 会自动识别为 pose 数据集。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="showImport = false">取消</button>
        <button class="primary" :disabled="!importForm.path.trim() || busy" @click="doImport">
          {{ busy ? '导入中…' : '开始导入' }}
        </button>
      </template>
    </Modal>

    <!-- 裁减 -->
    <Modal v-if="tileTarget" :title="`按 imagz 裁减 · ${tileTarget.name}`" @close="tileTarget = null">
      <div class="stack">
        <div class="form-grid">
          <label class="field">imagz（裁减窗口）
            <input type="number" v-model.number="tileForm.imgsz" min="64" step="32" />
          </label>
          <label class="field">重叠比例
            <input type="number" v-model.number="tileForm.overlap" min="0" max="0.9" step="0.05" />
          </label>
          <label class="field">最小保留比例
            <input type="number" v-model.number="tileForm.min_contain" min="0" max="1" step="0.05" />
          </label>
          <label class="field">窗口上限 max_tile
            <input type="number" v-model.number="tileForm.max_tile" min="64" step="32" />
          </label>
          <label class="field">目标比 imagz 大时
            <select v-model="tileForm.oversize">
              <option value="expand">放大窗口装下目标（推荐）</option>
              <option value="downscale">整图等比缩放</option>
              <option value="skip">跳过该图</option>
            </select>
          </label>
          <label class="field">新数据集名称
            <input v-model="tileForm.name" />
          </label>
        </div>
        <div class="small faint">
          规则：图片不超过 imagz 时不裁；目标比 imagz 大时把窗口放大到能完整装下（最多 max_tile）；
          目标比 imagz 小时按 imagz 切窗并平移避开目标。只裁不缩，训练分辨率不受影响。
        </div>
      </div>
      <template #foot>
        <button class="ghost" @click="tileTarget = null">取消</button>
        <button class="primary" :disabled="busy" @click="startTile">开始裁减</button>
      </template>
    </Modal>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import { TASKS, fmtBytes, loadCatalog, navigate, state, toast } from '../state'
import Modal from '../components/Modal.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const datasets = ref([])
const filter = ref('')
const filteredDatasets = computed(() => {
  const q = filter.value.trim().toLowerCase()
  if (!q) return datasets.value
  return datasets.value.filter(d => d.name.toLowerCase().includes(q) || d.id.toLowerCase().includes(q))
})
const showCreate = ref(false)
const showImport = ref(false)
const busy = ref(false)

const uploadDataset = ref('')
const uploadSplit = ref('auto')
const keepRaw = ref(true)
const dragOver = ref(false)
const uploading = ref(false)
const uploadProgress = ref('')
const uploadResult = ref(null)
const fileInput = ref(null)

const imageFormat = computed(() => state.catalog?.defaults?.images?.format || 'jpg')
const maxUploadMb = computed(() => props.meta?.web?.max_upload_mb || 512)

const createForm = reactive({ name: '', task: '', classes: '', description: '', autoDiscover: false })
const importForm = reactive({ path: '', name: '', classes: '' })

const tileTarget = ref(null)
const tileForm = reactive({ imgsz: 640, overlap: 0.2, min_contain: 0.6, max_tile: 1536, oversize: 'expand', name: '' })

function sourceLabel(ds) {
  const kind = ds.source?.kind
  if (kind === 'tile') return '裁减产物'
  if (kind === 'import') return '导入'
  if (kind === 'upload') return '上传'
  return '本地'
}

async function load() {
  if (!props.projectId) return
  datasets.value = await api.datasets(props.projectId)
}
function open(ds) { navigate(`datasets/${ds.id}`) }

async function create() {
  try {
    const task = createForm.task || undefined
    const classes = createForm.classes.split(',').map(s => s.trim()).filter(Boolean)
    const ds = await api.createDataset(props.projectId, {
      name: createForm.name.trim(), task,
      // autoDiscover = 明确发送空数组（不要类别），而不是 undefined（会继承项目类别）
      classes: createForm.autoDiscover ? [] : (classes.length ? classes : undefined),
      description: createForm.description.trim(),
    })
    showCreate.value = false
    createForm.name = ''
    createForm.classes = ''
    createForm.description = ''
    await load()
    navigate(`datasets/${ds.id}`)
    toast('数据集已创建', 'ok')
  } catch (e) { toast('创建失败：' + e.message, 'err') }
}

async function doImport() {
  busy.value = true
  try {
    const classes = importForm.classes.split(',').map(s => s.trim()).filter(Boolean)
    const res = await api.importDataset(props.projectId, {
      path: importForm.path.trim(), name: importForm.name.trim() || undefined,
      classes: classes.length ? classes : undefined,
    })
    showImport.value = false
    importForm.path = ''
    importForm.name = ''
    await load()
    toast(`导入完成：${res.images} 张图片`, 'ok')
    navigate(`datasets/${res.dataset.id}`)
  } catch (e) { toast('导入失败：' + e.message, 'err') }
  finally { busy.value = false }
}

function onFilePick(e) {
  const files = Array.from(e.target.files || [])
  e.target.value = ''
  doUpload(files)
}
function onDrop(e) {
  dragOver.value = false
  const files = Array.from(e.dataTransfer?.files || []).filter(f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|tif|tiff|gif|jfif|ppm)$/i.test(f.name))
  doUpload(files)
}

async function doUpload(files) {
  if (!files.length) return
  if (!uploadDataset.value) {
    if (datasets.value.length === 1) uploadDataset.value = datasets.value[0].id
    else { toast('请先选择目标数据集（或新建一个）', 'warn'); return }
  }
  uploading.value = true
  uploadResult.value = null
  try {
    let done = 0
    const chunkSize = 20
    const aggregate = { added: 0, skipped: 0, errors: [], stats: null }
    for (let i = 0; i < files.length; i += chunkSize) {
      const chunk = files.slice(i, i + chunkSize)
      uploadProgress.value = `${done}/${files.length}`
      const res = await api.upload(props.projectId, uploadDataset.value, chunk,
        { split: uploadSplit.value, keepRaw: keepRaw.value })
      aggregate.added += res.added
      aggregate.skipped += res.skipped
      aggregate.errors.push(...(res.errors || []))
      aggregate.stats = res.stats
      done += chunk.length
    }
    uploadResult.value = aggregate
    toast(`上传完成：新增 ${aggregate.added} 张`, 'ok')
    await load()
  } catch (e) {
    toast('上传失败：' + e.message, 'err')
  } finally {
    uploading.value = false
    uploadProgress.value = ''
  }
}

function openTile(ds) {
  tileTarget.value = ds
  tileForm.imgsz = 640
  tileForm.name = `${ds.name} @ 640px`
}

async function startTile() {
  busy.value = true
  try {
    const job = await api.runTile(props.projectId, tileTarget.value.id, {
      imgsz: tileForm.imgsz, overlap: tileForm.overlap, min_contain: tileForm.min_contain,
      max_tile: tileForm.max_tile, oversize: tileForm.oversize, name: tileForm.name,
    })
    tileTarget.value = null
    toast(`裁减作业已创建：${job.job.id}（在「作业日志」里看进度）`, 'ok')
    navigate('jobs')
  } catch (e) { toast('创建失败：' + e.message, 'err') }
  finally { busy.value = false }
}

async function remove(ds) {
  if (!confirm(`删除数据集「${ds.name}」？图片和标注都会删除，且不可恢复。`)) return
  try {
    await api.deleteDataset(props.projectId, ds.id)
    await load()
    toast('数据集已删除', 'ok')
  } catch (e) { toast('删除失败：' + e.message, 'err') }
}

watch(() => props.projectId, load)
onMounted(async () => {
  await load()
  try { await loadCatalog() } catch (e) { /* catalog 仅用于展示默认值 */ }
})
</script>

<style scoped>
.dropzone {
  margin-top: 12px; border: 1px dashed var(--border); border-radius: 10px;
  padding: 24px; text-align: center; color: var(--text-dim); transition: border-color .15s, background .15s;
}
.dropzone.over { border-color: var(--accent); background: #16203a; }
</style>
