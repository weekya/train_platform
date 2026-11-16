<template>
  <div>
    <div class="row" style="margin-bottom: 10px">
      <button :class="{ primary: mode === 'box' }" @click="mode = 'box'">✏️ 画框</button>
      <button :class="{ primary: mode === 'select' }" @click="mode = 'select'">🖐 选择 / 移动</button>
      <button v-if="task === 'pose'" :class="{ primary: mode === 'kpt' }" @click="mode = 'kpt'">📍 关键点</button>
      <select v-model.number="activeClass" style="width: auto; min-width: 110px">
        <option v-for="(c, i) in classes" :key="c" :value="i">{{ c }}</option>
      </select>
      <button class="tiny danger" :disabled="sel < 0" @click="removeSelected">删除选中</button>
      <button class="tiny ghost" @click="clearAll">清空</button>
      <span class="right row tight">
        <button class="tiny ghost" @click="$emit('navigate', -1)">← 上一张</button>
        <button class="tiny ghost" @click="$emit('navigate', 1)">下一张 →</button>
      </span>
    </div>

    <div class="row small muted" style="margin-bottom: 8px">
      <span class="mono">{{ image.name }}</span>
      <span>·</span><span>{{ image.width }}×{{ image.height }}</span>
      <span>·</span><span>{{ items.length }} 个标注</span>
      <span v-if="dirty" class="chip warn">未保存</span>
      <span class="right row tight">
        <button class="tiny primary" :disabled="saving" @click="save">{{ saving ? '保存中…' : '保存标注' }}</button>
      </span>
    </div>

    <div class="label-compare">
      <div class="label-pane">
        <div class="pane-title">
          <span class="chip">原图</span>
          <span class="faint">图片文件本身，没有任何框</span>
        </div>
        <div class="label-stage">
          <img :src="imageSrc" class="label-img" alt="原图" />
        </div>
      </div>
      <div class="label-pane">
        <div class="pane-title">
          <span class="chip ok">标注图</span>
          <span class="faint">框由 labels 实时绘制，可直接编辑</span>
        </div>
        <div class="label-stage" ref="stage">
          <img :src="imageSrc" class="label-img" @load="onImgLoad" @error="onImgError" />
          <canvas
            ref="canvas"
            class="label-canvas"
            :class="{ 'cursor-cross': mode !== 'select' }"
            @pointerdown="onDown"
            @pointermove="onMove"
            @pointerup="onUp"
            @pointerleave="onLeave"
          />
          <div v-if="loading" class="stage-loading">加载图片…</div>
          <div v-else-if="error" class="stage-loading" style="color: var(--err)">{{ error }}</div>
        </div>
      </div>
    </div>
    <div class="small faint mt">
      左边是原图文件本身（永远没有框）；右边是平台按 <span class="mono">labels/&lt;文件名&gt;.txt</span> 实时绘制的标注图，可编辑。
      画框模式下拖拽即可新建框；选择模式可拖动框微调；pose 数据集选中框后用「关键点」模式依次点击。
      <span v-if="error" style="color: var(--err)">· {{ error }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api, urls } from '../api'
import { toast } from '../state'

const props = defineProps({
  project: { type: String, required: true },
  dataset: { type: String, required: true },
  image: { type: Object, required: true },
  task: { type: String, default: 'detect' },
  classes: { type: Array, default: () => [] },
  kptShape: { type: Array, default: () => [17, 3] },
  initialItems: { type: Array, default: () => [] },
})
const emit = defineEmits(['navigate', 'saved'])

const canvas = ref(null)
const stage = ref(null)
const items = ref(JSON.parse(JSON.stringify(props.initialItems || [])))
const sel = ref(-1)
const mode = ref('box')
const activeClass = ref(0)
const loading = ref(true)
const saving = ref(false)
const dirty = ref(false)
const error = ref('')
const imgEl = ref(null)
const imgSize = ref({ w: 0, h: 0 })
const viewScale = ref(1)
const MAX_CANVAS_SIDE = 2400
let drag = null

const task = computed(() => props.task || 'detect')
const imageSrc = computed(() => urls.image(props.project, props.dataset, props.image.name))
const kptCount = computed(() => Number(props.kptShape?.[0] || 17))

function resetFromProps() {
  items.value = JSON.parse(JSON.stringify(props.initialItems || []))
  sel.value = -1
  dirty.value = false
  error.value = ''
  draw()
}

watch(() => props.image.name, () => {
  resetFromProps()
  loading.value = true
})
// 父组件异步拉取标注后（换图时先清空再填充），这里要跟着刷新
watch(() => props.initialItems, () => {
  if (!dirty.value) resetFromProps()
}, { deep: true })

function onImgLoad(e) {
  imgEl.value = e.target
  const img = e.target
  // 以数据集里记录的原始尺寸为准：预览图可能是缩略版，但标注坐标始终是原图像素
  const w = Number(props.image.width) || img.naturalWidth
  const h = Number(props.image.height) || img.naturalHeight
  imgSize.value = { w, h }
  loading.value = false
  // 超大图不在 canvas 里按原分辨率分配（几千万像素会吃掉大量内存），
  // 只在显示层缩放；标注坐标始终是原图像素，保存/换算不受影响。
  viewScale.value = Math.min(1, MAX_CANVAS_SIDE / Math.max(w, h))
  if (canvas.value) {
    canvas.value.width = Math.max(1, Math.round(w * viewScale.value))
    canvas.value.height = Math.max(1, Math.round(h * viewScale.value))
  }
  draw()
}

function onImgError() {
  loading.value = false
  error.value = '图片加载失败（文件可能已被删除）'
}

function toImage(evt) {
  const rect = canvas.value.getBoundingClientRect()
  const ratio = (canvas.value.width / rect.width) / (viewScale.value || 1)
  const x = (evt.clientX - rect.left) * ratio
  const y = (evt.clientY - rect.top) * ratio
  return {
    x: Math.max(0, Math.min(imgSize.value.w, x)),
    y: Math.max(0, Math.min(imgSize.value.h, y)),
  }
}

function hitTest(p) {
  for (let i = items.value.length - 1; i >= 0; i--) {
    const b = items.value[i].box
    if (!b) continue
    if (p.x >= b[0] && p.x <= b[2] && p.y >= b[1] && p.y <= b[3]) return i
  }
  return -1
}

function onDown(evt) {
  if (loading.value || !canvas.value) return
  const p = toImage(evt)
  if (mode.value === 'box') {
    drag = { kind: 'new', start: p, cur: p }
  } else if (mode.value === 'select') {
    const idx = hitTest(p)
    sel.value = idx
    if (idx >= 0) {
      drag = { kind: 'move', start: p, box: [...items.value[idx].box],
               poly: items.value[idx].polygon ? JSON.parse(JSON.stringify(items.value[idx].polygon)) : null,
               kps: items.value[idx].keypoints ? JSON.parse(JSON.stringify(items.value[idx].keypoints)) : null }
    }
    draw()
  } else if (mode.value === 'kpt') {
    if (sel.value < 0) { error.value = '先在选择模式下选中一个框'; return }
    const item = items.value[sel.value]
    const kps = item.keypoints || Array.from({ length: kptCount.value }, () => [0, 0, 0])
    const next = kps.findIndex(k => !k || Number(k[2]) === 0)
    const idx = next < 0 ? Math.min(kptCount.value - 1, kps.filter(k => Number(k[2]) > 0).length) : next
    kps[idx] = [p.x, p.y, 2]
    item.keypoints = kps
    dirty.value = true
    draw()
  }
}

function onMove(evt) {
  if (!drag) return
  const p = toImage(evt)
  if (drag.kind === 'new') {
    drag.cur = p
    draw()
  } else if (drag.kind === 'move') {
    const dx = p.x - drag.start.x
    const dy = p.y - drag.start.y
    const item = items.value[sel.value]
    const maxX = imgSize.value.w
    const maxY = imgSize.value.h
    const b = drag.box
    const w = b[2] - b[0]
    const h = b[3] - b[1]
    let nx1 = Math.max(0, Math.min(maxX - w, b[0] + dx))
    let ny1 = Math.max(0, Math.min(maxY - h, b[1] + dy))
    item.box = [nx1, ny1, nx1 + w, ny1 + h]
    if (drag.poly) item.polygon = drag.poly.map(([x, y]) => [x + nx1 - b[0], y + ny1 - b[1]])
    if (drag.kps) item.keypoints = drag.kps.map(k => [k[0] + nx1 - b[0], k[1] + ny1 - b[1], k[2]])
    dirty.value = true
    draw()
  }
}

function onUp() {
  if (!drag) return
  if (drag.kind === 'new') {
    const { start, cur } = drag
    const x1 = Math.min(start.x, cur.x)
    const y1 = Math.min(start.y, cur.y)
    const x2 = Math.max(start.x, cur.x)
    const y2 = Math.max(start.y, cur.y)
    if (x2 - x1 >= 3 && y2 - y1 >= 3) {
      const item = {
        cls: activeClass.value,
        box: [x1, y1, x2, y2],
        polygon: null,
        keypoints: task.value === 'pose'
          ? Array.from({ length: kptCount.value }, () => [0, 0, 0])
          : null,
        conf: null,
      }
      items.value.push(item)
      sel.value = items.value.length - 1
      dirty.value = true
    }
  }
  drag = null
  draw()
}

function onLeave() { if (drag) onUp() }

function removeSelected() {
  if (sel.value < 0) return
  items.value.splice(sel.value, 1)
  sel.value = -1
  dirty.value = true
  draw()
}

function clearAll() {
  items.value = []
  sel.value = -1
  dirty.value = true
  draw()
}

const COLORS = ['#ff6b6b', '#4dabf7', '#51cf66', '#ffd43b', '#c084fc', '#22d3ee', '#f783ac', '#a9e34b']

function draw() {
  const c = canvas.value
  if (!c || !imgSize.value.w) return
  const ctx = c.getContext('2d')
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, c.width, c.height)
  const vs = viewScale.value || 1
  // 之后所有绘制都用原图像素坐标，由 transform 缩放到画布
  ctx.setTransform(vs, 0, 0, vs, 0, 0)
  const minSide = Math.min(imgSize.value.w, imgSize.value.h)
  const lw = Math.max(2 / vs, Math.round(minSide / 320))
  const fs = Math.max(14 / vs, Math.round(minSide / 40))
  ctx.font = `${fs}px system-ui, sans-serif`
  ctx.textBaseline = 'bottom'

  items.value.forEach((item, idx) => {
    const color = COLORS[(item.cls || 0) % COLORS.length]
    const b = item.box
    if (!b) return
    if (item.polygon && item.polygon.length >= 3) {
      ctx.beginPath()
      item.polygon.forEach(([x, y], i) => { i ? ctx.lineTo(x, y) : ctx.moveTo(x, y) })
      ctx.closePath()
      ctx.fillStyle = color + '33'
      ctx.fill()
      ctx.strokeStyle = color
      ctx.lineWidth = idx === sel.value ? lw + 1 : lw * 0.6
      ctx.stroke()
    }
    ctx.strokeStyle = color
    ctx.lineWidth = idx === sel.value ? lw + 1 : lw
    if (idx === sel.value) ctx.setLineDash([lw * 2, lw])
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1])
    ctx.setLineDash([])
    const name = props.classes[item.cls] ?? `cls${item.cls}`
    const text = item.conf != null ? `${name} ${Number(item.conf).toFixed(2)}` : name
    const tw = ctx.measureText(text).width
    ctx.fillStyle = color
    ctx.fillRect(b[0], Math.max(0, b[1] - fs - 4), tw + 8, fs + 4)
    ctx.fillStyle = '#10151d'
    ctx.fillText(text, b[0] + 4, Math.max(fs, b[1] - 4))

    if (item.keypoints) {
      const r = Math.max(3, lw + 1)
      item.keypoints.forEach((kp) => {
        if (!kp || Number(kp[2]) === 0) return
        ctx.beginPath()
        ctx.arc(kp[0], kp[1], r, 0, Math.PI * 2)
        ctx.fillStyle = '#ff4d4d'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1
        ctx.stroke()
      })
    }
  })

  if (drag && drag.kind === 'new') {
    const { start, cur } = drag
    ctx.strokeStyle = '#ffffff'
    ctx.setLineDash([6 / vs, 4 / vs])
    ctx.lineWidth = lw
    ctx.strokeRect(Math.min(start.x, cur.x), Math.min(start.y, cur.y),
                   Math.abs(cur.x - start.x), Math.abs(cur.y - start.y))
    ctx.setLineDash([])
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)
}

async function save() {
  saving.value = true
  error.value = ''
  try {
    const res = await api.saveLabels(props.project, props.dataset, props.image.name, items.value)
    dirty.value = false
    emit('saved', res)
    toast('标注已保存', 'ok')
  } catch (e) {
    error.value = e.message
    toast('保存失败：' + e.message, 'err')
  } finally {
    saving.value = false
  }
}

function onKey(e) {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (sel.value >= 0) { e.preventDefault(); removeSelected() }
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', draw)
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('resize', draw)
})
</script>

<style scoped>
.label-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
.label-pane { min-width: 0; }
.pane-title { display: flex; align-items: center; gap: 6px; font-size: 12px; margin-bottom: 5px; }
.label-stage { position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #05070a; border: 1px solid var(--border); }
.label-img { display: block; width: 100%; height: auto; user-select: none; }
.label-canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
@media (max-width: 1000px) { .label-compare { grid-template-columns: 1fr; } }
</style>
