<template>
  <div class="stack" style="gap: 10px">
    <div v-if="!datasets.length" class="empty">这个项目还没有数据集，先去「数据集」页创建或上传图片。</div>
    <div v-for="ds in datasets" :key="ds.id" class="panel" style="padding: 12px"
         :style="{ borderColor: isSelected(ds.id) ? '#2c4a78' : '' }">
      <div class="row">
        <label class="check">
          <input type="checkbox" :checked="isSelected(ds.id)" @change="toggle(ds)" />
          <b>{{ ds.name }}</b>
        </label>
        <span class="chip muted">{{ ds.task }}</span>
        <span class="small muted">
          {{ ds.stats.images }} 张 · 已标注 {{ ds.stats.labeled }} ·
          train/val/test = {{ ds.stats.splits.train }}/{{ ds.stats.splits.val }}/{{ ds.stats.splits.test }}
        </span>
        <span class="right row tight" v-if="isSelected(ds.id)">
          <span class="small faint">{{ selectionLabel(ds) }}</span>
          <button class="tiny" @click="useAll(ds)">全选</button>
          <button class="tiny" @click="openPicker(ds)">手动选择</button>
        </span>
      </div>
      <div class="small faint mt" v-if="isSelected(ds.id) && ds.task === taskHint">
        类别：{{ (ds.classes || []).join(', ') }}
      </div>
    </div>

    <Modal v-if="picker" :title="`选择图片 · ${picker.name}`" wide @close="picker = null">
      <div class="row" style="margin-bottom: 10px">
        <input v-model="picker.q" placeholder="搜索文件名…" style="width: 200px" @keyup.enter="loadPage(0)" />
        <select v-model="picker.split" style="width: 130px" @change="loadPage(0)">
          <option value="">全部划分</option>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
        <span class="small muted">已选 {{ picker.selected.size }} / {{ picker.total }}</span>
        <span class="right row tight">
          <button class="tiny" :disabled="picker.loading" @click="selectPage">选择本页</button>
          <button class="tiny ghost" @click="picker.selected.clear()">清空选择</button>
        </span>
      </div>
      <div v-if="picker.loading" class="loading">加载图片…</div>
      <div v-else class="scroll-x" style="max-height: 52vh; overflow-y: auto; padding-right: 4px">
        <table class="image-list">
          <thead>
            <tr><th style="width: 36px"></th><th>文件名</th><th>划分</th><th class="num">标注</th><th>操作</th></tr>
          </thead>
          <tbody>
            <tr v-for="img in picker.images" :key="img.name" style="cursor: pointer"
                :class="{ selected: picker.selected.has(img.name) }" @click="toggleImage(img.name)">
              <td><span class="check-box" :class="{ on: picker.selected.has(img.name) }">{{ picker.selected.has(img.name) ? '✓' : '' }}</span></td>
              <td class="mono small">{{ img.name }}</td>
              <td><span class="chip" :class="img.labels ? 'ok' : 'warn'">{{ img.split || 'train' }}</span></td>
              <td class="num">{{ img.labels || 0 }}</td>
              <td @click.stop>
                <button class="tiny" @click="viewImage(img)">查看</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="row mt" v-if="picker.total > picker.limit">
        <button class="tiny" :disabled="picker.offset <= 0" @click="loadPage(picker.offset - picker.limit)">上一页</button>
        <span class="small muted">{{ picker.offset + 1 }} - {{ Math.min(picker.offset + picker.limit, picker.total) }} / {{ picker.total }}</span>
        <button class="tiny" :disabled="picker.offset + picker.limit >= picker.total" @click="loadPage(picker.offset + picker.limit)">下一页</button>
      </div>
      <template #foot>
        <button class="ghost" @click="picker = null">取消</button>
        <button class="primary" @click="confirmPicker">确定（{{ picker.selected.size }} 张）</button>
      </template>
    </Modal>

    <Modal v-if="viewer" :title="viewer.name" wide @close="viewer = null">
      <div class="small faint" style="margin-bottom: 8px">
        {{ viewer.width }}×{{ viewer.height }} · 标注 {{ viewer.labels }} 个 · 点关闭返回列表
      </div>
      <img :src="viewer.url" style="width: 100%; border-radius: 8px; display: block" />
    </Modal>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { api, urls } from '../api'
import { toast } from '../state'
import Modal from './Modal.vue'

const props = defineProps({
  project: { type: String, required: true },
  datasets: { type: Array, default: () => [] },
  modelValue: { type: Array, default: () => [] }, // [{dataset, images: 'all'|[names]}]
  taskHint: { type: String, default: '' },
})
const emit = defineEmits(['update:modelValue'])

const picker = ref(null)
const viewer = ref(null)

function viewImage(img) {
  viewer.value = {
    name: img.name,
    url: urls.image(props.project, picker.value.id, img.name),
    width: img.width,
    height: img.height,
    labels: img.labels || 0,
  }
}

function entry(did) { return props.modelValue.find(e => e.dataset === did) }
function isSelected(did) { return !!entry(did) }
function selectionLabel(ds) {
  const e = entry(ds.id)
  if (!e) return ''
  if (e.images === 'all') return `全部 ${ds.stats.images} 张`
  return `已选 ${e.images.length} 张`
}

function setEntries(list) { emit('update:modelValue', list) }

function toggle(ds) {
  const list = props.modelValue.filter(e => e.dataset !== ds.id)
  if (!isSelected(ds.id)) list.push({ dataset: ds.id, images: 'all' })
  setEntries(list)
}

function useAll(ds) {
  const list = props.modelValue.filter(e => e.dataset !== ds.id)
  list.push({ dataset: ds.id, images: 'all' })
  setEntries(list)
}

async function openPicker(ds) {
  const current = entry(ds.id)
  const selected = new Set(Array.isArray(current?.images) ? current.images : [])
  picker.value = {
    id: ds.id, name: ds.name, total: ds.stats.images, images: [],
    selected, q: '', split: '', offset: 0, limit: 20, loading: true,
  }
  await loadPage(0)
}

async function loadPage(offset) {
  const p = picker.value
  if (!p) return
  p.loading = true
  try {
    const res = await api.dataset(props.project, p.id, {
      offset, limit: p.limit, split: p.split || '', q: p.q || '',
    })
    p.images = res.images || []
    p.total = res.total || 0
    p.offset = offset
  } catch (e) {
    toast('加载图片失败：' + e.message, 'err')
  } finally {
    p.loading = false
  }
}

function toggleImage(name) {
  const p = picker.value
  if (p.selected.has(name)) p.selected.delete(name)
  else p.selected.add(name)
}

function selectPage() {
  const p = picker.value
  for (const img of p.images) p.selected.add(img.name)
}

function confirmPicker() {
  const p = picker.value
  const names = Array.from(p.selected)
  const list = props.modelValue.filter(e => e.dataset !== p.id)
  if (!names.length) {
    toast('没有选中任何图片，该数据集将不参与本次训练', 'warn')
  } else if (names.length >= p.total) {
    list.push({ dataset: p.id, images: 'all' })
  } else {
    list.push({ dataset: p.id, images: names })
  }
  setEntries(list)
  picker.value = null
}
</script>
