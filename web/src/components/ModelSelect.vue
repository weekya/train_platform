<template>
  <div class="stack" style="gap: 8px">
    <div class="row">
      <select :value="selectedPreset" aria-label="训练模型" @change="onPreset($event.target.value)" style="flex: 1">
        <option value="">选择模型…</option>
        <optgroup v-if="projectModels.length" label="项目模型（训练产物 / 上传）">
          <option v-for="m in projectModels" :key="m.id" :value="m.id">
            {{ m.name }} · {{ m.kind }} · {{ m.task }}
          </option>
        </optgroup>
        <optgroup v-for="f in families" :key="f.id" :label="f.label">
          <option v-for="w in weightsOf(f)" :key="w.name" :value="w.name">
            {{ f.id }} · {{ w.scale }} · {{ w.name }}{{ w.local ? '（本地）' : '（需下载）' }}
          </option>
        </optgroup>
      </select>
      <button v-if="allowCustom" class="tiny ghost" @click="custom = !custom">自定义</button>
    </div>
    <input v-if="allowCustom && custom" :value="modelValue" @input="$emit('update:modelValue', $event.target.value)"
           placeholder="权重路径或名字，例如 weights/pretrained/yolo11s.pt" />
    <div class="small faint">
      当前：<span class="mono">{{ modelValue || '未选择' }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  catalog: { type: Object, default: null },
  projectModels: { type: Array, default: () => [] },
  task: { type: String, default: 'detect' },
  modelValue: { type: String, default: '' },
  allowCustom: { type: Boolean, default: true },
})
const emit = defineEmits(['update:modelValue'])
const custom = ref(false)

const families = computed(() => {
  // 能力位由实现层声明（configs/models/_impls.yaml → impl/*.py）；
  // 没有 capabilities 的旧数据退回到 role 判断，保证向后兼容。
  const canTrain = (f) => Array.isArray(f.capabilities)
    ? f.capabilities.includes('train')
    : f.role !== 'segmenter'
  return (props.catalog?.families || [])
    .filter(canTrain)
    .filter(f => (f.weights || {})[props.task])
})
const selectedPreset = computed(() => {
  const inProject = props.projectModels.some(m => m.id === props.modelValue)
  if (inProject) return props.modelValue
  for (const f of families.value) {
    for (const w of weightsOf(f)) if (w.name === props.modelValue) return w.name
  }
  return ''
})
function weightsOf(family) {
  const mapping = family.weights?.[props.task] || {}
  return Object.entries(mapping).map(([scale, w]) => ({ scale, ...w }))
}
function onPreset(value) {
  if (!value) return
  emit('update:modelValue', value)
}
</script>
