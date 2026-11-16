<template>
  <span class="chip" :class="meta.kind">{{ label }}</span>
</template>

<script setup>
import { computed } from 'vue'
import { JOB_STATES } from '../state'

const props = defineProps({ state: { type: String, default: 'queued' } })
const meta = computed(() => JOB_STATES[props.state] || { label: props.state, kind: 'muted' })
const label = computed(() => meta.value.label)
</script>
