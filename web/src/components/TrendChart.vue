<template>
  <div>
    <svg :viewBox="`0 0 ${width} ${height}`" style="width: 100%; height: auto; display: block">
      <g>
        <line v-for="tick in yTicks" :key="'y' + tick" :x1="padL" :x2="width - padR"
              :y1="yScale(tick)" :y2="yScale(tick)" stroke="#22303f" stroke-dasharray="3 4" />
        <text v-for="tick in yTicks" :key="'yt' + tick" :x="padL - 6" :y="yScale(tick) + 3"
              fill="#6b7891" font-size="10" text-anchor="end">{{ fmt(tick) }}</text>
        <line v-for="tick in xTicks" :key="'x' + tick" :x1="xScale(tick)" :x2="xScale(tick)"
              :y1="padT" :y2="height - padB" stroke="#1c2532" />
        <text v-for="tick in xTicks" :key="'xt' + tick" :x="xScale(tick)" :y="height - padB + 14"
              fill="#6b7891" font-size="10" text-anchor="middle">{{ tick }}</text>
        <path v-for="s in series" :key="s.label" :d="path(s.points)"
              fill="none" :stroke="s.color" stroke-width="2" stroke-linejoin="round"
              stroke-linecap="round" />
      </g>
    </svg>
    <div class="row tight small muted" v-if="series.length">
      <span v-for="s in series" :key="s.label" class="row tight" style="gap: 4px">
        <i :style="{ display: 'inline-block', width: '10px', height: '3px', background: s.color, borderRadius: '2px' }" />
        {{ s.label }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  series: { type: Array, default: () => [] }, // [{label, color, points:[[x,y],...]}]
  height: { type: Number, default: 200 },
})
const width = 720
const padL = 44
const padR = 12
const padT = 10
const padB = 26

const all = computed(() => props.series.flatMap(s => s.points || []))
const maxY = computed(() => Math.max(1e-6, ...all.value.map(p => p[1])))
const maxX = computed(() => Math.max(1, ...all.value.map(p => p[0])))
const yTicks = computed(() => {
  const m = maxY.value
  return [0, m * 0.25, m * 0.5, m * 0.75, m].map(v => Math.round(v * 1000) / 1000)
})
const xTicks = computed(() => {
  const m = maxX.value
  const step = Math.max(1, Math.round(m / 6))
  const out = []
  for (let v = 1; v <= m; v += step) out.push(v)
  return out
})
const xScale = (x) => padL + (x / maxX.value) * (width - padL - padR)
const yScale = (y) => props.height - padB - (y / maxY.value) * (props.height - padT - padB)
const path = (points) => (points || []).map(([x, y], i) => `${i ? 'L' : 'M'}${xScale(x).toFixed(1)},${yScale(y).toFixed(1)}`).join(' ')
const fmt = (v) => (v >= 1 ? v.toFixed(1) : v.toFixed(3))
</script>
