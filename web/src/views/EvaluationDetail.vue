<template>
  <div v-if="record" class="stack">
    <div class="row">
      <button class="ghost" @click="navigate('eval')">← 评估记录</button>
      <b style="font-size: 15px">{{ record.name }}</b>
      <JobState :state="record.state" />
      <span class="chip muted">{{ record.task }}</span>
      <span class="spacer" />
      <span class="small faint" v-if="record.weights?.ref">权重 <span class="mono">{{ record.weights.ref }}</span></span>
    </div>

    <div class="cards">
      <div class="stat"><div class="label">mAP50-95</div><div class="value">{{ fmtNum(record.metrics?.map) }}</div></div>
      <div class="stat"><div class="label">mAP50</div><div class="value">{{ fmtNum(record.metrics?.map50) }}</div></div>
      <div class="stat"><div class="label">mAP75</div><div class="value">{{ fmtNum(record.metrics?.map75) }}</div></div>
      <div class="stat"><div class="label">Precision</div><div class="value">{{ fmtNum(record.metrics?.precision) }}</div></div>
      <div class="stat"><div class="label">Recall</div><div class="value">{{ fmtNum(record.metrics?.recall) }}</div></div>
      <div class="stat"><div class="label">推理速度</div>
        <div class="value">{{ record.metrics?.speed_ms?.inference ?? '—' }}<span class="small faint"> ms/图</span></div>
        <div class="hint">耗时 {{ fmtDuration(record.duration) }}</div></div>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>每类指标</h2><span class="spacer" />
        <span class="panel-note">{{ record.data?.val_images ?? '—' }} 张 · split={{ record.params?.split }}</span></div>
      <div v-if="!record.per_class?.length" class="empty">没有每类指标</div>
      <table v-else>
        <thead><tr><th>类别</th><th class="num">AP50-95</th><th class="num">AP50</th>
          <th class="num">Precision</th><th class="num">Recall</th></tr></thead>
        <tbody>
          <tr v-for="c in record.per_class" :key="c.class_id">
            <td>{{ c.name }}<span class="faint mono small"> #{{ c.class_id }}</span></td>
            <td class="num">{{ fmtNum(c.ap) }}</td>
            <td class="num">{{ fmtNum(c.ap50) }}</td>
            <td class="num">{{ fmtNum(c.precision) }}</td>
            <td class="num">{{ fmtNum(c.recall) }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <div class="grid-2">
      <section class="panel">
        <div class="panel-head"><h2>曲线 / 混淆矩阵</h2><span class="spacer" /><span class="panel-note">点击放大</span></div>
        <div v-if="!plotEntries.length" class="empty">没有可视化图</div>
        <div v-else class="grid-images">
          <div v-for="p in plotEntries" :key="p.name" class="tile" @click="view(p)">
            <img :src="fileUrl(p.path)" loading="lazy" />
            <div class="meta"><span>{{ p.name }}</span></div>
          </div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>作业日志</h2><span class="spacer" />
          <span class="panel-note mono" v-if="record.job">{{ record.job }}</span></div>
        <LogPane v-if="record.job" :project="projectId" :job-id="record.job" :show-head="false" />
        <div v-else class="empty">没有关联作业</div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>预测可视化</h2><span class="spacer" />
        <span class="panel-note">{{ (record.artifacts?.predictions || []).length }} 张（阈值 conf={{ record.params?.conf }}）</span></div>
      <div v-if="!(record.artifacts?.predictions || []).length" class="empty">没有预测图</div>
      <div v-else class="grid-images">
        <div v-for="p in visiblePredictions" :key="p" class="tile" @click="view({ name: p.split('/').pop(), path: p })">
          <img :src="fileUrl(p)" loading="lazy" />
          <div class="meta"><span>{{ p.split('/').pop() }}</span></div>
        </div>
      </div>
      <div class="row mt" v-if="(record.artifacts.predictions || []).length > visiblePredictions.length">
        <button class="tiny" @click="predLimit += 20">
          加载更多（{{ visiblePredictions.length }} / {{ record.artifacts.predictions.length }}）
        </button>
        <span class="small faint">一次只渲染 20 张，避免占用浏览器内存</span>
      </div>
    </section>

    <Modal v-if="viewing" :title="viewing.name" wide @close="viewing = null">
      <img :src="viewing.url" style="width: 100%; border-radius: 8px" />
    </Modal>
  </div>
  <div v-else class="loading">加载评估记录…</div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api, urls } from '../api'
import { fmtDuration, fmtNum, navigate, toast } from '../state'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'
import Modal from '../components/Modal.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const record = ref(null)
const viewing = ref(null)
const predLimit = ref(20)

const plotEntries = computed(() => Object.entries(record.value?.artifacts?.plots || {}).map(([name, path]) => ({ name, path })))
const visiblePredictions = computed(() => (record.value?.artifacts?.predictions || []).slice(0, predLimit.value))
const fileUrl = (path) => urls.file(props.projectId, path)
function view(p) { viewing.value = { name: p.name, url: fileUrl(p.path) } }

async function load() {
  if (!props.projectId || !props.routeId) return
  try { record.value = await api.evaluation(props.projectId, props.routeId) }
  catch (e) { toast('加载失败：' + e.message, 'err') }
}
watch(() => props.routeId, load)
onMounted(load)
</script>
