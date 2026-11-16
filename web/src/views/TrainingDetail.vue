<template>
  <div v-if="record" class="stack">
    <div class="row">
      <button class="ghost" @click="navigate('train')">← 训练记录</button>
      <b style="font-size: 15px">{{ record.name }}</b>
      <JobState :state="record.state" />
      <span class="chip muted">{{ record.task }}</span>
      <span class="spacer" />
      <button class="tiny primary" v-if="record.model_id" @click="navigate(`eval/${record.model_id}`)">
        用这个权重去评估
      </button>
      <button class="tiny ghost" @click="showParams = !showParams">{{ showParams ? '隐藏参数' : '查看参数' }}</button>
    </div>

    <div class="cards">
      <div class="stat"><div class="label">mAP50-95</div><div class="value">{{ fmtNum(record.metrics?.map) }}</div>
        <div class="hint">best epoch {{ record.metrics?.best_epoch ?? '—' }}</div></div>
      <div class="stat"><div class="label">mAP50</div><div class="value">{{ fmtNum(record.metrics?.map50) }}</div>
        <div class="hint">IoU=0.5</div></div>
      <div class="stat"><div class="label">Precision</div><div class="value">{{ fmtNum(record.metrics?.precision) }}</div></div>
      <div class="stat"><div class="label">Recall</div><div class="value">{{ fmtNum(record.metrics?.recall) }}</div></div>
      <div class="stat"><div class="label">轮数</div><div class="value">{{ record.metrics?.epochs_run ?? record.params?.epochs }}</div>
        <div class="hint">{{ record.params?.imgsz }}px · batch {{ record.params?.batch }}</div></div>
      <div class="stat"><div class="label">耗时</div><div class="value">{{ fmtDuration(record.duration) }}</div>
        <div class="hint">{{ record.params?.device || '—' }}</div></div>
    </div>

    <div class="grid-2">
      <section class="panel">
        <div class="panel-head"><h2>训练曲线</h2><span class="spacer" />
          <span class="panel-note" v-if="history.length">{{ history.length }} 个 epoch</span></div>
        <div v-if="!history.length" class="empty">还没有曲线数据（训练刚开始或没有 results.csv）</div>
        <TrendChart v-else :series="chartSeries" :height="230" />
      </section>
      <section class="panel">
        <div class="panel-head"><h2>作业与产物</h2><span class="spacer" />
          <span class="panel-note mono" v-if="record.job">{{ record.job }}</span></div>
        <div class="kv">
          <dt>模型</dt><dd><span class="mono">{{ record.model?.ref }}</span>
            <span v-if="record.model?.head" class="chip" style="margin-left: 6px">{{ record.model.head }}</span>
            <span v-if="record.model?.family" class="small faint" style="margin-left: 6px">族 {{ record.model.family }}</span>
          </dd>
          <dt>模型权重</dt><dd class="mono">{{ record.model_id || '（未登记）' }}</dd>
          <dt>训练集</dt><dd>{{ record.data?.train_images ?? '—' }} 张（{{ record.data?.note }}）</dd>
          <dt>验证集</dt><dd>{{ record.data?.val_images ?? '—' }} 张</dd>
          <dt>类别</dt><dd>{{ (record.classes || []).join(', ') }}</dd>
          <dt>数据来源</dt><dd>{{ (record.data?.datasets || []).map(d => d.name).join(' + ') || '—' }}</dd>
          <dt>训练环境</dt><dd class="mono">{{ record.env }}</dd>
        </div>
        <div class="row mt">
          <a v-if="record.artifacts?.results_csv" :href="fileUrl(record.artifacts.results_csv)" target="_blank">results.csv</a>
          <a v-if="record.artifacts?.args_yaml" :href="fileUrl(record.artifacts.args_yaml)" target="_blank">args.yaml</a>
          <span v-if="record.error" class="chip err">{{ record.error }}</span>
        </div>
        <div class="divider" />
        <LogPane v-if="record.job" :project="projectId" :job-id="record.job" :small="true" :show-head="false" />
      </section>
    </div>

    <section class="panel" v-if="showParams">
      <div class="panel-head"><h2>训练参数</h2></div>
      <div class="form-grid">
        <div v-for="(v, k) in record.params" :key="k" class="kv" style="display: contents">
          <dt>{{ k }}</dt><dd class="mono">{{ v }}</dd>
        </div>
      </div>
    </section>

    <section class="panel" v-if="plotEntries.length">
      <div class="panel-head"><h2>训练/验证可视化</h2><span class="spacer" />
        <span class="panel-note">点击放大</span></div>
      <div class="grid-images">
        <div v-for="p in plotEntries" :key="p.name" class="tile" style="aspect-ratio: 4/3"
             @click="openImage(p.path)">
          <img :src="fileUrl(p.path)" loading="lazy" />
          <div class="meta"><span>{{ p.name }}</span></div>
        </div>
      </div>
    </section>

    <Modal v-if="viewing" :title="viewing.name" wide @close="viewing = null">
      <img :src="viewing.url" style="width: 100%; border-radius: 8px" />
    </Modal>
  </div>
  <div v-else class="loading">加载训练记录…</div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api, urls } from '../api'
import { fmtDuration, fmtNum, navigate, toast } from '../state'
import JobState from '../components/JobState.vue'
import LogPane from '../components/LogPane.vue'
import Modal from '../components/Modal.vue'
import TrendChart from '../components/TrendChart.vue'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const record = ref(null)
const showParams = ref(false)
const viewing = ref(null)

const history = computed(() => (record.value?.history || []).filter(h => Number.isFinite(Number(h.map50_95))))
const chartSeries = computed(() => [
  { label: 'mAP50-95', color: '#4d8df6', points: history.value.map(h => [h.epoch, Number(h.map50_95) || 0]) },
  { label: 'mAP50', color: '#3ecf8e', points: history.value.map(h => [h.epoch, Number(h.map50) || 0]) },
  { label: 'Precision', color: '#f5b544', points: history.value.map(h => [h.epoch, Number(h.precision) || 0]) },
  { label: 'Recall', color: '#c084fc', points: history.value.map(h => [h.epoch, Number(h.recall) || 0]) },
])
const plotEntries = computed(() => Object.entries(record.value?.artifacts?.plots || {}).map(([name, path]) => ({ name, path })))
const fileUrl = (path) => urls.file(props.projectId, path)
function openImage(path) { viewing.value = { name: path.split('/').pop(), url: fileUrl(path) } }

async function load() {
  if (!props.projectId || !props.routeId) return
  try { record.value = await api.training(props.projectId, props.routeId) }
  catch (e) { toast('加载失败：' + e.message, 'err') }
}
watch(() => props.routeId, load)
onMounted(load)
</script>
