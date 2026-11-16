<template>
  <div class="stack">
    <section class="panel">
      <div class="panel-head"><h2>运行环境</h2><span class="spacer" />
        <button class="tiny ghost" :disabled="probing" @click="probeAll">
          {{ probing ? '探测中…' : '重新探测全部' }}
        </button></div>
      <div class="cards">
        <div v-for="e in envs" :key="e.name" class="panel" style="padding: 14px">
          <div class="row">
            <span class="chip" :class="e.role === 'main' ? 'info' : 'muted'">{{ roleLabel(e.role) }}</span>
            <b class="mono">{{ e.name }}</b>
            <span class="spacer" />
            <span class="chip" :class="e.exists ? 'ok' : 'err'">{{ e.exists ? '可用' : '缺失' }}</span>
          </div>
          <div class="kv mt">
            <dt>python</dt><dd>{{ e.python_version || '—' }}</dd>
            <dt>torch</dt><dd>{{ e.torch || '—' }}</dd>
            <dt>CUDA</dt><dd>{{ e.cuda ? '可用' : '不可用' }}</dd>
            <dt>GPU</dt><dd>{{ e.gpu || '—' }}{{ e.capability ? ` (sm_${e.capability.replace('.', '')})` : '' }}</dd>
            <dt>JIT</dt><dd>{{ e.jiterator_ok == null ? '未探测' : e.jiterator_ok ? '正常' : '失败' }}</dd>
            <dt>路径</dt><dd class="mono small">{{ e.path }}</dd>
          </div>
          <div class="row mt">
            <button class="tiny" :disabled="probing" @click="probe(e.name)">探测</button>
            <span class="small faint" v-if="e.probed_at">{{ timeAgo(e.probed_at) }}探测</span>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>体检</h2><span class="spacer" />
        <span class="chip" :class="doctor.ok ? 'ok' : 'warn'">{{ doctor.ok ? '正常' : '有问题' }}</span></div>
      <div class="kv">
        <dt>默认环境</dt><dd class="mono">{{ doctor.default_python || '—' }}</dd>
        <dt>缺失角色</dt><dd>{{ (doctor.missing_roles || []).join(', ') || '无' }}</dd>
        <dt>未登记前缀</dt><dd class="mono small">{{ (doctor.unregistered_prefixes || []).join(', ') || '无' }}</dd>
      </div>
      <div class="small faint mt">
        四个环境都在 <span class="mono">envs/prefixes/</span> 下，随项目目录一起迁移。重建：
        <span class="mono">bash envs/recipes/label_system.sh</span>（克隆母本 main_system）。
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>阶段与环境的对应关系</h2></div>
      <table>
        <thead><tr><th>阶段</th><th>作业类型</th><th>环境</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>启动系统</td><td class="mono">web / runner / cli</td><td class="mono">main_system</td>
            <td>FastAPI + 调度器 + 命令行，所有环境不可用时的兜底</td></tr>
          <tr><td>自动标注 / 数据生成</td><td class="mono">label / tile</td><td class="mono">label_system</td>
            <td>detect / pose / SAM / VLM 推理，imgz 自适应裁减</td></tr>
          <tr><td>模型训练</td><td class="mono">train</td><td class="mono">train_system</td>
            <td>ultralytics 训练，每轮指标实时回传</td></tr>
          <tr><td>模型评估</td><td class="mono">evaluate</td><td class="mono">test_system</td>
            <td>mAP / PR 曲线 / 混淆矩阵 / 预测可视化</td></tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { refreshMeta, timeAgo, toast } from '../state'

const props = defineProps({ projectId: String, routeId: String, meta: Object })
const envs = ref([])
const doctor = ref({})
const probing = ref(false)

const ROLE_LABEL = { main: '启动系统', label: '自动标注', train: '模型训练', test: '模型评估' }
const roleLabel = (r) => ROLE_LABEL[r] || r || '环境'

async function load() {
  try {
    const res = await api.envs()
    envs.value = res.envs || []
    doctor.value = res.doctor || {}
  } catch (e) { toast('读取环境失败：' + e.message, 'err') }
}

async function probe(name) {
  probing.value = true
  try {
    await api.probeEnv(name || null)
    await load()
    toast(name ? `${name} 探测完成` : '全部环境探测完成', 'ok')
  } catch (e) { toast('探测失败：' + e.message, 'err') }
  finally { probing.value = false }
}
function probeAll() { probe(null) }

watch(() => props.meta, load)
onMounted(load)
</script>
