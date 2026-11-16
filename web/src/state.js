import { reactive } from 'vue'
import { api } from './api'

export const state = reactive({
  ready: false,
  loading: false,
  meta: null,
  projects: [],
  projectId: localStorage.getItem('mlp.project') || '',
  route: { name: 'overview', id: '' },
  toasts: [],
  catalog: null,
  error: '',
})

let toastSeq = 0
export function toast(message, kind = 'info', ttl = 4200) {
  const id = ++toastSeq
  state.toasts.push({ id, message: String(message), kind })
  setTimeout(() => {
    const idx = state.toasts.findIndex(t => t.id === id)
    if (idx >= 0) state.toasts.splice(idx, 1)
  }, ttl)
}

export function currentProject() {
  return state.projects.find(p => p.id === state.projectId) || null
}

export function navigate(path, { replace = false } = {}) {
  const hash = '#/' + String(path).replace(/^#?\/?/, '')
  if (replace) history.replaceState(null, '', hash)
  else location.hash = hash
  syncRoute()
}

export function syncRoute() {
  const raw = location.hash.replace(/^#\/?/, '')
  const [name, id] = raw.split('/')
  state.route = { name: name || 'overview', id: id || '' }
}

export function selectProject(pid) {
  state.projectId = pid || ''
  if (pid) localStorage.setItem('mlp.project', pid)
  else localStorage.removeItem('mlp.project')
}

export async function refreshMeta({ keepProject = true } = {}) {
  state.loading = true
  try {
    const meta = await api.meta()
    state.meta = meta
    state.projects = meta.projects || []
    if (keepProject) {
      const exists = state.projects.some(p => p.id === state.projectId)
      if (!exists) {
        const stored = localStorage.getItem('mlp.project')
        const storedOk = stored && state.projects.some(p => p.id === stored)
        selectProject(storedOk ? stored : (state.projects[0] ? state.projects[0].id : ''))
      }
    }
    state.ready = true
    return meta
  } finally {
    state.loading = false
  }
}

export async function loadCatalog() {
  if (!state.catalog) state.catalog = await api.catalog()
  return state.catalog
}

export function fmtPercent(v) {
  const n = Number(v)
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : '—'
}

export function fmtNum(v, digits = 4) {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

export function fmtDuration(sec) {
  const n = Number(sec)
  if (!Number.isFinite(n) || n <= 0) return '—'
  if (n < 60) return `${n.toFixed(0)}s`
  if (n < 3600) return `${Math.floor(n / 60)}m${Math.round(n % 60)}s`
  return `${Math.floor(n / 3600)}h${Math.round((n % 3600) / 60)}m`
}

export function fmtBytes(bytes) {
  const n = Number(bytes)
  if (!Number.isFinite(n)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = n
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i++ }
  return `${value.toFixed(i ? 1 : 0)}${units[i]}`
}

export function timeAgo(iso) {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (!Number.isFinite(t)) return '—'
  const diff = Math.max(0, (Date.now() - t) / 1000)
  if (diff < 60) return `${Math.round(diff)} 秒前`
  if (diff < 3600) return `${Math.round(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.round(diff / 3600)} 小时前`
  return `${Math.round(diff / 86400)} 天前`
}

export const JOB_STATES = {
  queued: { label: '排队中', kind: 'muted' },
  running: { label: '运行中', kind: 'info' },
  canceling: { label: '取消中', kind: 'warn' },
  succeeded: { label: '成功', kind: 'ok' },
  failed: { label: '失败', kind: 'err' },
  cancelled: { label: '已取消', kind: 'muted' },
}

export const TASKS = [
  { id: 'detect', label: '目标检测' },
  { id: 'segment', label: '实例分割' },
  { id: 'pose', label: '姿态关键点' },
  { id: 'classify', label: '图像分类' },
]
