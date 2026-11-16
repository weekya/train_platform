// 极简 API 客户端：所有请求都走同源 /api，错误统一抛出带 message 的 Error
const BASE = ''

async function request(path, { method = 'GET', body, form, signal } = {}) {
  const opts = { method, headers: {}, signal }
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  if (form) opts.body = form
  const res = await fetch(BASE + path, opts)
  const text = await res.text()
  let data = text
  try { data = text ? JSON.parse(text) : null } catch (e) { /* 非 JSON */ }
  if (!res.ok) {
    let msg = data
    if (data && typeof data === 'object') msg = data.error || data.detail || JSON.stringify(data)
    if (Array.isArray(msg)) msg = msg.map(m => m.msg || JSON.stringify(m)).join('; ')
    const err = new Error(String(msg || res.statusText))
    err.status = res.status
    throw err
  }
  return data
}

const qs = (params = {}) => {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.set(k, v)
  }
  const s = p.toString()
  return s ? `?${s}` : ''
}

const P = (pid) => `/api/projects/${encodeURIComponent(pid)}`

export const api = {
  meta: () => request('/api/meta'),
  catalog: () => request('/api/catalog'),
  envs: () => request('/api/envs'),
  probeEnv: (name) => request('/api/envs/probe', { method: 'POST', body: { name: name || null } }),

  projects: () => request('/api/projects'),
  createProject: (body) => request('/api/projects', { method: 'POST', body }),
  project: (pid) => request(P(pid)),
  updateProject: (pid, body) => request(P(pid), { method: 'PATCH', body }),
  deleteProject: (pid) => request(P(pid), { method: 'DELETE' }),
  runnerStart: (pid) => request(`${P(pid)}/runner/start`, { method: 'POST', body: {} }),
  runnerStop: (pid, body = {}) => request(`${P(pid)}/runner/stop`, { method: 'POST', body }),
  runnerStatus: (pid) => request(`${P(pid)}/runner`),

  datasets: (pid) => request(`${P(pid)}/datasets`),
  createDataset: (pid, body) => request(`${P(pid)}/datasets`, { method: 'POST', body }),
  importDataset: (pid, body) => request(`${P(pid)}/datasets/import`, { method: 'POST', body }),
  dataset: (pid, did, params) => request(`${P(pid)}/datasets/${did}${qs(params)}`),
  updateDataset: (pid, did, body) => request(`${P(pid)}/datasets/${did}`, { method: 'PATCH', body }),
  deleteDataset: (pid, did) => request(`${P(pid)}/datasets/${did}`, { method: 'DELETE' }),
  upload: (pid, did, files, { split = 'auto', keepRaw = true } = {}) => {
    const form = new FormData()
    for (const f of files) form.append('files', f)
    form.append('split', split)
    form.append('keep_raw', String(keepRaw))
    return request(`${P(pid)}/datasets/${did}/upload`, { method: 'POST', form })
  },
  recomputeDataset: (pid, did) => request(`${P(pid)}/datasets/${did}/recompute`, { method: 'POST' }),
  addClasses: (pid, did, add) => request(`${P(pid)}/datasets/${did}/classes`, { method: 'POST', body: { add } }),
  labels: (pid, did, name) => request(`${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}/labels`),
  saveLabels: (pid, did, name, items) => request(`${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}/labels`, { method: 'PUT', body: { items } }),
  setSplit: (pid, did, name, split) => request(`${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}/split`, { method: 'POST', body: { split } }),
  deleteImage: (pid, did, name) => request(`${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  runLabel: (pid, did, body) => request(`${P(pid)}/datasets/${did}/label`, { method: 'POST', body }),
  runTile: (pid, did, body) => request(`${P(pid)}/datasets/${did}/tile`, { method: 'POST', body }),
  runRelation: (pid, did, body) => request(`${P(pid)}/datasets/${did}/relate`, { method: 'POST', body }),

  jobs: (pid, params) => request(`${P(pid)}/jobs${qs(params)}`),
  job: (pid, jid) => request(`${P(pid)}/jobs/${jid}`),
  jobLog: (pid, jid, offset = 0) => request(`${P(pid)}/jobs/${jid}/log${qs({ offset })}`),
  cancelJob: (pid, jid) => request(`${P(pid)}/jobs/${jid}/cancel`, { method: 'POST' }),
  deleteJob: (pid, jid) => request(`${P(pid)}/jobs/${jid}`, { method: 'DELETE' }),

  vlm: () => request('/api/vlm'),
  vlmTest: (body) => request('/api/vlm/test', { method: 'POST', body }),

  evalPresets: (pid) => request(`${P(pid)}/eval-presets`),
  saveEvalPreset: (pid, body) => request(`${P(pid)}/eval-presets`, { method: 'POST', body }),
  deleteEvalPreset: (pid, id) => request(`${P(pid)}/eval-presets/${id}`, { method: 'DELETE' }),

  models: (pid) => request(`${P(pid)}/models`),
  uploadModel: (pid, file, { name = '', task = 'detect', classes = '' } = {}) => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    form.append('task', task)
    form.append('classes', classes)
    return request(`${P(pid)}/models/upload`, { method: 'POST', form })
  },
  renameModel: (pid, mid, body) => request(`${P(pid)}/models/${mid}`, { method: 'PATCH', body }),
  deleteModel: (pid, mid) => request(`${P(pid)}/models/${mid}`, { method: 'DELETE' }),

  trainings: (pid) => request(`${P(pid)}/trainings`),
  training: (pid, tid) => request(`${P(pid)}/trainings/${tid}`),
  startTraining: (pid, body) => request(`${P(pid)}/trainings`, { method: 'POST', body }),
  deleteTraining: (pid, tid) => request(`${P(pid)}/trainings/${tid}`, { method: 'DELETE' }),

  evaluations: (pid) => request(`${P(pid)}/evaluations`),
  evaluation: (pid, eid) => request(`${P(pid)}/evaluations/${eid}`),
  startEvaluation: (pid, body) => request(`${P(pid)}/evaluations`, { method: 'POST', body }),
  deleteEvaluation: (pid, eid) => request(`${P(pid)}/evaluations/${eid}`, { method: 'DELETE' }),
}

export const urls = {
  image: (pid, did, name) => `${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}`,
  preview: (pid, did, name) => `${P(pid)}/datasets/${did}/images/${encodeURIComponent(name)}/preview`,
  file: (pid, path) => `${P(pid)}/file?path=${encodeURIComponent(path)}`,
  modelFile: (pid, mid) => `${P(pid)}/models/${mid}/download`,
}
