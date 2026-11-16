# Web 前端（Vue 3 + Vite）

后端在 `src/mlplatform/`，本目录只是**源码**；构建产物写到 `web/dist/`，由 FastAPI 直接托管。
日常运行不需要 node（`start.sh` 只启动 Python 服务）。

## 改界面

```bash
cd web
pnpm install          # 首次；本机 pnpm 11 需要 CI=true 或 --config.strictDepBuilds=false
pnpm build            # 产物 -> web/dist，刷新页面即生效
pnpm dev              # 开发模式（5173，/api 代理到 127.0.0.1:8600）
```

## 目录

```
web/
├── index.html
├── vite.config.js       # outDir = web/dist；dev 代理 /api -> :8600
└── src/
    ├── main.js
    ├── api.js           # 所有接口 + 图片/文件 URL
    ├── state.js         # 全局状态（项目选择、hash 路由、toast、格式化函数）
    ├── styles.css       # 设计系统（深色主题）
    ├── App.vue          # 侧边导航 + 顶栏（项目切换 / 调度器开关）+ 路由分发
    ├── components/
    │   ├── JobState.vue      状态徽章
    │   ├── LogPane.vue       增量拉取作业日志 + 进度条 + 取消
    │   ├── Modal.vue         弹层
    │   ├── TrendChart.vue    训练曲线（SVG）
    │   ├── ModelSelect.vue   模型选择（项目模型 / 预训练目录 / 自定义）
    │   ├── DatasetPicker.vue 数据集多选 + 逐图勾选
    │   └── LabelEditor.vue   canvas 标注编辑器（画框/拖动/关键点/删除）
    └── views/
        ├── Overview.vue        项目概览 + 最近作业 + 环境
        ├── Datasets.vue        数据集列表 + 上传 + 导入 + 裁减
        ├── DatasetDetail.vue   图片网格 + 标注编辑 + 裁减
        ├── Labeling.vue        自动标注表单 + 作业进度
        ├── Training.vue        训练表单（数据/模型/参数）+ 训练记录
        ├── TrainingDetail.vue  指标卡 + 曲线 + 产物 + 日志
        ├── Evaluation.vue      评估表单 + 记录
        ├── EvaluationDetail.vue 每类指标 + 曲线 + 预测画廊
        ├── Environments.vue    四个环境 + 探测 + 体检
        └── Jobs.vue            作业列表 + 日志 + 取消
```

## 约定

- 不引入路由库：`location.hash`（`#/datasets/<id>`）由 `state.syncRoute` 解析。
- 不引入 UI 库：`styles.css` 里的 `.panel / .stat / .chip / .grid-images / .log / .modal` 等类。
- 轮询而非 WebSocket：作业进度来自 `progress.json`，日志用 `offset` 增量拉取。
- 所有写操作都有 toast 反馈；接口错误会显示后端返回的 `error` 字段。
