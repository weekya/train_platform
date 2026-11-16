#!/usr/bin/env bash
# ============================================================================
# start.sh —— 启动整个平台（在 main_system 环境里跑 Web 服务 + 各项目调度器）
#
#   ./start.sh                       默认 http://127.0.0.1:8600/
#   ./start.sh --port 9000           换端口
#   ./start.sh --host 0.0.0.0 --token secret   对外监听（务必带 token）
#   ./start.sh --foreground          前台运行（Ctrl-C 退出）
#   ./start.sh --no-wait             不等健康检查
#   ./start.sh --no-runners          只起 Web，不自动拉起项目调度器
#
# 运行时文件：state/web.pid；日志统一在 running_time/log/（可整目录删除）
#   running_time/log/web/web-<pid>.log    结构化日志（级别/轮转，见 docs/LOGGING.md）
#   running_time/log/startup.log          start.sh 的启动输出 + logger 就绪前的崩溃
# 停止：./stop.sh
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/state"
LOGDIR="$ROOT/running_time/log"
PIDFILE="$STATE/web.pid"
# 启动输出（含 logger 就绪前的报错）单独一份，和结构化日志放同一个目录
LOGFILE="$LOGDIR/startup.log"
mkdir -p "$STATE" "$LOGDIR"

PY_REL=""
[[ -f "$ROOT/envs/.default_python" ]] && PY_REL="$(head -n1 "$ROOT/envs/.default_python" | tr -d '[:space:]')"
PY="$ROOT/$PY_REL"
if [[ -z "$PY_REL" || ! -x "$PY" ]]; then
  echo "[error] 找不到项目内环境解释器：envs/.default_python = ${PY_REL:-<空>}" >&2
  echo "        先建环境：bash envs/recipes/main_system.sh" >&2
  exit 2
fi

cfg() {  # cfg <dotted.key> -> value
  PYTHONNOUSERSITE=1 "$PY" -c "import sys; sys.path.insert(0,'$ROOT/src')
from mlplatform import config
value = config.get('$1', '')
print('' if value is None else value)" 2>/dev/null || true
}

HOST="${TP_WEB_HOST:-$(cfg web.host)}"; HOST="${HOST:-127.0.0.1}"
PORT="${TP_WEB_PORT:-$(cfg web.port)}"; PORT="${PORT:-8600}"
TIMEOUT="${TP_WEB_START_TIMEOUT:-$(cfg web.start_timeout)}"; TIMEOUT="${TIMEOUT:-90}"
TOKEN="${TP_WEB_TOKEN:-$(cfg web.token)}"
FOREGROUND=0
WAIT=1
RUNNERS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        HOST="${2:?}"; shift 2 ;;
    --host=*)      HOST="${1#*=}"; shift ;;
    --port)        PORT="${2:?}"; shift 2 ;;
    --port=*)      PORT="${1#*=}"; shift ;;
    --token)       TOKEN="${2:?}"; shift 2 ;;
    --token=*)     TOKEN="${1#*=}"; shift ;;
    --foreground)  FOREGROUND=1; shift ;;
    --no-wait)     WAIT=0; shift ;;
    --no-runners)  RUNNERS=0; shift ;;
    -h|--help)     sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[error] 未知参数: $1" >&2; exit 2 ;;
  esac
done

PROBE_HOST="$HOST"
case "$HOST" in 0.0.0.0|::|"[::]") PROBE_HOST="127.0.0.1" ;; esac
BASE_URL="http://$PROBE_HOST:$PORT"
SHOW_URL="http://$HOST:$PORT/"

alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

# 依赖自检（只检查 Web 依赖，训练环境由各自阶段检查）
if ! PYTHONNOUSERSITE=1 "$PY" -c "import fastapi, uvicorn, multipart" >/dev/null 2>&1; then
  echo "[error] main_system 环境缺少 Web 依赖（fastapi/uvicorn/python-multipart）" >&2
  echo "        修复： PYTHONNOUSERSITE=1 $PY -m pip install -U fastapi uvicorn python-multipart" >&2
  exit 2
fi

if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  echo "[warn] 前端尚未构建（web/dist/index.html 缺失），先显示后端提示页。" >&2
  echo "       构建： cd $ROOT/web && pnpm install && pnpm build" >&2
fi

start_runners() {
  [[ "$RUNNERS" == "1" ]] || return 0
  shopt -s nullglob
  local started=0
  for meta in "$ROOT"/projects/*/project.yaml; do
    local pid_dir
    pid_dir="$(dirname "$meta")"
    local pid
    pid="$(basename "$pid_dir")"
    if "$ROOT/bin/tp" project start "$pid" --no-web >/dev/null 2>&1; then
      started=$((started + 1))
    fi
  done
  [[ "$started" -gt 0 ]] && echo "  调度器  : 已为 $started 个项目启动作业调度器"
}

# 已在运行：直接报告（顺便补齐调度器）
if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(head -n1 "$PIDFILE" | tr -dc '0-9')"
  if alive "$OLD_PID"; then
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "$BASE_URL/api/ping" >/dev/null 2>&1; then
      echo "[skip] 服务已在运行（pid=$OLD_PID）： $SHOW_URL"
      start_runners
      echo "       日志: $LOGFILE   停止: ./stop.sh"
      exit 0
    fi
  fi
  rm -f "$PIDFILE"
fi

ARGS=(web --host "$HOST" --port "$PORT" --quiet)
[[ -n "$TOKEN" ]] && ARGS+=(--token "$TOKEN")

if [[ "$FOREGROUND" == "1" ]]; then
  start_runners
  echo "== mlplatform（前台）==  $SHOW_URL"
  exec "$ROOT/bin/tp" "${ARGS[@]}"
fi

{
  printf '\n===== start.sh %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
} >>"$LOGFILE"

MLPLATFORM_ROOT="$ROOT" nohup "$ROOT/bin/tp" "${ARGS[@]}" >>"$LOGFILE" 2>&1 &
PID=$!
echo "$PID" >"$PIDFILE"

echo "== mlplatform =="
echo "  进程    : pid=$PID"
echo "  URL     : $SHOW_URL"
echo "  API 文档: ${BASE_URL}/api/docs"
echo "  环境    : main_system（$PY_REL）"
echo "  日志    : running_time/log/（Web: running_time/log/web/web-<pid>.log）"
echo "  停止    : ./stop.sh"

if [[ "$WAIT" != "1" ]]; then
  start_runners
  exit 0
fi

printf '  等待就绪'
READY=0
for _ in $(seq 1 $((TIMEOUT * 2))); do
  if ! alive "$PID"; then
    echo
    echo "[error] 启动失败（进程已退出）。最近 30 行日志：" >&2
    tail -n 30 "$LOGFILE" >&2 || true
    rm -f "$PIDFILE"
    exit 1
  fi
  if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "$BASE_URL/api/ping" >/dev/null 2>&1; then
    READY=1; break
  fi
  printf '.'; sleep 0.5
done
echo

if [[ "$READY" == "1" ]]; then
  start_runners
  echo "  ✔ 已就绪：在浏览器打开 $SHOW_URL"
  exit 0
fi

echo "[warn] ${TIMEOUT}s 内没等到健康检查（端口被占用？日志见 running_time/log/startup.log）" >&2
tail -n 20 "$LOGFILE" >&2 || true
exit 1
