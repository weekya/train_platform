#!/usr/bin/env bash
# ============================================================================
# stop.sh —— 停止平台的全部进程
#
# 停止范围：
#   1. 各项目的作业调度器（bin/tp project stop，默认会先取消正在跑的作业）
#   2. start.sh 起的 Web 服务（state/web.pid）
#   3. 残留的阶段 worker（mlplatform.stages）与调度器进程（mlplatform.runner）
#
#   ./stop.sh               优雅停止（SIGTERM，最多等 15 秒，再 SIGKILL）
#   ./stop.sh --force       直接 SIGKILL
#   ./stop.sh --grace 30    改优雅等待时间
#   ./stop.sh --dry-run     只列出会停哪些进程
#   ./stop.sh --web-only    只停 Web 服务，保留调度器和作业
#   ./stop.sh --keep-jobs   停调度器和 Web，但不取消正在运行的作业
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/state"
PIDFILE="$STATE/web.pid"
GRACE="${TP_STOP_GRACE:-15}"
FORCE=0
DRY=0
WEB_ONLY=0
KEEP_JOBS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)   FORCE=1; shift ;;
    --grace)      GRACE="${2:?}"; shift 2 ;;
    --grace=*)    GRACE="${1#*=}"; shift ;;
    --dry-run)    DRY=1; shift ;;
    --web-only)   WEB_ONLY=1; shift ;;
    --keep-jobs)  KEEP_JOBS=1; shift ;;
    -h|--help)    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[error] 未知参数: $1" >&2; exit 2 ;;
  esac
done

alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

# 保护：绝不杀自己和祖先进程
declare -A PROTECT=()
add_ancestors() {
  local pid=$1 guard=0
  while [[ -n "$pid" && "$pid" != "0" && "$guard" -lt 20 ]]; do
    PROTECT["$pid"]=1
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    guard=$((guard + 1))
  done
}
add_ancestors "$$"

inside_root() {  # <pid>
  local pid=$1 cwd cmd
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ -n "$cwd" && ( "$cwd" == "$ROOT" || "$cwd" == "$ROOT"/* ) ]] && return 0
  cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmd" == *"$ROOT/src/mlplatform"* || "$cmd" == *"$ROOT/bin/tp"* ]]
}

declare -A TARGET=()
add_target() {  # <pid> <label>
  local pid=$1 label=${2:-}
  [[ -z "$pid" || -n "${PROTECT[$pid]:-}" ]] && return 0
  alive "$pid" || return 0
  [[ -n "${TARGET[$pid]:-}" ]] && return 0
  TARGET["$pid"]="$label"
}

echo "== mlplatform 停止 =="

# 1) 项目调度器（默认先取消正在跑的作业）
if [[ "$WEB_ONLY" != "1" ]]; then
  shopt -s nullglob
  for meta in "$ROOT"/projects/*/project.yaml; do
    proj="$(basename "$(dirname "$meta")")"
    if [[ "$DRY" == "1" ]]; then
      echo "  [dry-run] 将停止项目调度器: $proj"
      continue
    fi
    if [[ "$KEEP_JOBS" == "1" ]]; then
      "$ROOT/bin/tp" project stop "$proj" --keep-jobs >/dev/null 2>&1 || true
    else
      "$ROOT/bin/tp" project stop "$proj" >/dev/null 2>&1 || true
    fi
  done
fi

# 2) Web 服务
if [[ -f "$PIDFILE" ]]; then
  WEB_PID="$(head -n1 "$PIDFILE" | tr -dc '0-9')"
  add_target "$WEB_PID" "web 服务"
fi

# 3) 残留进程（阶段 worker / 调度器 / 手动跑的 tp 命令）
if [[ "$WEB_ONLY" != "1" ]]; then
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    inside_root "$pid" && add_target "$pid" "$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)"
  done < <(pgrep -f 'mlplatform\.(stages|runner|cli)' 2>/dev/null | sort -n || true)
fi

if [[ ${#TARGET[@]} -eq 0 ]]; then
  echo "  [info] 没有发现运行中的平台进程"
  rm -f "$PIDFILE"
  exit 0
fi

echo "-- 待停止（${#TARGET[@]} 个）--"
for pid in "${!TARGET[@]}"; do
  printf '  %-8s %s\n' "$pid" "${TARGET[$pid]}"
done

if [[ "$DRY" == "1" ]]; then
  echo "[dry-run] 未执行任何 kill"
  exit 0
fi

signal_pid() {  # <pid> <sig>
  local pid=$1 sig=$2 pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [[ -n "$pgid" && "$pgid" == "$pid" && "$pgid" != "$$" ]]; then
    kill "-$sig" "-$pgid" 2>/dev/null || kill "-$sig" "$pid" 2>/dev/null || true
  else
    kill "-$sig" "$pid" 2>/dev/null || true
  fi
}

if [[ "$FORCE" == "1" ]]; then
  echo "-- SIGKILL --"
  for pid in "${!TARGET[@]}"; do signal_pid "$pid" KILL; done
else
  echo "-- SIGTERM（最多等 ${GRACE}s）--"
  for pid in "${!TARGET[@]}"; do signal_pid "$pid" TERM; done
  for _ in $(seq 1 "$GRACE"); do
    left=0
    for pid in "${!TARGET[@]}"; do alive "$pid" && left=$((left + 1)); done
    [[ "$left" -eq 0 ]] && break
    sleep 1
  done
  for pid in "${!TARGET[@]}"; do
    if alive "$pid"; then
      echo "  [kill -9] $pid ${TARGET[$pid]}"
      signal_pid "$pid" KILL
    fi
  done
fi

sleep 0.5
LEFT=()
for pid in "${!TARGET[@]}"; do alive "$pid" && LEFT+=("$pid"); done
rm -f "$PIDFILE"

if [[ ${#LEFT[@]} -gt 0 ]]; then
  echo "[warn] 仍有进程未退出: ${LEFT[*]}" >&2
  echo "       重试: ./stop.sh --force" >&2
  exit 1
fi
echo "  ✔ 已停止 ${#TARGET[@]} 个进程"
