#!/usr/bin/env bash
# 建阶段环境 label_system（由 main_system 克隆；独立可装包）
#   bash envs/recipes/label_system.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$HERE/envs/recipes/conda_env.sh" "label_system"
