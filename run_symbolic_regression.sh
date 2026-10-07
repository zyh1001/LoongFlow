#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="modified_symbolic_regression"
TASK_DIR="${ROOT}/agents/math_agent/examples/${TASK_NAME}"

CONFIG="${TASK_DIR}/task_config.yaml"
DATASET="${ROOT}/datasets/disk_n2_Q6_kapa_100/dataset_uniform.npz"
OUTPUT_DIR="${ROOT}/output/n2_Q6_kapa100_iter6"
MAX_ITERATIONS="5"
TARGET_SCORE="0.995"
MAX_NFEV="120"
LOG_LEVEL="INFO"
CHECKPOINT=""
VISUALIZER=1
VISUALIZER_HOST="127.0.0.1"
VISUALIZER_PORT="8888"
BACKGROUND=0
ACTION="start"
EXTRA_ARGS=()
ORIGINAL_ARGS=("$@")

BACKGROUND_WORKER="${LOONGFLOW_SR_BACKGROUND_WORKER:-0}"
unset LOONGFLOW_SR_BACKGROUND_WORKER

usage() {
  cat <<'EOF'
用法：
  ./run_symbolic_regression.sh [选项] [-- LoongFlow原生参数...]
  ./run_symbolic_regression.sh stop [--output-dir DIR]

选项：
  --config FILE             配置文件
  --dataset FILE            符号回归数据集
  --output-dir DIR          输出目录
  --max-iterations N        最大迭代数，默认 5
  --target-score FLOAT      提前停止分数，默认 0.995
  --max-nfev N              每个候选的参数拟合次数，默认 120
  --checkpoint-path DIR     从 checkpoint 恢复
  --log-level LEVEL         DEBUG/INFO/WARNING/ERROR/CRITICAL
  --visualizer-host HOST    visualizer 监听地址，默认 127.0.0.1
  --visualizer-port PORT    visualizer 端口，默认 8888
  --no-visualize            不启动 visualizer
  --background              后台运行
  -h, --help                显示帮助

示例：
  ./run_symbolic_regression.sh --max-iterations 20
  ./run_symbolic_regression.sh --background --max-iterations 200
  ./run_symbolic_regression.sh -- --executor evolve_executor_fuse
EOF
}

die() {
  echo "错误：$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 缺少参数值"
}

absolute_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT" "$path"
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  ACTION="stop"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --dataset) need_value "$@"; DATASET="$2"; shift 2 ;;
    --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --max-iterations) need_value "$@"; MAX_ITERATIONS="$2"; shift 2 ;;
    --target-score) need_value "$@"; TARGET_SCORE="$2"; shift 2 ;;
    --max-nfev) need_value "$@"; MAX_NFEV="$2"; shift 2 ;;
    --checkpoint-path) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --log-level) need_value "$@"; LOG_LEVEL="$2"; shift 2 ;;
    --visualizer-host) need_value "$@"; VISUALIZER_HOST="$2"; shift 2 ;;
    --visualizer-port) need_value "$@"; VISUALIZER_PORT="$2"; shift 2 ;;
    --no-visualize) VISUALIZER=0; shift ;;
    --background) BACKGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "未知参数：$1（LoongFlow 原生参数请放在 -- 后）" ;;
  esac
done

CONFIG="$(absolute_path "$CONFIG")"
DATASET="$(absolute_path "$DATASET")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
[[ -z "$CHECKPOINT" ]] || CHECKPOINT="$(absolute_path "$CHECKPOINT")"

PID_FILE="${OUTPUT_DIR}/run.pid"
RUN_LOG="${OUTPUT_DIR}/run.log"
VISUALIZER_LOG="${OUTPUT_DIR}/visualizer.log"

if [[ "$ACTION" == "stop" ]]; then
  [[ -f "$PID_FILE" ]] || die "找不到 PID 文件：${PID_FILE}"
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || die "PID 文件内容无效：${PID_FILE}"
  if kill -0 "$pid" 2>/dev/null; then
    kill -- "-$pid" 2>/dev/null || kill "$pid"
    echo "已停止符号回归任务（PID ${pid}）。"
  else
    echo "任务已经停止；清理过期 PID 文件。"
  fi
  rm -f -- "$PID_FILE"
  exit 0
fi

[[ -x "${ROOT}/run_math.sh" ]] || die "找不到可执行的 run_math.sh"
[[ -x "${ROOT}/.venv/bin/python" ]] || die "找不到 .venv/bin/python"
[[ -f "$CONFIG" ]] || die "配置文件不存在：${CONFIG}"
[[ -f "$DATASET" ]] || die "数据集不存在：${DATASET}"
[[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "--max-iterations 必须是正整数"
[[ "$MAX_NFEV" =~ ^[1-9][0-9]*$ ]] || die "--max-nfev 必须是正整数"
[[ "$VISUALIZER_PORT" =~ ^[1-9][0-9]*$ ]] || die "--visualizer-port 必须是正整数"
[[ -z "$CHECKPOINT" || -d "$CHECKPOINT" ]] || die "checkpoint 不存在：${CHECKPOINT}"
case "$LOG_LEVEL" in
  DEBUG|INFO|WARNING|ERROR|CRITICAL) ;;
  *) die "无效的 --log-level：${LOG_LEVEL}" ;;
esac

mkdir -p -- "${OUTPUT_DIR}/database/checkpoints"

if [[ "$BACKGROUND" == "1" && "$BACKGROUND_WORKER" != "1" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(<"$PID_FILE")"
    if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "任务已在运行（PID ${old_pid}）"
    fi
    rm -f -- "$PID_FILE"
  fi

  nohup setsid env LOONGFLOW_SR_BACKGROUND_WORKER=1 \
    bash "$0" "${ORIGINAL_ARGS[@]}" \
    </dev/null >"$RUN_LOG" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  echo "后台任务已启动："
  echo "  PID：${pid}"
  echo "  日志：${RUN_LOG}"
  echo "  可视化：http://${VISUALIZER_HOST}:${VISUALIZER_PORT}"
  echo "  停止：./run_symbolic_regression.sh stop --output-dir '${OUTPUT_DIR}'"
  exit 0
fi

RUNTIME_CONFIG=""
VISUALIZER_PID=""
cleanup() {
  if [[ -n "$VISUALIZER_PID" ]]; then
    kill "$VISUALIZER_PID" 2>/dev/null || true
    wait "$VISUALIZER_PID" 2>/dev/null || true
  fi
  [[ -z "$RUNTIME_CONFIG" ]] || rm -f -- "$RUNTIME_CONFIG"
  if [[ "$BACKGROUND_WORKER" == "1" && -f "$PID_FILE" ]]; then
    [[ "$(<"$PID_FILE")" != "$$" ]] || rm -f -- "$PID_FILE"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# 保持 runtime config 与原配置处于同一目录，确保多岛 seed 的相对路径有效。
umask 077
RUNTIME_CONFIG="$(mktemp "$(dirname -- "$CONFIG")/.task_config.runtime.XXXXXX.yaml")"
if grep -Fq '${DEEPSEEK_API_KEY}' "$CONFIG"; then
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || die "请先设置 DEEPSEEK_API_KEY"
  command -v envsubst >/dev/null 2>&1 || die "找不到 envsubst"
  envsubst '${DEEPSEEK_API_KEY}' <"$CONFIG" >"$RUNTIME_CONFIG"
else
  cp -- "$CONFIG" "$RUNTIME_CONFIG"
fi

"${ROOT}/.venv/bin/python" - "$RUNTIME_CONFIG" "$OUTPUT_DIR" <<'PY_RUNTIME_CONFIG'
import sys

import yaml

config_path, output_dir = sys.argv[1:]
with open(config_path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config["workspace_path"] = output_dir
with open(config_path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
PY_RUNTIME_CONFIG

export PATH="${ROOT}/.venv/bin:${PATH}"
export LOONGFLOW_SR_DATASET="$DATASET"
export LOONGFLOW_SR_MAX_NFEV="$MAX_NFEV"
export LOONGFLOW_SR_PARETO_ARCHIVE="${OUTPUT_DIR}/pareto_archive.json"
export PYTHONUNBUFFERED=1

if [[ "$BACKGROUND_WORKER" == "1" ]]; then
  printf '%s\n' "$$" >"$PID_FILE"
fi

if [[ "$VISUALIZER" == "1" ]]; then
  "${ROOT}/.venv/bin/python" \
    "${ROOT}/agents/math_agent/visualizer/visualizer.py" \
    --checkpoint-path "${OUTPUT_DIR}/database/checkpoints" \
    --host "$VISUALIZER_HOST" \
    --port "$VISUALIZER_PORT" \
    >"$VISUALIZER_LOG" 2>&1 &
  VISUALIZER_PID="$!"
  echo "Visualizer：http://${VISUALIZER_HOST}:${VISUALIZER_PORT}"
  echo "Visualizer 日志：${VISUALIZER_LOG}"
fi

COMMAND=(
  "${ROOT}/run_math.sh" "$TASK_NAME"
  --config "$RUNTIME_CONFIG"
  --workspace-path "$OUTPUT_DIR"
  --max-iterations "$MAX_ITERATIONS"
  --target-score "$TARGET_SCORE"
  --log-level "$LOG_LEVEL"
)
[[ -z "$CHECKPOINT" ]] || COMMAND+=(--checkpoint-path "$CHECKPOINT")
COMMAND+=("${EXTRA_ARGS[@]}")

echo "输出目录：${OUTPUT_DIR}"
echo "数据集：${DATASET}"
printf '运行命令：'; printf ' %q' "${COMMAND[@]}"; printf '\n'

cd "$ROOT"
"${COMMAND[@]}"
