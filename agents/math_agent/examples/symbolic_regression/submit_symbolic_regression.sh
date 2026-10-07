#!/usr/bin/env bash
# 在普通 Linux 服务器上运行 LoongFlow 符号回归。
#
# 最简单的运行方式：
#   bash agents/math_agent/examples/symbolic_regression/submit_symbolic_regression.sh \
#     --background --output-dir /path/to/output

set -Eeuo pipefail

# 作业不继承用户 shell 中设置的代理，使用主机默认网络设置。
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

readonly START_DIR="${PWD}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SR_LAUNCHER_PATH="${SCRIPT_DIR}/submit_symbolic_regression.sh"
readonly DEFAULT_LOONGFLOW_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
readonly ORIGINAL_ARGS=("$@")

BACKGROUND_WORKER="${LOONGFLOW_SR_BACKGROUND_WORKER:-0}"
unset LOONGFLOW_SR_BACKGROUND_WORKER

LOONGFLOW_ROOT="${LOONGFLOW_ROOT:-${DEFAULT_LOONGFLOW_ROOT}}"
VENV_DIR=""
CONFIG_FILE=""
INITIAL_FILE=""
EVAL_FILE=""
DATASET=""
OUTPUT_DIR=""
MAX_ITERATIONS="10"
TARGET_SCORE="0.995"
MAX_NFEV="120"
CHECKPOINT_PATH=""
TASK_FILE=""
PLANNER=""
EXECUTOR=""
SUMMARY=""
LOG_LEVEL="INFO"
DRY_RUN=0
BACKGROUND=0
EXTRA_ARGS=()
RUNTIME_CONFIG=""
VISUALIZER_PID=""

usage() {
  cat <<'EOF'
用法：
  bash submit_symbolic_regression.sh [任务参数]

常用任务参数：
  --dataset PATH          数据集 .npz 文件。
  --output-dir PATH       本次运行的输出目录；应为一个新的目录。
  --max-iterations N      LoongFlow 最大迭代次数，默认 200。
  --target-score FLOAT    提前停止的目标分数，范围 [0, 1]，默认 0.995。
  --max-nfev N            每个候选公式拟合常数时的最大函数评估次数，默认 120。
  --checkpoint-path PATH  从已有 LoongFlow 数据库断点继续运行。
  --task-file PATH        用文本文件替换 config 中的任务提示词。

文件和环境参数：
  --config PATH           基础 task_config.yaml。
  --initial-file PATH     初始候选程序。
  --eval-file PATH        评估程序。
  --loongflow-root PATH   LoongFlow 仓库目录。
  --venv PATH             Python 虚拟环境目录，默认 <loongflow-root>/.venv。

高级参数：
  --background            在独立后台会话中运行，SSH 断开后继续执行。
  --planner NAME          覆盖 planner 名称。
  --executor NAME         覆盖 executor 名称。
  --summary NAME          覆盖 summary 名称。
  --log-level LEVEL       DEBUG、INFO、WARNING、ERROR 或 CRITICAL，默认 INFO。
  --dry-run               只检查参数并显示命令，不启动 LoongFlow。
  -h, --help              显示本帮助。
  --                      后面的参数原样传给 math_evolve_agent.py。

默认数据集：
  <loongflow-root>/datasets/disk_n6_Q18_kapa1/dataset_uniform_4096.npz
  若不存在，则自动尝试 <loongflow-root>/../datasets/ 下的同一路径。

默认输出和可视化：
  输出写入 <loongflow-root>/../output。
  作业运行期间可通过 http://<主机>:8888 查看实时可视化结果。
EOF
}

die() {
  echo "错误：$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2}" ]] || die "参数 ${1} 缺少值"
}

absolute_from_start_dir() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${START_DIR}" "${path}"
  fi
}

show_command() {
  printf '运行命令：'
  printf ' %q' "$@"
  printf '\n'
}

cleanup() {
  if [[ -n "${VISUALIZER_PID}" ]]; then
    kill "${VISUALIZER_PID}" 2>/dev/null || true
    wait "${VISUALIZER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RUNTIME_CONFIG}" && -f "${RUNTIME_CONFIG}" ]]; then
    rm -f -- "${RUNTIME_CONFIG}"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
if [[ "${BACKGROUND_WORKER}" == "1" ]]; then
  trap '' HUP
else
  trap 'exit 143' HUP
fi
trap 'exit 143' TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      need_value "$@"
      DATASET="$2"
      shift 2
      ;;
    --output-dir)
      need_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-iterations)
      need_value "$@"
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --target-score)
      need_value "$@"
      TARGET_SCORE="$2"
      shift 2
      ;;
    --max-nfev)
      need_value "$@"
      MAX_NFEV="$2"
      shift 2
      ;;
    --checkpoint-path)
      need_value "$@"
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --task-file)
      need_value "$@"
      TASK_FILE="$2"
      shift 2
      ;;
    --config)
      need_value "$@"
      CONFIG_FILE="$2"
      shift 2
      ;;
    --initial-file)
      need_value "$@"
      INITIAL_FILE="$2"
      shift 2
      ;;
    --eval-file)
      need_value "$@"
      EVAL_FILE="$2"
      shift 2
      ;;
    --loongflow-root)
      need_value "$@"
      LOONGFLOW_ROOT="$2"
      shift 2
      ;;
    --venv)
      need_value "$@"
      VENV_DIR="$2"
      shift 2
      ;;
    --planner)
      need_value "$@"
      PLANNER="$2"
      shift 2
      ;;
    --executor)
      need_value "$@"
      EXECUTOR="$2"
      shift 2
      ;;
    --summary)
      need_value "$@"
      SUMMARY="$2"
      shift 2
      ;;
    --log-level)
      need_value "$@"
      LOG_LEVEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --background)
      BACKGROUND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die "未知参数：$1（使用 --help 查看帮助）"
      ;;
  esac
done

LOONGFLOW_ROOT="$(absolute_from_start_dir "${LOONGFLOW_ROOT}")"
readonly TASK_DIR="${LOONGFLOW_ROOT}/agents/math_agent/examples/symbolic_regression"

if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${LOONGFLOW_ROOT}/.venv"
else
  VENV_DIR="$(absolute_from_start_dir "${VENV_DIR}")"
fi
if [[ -z "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="${TASK_DIR}/task_config.yaml"
else
  CONFIG_FILE="$(absolute_from_start_dir "${CONFIG_FILE}")"
fi
if [[ -z "${INITIAL_FILE}" ]]; then
  INITIAL_FILE="${TASK_DIR}/initial_program.py"
else
  INITIAL_FILE="$(absolute_from_start_dir "${INITIAL_FILE}")"
fi
if [[ -z "${EVAL_FILE}" ]]; then
  EVAL_FILE="${TASK_DIR}/eval_program.py"
else
  EVAL_FILE="$(absolute_from_start_dir "${EVAL_FILE}")"
fi
if [[ -z "${DATASET}" ]]; then
  DATASET="${LOONGFLOW_ROOT}/datasets/disk_n6_Q18_kapa1/dataset_uniform_4096.npz"
  if [[ ! -f "${DATASET}" ]]; then
    DATASET="${LOONGFLOW_ROOT}/../datasets/disk_n6_Q18_kapa1/dataset_uniform_4096.npz"
  fi
else
  DATASET="$(absolute_from_start_dir "${DATASET}")"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="$(dirname -- "${LOONGFLOW_ROOT%/}")/output"
else
  OUTPUT_DIR="$(absolute_from_start_dir "${OUTPUT_DIR}")"
fi
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(absolute_from_start_dir "${CHECKPOINT_PATH}")"
fi
if [[ -n "${TASK_FILE}" ]]; then
  TASK_FILE="$(absolute_from_start_dir "${TASK_FILE}")"
fi

[[ -d "${LOONGFLOW_ROOT}" ]] || die "LoongFlow 目录不存在：${LOONGFLOW_ROOT}"
[[ -f "${LOONGFLOW_ROOT}/agents/math_agent/math_evolve_agent.py" ]] || \
  die "找不到 math_evolve_agent.py：${LOONGFLOW_ROOT}"
[[ -f "${VENV_DIR}/bin/activate" ]] || die "虚拟环境不存在：${VENV_DIR}"
[[ -f "${CONFIG_FILE}" ]] || die "配置文件不存在：${CONFIG_FILE}"
[[ -f "${INITIAL_FILE}" ]] || die "初始程序不存在：${INITIAL_FILE}"
[[ -f "${EVAL_FILE}" ]] || die "评估程序不存在：${EVAL_FILE}"
[[ -f "${DATASET}" ]] || die "数据集不存在：${DATASET}"
[[ -r "${DATASET}" ]] || die "数据集不可读：${DATASET}"
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "--max-iterations 必须是正整数"
[[ "${MAX_NFEV}" =~ ^[1-9][0-9]*$ ]] || die "--max-nfev 必须是正整数"
if [[ -n "${CHECKPOINT_PATH}" && ! -d "${CHECKPOINT_PATH}" ]]; then
  die "断点目录不存在：${CHECKPOINT_PATH}"
fi
if [[ -n "${TASK_FILE}" && ! -f "${TASK_FILE}" ]]; then
  die "任务提示词文件不存在：${TASK_FILE}"
fi
case "${LOG_LEVEL}" in
  DEBUG|INFO|WARNING|ERROR|CRITICAL) ;;
  *) die "--log-level 必须是 DEBUG、INFO、WARNING、ERROR 或 CRITICAL" ;;
esac

# 使用 LoongFlow 自带的虚拟环境。
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
command -v python >/dev/null 2>&1 || die "虚拟环境中找不到 python"

python - "${TARGET_SCORE}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("错误：--target-score 必须是数字") from exc
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit("错误：--target-score 必须在 [0, 1] 范围内")
PY

readonly CHECKPOINT_DIR="${OUTPUT_DIR}/database/checkpoints"
readonly VISUALIZER_LOG="${OUTPUT_DIR}/visualizer.log"
readonly RUN_LOG="${OUTPUT_DIR}/run.log"
readonly RUN_PID="${OUTPUT_DIR}/run.pid"
mkdir -p -- "${CHECKPOINT_DIR}"

if [[ "${BACKGROUND}" == "1" && "${BACKGROUND_WORKER}" != "1" && "${DRY_RUN}" != "1" ]]; then
  command -v nohup >/dev/null 2>&1 || die "后台运行需要 nohup"
  command -v setsid >/dev/null 2>&1 || die "后台运行需要 setsid"
  nohup setsid env LOONGFLOW_SR_BACKGROUND_WORKER=1 \
    bash "${SR_LAUNCHER_PATH}" "${ORIGINAL_ARGS[@]}" \
    </dev/null >"${RUN_LOG}" 2>&1 &
  background_pid="$!"
  printf '%s\n' "${background_pid}" >"${RUN_PID}"
  cat <<EOF
后台任务已启动，SSH 或 VS Code 断开后仍会继续运行。
  PID： ${background_pid}
  日志：${RUN_LOG}
  查看：tail -f "${RUN_LOG}"
EOF
  exit 0
fi

# task_config.yaml 中含有 API key。临时配置仅写入系统临时目录，
# 权限设为 600，并在任务退出时删除，避免在输出目录中复制密钥。
umask 077
readonly TMP_BASE="${TMPDIR:-/tmp}"
RUNTIME_CONFIG="$(mktemp "${TMP_BASE}/loongflow-sr.XXXXXX.yaml")"

export LOONGFLOW_BASE_CONFIG="${CONFIG_FILE}"
export LOONGFLOW_RUNTIME_CONFIG="${RUNTIME_CONFIG}"
export LOONGFLOW_RUN_OUTPUT="${OUTPUT_DIR}"
python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["LOONGFLOW_BASE_CONFIG"])
destination = Path(os.environ["LOONGFLOW_RUNTIME_CONFIG"])
output_dir = str(Path(os.environ["LOONGFLOW_RUN_OUTPUT"]))

with source.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise SystemExit(f"错误：配置文件不是 YAML 映射：{source}")

config["workspace_path"] = output_dir

# 清空可能存在的子目录覆盖值，让 LoongFlow 将日志、数据库和评估结果
# 全部放到本次 --output-dir 下。基础配置文件本身不会被修改。
evolve = config.setdefault("evolve", {})
evolve["workspace_path"] = None
evaluator = evolve.setdefault("evaluator", {})
evaluator["workspace_path"] = None
database = evolve.setdefault("database", {})
database["output_path"] = None
logger = config.setdefault("logger", {})
logger["log_path"] = None

with destination.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
destination.chmod(0o600)
PY

export PYTHONUNBUFFERED=1
export PYTHONPATH="${LOONGFLOW_ROOT}/src:${LOONGFLOW_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LOONGFLOW_SR_DATASET="${DATASET}"
export LOONGFLOW_SR_MAX_NFEV="${MAX_NFEV}"
export OMP_NUM_THREADS="${LOONGFLOW_SR_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${LOONGFLOW_SR_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${LOONGFLOW_SR_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${LOONGFLOW_SR_NUM_THREADS:-8}"

COMMAND=(
  "python"
  "agents/math_agent/math_evolve_agent.py"
  "--config" "${RUNTIME_CONFIG}"
  "--initial-file" "${INITIAL_FILE}"
  "--eval-file" "${EVAL_FILE}"
  "--max-iterations" "${MAX_ITERATIONS}"
  "--target-score" "${TARGET_SCORE}"
  "--log-level" "${LOG_LEVEL}"
)
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  COMMAND+=("--checkpoint-path" "${CHECKPOINT_PATH}")
fi
if [[ -n "${TASK_FILE}" ]]; then
  COMMAND+=("--task-file" "${TASK_FILE}")
fi
if [[ -n "${PLANNER}" ]]; then
  COMMAND+=("--planner" "${PLANNER}")
fi
if [[ -n "${EXECUTOR}" ]]; then
  COMMAND+=("--executor" "${EXECUTOR}")
fi
if [[ -n "${SUMMARY}" ]]; then
  COMMAND+=("--summary" "${SUMMARY}")
fi
COMMAND+=("${EXTRA_ARGS[@]}")

VISUALIZER_COMMAND=(
  "python"
  "agents/math_agent/visualizer/visualizer.py"
  "--port" "8888"
  "--checkpoint-path" "${CHECKPOINT_DIR}"
)

cat <<EOF
LoongFlow 符号回归任务
  主机：        $(hostname)
  数据集：      ${DATASET}
  输出目录：    ${OUTPUT_DIR}
  最大迭代数：  ${MAX_ITERATIONS}
  目标分数：    ${TARGET_SCORE}
  拟合 max_nfev：${MAX_NFEV}
  CPU 线程数：  ${LOONGFLOW_SR_NUM_THREADS:-8}
  可视化地址：  http://$(hostname):8888
EOF

cd "${LOONGFLOW_ROOT}"
show_command "${COMMAND[@]}"
show_command "${VISUALIZER_COMMAND[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "参数检查通过；--dry-run 未启动 LoongFlow。"
  exit 0
fi

"${VISUALIZER_COMMAND[@]}" >"${VISUALIZER_LOG}" 2>&1 &
VISUALIZER_PID="$!"
echo "可视化服务已启动；日志：${VISUALIZER_LOG}"

"${COMMAND[@]}"
