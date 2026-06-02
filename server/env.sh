# 登录服务器后执行: source /data1/ml-2-3/code/handwritten-to-data/env.sh
# 请按实际路径修改 ML_ROOT

export ML_ROOT="/data1/ml-2-3"

# 缓存与临时文件全部放数据盘
export HF_HOME="${ML_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${ML_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${ML_ROOT}/cache/huggingface"
export TORCH_HOME="${ML_ROOT}/cache/torch"
export XDG_CACHE_HOME="${ML_ROOT}/cache/xdg"
export TMPDIR="${ML_ROOT}/cache/tmp"
export PIP_CACHE_DIR="${ML_ROOT}/cache/pip"

# 离线模式（服务器无外网时必须）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 只用一张卡（助教要求每组单卡）
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${ML_ROOT}/models" "${ML_ROOT}/output"
