#!/bin/bash
# ============================================
# 模型评估模板
# 
# 使用方法：
#   1. 复制此文件到 experiments/exp_NNN_name/run_eval.sh
#   2. 修改下方 CONFIG 区域的参数
#   3. 运行: ./run_eval.sh
#
# ⚠️ 严禁直接修改此模板文件！
# ============================================

set -e

# ============================================
# CONFIG - 修改此区域的参数
# ============================================

EXP_NAME="exp_NNN_name"           # 实验名称
POLICY_TYPE="diffusion"           # 策略类型: diffusion 或 act
N_EPISODES=50                     # 评估 episode 数

# ============================================
# 以下内容无需修改
# ============================================

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }

# 设置环境
cd /home/james/ai_projects/lerobot
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

# 修复依赖
pip install pymunk==6.4.0 -q 2>/dev/null || true

MODEL_DIR="experiments/${EXP_NAME}"

# 检查模型是否存在
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    error "模型不存在: ${MODEL_DIR}/model.safetensors"
fi

log "============================================"
log "📊 模型评估"
log "============================================"
log "📝 实验: ${EXP_NAME}"
log "📁 模型: ${MODEL_DIR}"
log "🎮 Episodes: ${N_EPISODES}"
log "============================================"

# 运行评估
python scripts/eval_model.py \
    --model_path "${MODEL_DIR}" \
    --policy_type "${POLICY_TYPE}" \
    --n_episodes ${N_EPISODES} \
    --output "${MODEL_DIR}/eval_result.json" \
    2>&1 | tee "${MODEL_DIR}/eval.log"

log "✅ 评估完成"
log "📁 结果: ${MODEL_DIR}/eval_result.json"

# 显示结果摘要
if [ -f "${MODEL_DIR}/eval_result.json" ]; then
    log ""
    log "📊 结果摘要:"
    cat "${MODEL_DIR}/eval_result.json"
fi
