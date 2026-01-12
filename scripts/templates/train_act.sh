#!/bin/bash
# ============================================
# ACT (Action Chunking Transformer) 训练模板
# 
# 使用方法：
#   1. 复制此文件到 experiments/exp_NNN_name/run_train.sh
#   2. 修改下方 CONFIG 区域的参数
#   3. 运行: ./run_train.sh
#
# ⚠️ 严禁直接修改此模板文件！
# ============================================

set -e

# ============================================
# CONFIG - 修改此区域的参数
# ============================================

EXP_NAME="exp_NNN_name"           # 实验名称
PARENT_EXP="act_005"              # 父实验 (用于追溯)

# 训练参数 (参考 experiment_registry.md)
TRAINING_STEPS=20000              # 训练步数
BATCH_SIZE=32                     # 批量大小

# 模型参数
DIM_MODEL=1024                    # Transformer 维度 (最佳: 1024)
N_DECODER_LAYERS=4                # Decoder 层数 (最佳: 4)

# ⚠️ 关键参数 - 除非有明确实验目的，否则不要修改！
CHUNK_SIZE=10                     # Action Chunk 大小 (最佳: 10)
N_ACTION_STEPS=10                 # 执行动作数 (必须与 chunk_size 匹配!)

# ============================================
# 以下内容无需修改
# ============================================

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️ $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }

# ⚠️ 安全检查：chunk_size 和 n_action_steps 必须匹配
if [ "${CHUNK_SIZE}" != "${N_ACTION_STEPS}" ]; then
    warn "chunk_size (${CHUNK_SIZE}) != n_action_steps (${N_ACTION_STEPS})"
    warn "这可能导致性能问题，请确认是有意为之"
fi

# 设置环境
cd /home/james/ai_projects/lerobot
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

# 修复 pymunk 依赖
pip uninstall pymunk -y 2>/dev/null || true
pip install pymunk==6.4.0 -q 2>/dev/null || true

# 输出目录
OUTPUT_DIR="experiments/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

# 生成 config.yaml
cat > "${OUTPUT_DIR}/config.yaml" << EOF
experiment:
  name: ${EXP_NAME}
  parent: ${PARENT_EXP}
  date: $(date '+%Y-%m-%d %H:%M:%S')
  policy: act

training:
  steps: ${TRAINING_STEPS}
  batch_size: ${BATCH_SIZE}

model:
  dim_model: ${DIM_MODEL}
  n_decoder_layers: ${N_DECODER_LAYERS}
  chunk_size: ${CHUNK_SIZE}
  n_action_steps: ${N_ACTION_STEPS}
EOF

log "============================================"
log "🚀 ACT (Action Chunking Transformer) 训练"
log "============================================"
log "📝 实验: ${EXP_NAME} (基于 ${PARENT_EXP})"
log "📁 输出: ${OUTPUT_DIR}"
log "📊 参数:"
log "   - steps: ${TRAINING_STEPS}"
log "   - dim_model: ${DIM_MODEL}"
log "   - n_decoder_layers: ${N_DECODER_LAYERS}"
log "   - chunk_size: ${CHUNK_SIZE} ⚠️"
log "   - n_action_steps: ${N_ACTION_STEPS}"
log "   - batch_size: ${BATCH_SIZE}"
log "============================================"

# 创建临时训练脚本
TRAIN_SCRIPT="/tmp/train_act_${EXP_NAME}.py"
cp examples/tutorial/act/act_training_pusht.py "${TRAIN_SCRIPT}"

# 替换参数
sed -i "s|output_directory = Path(\"outputs/act_pusht_demo\")|output_directory = Path(\"${OUTPUT_DIR}\")|" "${TRAIN_SCRIPT}"
sed -i "s/training_steps = [0-9]*/training_steps = ${TRAINING_STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/dim_model=[0-9]*/dim_model=${DIM_MODEL}/" "${TRAIN_SCRIPT}"
sed -i "s/n_decoder_layers=[0-9]*/n_decoder_layers=${N_DECODER_LAYERS}/" "${TRAIN_SCRIPT}"
sed -i "s/chunk_size=[0-9]*/chunk_size=${CHUNK_SIZE}/" "${TRAIN_SCRIPT}"
sed -i "s/n_action_steps=[0-9]*/n_action_steps=${N_ACTION_STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/batch_size = [0-9]*/batch_size = ${BATCH_SIZE}/" "${TRAIN_SCRIPT}"

# 开始训练
log "🏋️ 开始训练..."
python "${TRAIN_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

# 验证模型保存
if [ ! -f "${OUTPUT_DIR}/model.safetensors" ]; then
    error "模型保存失败！"
fi

log "✅ 训练完成: ${OUTPUT_DIR}"
log "💡 下一步: 运行 run_eval.sh 评估模型"
