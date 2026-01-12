#!/bin/bash
# ============================================
# Diffusion Policy 训练模板
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
PARENT_EXP="exp_006"              # 父实验 (用于追溯)

# 训练参数 (参考 experiment_registry.md)
TRAINING_STEPS=100000             # 训练步数
HORIZON=32                        # 预测序列长度 (最佳: 32)
N_ACTION_STEPS=8                  # 执行动作数 (最佳: 8)
BATCH_SIZE=32                     # 批量大小

# 模型参数
DOWN_DIMS="[256, 512, 1024]"      # U-Net 下采样维度

# ============================================
# 以下内容无需修改
# ============================================

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }

# 设置环境
cd /home/james/ai_projects/lerobot
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

# 输出目录
OUTPUT_DIR="experiments/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

# 生成 config.yaml
cat > "${OUTPUT_DIR}/config.yaml" << EOF
experiment:
  name: ${EXP_NAME}
  parent: ${PARENT_EXP}
  date: $(date '+%Y-%m-%d %H:%M:%S')
  policy: diffusion

training:
  steps: ${TRAINING_STEPS}
  batch_size: ${BATCH_SIZE}

model:
  horizon: ${HORIZON}
  n_action_steps: ${N_ACTION_STEPS}
  down_dims: ${DOWN_DIMS}
EOF

log "============================================"
log "🚀 Diffusion Policy 训练"
log "============================================"
log "📝 实验: ${EXP_NAME} (基于 ${PARENT_EXP})"
log "📁 输出: ${OUTPUT_DIR}"
log "📊 参数:"
log "   - steps: ${TRAINING_STEPS}"
log "   - horizon: ${HORIZON}"
log "   - n_action_steps: ${N_ACTION_STEPS}"
log "   - batch_size: ${BATCH_SIZE}"
log "============================================"

# 创建临时训练脚本
TRAIN_SCRIPT="/tmp/train_diffusion_${EXP_NAME}.py"
cp examples/tutorial/diffusion/diffusion_training_pusht.py "${TRAIN_SCRIPT}"

# 替换参数
sed -i "s|output_directory = Path(\"outputs/diffusion_pusht_demo\")|output_directory = Path(\"${OUTPUT_DIR}\")|" "${TRAIN_SCRIPT}"
sed -i "s/training_steps = [0-9]*/training_steps = ${TRAINING_STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/horizon=[0-9]*/horizon=${HORIZON}/" "${TRAIN_SCRIPT}"
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
