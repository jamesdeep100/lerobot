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

# ============================================
# 记录元数据
# ============================================

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
HAS_UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l)
PYTHON_VERSION=$(python --version 2>&1)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)

# 保存训练代码快照
cp "${TRAIN_SCRIPT}" "${OUTPUT_DIR}/train_snapshot.py"

# 开始训练
log "🏋️ 开始训练..."
python "${TRAIN_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

# 验证模型保存
if [ ! -f "${OUTPUT_DIR}/model.safetensors" ]; then
    error "模型保存失败！"
fi

# 提取最终 loss
FINAL_LOSS=$(grep -oP 'Loss: \K[0-9.]+' "${OUTPUT_DIR}/train.log" | tail -1 || echo "unknown")

# 生成完整 metadata.yaml
cat > "${OUTPUT_DIR}/metadata.yaml" << EOF
experiment:
  id: ${EXP_NAME}
  created: ${START_TIME}
  parent: ${PARENT_EXP}
  policy: diffusion

code:
  commit_hash: ${COMMIT_HASH}
  branch: $(git branch --show-current 2>/dev/null || echo "unknown")
  has_uncommitted: $([[ ${HAS_UNCOMMITTED} -gt 0 ]] && echo "true" || echo "false")
  experiment_branch: exp/${EXP_NAME}

environment:
  python: ${PYTHON_VERSION}
  torch: ${TORCH_VERSION}
  cuda_driver: ${CUDA_VERSION}
  gpu: ${GPU_NAME}

training:
  start_time: ${START_TIME}
  end_time: ${END_TIME}
  duration_seconds: ${DURATION}
  final_loss: ${FINAL_LOSS}
  steps: ${TRAINING_STEPS}
  batch_size: ${BATCH_SIZE}

model:
  horizon: ${HORIZON}
  n_action_steps: ${N_ACTION_STEPS}
  down_dims: ${DOWN_DIMS}

# 评估结果（评估后填充）
evaluation:
  n_episodes: null
  success_rate: null
  avg_sum_reward: null
  avg_max_reward: null
EOF

# 创建实验专属分支（代码归档）
log "📦 归档代码到分支 exp/${EXP_NAME}..."
git stash -q 2>/dev/null || true
git branch "exp/${EXP_NAME}" 2>/dev/null || log "分支已存在，跳过创建"
git stash pop -q 2>/dev/null || true

log "✅ 训练完成: ${OUTPUT_DIR}"
log "📊 最终 Loss: ${FINAL_LOSS}"
log "⏱️  训练时长: ${DURATION} 秒"
log "💡 下一步: 运行 run_eval.sh 评估模型"
