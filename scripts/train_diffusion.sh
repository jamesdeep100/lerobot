#!/bin/bash
# ============================================
# Diffusion Policy 训练脚本
# 
# 使用方法：
#   ./scripts/train_diffusion.sh <exp_name> [options]
#
# 参数：
#   --steps N          训练步数 (默认: 100000)
#   --horizon N        预测序列长度 (默认: 32)
#   --n_action_steps N 执行动作数 (默认: 8)
#   --batch_size N     批量大小 (默认: 32)
#   --no-eval          跳过评估（默认自动评估）
#   --eval_episodes N  评估轮数 (默认: 50)
# ============================================

set -e

# ============================================
# 默认配置 (基于 experiment_registry.md 最优)
# ============================================

EXP_NAME="${1:-exp_unnamed}"      # 第一个参数是实验名称
PARENT_EXP="exp_006"              # 父实验 (用于追溯)

# 训练参数 (参考 experiment_registry.md)
TRAINING_STEPS=100000             # 训练步数
HORIZON=32                        # 预测序列长度 (最佳: 32)
N_ACTION_STEPS=8                  # 执行动作数 (最佳: 8)
BATCH_SIZE=32                     # 批量大小

# 模型参数
DOWN_DIMS="[256, 512, 1024]"      # U-Net 下采样维度

# 评估选项（默认开启）
DO_EVAL=true
EVAL_EPISODES=50

# 输出目录（可通过参数覆盖）
OUTPUT_DIR=""

# ============================================
# 解析命令行参数
# ============================================

shift  # 跳过第一个参数 (exp_name)

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --n_action_steps)
            N_ACTION_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-eval)
            DO_EVAL=false
            shift
            ;;
        --eval_episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# 如果没有指定输出目录，使用默认（独立实验仓库）
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/home/james/ai_projects/lerobot-experiments/${EXP_NAME}"
fi

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

# 创建输出目录
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

# 创建实验专属分支（基于当前 HEAD 的快照）
log "📦 归档代码到分支 exp/${EXP_NAME}..."

# 直接创建分支指向当前 HEAD（不切换，不提交）
if git show-ref --verify --quiet "refs/heads/exp/${EXP_NAME}"; then
    log "分支已存在，跳过创建"
else
    git branch "exp/${EXP_NAME}" HEAD 2>/dev/null || true
    log "✅ 分支 exp/${EXP_NAME} 已创建"
fi

log "✅ 训练完成: ${OUTPUT_DIR}"
log "📊 最终 Loss: ${FINAL_LOSS}"
log "⏱️  训练时长: ${DURATION} 秒"

# ============================================
# 评估（如果启用）
# ============================================

if [ "$DO_EVAL" = true ]; then
    log "🎯 开始评估 (${EVAL_EPISODES} episodes)..."
    
    python scripts/eval_model.py \
        --model_path "${OUTPUT_DIR}" \
        --policy_type diffusion \
        --n_episodes ${EVAL_EPISODES} \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
    
    # 复制评估代码快照
    cp scripts/eval_model.py "${OUTPUT_DIR}/eval_snapshot.py"
    
    log "✅ 评估完成"
    
    # 自动更新 leaderboard 并推送
    if [ -f "scripts/update_leaderboard.sh" ]; then
        bash scripts/update_leaderboard.sh "${OUTPUT_DIR}" "diffusion"
    fi
else
    log "💡 下一步: 运行评估脚本"
fi
