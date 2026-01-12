#!/bin/bash
# ============================================
# ACT (Action Chunking Transformer) 训练脚本
# 
# 使用方法：
#   ./scripts/train_act.sh <exp_name> [options]
#
# 参数：
#   --steps N          训练步数 (默认: 20000)
#   --dim_model N      Transformer 维度 (默认: 1024)
#   --n_decoder_layers N  Decoder 层数 (默认: 4)
#   --chunk_size N     Action Chunk 大小 (默认: 10)
#   --n_action_steps N 执行动作数 (默认: 10)
#   --batch_size N     批量大小 (默认: 32)
#   --eval             训练后自动评估
#   --eval_episodes N  评估轮数 (默认: 50)
# ============================================

set -e

# ============================================
# 默认配置 (基于 experiment_registry.md 最优)
# ============================================

EXP_NAME="${1:-exp_unnamed}"      # 第一个参数是实验名称
PARENT_EXP="act_005"              # 父实验 (用于追溯)

# 训练参数 (参考 experiment_registry.md)
TRAINING_STEPS=20000              # 训练步数
BATCH_SIZE=32                     # 批量大小

# 模型参数
DIM_MODEL=1024                    # Transformer 维度 (最佳: 1024)
N_DECODER_LAYERS=4                # Decoder 层数 (最佳: 4)

# 动作序列参数
CHUNK_SIZE=10                     # Action Chunk 大小
N_ACTION_STEPS=10                 # 执行动作数

# 评估选项
DO_EVAL=false
EVAL_EPISODES=50

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
        --dim_model)
            DIM_MODEL="$2"
            shift 2
            ;;
        --n_decoder_layers)
            N_DECODER_LAYERS="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
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
        --eval)
            DO_EVAL=true
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
  policy: act

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
  dim_model: ${DIM_MODEL}
  n_decoder_layers: ${N_DECODER_LAYERS}
  chunk_size: ${CHUNK_SIZE}
  n_action_steps: ${N_ACTION_STEPS}

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

# ============================================
# 评估（如果启用）
# ============================================

if [ "$DO_EVAL" = true ]; then
    log "🎯 开始评估 (${EVAL_EPISODES} episodes)..."
    
    python scripts/eval_model.py \
        --model_path "${OUTPUT_DIR}" \
        --policy_type act \
        --n_episodes ${EVAL_EPISODES} \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
    
    # 复制评估代码快照
    cp scripts/eval_model.py "${OUTPUT_DIR}/eval_snapshot.py"
    
    log "✅ 评估完成"
else
    log "💡 下一步: 运行评估脚本"
fi
