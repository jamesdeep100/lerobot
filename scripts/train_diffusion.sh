#!/bin/bash
# ============================================
# Diffusion Policy 通用训练脚本
# 用法: ./train_diffusion.sh <实验名> [选项]
#
# 示例:
#   ./train_diffusion.sh exp1_100k --steps 100000
#   ./train_diffusion.sh exp2_wide --steps 50000 --down_dims "1024,2048,4096"
#   ./train_diffusion.sh exp3_h32 --horizon 32 --n_action_steps 8
#
# 选项:
#   --steps N          训练步数 (默认: 50000)
#   --horizon N        预测时长 (默认: 16)
#   --n_action_steps N 执行步数 (默认: 8)
#   --batch_size N     批量大小 (默认: 32)
#   --down_dims "a,b,c" UNet 维度 (默认: "512,1024,2048")
#   --eval             训练后自动评估
#   --eval_episodes N  评估 episode 数 (默认: 50)
# ============================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️ $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }

# 默认参数
EXP_NAME=""
STEPS=50000
HORIZON=16
N_ACTION_STEPS=8
BATCH_SIZE=32
DOWN_DIMS="512,1024,2048"
DO_EVAL=false
EVAL_EPISODES=50

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        --n_action_steps) N_ACTION_STEPS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --down_dims) DOWN_DIMS="$2"; shift 2 ;;
        --eval) DO_EVAL=true; shift ;;
        --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
        -*) error "未知选项: $1" ;;
        *) EXP_NAME="$1"; shift ;;
    esac
done

# 检查必需参数
if [ -z "$EXP_NAME" ]; then
    error "请提供实验名称！用法: ./train_diffusion.sh <实验名> [选项]"
fi

# 设置环境
cd /home/james/ai_projects/lerobot
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

# 输出目录
OUTPUT_DIR="outputs/diffusion_exp/${EXP_NAME}"
TRAIN_SCRIPT="/tmp/train_${EXP_NAME}.py"

log "============================================"
log "🚀 Diffusion Policy 训练"
log "============================================"
log "📝 实验名称: ${EXP_NAME}"
log "📁 输出目录: ${OUTPUT_DIR}"
log "📊 配置:"
log "   - training_steps: ${STEPS}"
log "   - horizon: ${HORIZON}"
log "   - n_action_steps: ${N_ACTION_STEPS}"
log "   - batch_size: ${BATCH_SIZE}"
log "   - down_dims: (${DOWN_DIMS})"
log "============================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 复制并修改训练脚本
cp examples/tutorial/diffusion/diffusion_training_pusht.py "${TRAIN_SCRIPT}"

# 关键：使用正确的变量名 output_directory
sed -i "s|output_directory = Path(\"outputs/diffusion_pusht_demo\")|output_directory = Path(\"${OUTPUT_DIR}\")|" "${TRAIN_SCRIPT}"
sed -i "s/training_steps = [0-9]*/training_steps = ${STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/horizon=[0-9]*/horizon=${HORIZON}/" "${TRAIN_SCRIPT}"
sed -i "s/n_action_steps=[0-9]*/n_action_steps=${N_ACTION_STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/batch_size = [0-9]*/batch_size = ${BATCH_SIZE}/" "${TRAIN_SCRIPT}"
sed -i "s/down_dims=([0-9, ]*)/down_dims=(${DOWN_DIMS})/" "${TRAIN_SCRIPT}"

# ⚠️ 验证 sed 替换是否成功
log "🔍 验证配置替换..."
if ! grep -q "output_directory = Path(\"${OUTPUT_DIR}\")" "${TRAIN_SCRIPT}"; then
    error "输出目录替换失败！请检查脚本。"
fi
log "✅ 配置验证通过"

# 记录配置到文件
cat > "${OUTPUT_DIR}/config.txt" << EOF
experiment: ${EXP_NAME}
date: $(date)
training_steps: ${STEPS}
horizon: ${HORIZON}
n_action_steps: ${N_ACTION_STEPS}
batch_size: ${BATCH_SIZE}
down_dims: (${DOWN_DIMS})
EOF

# 开始训练
log "🏋️ 开始训练..."
python "${TRAIN_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

# 验证模型是否保存成功
if [ ! -f "${OUTPUT_DIR}/model.safetensors" ]; then
    error "模型保存失败！未找到 ${OUTPUT_DIR}/model.safetensors"
fi
log "✅ 模型保存成功: ${OUTPUT_DIR}/model.safetensors"

# 可选评估
if [ "$DO_EVAL" = true ]; then
    log ""
    log "📊 开始评估 (${EVAL_EPISODES} episodes)..."
    python scripts/eval_model.py \
        --model_path "${OUTPUT_DIR}" \
        --policy_type diffusion \
        --n_episodes ${EVAL_EPISODES} \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
fi

log ""
log "============================================"
log "✅ 完成时间: $(date)"
log "📁 模型路径: ${OUTPUT_DIR}"
log "============================================"
