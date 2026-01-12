#!/bin/bash
# ============================================
# ACT (Action Chunking Transformer) 通用训练脚本
# 用法: ./train_act.sh <实验名> [选项]
#
# 示例:
#   ./train_act.sh exp1_50k --steps 50000
#   ./train_act.sh exp2_large --steps 100000 --dim_model 1024 --n_decoder_layers 4
#
# 选项:
#   --steps N            训练步数 (默认: 50000)
#   --dim_model N        模型维度 (默认: 512)
#   --n_decoder_layers N 解码器层数 (默认: 1)
#   --batch_size N       批量大小 (默认: 32)
#   --chunk_size N       动作块大小 (默认: 100)
#   --eval               训练后自动评估
#   --eval_episodes N    评估 episode 数 (默认: 50)
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
DIM_MODEL=512
N_DECODER_LAYERS=1
BATCH_SIZE=32
CHUNK_SIZE=100
DO_EVAL=false
EVAL_EPISODES=50

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --dim_model) DIM_MODEL="$2"; shift 2 ;;
        --n_decoder_layers) N_DECODER_LAYERS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
        --eval) DO_EVAL=true; shift ;;
        --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
        -*) error "未知选项: $1" ;;
        *) EXP_NAME="$1"; shift ;;
    esac
done

# 检查必需参数
if [ -z "$EXP_NAME" ]; then
    error "请提供实验名称！用法: ./train_act.sh <实验名> [选项]"
fi

# 设置环境
cd /home/james/ai_projects/lerobot
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

# 修复 pymunk (ACT 评估需要)
pip uninstall pymunk -y 2>/dev/null || true
pip install pymunk==6.4.0 -q 2>/dev/null || true

# 输出目录
OUTPUT_DIR="outputs/act_exp/${EXP_NAME}"
TRAIN_SCRIPT="/tmp/train_act_${EXP_NAME}.py"

log "============================================"
log "🚀 ACT (Action Chunking Transformer) 训练"
log "============================================"
log "📝 实验名称: ${EXP_NAME}"
log "📁 输出目录: ${OUTPUT_DIR}"
log "📊 配置:"
log "   - training_steps: ${STEPS}"
log "   - dim_model: ${DIM_MODEL}"
log "   - n_decoder_layers: ${N_DECODER_LAYERS}"
log "   - batch_size: ${BATCH_SIZE}"
log "   - chunk_size: ${CHUNK_SIZE}"
log "============================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 复制并修改训练脚本
cp examples/tutorial/act/act_training_pusht.py "${TRAIN_SCRIPT}"

# 关键：使用正确的变量名 output_directory
sed -i "s|output_directory = Path(\"outputs/act_pusht_demo\")|output_directory = Path(\"${OUTPUT_DIR}\")|" "${TRAIN_SCRIPT}"
sed -i "s/training_steps = [0-9]*/training_steps = ${STEPS}/" "${TRAIN_SCRIPT}"
sed -i "s/dim_model=[0-9]*/dim_model=${DIM_MODEL}/" "${TRAIN_SCRIPT}"
sed -i "s/n_decoder_layers=[0-9]*/n_decoder_layers=${N_DECODER_LAYERS}/" "${TRAIN_SCRIPT}"
sed -i "s/batch_size = [0-9]*/batch_size = ${BATCH_SIZE}/" "${TRAIN_SCRIPT}"
sed -i "s/chunk_size=[0-9]*/chunk_size=${CHUNK_SIZE}/" "${TRAIN_SCRIPT}"

# ⚠️ 验证 sed 替换是否成功
log "🔍 验证配置替换..."
if ! grep -q "output_directory = Path(\"${OUTPUT_DIR}\")" "${TRAIN_SCRIPT}"; then
    warn "输出目录替换可能失败，检查变量名..."
    # 尝试其他可能的变量名
    sed -i "s|output_dir = Path(\"outputs/act_pusht_demo\")|output_dir = Path(\"${OUTPUT_DIR}\")|" "${TRAIN_SCRIPT}"
fi
log "✅ 配置验证通过"

# 记录配置到文件
cat > "${OUTPUT_DIR}/config.txt" << EOF
experiment: ${EXP_NAME}
date: $(date)
training_steps: ${STEPS}
dim_model: ${DIM_MODEL}
n_decoder_layers: ${N_DECODER_LAYERS}
batch_size: ${BATCH_SIZE}
chunk_size: ${CHUNK_SIZE}
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
        --policy_type act \
        --n_episodes ${EVAL_EPISODES} \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
fi

log ""
log "============================================"
log "✅ 完成时间: $(date)"
log "📁 模型路径: ${OUTPUT_DIR}"
log "============================================"
