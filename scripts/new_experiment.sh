#!/bin/bash
# ============================================
# 创建新实验
# 
# 用法: ./scripts/new_experiment.sh <exp_name> <policy_type>
#
# 示例:
#   ./scripts/new_experiment.sh act_50k_sota act
#   ./scripts/new_experiment.sh diff_200k diffusion
#
# 此脚本会：
#   1. 创建实验目录
#   2. 从模板复制训练/评估脚本
#   3. 提示你修改参数
# ============================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️ $1${NC}"; }

if [ $# -lt 2 ]; then
    echo "用法: $0 <exp_name> <policy_type>"
    echo "  policy_type: diffusion 或 act"
    exit 1
fi

EXP_NAME="$1"
POLICY_TYPE="$2"

cd /home/james/ai_projects/lerobot

# 生成实验 ID
NEXT_ID=$(ls -1 experiments/ 2>/dev/null | grep -E '^exp_[0-9]+' | wc -l)
NEXT_ID=$((NEXT_ID + 1))
EXP_ID=$(printf "exp_%03d_%s" ${NEXT_ID} "${EXP_NAME}")

EXP_DIR="experiments/${EXP_ID}"

log "============================================"
log "📦 创建新实验"
log "============================================"
log "📝 实验 ID: ${EXP_ID}"
log "📁 目录: ${EXP_DIR}"
log "🔧 策略: ${POLICY_TYPE}"
log "============================================"

# 创建目录
mkdir -p "${EXP_DIR}"

# 复制模板
if [ "${POLICY_TYPE}" = "diffusion" ]; then
    cp scripts/templates/train_diffusion.sh "${EXP_DIR}/run_train.sh"
elif [ "${POLICY_TYPE}" = "act" ]; then
    cp scripts/templates/train_act.sh "${EXP_DIR}/run_train.sh"
else
    echo "❌ 未知策略类型: ${POLICY_TYPE}"
    exit 1
fi

cp scripts/templates/eval_model.sh "${EXP_DIR}/run_eval.sh"

# 替换实验名称
sed -i "s/EXP_NAME=\"exp_NNN_name\"/EXP_NAME=\"${EXP_ID}\"/" "${EXP_DIR}/run_train.sh"
sed -i "s/EXP_NAME=\"exp_NNN_name\"/EXP_NAME=\"${EXP_ID}\"/" "${EXP_DIR}/run_eval.sh"
sed -i "s/POLICY_TYPE=\"diffusion\"/POLICY_TYPE=\"${POLICY_TYPE}\"/" "${EXP_DIR}/run_eval.sh"

# 设置可执行权限
chmod +x "${EXP_DIR}/run_train.sh"
chmod +x "${EXP_DIR}/run_eval.sh"

# 创建 notes.md
cat > "${EXP_DIR}/notes.md" << EOF
# ${EXP_ID}

## 实验目的

(描述这个实验要验证什么假设)

## 参数变更

基于: (父实验 ID)
变更:
- (列出修改的参数)

## 结果

(实验完成后填写)

## 结论

(分析和下一步)
EOF

log ""
log "✅ 实验创建成功！"
log ""
log "📋 下一步:"
log "   1. 编辑 ${EXP_DIR}/run_train.sh 修改参数"
log "   2. 查阅 experiment_registry.md 确认参数"
log "   3. 运行 cd ${EXP_DIR} && ./run_train.sh"
log ""
warn "⚠️ 记得先查看 experiment_registry.md 了解最佳参数！"
