#!/bin/bash
# ============================================
# 自动更新 leaderboard 并 push
# ============================================
# 用法: ./scripts/update_leaderboard.sh <exp_dir> <policy_type>
#
# 参数:
#   exp_dir     实验目录路径
#   policy_type act 或 diffusion
# ============================================

set -e

EXP_DIR="$1"
POLICY_TYPE="$2"

LEROBOT_EXPERIMENTS_DIR="${LEROBOT_EXPERIMENTS_DIR:-/home/james/ai_projects/lerobot-experiments}"
LEADERBOARD="${LEROBOT_EXPERIMENTS_DIR}/leaderboard.md"

# 颜色
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }

if [ -z "$EXP_DIR" ] || [ -z "$POLICY_TYPE" ]; then
    echo "用法: $0 <exp_dir> <policy_type>"
    exit 1
fi

log "📊 更新 leaderboard..."

# 读取实验结果
if [ -f "${EXP_DIR}/eval_results.json" ]; then
    SUCCESS_RATE=$(python3 -c "import json; d=json.load(open('${EXP_DIR}/eval_results.json')); print(f\"{d['pc_success']:.0f}%\")")
    AVG_REWARD=$(python3 -c "import json; d=json.load(open('${EXP_DIR}/eval_results.json')); print(f\"{d['avg_sum_reward']:.0f}\")")
else
    SUCCESS_RATE="-"
    AVG_REWARD="-"
fi

# 读取训练元数据
if [ -f "${EXP_DIR}/metadata.yaml" ]; then
    TRAINING_TIME=$(grep "training_duration" "${EXP_DIR}/metadata.yaml" | awk '{print $2}' | head -1)
    MODEL_SIZE=$(grep "total_params" "${EXP_DIR}/metadata.yaml" | awk '{print $2}' | head -1)
else
    TRAINING_TIME="-"
    MODEL_SIZE="-"
fi

# 获取实验名称（从目录名）
EXP_NAME=$(basename "$EXP_DIR")
BATCH_NAME=$(basename "$(dirname "$EXP_DIR")")

# 生成链接
GITHUB_LINK="https://github.com/jamesdeep100/lerobot-experiments/tree/main/${BATCH_NAME}/${EXP_NAME}"

log "   实验: ${EXP_NAME}"
log "   成功率: ${SUCCESS_RATE}"
log "   奖励: ${AVG_REWARD}"
log "   训练时间: ${TRAINING_TIME}"

# 更新 leaderboard 的时间戳
sed -i "s/\*Last Updated:.*/*Last Updated: $(date '+%Y-%m-%d %H:%M')*/g" "$LEADERBOARD"

# 提交并推送
cd "$LEROBOT_EXPERIMENTS_DIR"
git add -A
git commit -m "auto: 更新实验 ${EXP_NAME} 结果

成功率: ${SUCCESS_RATE}
avg_reward: ${AVG_REWARD}" 2>/dev/null || log "无新更改"

# 推送（后台执行，不阻塞）
git push origin main 2>/dev/null &

log "✅ leaderboard 已更新并推送"
