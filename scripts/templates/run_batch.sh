#!/bin/bash
# ============================================
# 多机实验调度脚本
# 用法: ./run_experiments.sh <配置文件>
#
# 或者直接调用函数:
#   source scripts/run_experiments.sh
#   add_laptop_exp "diffusion" "exp1" "--steps 100000"
#   add_desktop_exp "act" "exp1" "--steps 50000"
#   run_all
# ============================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
LAPTOP_HOST="localhost"
# 使用 Tailscale IP，支持远程连接
DESKTOP_HOST="james@100.67.100.43"
LEROBOT_DIR="/home/james/ai_projects/lerobot"

# 调度临时文件目录
SCHEDULER_DIR="${LEROBOT_DIR}/outputs/.scheduler"

# 实验列表
declare -a LAPTOP_EXPS
declare -a DESKTOP_EXPS

# 当前批次时间戳 (在 source 时生成)
BATCH_TIMESTAMP=$(date '+%m%d_%H%M')

# 训练速度估算 (ms/step)
DIFFUSION_SPEED_LAPTOP=100   # RTX 5090
DIFFUSION_SPEED_DESKTOP=190  # RTX 3060 Ti
ACT_SPEED_LAPTOP=65
ACT_SPEED_DESKTOP=100

# 评估速度 (s/episode)
EVAL_SPEED_LAPTOP=31
EVAL_SPEED_DESKTOP=55

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️ $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; }

# 生成带时间戳的实验名
get_exp_name() {
    local base_name="$1"
    echo "${BATCH_TIMESTAMP}_${base_name}"
}

# 估算训练时间
estimate_time() {
    local policy_type="$1"
    local steps="$2"
    local machine="$3"
    local eval_episodes="${4:-50}"
    
    local speed=0
    local eval_speed=0
    
    if [ "$policy_type" == "diffusion" ]; then
        if [ "$machine" == "laptop" ]; then
            speed=$DIFFUSION_SPEED_LAPTOP
            eval_speed=$EVAL_SPEED_LAPTOP
        else
            speed=$DIFFUSION_SPEED_DESKTOP
            eval_speed=$EVAL_SPEED_DESKTOP
        fi
    else
        if [ "$machine" == "laptop" ]; then
            speed=$ACT_SPEED_LAPTOP
            eval_speed=$EVAL_SPEED_LAPTOP
        else
            speed=$ACT_SPEED_DESKTOP
            eval_speed=$EVAL_SPEED_DESKTOP
        fi
    fi
    
    local train_time=$((steps * speed / 1000))
    local eval_time=$((eval_episodes * eval_speed))
    local total_time=$((train_time + eval_time))
    
    echo "$total_time"
}

# 格式化时间
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# 添加笔记本实验
add_laptop_exp() {
    local policy_type="$1"
    local base_name="$2"
    local options="$3"
    
    local exp_name=$(get_exp_name "$base_name")
    LAPTOP_EXPS+=("${policy_type}|${exp_name}|${options}")
}

# 添加台式机实验
add_desktop_exp() {
    local policy_type="$1"
    local base_name="$2"
    local options="$3"
    
    local exp_name=$(get_exp_name "$base_name")
    DESKTOP_EXPS+=("${policy_type}|${exp_name}|${options}")
}

# 解析选项获取步数
get_steps_from_options() {
    local options="$1"
    echo "$options" | grep -oP '(?<=--steps\s)\d+' || echo "50000"
}

# 解析选项获取评估数
get_eval_episodes_from_options() {
    local options="$1"
    echo "$options" | grep -oP '(?<=--eval_episodes\s)\d+' || echo "50"
}

# 显示实验计划
show_plan() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    🚀 实验计划                                  ║${NC}"
    echo -e "${BLUE}║                    批次: ${BATCH_TIMESTAMP}                              ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    
    local laptop_total=0
    local desktop_total=0
    
    # 笔记本实验
    echo -e "${BLUE}║${NC} ${CYAN}📱 笔记本 (RTX 5090)${NC}"
    if [ ${#LAPTOP_EXPS[@]} -eq 0 ]; then
        echo -e "${BLUE}║${NC}    (无实验)"
    else
        for exp in "${LAPTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "laptop" "$eval_eps")
            laptop_total=$((laptop_total + est_time))
            
            echo -e "${BLUE}║${NC}    ├─ ${GREEN}${exp_name}${NC}"
            echo -e "${BLUE}║${NC}    │  策略: ${policy_type}, 步数: ${steps}, 预估: $(format_time $est_time)"
        done
    fi
    
    echo -e "${BLUE}║${NC}"
    
    # 台式机实验
    echo -e "${BLUE}║${NC} ${CYAN}🖥️ 台式机 (RTX 3060 Ti)${NC}"
    if [ ${#DESKTOP_EXPS[@]} -eq 0 ]; then
        echo -e "${BLUE}║${NC}    (无实验)"
    else
        for exp in "${DESKTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "desktop" "$eval_eps")
            desktop_total=$((desktop_total + est_time))
            
            echo -e "${BLUE}║${NC}    ├─ ${GREEN}${exp_name}${NC}"
            echo -e "${BLUE}║${NC}    │  策略: ${policy_type}, 步数: ${steps}, 预估: $(format_time $est_time)"
        done
    fi
    
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} ${YELLOW}⏱️ 预估总时长${NC}"
    echo -e "${BLUE}║${NC}    笔记本: $(format_time $laptop_total)"
    echo -e "${BLUE}║${NC}    台式机: $(format_time $desktop_total)"
    echo -e "${BLUE}║${NC}    并行时间: $(format_time $(($laptop_total > $desktop_total ? $laptop_total : $desktop_total)))"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 生成统筹日志
generate_batch_log() {
    local batch_log="${SCHEDULER_DIR}/batch_${BATCH_TIMESTAMP}.md"
    
    cat > "$batch_log" << EOF
# 实验批次: ${BATCH_TIMESTAMP}

> 创建时间: $(date '+%Y-%m-%d %H:%M:%S')

## 📋 实验清单

### 📱 笔记本 (RTX 5090)
EOF
    
    if [ ${#LAPTOP_EXPS[@]} -eq 0 ]; then
        echo "无实验" >> "$batch_log"
    else
        for exp in "${LAPTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "laptop" "$eval_eps")
            
            if [ "$policy_type" == "diffusion" ]; then
                local output_dir="outputs/diffusion_exp/${exp_name}"
            else
                local output_dir="outputs/act_exp/${exp_name}"
            fi
            
            cat >> "$batch_log" << EOF

#### ${exp_name}
- **策略**: ${policy_type}
- **参数**: ${options}
- **预估时长**: $(format_time $est_time)
- **模型路径**: \`${output_dir}\`
- **状态**: 🔵 待运行
EOF
        done
    fi
    
    cat >> "$batch_log" << EOF

### 🖥️ 台式机 (RTX 3060 Ti)
EOF
    
    if [ ${#DESKTOP_EXPS[@]} -eq 0 ]; then
        echo "无实验" >> "$batch_log"
    else
        for exp in "${DESKTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "desktop" "$eval_eps")
            
            if [ "$policy_type" == "diffusion" ]; then
                local output_dir="outputs/diffusion_exp/${exp_name}"
            else
                local output_dir="outputs/act_exp/${exp_name}"
            fi
            
            cat >> "$batch_log" << EOF

#### ${exp_name}
- **策略**: ${policy_type}
- **参数**: ${options}
- **预估时长**: $(format_time $est_time)
- **模型路径**: \`${output_dir}\`
- **状态**: 🔵 待运行
EOF
        done
    fi
    
    cat >> "$batch_log" << EOF

---

## 📁 文件位置

| 类型 | 路径 |
|------|------|
| 批次日志 | \`${batch_log}\` |
| 笔记本执行脚本 | \`${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.sh\` |
| 笔记本运行日志 | \`${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.log\` |
| 台式机执行脚本 | \`${SCHEDULER_DIR}/desktop_${BATCH_TIMESTAMP}.sh\` |
| 台式机运行日志 | \`${SCHEDULER_DIR}/desktop_${BATCH_TIMESTAMP}.log\` |

---

## 🕐 时间线

- $(date '+%H:%M:%S') - 批次创建
EOF
    
    echo "$batch_log"
}

# 更新批次日志
update_batch_log() {
    local batch_log="${SCHEDULER_DIR}/batch_${BATCH_TIMESTAMP}.md"
    local message="$1"
    
    echo "- $(date '+%H:%M:%S') - ${message}" >> "$batch_log"
}

# 在笔记本上运行实验
run_laptop_experiments() {
    if [ ${#LAPTOP_EXPS[@]} -eq 0 ]; then
        log "笔记本无实验"
        return
    fi
    
    log "🚀 启动笔记本实验 (${#LAPTOP_EXPS[@]} 个)..."
    
    local script_content="#!/bin/bash
cd ${LEROBOT_DIR}
eval \"\$(~/miniconda3/bin/conda shell.bash hook)\"
conda activate lerobot

echo \"============================================\"
echo \"🌙 笔记本实验开始: \$(date)\"
echo \"批次: ${BATCH_TIMESTAMP}\"
echo \"============================================\"
"
    
    for exp in "${LAPTOP_EXPS[@]}"; do
        IFS='|' read -r policy_type exp_name options <<< "$exp"
        
        script_content+="
echo \"\"
echo \"======================================\"
echo \"🔬 实验: ${exp_name} (${policy_type})\"
echo \"======================================\"
./scripts/templates/train_${policy_type}.sh ${exp_name} ${options} --eval
"
    done
    
    script_content+="
echo \"\"
echo \"============================================\"
echo \"✅ 笔记本实验完成: \$(date)\"
echo \"============================================\"
"
    
    # 保存脚本到调度目录
    local script_file="${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.sh"
    echo "$script_content" > "$script_file"
    chmod +x "$script_file"
    
    # 后台运行
    local log_file="${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.log"
    nohup bash "$script_file" > "$log_file" 2>&1 &
    local pid=$!
    
    log "✅ 笔记本实验已启动"
    log "   PID: $pid"
    log "   脚本: $script_file"
    log "   日志: $log_file"
    
    echo "$pid" > "${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.pid"
    
    update_batch_log "笔记本实验启动 (PID: $pid)"
}

# 在台式机上运行实验
run_desktop_experiments() {
    if [ ${#DESKTOP_EXPS[@]} -eq 0 ]; then
        log "台式机无实验"
        return
    fi
    
    log "🚀 启动台式机实验 (${#DESKTOP_EXPS[@]} 个)..."
    
    local script_content="#!/bin/bash
cd ${LEROBOT_DIR}
eval \"\$(~/miniconda3/bin/conda shell.bash hook)\"
conda activate lerobot

# 修复 pymunk
pip uninstall pymunk -y 2>/dev/null || true
pip install pymunk==6.4.0 -q 2>/dev/null || true

echo \"============================================\"
echo \"🌙 台式机实验开始: \$(date)\"
echo \"批次: ${BATCH_TIMESTAMP}\"
echo \"============================================\"
"
    
    for exp in "${DESKTOP_EXPS[@]}"; do
        IFS='|' read -r policy_type exp_name options <<< "$exp"
        
        script_content+="
echo \"\"
echo \"======================================\"
echo \"🔬 实验: ${exp_name} (${policy_type})\"
echo \"======================================\"
./scripts/templates/train_${policy_type}.sh ${exp_name} ${options} --eval
"
    done
    
    script_content+="
echo \"\"
echo \"============================================\"
echo \"✅ 台式机实验完成: \$(date)\"
echo \"============================================\"
"
    
    # 确保远程调度目录存在
    ssh ${DESKTOP_HOST} "mkdir -p ${SCHEDULER_DIR}"
    
    # 通过 SSH 创建并运行脚本
    local remote_script="${SCHEDULER_DIR}/desktop_${BATCH_TIMESTAMP}.sh"
    local remote_log="${SCHEDULER_DIR}/desktop_${BATCH_TIMESTAMP}.log"
    
    # 创建远程脚本
    ssh ${DESKTOP_HOST} "cat > ${remote_script}" << EOF
${script_content}
EOF
    
    ssh ${DESKTOP_HOST} "chmod +x ${remote_script}"
    
    # 后台运行
    ssh ${DESKTOP_HOST} "cd ${LEROBOT_DIR} && nohup bash ${remote_script} > ${remote_log} 2>&1 &"
    
    log "✅ 台式机实验已启动"
    log "   脚本: ${remote_script}"
    log "   日志: ${remote_log}"
    log "   查看: ssh ${DESKTOP_HOST} 'tail -f ${remote_log}'"
    
    update_batch_log "台式机实验启动"
}

# 运行所有实验
run_all() {
    # 创建调度目录
    mkdir -p "${SCHEDULER_DIR}"
    
    show_plan
    
    # 生成批次日志
    local batch_log=$(generate_batch_log)
    log "📝 批次日志: $batch_log"
    
    update_batch_log "启动实验"
    
    run_laptop_experiments
    run_desktop_experiments
    
    echo ""
    log "============================================"
    log "🌙 所有实验已启动！"
    log "============================================"
    log ""
    log "📋 批次日志: ${SCHEDULER_DIR}/batch_${BATCH_TIMESTAMP}.md"
    log ""
    log "📋 检查命令:"
    log "   笔记本: tail -f ${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.log"
    log "   台式机: ssh ${DESKTOP_HOST} 'tail -f ${SCHEDULER_DIR}/desktop_${BATCH_TIMESTAMP}.log'"
    log ""
    log "🛑 停止命令:"
    log "   笔记本: kill \$(cat ${SCHEDULER_DIR}/laptop_${BATCH_TIMESTAMP}.pid)"
    log "   台式机: ssh ${DESKTOP_HOST} 'pkill -f desktop_${BATCH_TIMESTAMP}'"
    
    update_batch_log "所有实验启动完成"
}

# 检查实验状态
check_status() {
    echo ""
    log "============================================"
    log "📊 实验状态检查"
    log "============================================"
    
    # 查找最新的批次
    local latest_batch=$(ls -t ${SCHEDULER_DIR}/batch_*.md 2>/dev/null | head -1)
    if [ -n "$latest_batch" ]; then
        local batch_id=$(basename "$latest_batch" .md | sed 's/batch_//')
        log "最新批次: $batch_id"
        echo ""
    fi
    
    echo ""
    info "📱 笔记本:"
    local latest_pid_file=$(ls -t ${SCHEDULER_DIR}/laptop_*.pid 2>/dev/null | head -1)
    if [ -f "$latest_pid_file" ]; then
        local pid=$(cat "$latest_pid_file")
        local batch_id=$(basename "$latest_pid_file" .pid | sed 's/laptop_//')
        if ps -p $pid > /dev/null 2>&1; then
            echo "   状态: 🟢 运行中 (PID: $pid, 批次: $batch_id)"
            echo "   日志尾部:"
            tail -5 "${SCHEDULER_DIR}/laptop_${batch_id}.log" 2>/dev/null | sed 's/^/   /'
        else
            echo "   状态: ⚪ 已完成或停止 (批次: $batch_id)"
        fi
    else
        echo "   状态: ⚪ 无运行记录"
    fi
    
    echo ""
    info "🖥️ 台式机:"
    ssh ${DESKTOP_HOST} "
        latest=\$(ls -t ${SCHEDULER_DIR}/desktop_*.sh 2>/dev/null | head -1)
        if [ -n \"\$latest\" ]; then
            batch_id=\$(basename \"\$latest\" .sh | sed 's/desktop_//')
            if pgrep -f \"desktop_\${batch_id}\" > /dev/null; then
                echo \"   状态: 🟢 运行中 (批次: \$batch_id)\"
                echo '   日志尾部:'
                tail -5 ${SCHEDULER_DIR}/desktop_\${batch_id}.log 2>/dev/null | sed 's/^/   /'
            else
                echo \"   状态: ⚪ 已完成或停止 (批次: \$batch_id)\"
            fi
        else
            echo '   状态: ⚪ 无运行记录'
        fi
    " 2>/dev/null || echo "   状态: 🔴 无法连接"
}

# 收集结果
collect_results() {
    echo ""
    log "============================================"
    log "📊 收集实验结果"
    log "============================================"
    
    echo ""
    info "📱 笔记本结果 (最近24小时):"
    find ${LEROBOT_DIR}/outputs -name "eval_results.json" -mtime -1 2>/dev/null | while read f; do
        echo ""
        echo "   📁 $f"
        cat "$f" | sed 's/^/      /'
    done
    
    echo ""
    info "🖥️ 台式机结果 (最近24小时):"
    ssh ${DESKTOP_HOST} "find ${LEROBOT_DIR}/outputs -name 'eval_results.json' -mtime -1 2>/dev/null" | while read f; do
        echo ""
        echo "   📁 $f"
        ssh ${DESKTOP_HOST} "cat $f" | sed 's/^/      /'
    done
}

# 列出所有批次
list_batches() {
    echo ""
    log "============================================"
    log "📋 历史批次列表"
    log "============================================"
    echo ""
    
    for batch_file in $(ls -t ${SCHEDULER_DIR}/batch_*.md 2>/dev/null); do
        local batch_id=$(basename "$batch_file" .md | sed 's/batch_//')
        local create_time=$(head -5 "$batch_file" | grep "创建时间" | cut -d: -f2-)
        echo "   📦 ${batch_id} - ${create_time}"
    done
}

# 如果直接运行脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        status)
            check_status
            ;;
        results)
            collect_results
            ;;
        list)
            list_batches
            ;;
        *)
            echo "用法: $0 {status|results|list}"
            echo ""
            echo "命令:"
            echo "  status  - 检查当前实验状态"
            echo "  results - 收集最近24小时的实验结果"
            echo "  list    - 列出历史批次"
            echo ""
            echo "或者 source 后使用函数:"
            echo "  source $0"
            echo "  add_laptop_exp 'diffusion' 'exp1' '--steps 100000'"
            echo "  add_desktop_exp 'act' 'exp1' '--steps 50000'"
            echo "  run_all"
            ;;
    esac
fi
