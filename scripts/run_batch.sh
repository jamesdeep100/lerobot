#!/bin/bash
# ============================================
# 多机实验调度脚本 (CNEP v1.2)
#
# 用法:
#   source scripts/run_batch.sh
#   add_laptop_exp "act" "exp_name" "--steps 50000 --eval"
#   add_desktop_exp "diffusion" "exp_name" "--steps 100000 --eval"
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
DESKTOP_HOST="james@100.67.100.43"
LEROBOT_DIR="/home/james/ai_projects/lerobot"
EXPERIMENTS_DIR="/home/james/ai_projects/lerobot-experiments"

# 当前批次时间戳 (YYYYMMDD_HHMM 格式)
BATCH_TIMESTAMP=$(date '+%Y%m%d_%H%M')

# 批次目录 (保存到独立的实验仓库)
BATCH_DIR="${EXPERIMENTS_DIR}/${BATCH_TIMESTAMP}_batch"

# 实验列表
declare -a LAPTOP_EXPS
declare -a DESKTOP_EXPS

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

# 批次目的（在 run_all 前设置）
BATCH_PURPOSE=""

# 设置批次目的
set_batch_purpose() {
    BATCH_PURPOSE="$1"
}

# 添加笔记本实验
# 用法: add_laptop_exp "policy" "name" "options" ["purpose"]
add_laptop_exp() {
    local policy_type="$1"
    local base_name="$2"
    local options="$3"
    local purpose="${4:-}"
    
    # 子实验名称：YYYYMMDD_HHMM_name
    local exp_name="${BATCH_TIMESTAMP}_${base_name}"
    LAPTOP_EXPS+=("${policy_type}|${exp_name}|${options}|${purpose}")
}

# 添加台式机实验
# 用法: add_desktop_exp "policy" "name" "options" ["purpose"]
add_desktop_exp() {
    local policy_type="$1"
    local base_name="$2"
    local options="$3"
    local purpose="${4:-}"
    
    # 子实验名称：YYYYMMDD_HHMM_name
    local exp_name="${BATCH_TIMESTAMP}_${base_name}"
    DESKTOP_EXPS+=("${policy_type}|${exp_name}|${options}|${purpose}")
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
            IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "laptop" "$eval_eps")
            laptop_total=$((laptop_total + est_time))
            
            echo -e "${BLUE}║${NC}    ├─ ${GREEN}${exp_name}${NC} (${policy_type})"
            echo -e "${BLUE}║${NC}    │  步数: ${steps}, 预估: $(format_time $est_time)"
        done
    fi
    
    echo -e "${BLUE}║${NC}"
    
    # 台式机实验
    echo -e "${BLUE}║${NC} ${CYAN}🖥️ 台式机 (RTX 3060 Ti)${NC}"
    if [ ${#DESKTOP_EXPS[@]} -eq 0 ]; then
        echo -e "${BLUE}║${NC}    (无实验)"
    else
        for exp in "${DESKTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "desktop" "$eval_eps")
            desktop_total=$((desktop_total + est_time))
            
            echo -e "${BLUE}║${NC}    ├─ ${GREEN}${exp_name}${NC} (${policy_type})"
            echo -e "${BLUE}║${NC}    │  步数: ${steps}, 预估: $(format_time $est_time)"
        done
    fi
    
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} ${YELLOW}⏱️ 预估总时长${NC}"
    echo -e "${BLUE}║${NC}    笔记本: $(format_time $laptop_total)"
    echo -e "${BLUE}║${NC}    台式机: $(format_time $desktop_total)"
    echo -e "${BLUE}║${NC}    并行时间: $(format_time $(($laptop_total > $desktop_total ? $laptop_total : $desktop_total)))"
    echo -e "${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}📁 批次目录: ${BATCH_DIR}${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 生成批次日志
generate_batch_log() {
    local batch_log="${BATCH_DIR}/batch.md"
    
    cat > "$batch_log" << EOF
# 实验批次: ${BATCH_TIMESTAMP}_batch

> 创建时间: $(date '+%Y-%m-%d %H:%M:%S')
> 目录: ${BATCH_DIR}

## 🎯 批次目的

${BATCH_PURPOSE:-（未设置）}

## 📋 实验清单

### 📱 笔记本 (RTX 5090)
EOF
    
    if [ ${#LAPTOP_EXPS[@]} -eq 0 ]; then
        echo "无实验" >> "$batch_log"
    else
        for exp in "${LAPTOP_EXPS[@]}"; do
            IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "laptop" "$eval_eps")
            
            cat >> "$batch_log" << EOF

#### ${exp_name}
- **目的**: ${purpose:-（未设置）}
- **策略**: ${policy_type}
- **参数**: ${options}
- **预估时长**: $(format_time $est_time)
- **目录**: \`${exp_name}/\`
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
            IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
            local steps=$(get_steps_from_options "$options")
            local eval_eps=$(get_eval_episodes_from_options "$options")
            local est_time=$(estimate_time "$policy_type" "$steps" "desktop" "$eval_eps")
            
            cat >> "$batch_log" << EOF

#### ${exp_name}
- **目的**: ${purpose:-（未设置）}
- **策略**: ${policy_type}
- **参数**: ${options}
- **预估时长**: $(format_time $est_time)
- **目录**: \`${exp_name}/\`
- **状态**: 🔵 待运行
EOF
        done
    fi
    
    cat >> "$batch_log" << EOF

---

## 📁 文件位置

| 类型 | 路径 |
|------|------|
| 批次日志 | \`batch.md\` |
| 笔记本执行脚本 | \`laptop.sh\` |
| 笔记本运行日志 | \`laptop.log\` |
| 台式机执行脚本 | \`desktop.sh\` |
| 台式机运行日志 | \`desktop.log\` |

---

## 🕐 时间线

- $(date '+%H:%M:%S') - 批次创建
EOF
    
    echo "$batch_log"
}

# 更新批次日志
update_batch_log() {
    local message="$1"
    echo "- $(date '+%H:%M:%S') - ${message}" >> "${BATCH_DIR}/batch.md"
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
echo \"批次: batch_${BATCH_TIMESTAMP}\"
echo \"目录: ${BATCH_DIR}\"
echo \"============================================\"
"
    
    for exp in "${LAPTOP_EXPS[@]}"; do
        IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
        local exp_dir="${BATCH_DIR}/${exp_name}"
        
        script_content+="
echo \"\"
echo \"======================================\"
echo \"🔬 实验: ${exp_name} (${policy_type})\"
echo \"======================================\"
./scripts/train_${policy_type}.sh ${exp_name} --output_dir ${exp_dir} ${options}
"
    done
    
    script_content+="
echo \"\"
echo \"============================================\"
echo \"✅ 笔记本实验完成: \$(date)\"
echo \"============================================\"
"
    
    # 保存脚本到批次目录
    local script_file="${BATCH_DIR}/laptop.sh"
    echo "$script_content" > "$script_file"
    chmod +x "$script_file"
    
    # 后台运行
    local log_file="${BATCH_DIR}/laptop.log"
    nohup bash "$script_file" > "$log_file" 2>&1 &
    local pid=$!
    
    log "✅ 笔记本实验已启动"
    log "   PID: $pid"
    log "   脚本: $script_file"
    log "   日志: $log_file"
    
    echo "$pid" > "${BATCH_DIR}/laptop.pid"
    
    update_batch_log "笔记本实验启动 (PID: $pid)"
}

# 在台式机上运行实验
run_desktop_experiments() {
    if [ ${#DESKTOP_EXPS[@]} -eq 0 ]; then
        log "台式机无实验"
        return
    fi
    
    log "🚀 启动台式机实验 (${#DESKTOP_EXPS[@]} 个)..."
    
    # 远程批次目录 (台式机上也使用独立的实验仓库)
    local remote_batch_dir="${EXPERIMENTS_DIR}/${BATCH_TIMESTAMP}_batch"
    
    local script_content="#!/bin/bash
cd ${LEROBOT_DIR}
eval \"\$(~/miniconda3/bin/conda shell.bash hook)\"
conda activate lerobot

# 修复 pymunk
pip uninstall pymunk -y 2>/dev/null || true
pip install pymunk==6.4.0 -q 2>/dev/null || true

echo \"============================================\"
echo \"🌙 台式机实验开始: \$(date)\"
echo \"批次: batch_${BATCH_TIMESTAMP}\"
echo \"目录: ${remote_batch_dir}\"
echo \"============================================\"
"
    
    for exp in "${DESKTOP_EXPS[@]}"; do
        IFS='|' read -r policy_type exp_name options purpose <<< "$exp"
        local exp_dir="${remote_batch_dir}/${exp_name}"
        
        script_content+="
echo \"\"
echo \"======================================\"
echo \"🔬 实验: ${exp_name} (${policy_type})\"
echo \"======================================\"
./scripts/train_${policy_type}.sh ${exp_name} --output_dir ${exp_dir} ${options}
"
    done
    
    script_content+="
echo \"\"
echo \"============================================\"
echo \"✅ 台式机实验完成: \$(date)\"
echo \"============================================\"
"
    
    # 确保远程批次目录存在
    ssh ${DESKTOP_HOST} "mkdir -p ${remote_batch_dir}"
    
    # 通过 SSH 创建并运行脚本
    local remote_script="${remote_batch_dir}/desktop.sh"
    local remote_log="${remote_batch_dir}/desktop.log"
    
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
    # 创建批次目录
    mkdir -p "${BATCH_DIR}"
    
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
    log "📁 批次目录: ${BATCH_DIR}"
    log ""
    log "📋 检查命令:"
    log "   笔记本: tail -f ${BATCH_DIR}/laptop.log"
    log "   台式机: ssh ${DESKTOP_HOST} 'tail -f ${BATCH_DIR}/desktop.log'"
    log ""
    log "🛑 停止命令:"
    log "   笔记本: kill \$(cat ${BATCH_DIR}/laptop.pid)"
    log "   台式机: ssh ${DESKTOP_HOST} 'pkill -f batch_${BATCH_TIMESTAMP}'"
    
    update_batch_log "所有实验启动完成"
}

# 检查实验状态
check_status() {
    echo ""
    log "============================================"
    log "📊 实验状态检查"
    log "============================================"
    
    # 查找最新的批次
    local latest_batch=$(ls -td ${LEROBOT_DIR}/experiments/batch_* 2>/dev/null | head -1)
    if [ -n "$latest_batch" ]; then
        local batch_name=$(basename "$latest_batch")
        log "最新批次: $batch_name"
        log "目录: $latest_batch"
        echo ""
    fi
    
    echo ""
    info "📱 笔记本:"
    if [ -f "$latest_batch/laptop.pid" ]; then
        local pid=$(cat "$latest_batch/laptop.pid")
        if ps -p $pid > /dev/null 2>&1; then
            echo "   状态: 🟢 运行中 (PID: $pid)"
            echo "   日志尾部:"
            tail -5 "$latest_batch/laptop.log" 2>/dev/null | sed 's/^/   /'
        else
            echo "   状态: ⚪ 已完成或停止"
        fi
    else
        echo "   状态: ⚪ 无运行记录"
    fi
    
    echo ""
    info "🖥️ 台式机:"
    if [ -n "$latest_batch" ]; then
        ssh ${DESKTOP_HOST} "
            batch_dir='$latest_batch'
            if [ -f \"\$batch_dir/desktop.log\" ]; then
                if pgrep -f 'batch_' > /dev/null; then
                    echo '   状态: 🟢 运行中'
                    echo '   日志尾部:'
                    tail -5 \"\$batch_dir/desktop.log\" 2>/dev/null | sed 's/^/   /'
                else
                    echo '   状态: ⚪ 已完成或停止'
                fi
            else
                echo '   状态: ⚪ 无运行记录'
            fi
        " 2>/dev/null || echo "   状态: 🔴 无法连接"
    fi
}

# 收集结果
collect_results() {
    echo ""
    log "============================================"
    log "📊 收集实验结果"
    log "============================================"
    
    local latest_batch=$(ls -td ${LEROBOT_DIR}/experiments/batch_* 2>/dev/null | head -1)
    
    if [ -z "$latest_batch" ]; then
        error "未找到批次目录"
        return
    fi
    
    log "批次: $(basename $latest_batch)"
    echo ""
    
    for exp_dir in "$latest_batch"/*/; do
        if [ -d "$exp_dir" ]; then
            local exp_name=$(basename "$exp_dir")
            if [ -f "$exp_dir/eval_results.json" ]; then
                echo "📁 $exp_name:"
                cat "$exp_dir/eval_results.json" | sed 's/^/   /'
                echo ""
            fi
        fi
    done
}

# 列出所有批次
list_batches() {
    echo ""
    log "============================================"
    log "📋 历史批次列表"
    log "============================================"
    echo ""
    
    for batch_dir in $(ls -td ${LEROBOT_DIR}/experiments/batch_* 2>/dev/null); do
        local batch_name=$(basename "$batch_dir")
        local create_time=""
        if [ -f "$batch_dir/batch.md" ]; then
            create_time=$(head -5 "$batch_dir/batch.md" | grep "创建时间" | cut -d: -f2-)
        fi
        local exp_count=$(find "$batch_dir" -maxdepth 1 -type d | wc -l)
        exp_count=$((exp_count - 1))  # 减去批次目录本身
        echo "   📦 ${batch_name} - ${create_time} (${exp_count} 个实验)"
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
            echo "  results - 收集最新批次的实验结果"
            echo "  list    - 列出历史批次"
            echo ""
            echo "或者 source 后使用函数:"
            echo "  source $0"
            echo "  add_laptop_exp 'act' 'test1' '--steps 50000 --eval'"
            echo "  add_desktop_exp 'diffusion' 'test1' '--steps 100000 --eval'"
            echo "  run_all"
            ;;
    esac
fi
