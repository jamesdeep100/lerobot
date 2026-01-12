# 🛠️ 训练和评估脚本

本目录包含 Diffusion Policy 和 ACT 的通用训练和评估脚本。

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `train_diffusion.sh` | Diffusion Policy 训练脚本 |
| `train_act.sh` | ACT 训练脚本 |
| `eval_model.py` | 通用模型评估脚本 |
| `run_experiments.sh` | **多机实验调度脚本** |

---

## 🚀 使用方法

### Diffusion Policy 训练

```bash
# 基本用法
./scripts/train_diffusion.sh exp1_100k --steps 100000

# 完整参数
./scripts/train_diffusion.sh exp2_wide \
    --steps 50000 \
    --horizon 16 \
    --n_action_steps 8 \
    --batch_size 32 \
    --down_dims "512,1024,2048" \
    --eval \
    --eval_episodes 50
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | 50000 | 训练步数 |
| `--horizon` | 16 | 预测动作序列长度 |
| `--n_action_steps` | 8 | 每次执行的动作步数 |
| `--batch_size` | 32 | 批量大小 |
| `--down_dims` | "512,1024,2048" | UNet 下采样维度 |
| `--eval` | false | 训练后自动评估 |
| `--eval_episodes` | 50 | 评估 episode 数 |

### ACT 训练

```bash
# 基本用法
./scripts/train_act.sh exp1_50k --steps 50000

# 完整参数
./scripts/train_act.sh exp2_large \
    --steps 100000 \
    --dim_model 1024 \
    --n_decoder_layers 4 \
    --batch_size 32 \
    --eval
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | 50000 | 训练步数 |
| `--dim_model` | 512 | Transformer 维度 |
| `--n_decoder_layers` | 1 | 解码器层数 |
| `--batch_size` | 32 | 批量大小 |
| `--chunk_size` | 100 | 动作块大小 |
| `--eval` | false | 训练后自动评估 |

### 模型评估

```bash
# 评估 Diffusion Policy
python scripts/eval_model.py \
    --model_path outputs/diffusion_exp/exp1_100k \
    --policy_type diffusion \
    --n_episodes 50

# 评估 ACT
python scripts/eval_model.py \
    --model_path outputs/act_exp/exp1_50k \
    --policy_type act \
    --n_episodes 20
```

---

## ⚠️ 注意事项

### 1. 运行前检查清单

- [ ] 确认 conda 环境已正确配置
- [ ] 检查输出目录不会覆盖重要模型
- [ ] 对于长时间训练，使用 `nohup` 或 `tmux`

### 2. 后台运行

```bash
# 使用 nohup
nohup ./scripts/train_diffusion.sh exp1_100k --steps 100000 > train.log 2>&1 &

# 使用 tmux
tmux new -s training
./scripts/train_diffusion.sh exp1_100k --steps 100000
# Ctrl+B, D 分离
```

### 3. 验证模型保存

训练完成后，检查模型是否正确保存：

```bash
ls -la outputs/diffusion_exp/exp1_100k/
# 应该包含:
#   - model.safetensors
#   - config.json
#   - train.log
```

---

## 📊 推荐配置

### Diffusion Policy (PushT 任务)

```bash
# 最佳配置 (54% 成功率)
./scripts/train_diffusion.sh best_config \
    --steps 100000 \
    --horizon 16 \
    --n_action_steps 8 \
    --batch_size 32 \
    --down_dims "512,1024,2048" \
    --eval
```

### ACT (PushT 任务)

```bash
# 较好配置 (24% 成功率)
./scripts/train_act.sh best_config \
    --steps 20000 \
    --dim_model 1024 \
    --n_decoder_layers 4 \
    --eval
```

---

## 📁 输出目录结构

```
outputs/
├── diffusion_exp/
│   ├── exp1_100k/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── config.txt
│   │   ├── train.log
│   │   └── eval_results.json
│   └── exp2_wide/
│       └── ...
└── act_exp/
    ├── exp1_50k/
    │   └── ...
    └── exp2_large/
        └── ...
```

---

## 🌙 多机实验调度

### 快速使用

```bash
# 进入脚本目录
cd ~/ai_projects/lerobot

# 加载调度函数
source scripts/run_experiments.sh

# 添加笔记本实验 (自动添加时间戳到实验名)
add_laptop_exp "diffusion" "exp1" "--steps 100000 --horizon 16"
add_laptop_exp "diffusion" "exp2" "--steps 50000 --horizon 32"

# 添加台式机实验
add_desktop_exp "act" "exp1" "--steps 50000 --dim_model 1024"
add_desktop_exp "act" "exp2" "--steps 100000 --dim_model 1024 --n_decoder_layers 4"

# 显示计划并启动
run_all
```

### 实验名称格式

实验名称自动添加时间戳：`MMDD_HHMM_<base_name>`

例如：`0111_2230_exp1` 表示 01月11日 22:30 创建的 exp1

### 检查状态

```bash
./scripts/run_experiments.sh status
```

### 收集结果

```bash
./scripts/run_experiments.sh results
```

### 预估时长参考

| 策略 | 机器 | 训练速度 | 10万步预估 |
|------|------|----------|-----------|
| Diffusion | 笔记本 (RTX 5090) | ~100 ms/step | ~2.8h |
| Diffusion | 台式机 (RTX 3060 Ti) | ~190 ms/step | ~5.3h |
| ACT | 笔记本 | ~65 ms/step | ~1.8h |
| ACT | 台式机 | ~100 ms/step | ~2.8h |

---

*创建时间: 2026-01-11*
*更新: 添加多机实验调度*