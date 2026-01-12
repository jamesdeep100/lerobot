# Model Experiment Leaderboard

> **User Instruction:** Before starting any new experiment, READ this table to understand the current SOTA and parameter history.

## Current SOTA

| Task | Policy | Success Rate | Config | Experiment |
|------|--------|--------------|--------|------------|
| PushT | Diffusion | **54%** | 100k steps, h=32, a=8 | exp_006 |
| PushT | ACT | **24%** | 20k steps, dim=1024, dec=4, chunk=10 | exp_005 |

---

## Diffusion Policy Experiments (PushT)

| Exp ID | Date | Parent | Steps | horizon | n_action | down_dims | Success | avg_sum | avg_max | Status | Notes |
|--------|------|--------|-------|---------|----------|-----------|---------|---------|---------|--------|-------|
| exp_001 | 2026-01-10 | - | 20k | 16 | 8 | [256,512,1024] | 12% | 79.5 | 62.6% | Done | Baseline |
| exp_002 | 2026-01-10 | exp_001 | 50k | 16 | 8 | [256,512,1024] | 16% | 112.2 | 62.6% | Done | More steps |
| exp_003 | 2026-01-10 | exp_001 | 20k | **32** | 8 | [256,512,1024] | 18% | 86.1 | 54.1% | Done | Larger horizon |
| exp_004 | 2026-01-10 | exp_001 | 20k | 16 | **16** | [256,512,1024] | 2% | 53.6 | - | Done | More action steps (bad) |
| exp_005 | 2026-01-10 | exp_001 | 20k | 16 | **4** | [256,512,1024] | 6% | 76.4 | 54.1% | Done | Less action steps (bad) |
| **exp_006** | 2026-01-10 | exp_003 | **100k** | **32** | 8 | [256,512,1024] | **54%** | 97.14 | 96.5% | Done | **🏆 SOTA** |

### Key Findings (Diffusion)

1. `horizon=32` > `horizon=16` (+6pp)
2. `n_action_steps=8` is optimal
3. 100k steps = 125 epochs, significant improvement over 20k

---

## ACT Experiments (PushT)

| Exp ID | Date | Parent | Steps | dim_model | n_dec | chunk_size | n_action | Success | avg_sum | avg_max | Status | Notes |
|--------|------|--------|-------|-----------|-------|------------|----------|---------|---------|---------|--------|-------|
| act_001 | 2026-01-10 | - | 5k | 512 | 1 | **10** | **10** | ~2% | ~109 | - | Done | Baseline |
| act_002 | 2026-01-10 | act_001 | 20k | 512 | 1 | **10** | **10** | 10% | 99.9 | 65.2% | Done | More steps |
| act_003 | 2026-01-10 | act_001 | 20k | 512 | **2** | **10** | **10** | 2% | 116.3 | 65.8% | Done | Deeper decoder |
| act_004 | 2026-01-10 | act_001 | 20k | **1024** | **2** | **10** | **10** | 14% | 102.4 | 71.2% | Done | Wider + deeper |
| **act_005** | 2026-01-10 | act_004 | 20k | **1024** | **4** | **10** | **10** | **24%** | 95.5 | 71.6% | Done | **🏆 SOTA** |
| ~~act_batch_01~~ | 2026-01-11 | - | 100k | 512 | 1 | ~~100~~ | 10 | 0% | 17.25 | 7.7% | Failed | ❌ Wrong chunk_size! |
| ~~act_batch_02~~ | 2026-01-11 | - | 50k | 1024 | 4 | ~~100~~ | 10 | 0% | 14.97 | 5.7% | Failed | ❌ Wrong chunk_size! |
| ~~act_batch_03~~ | 2026-01-11 | - | 100k | 1024 | 4 | ~~100~~ | 10 | 0% | 23.21 | 7.9% | Failed | ❌ Wrong chunk_size! |

### Key Findings (ACT)

1. ⚠️ **chunk_size=10 is critical!** Do NOT change to 100!
2. `dim_model=1024` > `dim_model=512`
3. `n_decoder_layers=4` is optimal
4. More training (50k+) may further improve

### Critical Parameters (ACT)

```yaml
# NEVER change these without explicit experiment purpose:
chunk_size: 10        # Predicts 1 second of actions
n_action_steps: 10    # Must match chunk_size
```

---

## Failed Experiments Log

| Date | Issue | Root Cause | Lesson |
|------|-------|------------|--------|
| 2026-01-12 | 10 ACT experiments all 0% success | chunk_size changed from 10 to 100 | Always check experiment_registry.md before setting params |

---

## Next Experiment Plan

- [ ] ACT with 50k steps, chunk_size=10, dim=1024, dec=4 (extend SOTA)
- [ ] Diffusion with 200k steps (test ceiling)
- [ ] ACT with chunk_size=20 (controlled experiment)

---

## 归档规范

每个实验必须包含以下文件：

| 文件 | 说明 | 必须 |
|------|------|------|
| `config.yaml` | 参数配置 | ✅ |
| `metadata.yaml` | 环境元数据、代码版本 | ✅ |
| `run_train.sh` | 训练启动脚本 | ✅ |
| `run_eval.sh` | 评估启动脚本 | ✅ |
| `train_snapshot.py` | 训练代码快照 | ✅ |
| `eval_snapshot.py` | 评估代码快照 | ✅ |
| `train.log` | 训练日志 | ✅ |
| `eval.log` | 评估日志 | ✅ |
| `eval_result.json` | 评估结果 | ✅ |
| `model.safetensors` | 模型权重 | ✅ |
| `notes.md` | 实验结论 | ✅ |

### 代码追溯

- 每个实验对应一个 git 分支: `exp/exp_NNN_name`
- 训练完成后自动创建分支
- 通过 `git checkout exp/exp_NNN_name` 可恢复当时代码

---

*Last Updated: 2026-01-12*
*Protocol Version: CNEP v1.1*
