#!/usr/bin/env python3
"""
通用模型评估脚本
支持 Diffusion Policy 和 ACT

用法:
    python eval_model.py --model_path <模型路径> [选项]

示例:
    python eval_model.py --model_path outputs/diffusion_exp/exp1_100k --policy_type diffusion
    python eval_model.py --model_path outputs/act_exp/exp1 --policy_type act --n_episodes 20
"""

import argparse
import sys
import time
import json
from pathlib import Path

import torch
import numpy as np
import gymnasium as gym

# 确保输出实时刷新
def log(msg):
    print(msg, flush=True)


def load_policy(model_path: str, policy_type: str):
    """加载策略模型"""
    log(f"[{time.strftime('%H:%M:%S')}] 📦 加载 {policy_type} 模型: {model_path}")
    
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(model_path)
    elif policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(model_path)
    else:
        raise ValueError(f"不支持的策略类型: {policy_type}")
    
    policy.eval()
    policy.cuda()
    
    params = sum(p.numel() for p in policy.parameters())
    log(f"[{time.strftime('%H:%M:%S')}] ✅ 模型加载成功")
    log(f"   参数量: {params:,} ({params/1e9:.2f}B)")
    
    # 打印关键配置
    if hasattr(policy.config, 'horizon'):
        log(f"   horizon: {policy.config.horizon}")
    if hasattr(policy.config, 'n_action_steps'):
        log(f"   n_action_steps: {policy.config.n_action_steps}")
    if hasattr(policy.config, 'down_dims'):
        log(f"   down_dims: {policy.config.down_dims}")
    if hasattr(policy.config, 'dim_model'):
        log(f"   dim_model: {policy.config.dim_model}")
    if hasattr(policy.config, 'n_decoder_layers'):
        log(f"   n_decoder_layers: {policy.config.n_decoder_layers}")
    
    return policy


def evaluate(policy, n_episodes: int = 50, verbose: bool = True):
    """评估模型"""
    import gym_pusht  # 确保环境已注册
    
    log(f"\n[{time.strftime('%H:%M:%S')}] 🎮 创建 PushT 环境...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    successes = []
    sum_rewards = []
    max_rewards = []
    episode_times = []
    
    log(f"[{time.strftime('%H:%M:%S')}] 🚀 开始评估 ({n_episodes} episodes)...")
    log("=" * 60)
    
    total_start = time.time()
    
    for ep in range(n_episodes):
        ep_start = time.time()
        obs, info = env.reset()
        policy.reset()
        done = False
        episode_reward = 0
        max_reward = 0
        step = 0
        
        while not done:
            # 准备输入
            img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
            
            batch = {
                "observation.image": img.cuda(),
                "observation.state": state.cuda(),
            }
            
            # 推理
            with torch.no_grad():
                action = policy.select_action(batch)
            
            action = action.cpu().numpy().flatten()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            max_reward = max(max_reward, reward)
            step += 1
            
            # 进度显示（大模型每50步显示一次）
            if verbose and step % 50 == 0:
                log(f"  [Episode {ep+1}] Step {step}...")
        
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        
        success = max_reward >= 1.0
        successes.append(success)
        sum_rewards.append(episode_reward)
        max_rewards.append(max_reward)
        
        # 预估剩余时间
        avg_time = np.mean(episode_times)
        remaining = avg_time * (n_episodes - ep - 1)
        
        log(f"[{time.strftime('%H:%M:%S')}] Episode {ep+1}/{n_episodes}: "
            f"{'✅' if success else '❌'} reward={episode_reward:.1f}, "
            f"max={max_reward:.3f} | {ep_time:.1f}s | 剩余: {remaining:.0f}s")
    
    env.close()
    total_time = time.time() - total_start
    
    # 汇总结果
    results = {
        "pc_success": 100 * np.mean(successes),
        "avg_sum_reward": float(np.mean(sum_rewards)),
        "avg_max_reward": float(np.mean(max_rewards)),
        "n_episodes": n_episodes,
        "total_time_s": total_time,
        "avg_episode_time_s": float(np.mean(episode_times)),
    }
    
    log("\n" + "=" * 60)
    log(f"[{time.strftime('%H:%M:%S')}] 📊 评估结果:")
    log(f"   pc_success: {results['pc_success']:.1f}%")
    log(f"   avg_sum_reward: {results['avg_sum_reward']:.2f}")
    log(f"   avg_max_reward: {results['avg_max_reward']:.4f}")
    log(f"   总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    log(f"   平均每 episode: {results['avg_episode_time_s']:.1f}s")
    log("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="通用模型评估脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--policy_type", type=str, default="diffusion", 
                        choices=["diffusion", "act"], help="策略类型")
    parser.add_argument("--n_episodes", type=int, default=50, help="评估 episode 数")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径 (JSON)")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    
    args = parser.parse_args()
    
    log("=" * 60)
    log(f"🔬 模型评估")
    log(f"   模型路径: {args.model_path}")
    log(f"   策略类型: {args.policy_type}")
    log(f"   评估数量: {args.n_episodes} episodes")
    log("=" * 60)
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        log(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    if not (model_path / "model.safetensors").exists():
        log(f"❌ 模型文件不存在: {model_path / 'model.safetensors'}")
        sys.exit(1)
    
    # 加载模型
    policy = load_policy(str(model_path), args.policy_type)
    
    # 评估
    results = evaluate(policy, n_episodes=args.n_episodes, verbose=not args.quiet)
    
    # 保存结果
    output_path = args.output or (model_path / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n📁 结果已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
