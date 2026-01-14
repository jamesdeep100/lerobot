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
    """加载策略模型和后处理器"""
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
    
    # 加载预处理器（用于归一化输入）
    preprocessor = None
    preprocessor_path = Path(model_path) / "policy_preprocessor.json"
    if preprocessor_path.exists():
        try:
            from lerobot.processor import PolicyProcessorPipeline
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=model_path,
                config_filename="policy_preprocessor.json"
            )
            log(f"[{time.strftime('%H:%M:%S')}] ✅ 预处理器加载成功 (用于输入归一化)")
        except Exception as e:
            log(f"[{time.strftime('%H:%M:%S')}] ⚠️ 预处理器加载失败: {e}")
    
    # 加载后处理器（用于反归一化动作）
    postprocessor = None
    postprocessor_path = Path(model_path) / "policy_postprocessor.json"
    if postprocessor_path.exists():
        try:
            from lerobot.processor import PolicyProcessorPipeline
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=model_path,
                config_filename="policy_postprocessor.json"
            )
            log(f"[{time.strftime('%H:%M:%S')}] ✅ 后处理器加载成功 (用于动作反归一化)")
        except Exception as e:
            log(f"[{time.strftime('%H:%M:%S')}] ⚠️ 后处理器加载失败: {e}")
    
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
    
    return policy, preprocessor, postprocessor


def evaluate(policy, n_episodes: int = 50, verbose: bool = True, video_dir: str = None, n_video_episodes: int = 3, preprocessor=None, postprocessor=None):
    """评估模型
    
    Args:
        policy: 策略模型
        n_episodes: 评估轮数
        verbose: 是否显示详细日志
        preprocessor: 输入预处理器（用于归一化）
        postprocessor: 动作后处理器（用于反归一化）
        video_dir: 视频保存目录，None 则不录制
        n_video_episodes: 录制视频的 episode 数量
    """
    import gym_pusht  # 确保环境已注册
    from gymnasium.wrappers import RecordVideo
    
    log(f"\n[{time.strftime('%H:%M:%S')}] 🎮 创建 PushT 环境...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    # 添加视频录制
    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=video_dir,
            episode_trigger=lambda ep: ep < n_video_episodes,  # 只录制前几个 episode
            name_prefix="eval"
        )
        log(f"[{time.strftime('%H:%M:%S')}] 🎬 视频将保存到: {video_dir} (前{n_video_episodes}个episode)")
    
    successes = []
    avg_rewards = []  # 每个 episode 的平均奖励（reward_sum / steps）
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
            img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1) / 255.0
            state = torch.from_numpy(obs["agent_pos"]).float()
            
            batch = {
                "observation.image": img,
                "observation.state": state,
            }
            
            # 应用预处理器（归一化输入）
            if preprocessor is not None:
                batch = preprocessor(batch)
            else:
                # 如果没有预处理器，手动添加 batch 维度和移到 GPU
                batch = {
                    "observation.image": img.unsqueeze(0).cuda(),
                    "observation.state": state.unsqueeze(0).cuda(),
                }
            
            # 推理
            with torch.no_grad():
                action = policy.select_action(batch)
            
            # 应用后处理器（反归一化动作）
            if postprocessor is not None:
                action_dict = {"action": action}
                action_dict = postprocessor(action_dict)
                action = action_dict["action"]
            
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
        
        # 使用环境返回的 is_success（官方标准：coverage > 0.95）
        success = info.get("is_success", False)
        successes.append(success)
        avg_rewards.append(episode_reward / step)  # 该 episode 的平均奖励
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
        "avg_reward": float(np.mean(avg_rewards)),  # 所有 episode 平均奖励的平均值
        "avg_max_reward": float(np.mean(max_rewards)),
        "n_episodes": n_episodes,
        "total_time_s": total_time,
        "avg_episode_time_s": float(np.mean(episode_times)),
    }
    
    log("\n" + "=" * 60)
    log(f"[{time.strftime('%H:%M:%S')}] 📊 评估结果:")
    log(f"   pc_success: {results['pc_success']:.1f}%")
    log(f"   avg_reward: {results['avg_reward']:.4f}")  # 平均奖励，范围 0-1
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
    parser.add_argument("--video_dir", type=str, default=None, help="视频保存目录 (默认: 模型目录/videos)")
    parser.add_argument("--n_video_episodes", type=int, default=3, help="录制视频的 episode 数")
    
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
    
    # 加载模型、预处理器和后处理器
    policy, preprocessor, postprocessor = load_policy(str(model_path), args.policy_type)
    
    # 确定视频目录（默认保存到模型目录下的 videos/）
    video_dir = args.video_dir
    if video_dir is None:
        video_dir = str(model_path / "videos")
    
    # 评估
    results = evaluate(
        policy, 
        n_episodes=args.n_episodes, 
        verbose=not args.quiet,
        video_dir=video_dir,
        n_video_episodes=args.n_video_episodes,
        preprocessor=preprocessor,
        postprocessor=postprocessor
    )
    
    # 保存结果
    output_path = args.output or (model_path / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n📁 结果已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
