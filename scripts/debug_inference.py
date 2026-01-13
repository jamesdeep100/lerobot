#!/usr/bin/env python3
"""
调试推理脚本：检查模型在 PushT 环境中的行为
"""

import torch
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
import gym_pusht

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/james/ai_projects/lerobot-experiments/20260113_0000_batch/20260113_0000_sota_reproduce"
    
    print(f"=== 调试推理 ===")
    print(f"模型路径: {model_path}")
    
    # 加载模型
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(model_path)
    policy.eval()
    policy.cuda()
    
    print(f"\n模型配置:")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  dim_model: {policy.config.dim_model}")
    
    # 检查 policy 的 preprocessor 和 postprocessor
    print(f"\n预处理器/后处理器:")
    if hasattr(policy, 'policy_preprocessor'):
        print(f"  preprocessor: {policy.policy_preprocessor}")
    if hasattr(policy, 'policy_postprocessor'):
        print(f"  postprocessor: {policy.policy_postprocessor}")
    
    # 创建环境
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    print(f"\n环境信息:")
    print(f"  action_space: {env.action_space}")
    print(f"  observation_space: {env.observation_space}")
    
    # 重置环境
    obs, info = env.reset(seed=42)
    policy.reset()
    
    print(f"\n观察数据:")
    print(f"  pixels shape: {obs['pixels'].shape}, dtype: {obs['pixels'].dtype}")
    print(f"  pixels range: [{obs['pixels'].min()}, {obs['pixels'].max()}]")
    print(f"  agent_pos: {obs['agent_pos']}")
    
    # 准备输入
    img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
    
    print(f"\n模型输入:")
    print(f"  image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  state shape: {state.shape}, values: {state[0].numpy()}")
    
    batch = {
        "observation.image": img.cuda(),
        "observation.state": state.cuda(),
    }
    
    # 推理
    with torch.no_grad():
        action = policy.select_action(batch)
    
    action_np = action.cpu().numpy().flatten()
    
    print(f"\n模型输出:")
    print(f"  action shape: {action.shape}")
    print(f"  action values: {action_np}")
    print(f"  action range: [{action_np.min():.3f}, {action_np.max():.3f}]")
    
    # 检查动作是否在合理范围内
    # PushT 的动作空间是 [-1, 1] 或者 [0, 512] ?
    print(f"\n环境动作空间:")
    print(f"  low: {env.action_space.low}")
    print(f"  high: {env.action_space.high}")
    
    # 执行多步，观察轨迹
    print(f"\n=== 执行 20 步观察轨迹 ===")
    positions = [obs['agent_pos'].copy()]
    actions = []
    
    obs, info = env.reset(seed=42)
    policy.reset()
    
    for step in range(20):
        img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
        
        batch = {
            "observation.image": img.cuda(),
            "observation.state": state.cuda(),
        }
        
        with torch.no_grad():
            action = policy.select_action(batch)
        
        action_np = action.cpu().numpy().flatten()
        actions.append(action_np.copy())
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        positions.append(obs['agent_pos'].copy())
        
        print(f"  Step {step+1:2d}: pos={positions[-1]}, action={action_np}, reward={reward:.2f}")
    
    env.close()
    
    # 分析轨迹
    positions = np.array(positions)
    actions = np.array(actions)
    
    print(f"\n=== 轨迹分析 ===")
    print(f"位置变化:")
    print(f"  起点: {positions[0]}")
    print(f"  终点: {positions[-1]}")
    print(f"  位移: {positions[-1] - positions[0]}")
    
    print(f"\n动作统计:")
    print(f"  均值: {actions.mean(axis=0)}")
    print(f"  标准差: {actions.std(axis=0)}")
    print(f"  范围: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # 检查动作是否总是指向同一方向（问题症状）
    if np.all(actions[:, 0] < 0) or np.all(actions[:, 0] > 0):
        print(f"\n⚠️ 警告: 动作 x 分量全部同号 ({'+' if actions[0, 0] > 0 else '-'})")
    if np.all(actions[:, 1] < 0) or np.all(actions[:, 1] > 0):
        print(f"⚠️ 警告: 动作 y 分量全部同号 ({'+' if actions[0, 1] > 0 else '-'})")

if __name__ == "__main__":
    main()
