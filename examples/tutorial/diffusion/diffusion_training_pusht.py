"""
Diffusion Policy 训练脚本 - PushT 任务专用版

这是基于 LeRobot 官方示例修改的 PushT 训练 Demo：
- 使用 lerobot/pusht 数据集
- 适配 RTX 5090 + PyTorch nightly (使用 pyav 而非 torchcodec)
- 添加详细的训练信息输出
- 包含模型结构分析

运行方式：
    cd ~/ai_projects/lerobot
    python examples/tutorial/diffusion/diffusion_training_pusht.py

作者：James (LeRobot 学习笔记)
日期：2026-01-09
"""

from pathlib import Path
import time

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将帧索引转换为时间戳"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_architecture(policy, cfg):
    """打印模型结构详情"""
    print("\n" + "="*70)
    print("📐 Diffusion Policy 网络结构")
    print("="*70)
    
    total_params, trainable_params = count_parameters(policy)
    
    # 打印配置
    print(f"\n📊 模型配置:")
    print(f"   n_obs_steps:         {cfg.n_obs_steps}")
    print(f"   horizon:             {cfg.horizon}")
    print(f"   n_action_steps:      {cfg.n_action_steps}")
    print(f"   vision_backbone:     {cfg.vision_backbone}")
    print(f"   down_dims:           {cfg.down_dims}")
    print(f"   noise_scheduler:     {cfg.noise_scheduler_type}")
    print(f"   num_train_timesteps: {cfg.num_train_timesteps}")
    
    print(f"\n📊 参数统计:")
    print(f"   总参数量:     {total_params:,} ({total_params/1e6:.2f}M / {total_params/1e9:.4f}B)")
    print(f"   可训练参数:   {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 分析各组件参数
    print(f"\n📊 组件参数分布:")
    
    # RGB encoder
    if hasattr(policy, 'rgb_encoder'):
        rgb_params = sum(p.numel() for p in policy.rgb_encoder.parameters())
        print(f"   ├─ RGB Encoder:     {rgb_params:>10,} ({rgb_params/1e6:.2f}M)")
    
    # UNet
    if hasattr(policy, 'unet'):
        unet_params = sum(p.numel() for p in policy.unet.parameters())
        print(f"   ├─ UNet Diffusion:  {unet_params:>10,} ({unet_params/1e6:.2f}M)")
    
    print(f"   └─ 总计:            {total_params:>10,} ({total_params/1e6:.2f}M)")
    
    # ASCII 结构图
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    Diffusion Policy 网络结构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      输入 (Inputs)                                │   │
│  │  observation.image: [B, {cfg.n_obs_steps}, 3, 96, 96]              │   │
│  │  observation.state: [B, {cfg.n_obs_steps}, 2]                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  RGB Encoder (ResNet-18)                          │   │
│  │   输入: [{cfg.n_obs_steps} 帧] × [3, 96, 96]                         │   │
│  │   输出: Visual Features (SpatialSoftmax → Keypoints)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│                            │ Conditioning Features                      │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  1D Conditional UNet                              │   │
│  │   ┌─────────────────────────────────────────────────────────┐   │   │
│  │   │  Down Blocks: {cfg.down_dims}                           │   │   │
│  │   │  Kernel Size: {cfg.kernel_size}                                   │   │   │
│  │   │  FiLM Conditioning (视觉特征 + 状态)                      │   │   │
│  │   │  Diffusion Step Embedding (dim={cfg.diffusion_step_embed_dim})       │   │   │
│  │   └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │   输入: Noisy Actions [B, horizon, action_dim]                    │   │
│  │   输出: Denoised Actions [B, horizon, action_dim]                 │   │
│  │                                                                     │   │
│  │   训练: 预测噪声 ε (prediction_type="{cfg.prediction_type}")        │   │
│  │   推理: {cfg.num_train_timesteps} 步 DDPM 去噪 → 干净动作序列                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      输出 (Output)                                 │   │
│  │   Action Sequence: [B, {cfg.horizon}, 2]                           │   │
│  │   实际执行: 前 {cfg.n_action_steps} 步动作                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

🔄 Diffusion 过程:
   训练: x₀ (真实动作) → 加噪 → x_t → UNet 预测噪声 → MSE Loss
   推理: x_T (纯噪声)  → {cfg.num_train_timesteps} 步去噪 → x₀ (预测动作)
""")
    print("="*70 + "\n")


def main():
    # ==================== 配置 ====================
    output_directory = Path("outputs/diffusion_pusht_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 数据集选择 - PushT
    dataset_id = "lerobot/pusht"
    
    # 训练参数
    training_steps = 50000    # 训练步数
    batch_size = 32           # 批量大小
    log_freq = 100            # 日志频率
    
    print(f"\n📚 数据集: {dataset_id}")
    print(f"🎯 训练步数: {training_steps}")
    print(f"📦 批量大小: {batch_size}")
    
    # ==================== 数据集准备 ====================
    print("\n⏳ 加载数据集元数据...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 分离输入/输出特征
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"\n📊 数据集特征:")
    print(f"   FPS: {dataset_metadata.fps}")
    print(f"   输入特征: {list(input_features.keys())}")
    print(f"   输出特征: {list(output_features.keys())}")
    
    # ==================== 模型配置 ====================
    # Diffusion Policy 默认配置已经为 PushT 优化
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # ---- 架构参数（使用默认值，已针对 PushT 优化）----
        n_obs_steps=2,           # 观察帧数
        horizon=16,              # 预测时间范围
        n_action_steps=8,        # 实际执行步数
        vision_backbone="resnet18",
        down_dims=(512, 1024, 2048),  # UNet 通道数
        # ---- 扩散参数 ----
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        # ---- 学习率 ----
        optimizer_lr=1e-4,
    )
    
    # ==================== 模型创建 ====================
    print("\n⏳ 创建模型...")
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # 打印模型结构
    print_model_architecture(policy, cfg)
    
    # 创建预处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # ==================== 数据加载 ====================
    # 构建 delta_timestamps
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    
    # 添加图像特征的 delta_timestamps
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    
    print(f"\n📊 Delta Timestamps:")
    for k, v in delta_timestamps.items():
        print(f"   {k}: {v}")
    
    # 加载数据集 (使用 pyav 替代 torchcodec，兼容 RTX 5090)
    print("\n⏳ 加载数据集...")
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    print(f"✅ 数据集加载完成，共 {len(dataset)} 个样本")
    
    # 创建 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=4,
    )
    
    # ==================== 优化器 ====================
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    print(f"\n🔧 优化器: Adam (lr={cfg.optimizer_lr})")
    
    # ==================== 训练循环 ====================
    print("\n" + "="*50)
    print("🚀 开始训练 Diffusion Policy")
    print("="*50 + "\n")
    
    step = 0
    start_time = time.time()
    running_loss = 0.0
    
    done = False
    while not done:
        for batch in dataloader:
            # 移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 预处理
            batch = preprocessor(batch)
            
            # 前向传播 + 反向传播
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录损失
            running_loss += loss.item()
            
            # 日志输出
            if step > 0 and step % log_freq == 0:
                avg_loss = running_loss / log_freq
                elapsed = time.time() - start_time
                
                # 计算预估剩余时间
                steps_done = step
                steps_remaining = training_steps - step
                speed = elapsed / steps_done  # 秒/步
                eta_seconds = steps_remaining * speed
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | Time: {elapsed:.1f}s | ETA: {eta_min}m {eta_sec}s")
                if torch.cuda.is_available():
                    print(f"         | GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                running_loss = 0.0
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # ==================== 保存模型 ====================
    total_time = time.time() - start_time
    print("\n" + "-"*50)
    print(f"✅ 训练完成!")
    print(f"   总步数: {training_steps}")
    print(f"   总时间: {total_time:.1f}s")
    print(f"   平均速度: {total_time/training_steps*1000:.1f} ms/step")
    print(f"   最终 Loss: {loss.item():.4f}")
    
    print(f"\n💾 保存模型到: {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    
    print(f"\n🎉 Demo 完成！")
    print(f"   模型路径: {output_directory.absolute()}")
    print(f"   评估命令: lerobot-eval --policy.path={output_directory} --env.type=pusht --eval.n_episodes=10")


if __name__ == "__main__":
    main()
