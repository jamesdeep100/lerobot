#!/usr/bin/env python
"""
ACT 训练 Demo - PushT 数据集
============================
基于官方 act_training_example.py 定制，适配 RTX 5090

用法:
    conda activate lerobot
    cd /home/james/ai_projects/lerobot
    python examples/tutorial/act/act_training_pusht.py

作者: James (学习笔记)
日期: 2026-01-06
"""

from pathlib import Path
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将帧索引转换为时间戳偏移"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def print_model_architecture(policy):
    """打印 ACT 网络结构 ASCII 图"""
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    def format_params(n):
        if n >= 1e9:
            return f"{n/1e9:.3f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        else:
            return str(n)
    
    model = policy.model
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        ACT 网络结构 (Action Chunking Transformer)         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      输入 (Inputs)                                │   │
│  │  observation.image: [B, 3, 96, 96]    observation.state: [B, 2]  │   │
│  │  action (训练目标): [B, chunk_size, 2]                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                │                           │
│                            ▼                ▼                           │""")
    
    backbone_params = count_params(model.backbone)
    vae_params = count_params(model.vae_encoder) + 512 + 1536 + 1536 + 32832
    
    print(f"""│  ┌────────────────────────┐    ┌────────────────────────────────┐   │
│  │   Vision Backbone       │    │   VAE Encoder (训练时)          │   │
│  │   ResNet-18             │    │   4-layer Transformer           │   │
│  │   {format_params(backbone_params):>8}              │    │   {format_params(vae_params):>8}                     │   │
│  └────────────────────────┘    └────────────────────────────────┘   │
│            │                                   │                       │
│            │ Visual Tokens                     │ Latent z (dim=32)     │
│            ▼                                   ▼                       │""")
    
    encoder_params = count_params(model.encoder)
    
    print(f"""│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Transformer Encoder                            │   │
│  │   4-layer, dim=512, heads=8, ffn=3200                            │   │
│  │   输入: [Latent_z, Robot_State, Visual_Tokens]                    │   │
│  │   {format_params(encoder_params):>8}                                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    │ Memory (融合特征)                   │
│                                    ▼                                    │""")
    
    decoder_params = count_params(model.decoder)
    
    print(f"""│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Transformer Decoder                            │   │
│  │   1-layer, Cross-Attention                                       │   │
│  │   Query: Positional Embeddings (chunk_size=100)                  │   │
│  │   {format_params(decoder_params):>8}                                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Action Head (Linear)                         │   │
│  │   输出: [B, chunk_size, action_dim]  →  [B, chunk_size, action_dim]              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  组件参数统计:                                                           │""")
    
    print(f"""│    ├─ Vision Backbone (ResNet-18):     {format_params(backbone_params):>10}                     │
│    ├─ VAE Encoder (4-layer Trans):     {format_params(vae_params):>10}                     │
│    ├─ Transformer Encoder (4-layer):   {format_params(encoder_params):>10}                     │
│    ├─ Transformer Decoder (1-layer):   {format_params(decoder_params):>10}                     │
│    └─ 其他 (Projections, Embeds):      {format_params(count_params(policy) - backbone_params - vae_params - encoder_params - decoder_params):>10}                     │
│                                                                         │
│  总计: {format_params(count_params(policy)):>10} ({count_params(policy)/1e9:.4f}B)                                        │
└─────────────────────────────────────────────────────────────────────────┘
""")


def main():
    # ============================================================
    # 配置区 - 可自定义修改
    # ============================================================
    
    # 数据集选择（pusht 是最小的，适合快速测试）
    dataset_id = "lerobot/pusht"
    
    # 输出目录
    output_directory = Path("outputs/act_pusht_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # 设备选择（RTX 5090 用 cuda）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 训练参数
    training_steps = 5000       # 训练步数
    batch_size = 32             # 批大小
    log_freq = 10              # 日志频率
    
    # ============================================================
    # 模型配置 - 可自定义架构
    # ============================================================
    
    # 1. 获取数据集元数据（只下载几 MB）
    print(f"\n📦 加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    print(f"   FPS: {dataset_metadata.fps}")
    print(f"   Episodes: {dataset_metadata.total_episodes}")
    print(f"   Frames: {dataset_metadata.total_frames}")
    
    # 2. 自动提取特征（也可以手动指定，见下方注释）
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"\n📊 特征配置:")
    print(f"   输入特征: {list(input_features.keys())}")
    print(f"   输出特征: {list(output_features.keys())}")
    
    # 【可选】手动指定特征（完全控制模式）
    # input_features = {
    #     "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    # }
    # output_features = {
    #     "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    # }
    
    # 3. 创建 ACT 配置（可自定义架构参数）
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ---- 架构参数（可调整）----
        chunk_size=10,              # Action Chunking 大小（最优值）
        n_action_steps=10,          # 每次执行的动作步数
        dim_model=512,               # Transformer 隐藏维度
        n_heads=8,                   # 注意力头数
        n_encoder_layers=4,          # Encoder 层数
        n_decoder_layers=1,          # Decoder 层数
        use_vae=True,                # 是否使用 VAE
        latent_dim=32,               # VAE 隐变量维度
        vision_backbone="resnet18",   # 视觉骨干（pusht 有 96x96 RGB 图像）
        # ---- 优化器参数 ----
        optimizer_lr=1e-5,           # 学习率（默认值）
        optimizer_lr_backbone=1e-5,  # backbone 学习率
    )
    
    # ============================================================
    # 模型创建
    # ============================================================
    
    print(f"\n🤖 创建 ACT 模型...")
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # 统计参数量（用 B 作为单位，方便与大模型对比）
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,} ({total_params / 1e9:.4f}B)")
    print(f"   可训练: {trainable_params:,} ({trainable_params / 1e9:.4f}B)")
    
    # 打印网络结构概览
    print_model_architecture(policy)
    
    policy.train()
    policy.to(device)
    
    # ============================================================
    # 数据加载
    # ============================================================
    
    # delta_timestamps 配置（Action Chunking 核心）
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 如果有图像特征，添加历史帧
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    
    print(f"\n📂 加载数据集...")
    # 注意：RTX 5090 + PyTorch nightly 需要用 pyav 后端（torchcodec 不兼容）
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    print(f"   样本数: {len(dataset)}")
    
    # 创建 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    # ============================================================
    # 训练循环
    # ============================================================
    
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    
    print(f"\n🚀 开始训练 ({training_steps} 步)...")
    print("-" * 50)
    
    step = 0
    done = False
    start_time = time.time()
    losses = []
    
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            
            # 将数据移到 GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播
            loss, info = policy.forward(batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            
            if step % log_freq == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-log_freq:]) / len(losses[-log_freq:])
                
                # 计算预估剩余时间
                if step > 0:
                    steps_remaining = training_steps - step
                    speed = elapsed / step  # 秒/步
                    eta_seconds = steps_remaining * speed
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f" | ETA: {eta_min}m {eta_sec}s"
                else:
                    eta_str = ""
                
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | Time: {elapsed:.1f}s{eta_str}")
                
                # 显示显存使用
                if device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"         | GPU Memory: {mem_used:.2f} GB")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # ============================================================
    # 训练完成
    # ============================================================
    
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ 训练完成!")
    print(f"   总步数: {step}")
    print(f"   总时间: {total_time:.1f}s")
    print(f"   平均速度: {total_time/step*1000:.1f} ms/step")
    print(f"   最终 Loss: {losses[-1]:.4f}")
    
    # 保存模型
    print(f"\n💾 保存模型到: {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    
    print("\n🎉 Demo 完成！")
    print(f"   模型路径: {output_directory.absolute()}")
    print(f"   可用于推理: ACTPolicy.from_pretrained('{output_directory}')")


if __name__ == "__main__":
    main()

