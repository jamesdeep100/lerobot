# **扩散策略 (Diffusion Policy) 在 Push-T 任务中的深度技术分析报告：性能评估与超参数调优**

## **摘要**

本报告旨在对“Diffusion Policy: Visuomotor Policy Learning via Action Diffusion”一文及其在 Push-T（T 形块推移）任务中的应用进行详尽的技术分析。Push-T 任务作为机器人操作领域中具有高度代表性的基准测试，因其显著的多模态特性和对接触物理的高精度要求，成为了评估新一代视觉运动策略（Visuomotor Policy）的关键试金石。基于对原始论文及后续复现研究的深入剖析，本报告确认 Diffusion Policy 在该任务上取得了突破性的成功率。具体而言，采用 CLIP 预训练 ViT-B/16 视觉编码器的 Transformer 架构扩散策略实现了 **98%** 的最高成功率，而标准的 CNN 架构策略亦能稳定达到 **84%-90%** 的区间，显著优于传统的 LSTM-GMM（约 60%）和隐式行为克隆 IBC（约 20%）。

在超参数配置方面，该策略的成功高度依赖于一组精心调优的时间视界参数：**动作预测视界 ($T\_p$) 为 16 步**，**动作执行视界 ($T\_a$) 为 8 步**，以及极短的**观测视界 ($T\_o$) 为 2 步**。训练过程采用 AdamW 优化器，基础学习率设定为 **1e-4**，并在视觉编码器微调时采用 **1e-5** 的低学习率以防止灾难性遗忘。此外，报告还详细探讨了去噪迭代次数（训练时 100 次，推理时通过 DDIM 压缩至 16 次）以及条件调节机制（In-painting）的选择逻辑。本报告将通过系统性的章节，深入阐述这些技术细节及其背后的理论依据。

## ---

**第 1 章 引言：视觉运动策略学习的新范式**

### **1.1 机器人操作中的多模态挑战**

在机器人学习领域，尤其是模仿学习（Imitation Learning, IL）中，如何从人类专家的示范数据中学习复杂的灵巧操作技能一直是一个核心难题。传统的行为克隆（Behavioral Cloning, BC）方法通常将此问题建模为从状态空间到动作空间的监督回归问题。然而，这种简单的映射假设忽略了人类行为中固有的“多模态性”（Multimodality）。

多模态性是指在给定的环境状态下，存在多种互斥但同样有效的动作策略。以 Push-T 任务为例，当机器人面对一个 T 形块时，专家演示者可能会选择从左侧推动，也可能会选择从右侧推动。这两种策略都是最优的，分布在动作空间的两个不同区域。如果使用传统的均方误差（MSE）回归模型（如确定性策略网络），模型往往会输出这两个分布模态的平均值——即直接推向物体的中心。在 Push-T 场景中，这种平均化的动作会导致机器人直接撞击物体或完全错过接触点，从而导致任务失败。

### **1.2 现有方法的局限性**

在 Diffusion Policy 提出之前，学界尝试了多种方法来解决多模态分布建模问题，但均存在显著局限：

* **混合密度网络 (MDN) 与 LSTM-GMM**：这类方法试图通过高斯混合模型（GMM）来拟合动作分布。然而，在实际应用中，GMM 往往难以在高维动作空间中精确地分离模态，尤其是在接触丰富的操作任务中，模型容易在不同模态之间快速切换或“犹豫”，导致动作不连贯，或者在物体附近陷入局部极小值（Local Minima）。  
* **隐式行为克隆 (IBC)**：IBC 使用能量函数（Energy-Based Models, EBM）来表示策略，通过优化能量最小化来生成动作。虽然理论上能处理多模态，但 IBC 的训练过程类似于生成对抗网络（GAN），极其不稳定，且推理过程需要昂贵的迭代优化。在 Push-T 任务中，IBC 经常表现出过早停止或无法精细调整物体姿态的问题。  
* **行为 Transformer (BET)**：利用 Transformer 进行动作离散化和分类。虽然能处理多模态，但在连续控制任务中，离散化带来的精度损失以及对特定模态的承诺能力（Mode Commitment）不足，限制了其在精细操作中的表现。

### **1.3 扩散策略的技术突破**

Diffusion Policy 引入了一种全新的视角，将机器人策略学习建模为条件去噪扩散过程（Conditional Denoising Diffusion Process）。不同于直接预测动作，该策略学习的是动作分布的得分函数（Score Function）的梯度。在推理阶段，模型从高斯噪声开始，通过迭代去噪逐步生成动作序列。

这种范式在 Push-T 任务中展现了三大核心优势：

1. **完美的模态覆盖与承诺**：作为一种生成式模型，它能精确拟合任意复杂的分布。在推理时，由于去噪过程的随机性，模型会“承诺”并收敛到单一模态（例如，坚定地选择从左侧推），而不是在两者之间摇摆。  
2. **时间一致性**：通过预测一段未来的动作序列（Action Sequence）而非单步动作，扩散策略天然具有极高的动作平滑性和时间一致性，有效抑制了高频抖动。  
3. **训练稳定性**：相比于 IBC 和 GAN，扩散模型的训练基于更加稳定的去噪目标函数（MSE loss on noise），极易收敛且对超参数不那么敏感。

## ---

**第 2 章 Push-T 任务详解与评估体系**

### **2.1 任务物理特性与动力学**

Push-T 任务是一个经典的平面非抓取式操作（Non-prehensile Manipulation）任务。其设计初衷是模拟高精度的接触动力学控制。

* **环境设定**：在一个 2D 平面上，机器人由一个圆形的末端执行器（End-effector）表示。目标是一个 T 形的刚体滑块。  
* **目标定义**：机器人的任务是将 T 形块从随机的初始位置和角度，精确地推送到一个固定的 T 形目标区域。  
* **操作难点**：  
  * **欠驱动系统**：机器人只能通过单点接触施加推力，无法通过抓取来完全约束物体的 6 个自由度。这意味着机器人必须利用摩擦锥（Friction Cone）原理，精确控制接触点和推力方向来引导物体旋转和平移。  
  * **接触断续性**：任务过程中涉及频繁的接触建立与断开（Make and Break Contact）。  
  * **高精度要求**：T 形块的几何形状不对称，且目标区域与块的大小完全一致，容差极小。任何微小的角度偏差都会导致 IoU 大幅下降。

### **2.2 评估指标：IoU 与成功率**

为了量化策略的性能，研究采用了严格的评估标准：

* **IoU (Intersection over Union)**：计算 T 形块当前占用的像素区域与目标区域的交并比。这是衡量物体姿态对齐程度的核心几何指标。  
* **成功判定阈值**：只有当 episode 结束时，最终状态的 IoU 达到或超过 **0.95 (95%)** 时，该次尝试才被判定为“成功”。这是一个极高的标准，意味着物体必须几乎完美地与目标重合 1。  
* **最大重叠率 (Max Overlap)**：除了最终成功率，评估中也常记录整个过程中达到的最大 IoU，以分析策略是否曾接近成功但未能保持。

## ---

**第 3 章 性能评估与成功率深度分析**

本章将详细回答用户关于“作者在 PushT 上面的成功率是多少”的问题。根据论文原文及后续的实验复现数据，Diffusion Policy 在该任务上展现了压倒性的优势，但具体数值取决于所选用的视觉编码器（Vision Encoder）和网络架构。

### **3.1 仿真环境基准测试结果**

在标准的 Gym-PushT 仿真环境中，不同配置的 Diffusion Policy 表现如下：

#### **3.1.1 最佳性能配置：Transformer \+ ViT-B/16 (CLIP)**

当使用 Transformer 架构作为扩散模型的骨干网络（Backbone），并结合经由 CLIP 预训练的 ViT-B/16 作为视觉编码器时，Diffusion Policy 达到了目前的最高水平。

* **成功率**：**98% (0.98)** 2。  
* **分析**：这一惊人的成绩归功于 CLIP 编码器强大的语义理解能力和 Transformer 在处理长序列依赖时的优势。即便仅经过 50 个 epoch 的训练，该配置也能达到极高的成功率。

#### **3.1.2 标准配置：CNN-Based (ResNet-18)**

这是论文中作为主要基准对比的配置，通常被称为 "DiffusionPolicy-C" 或 "Hybrid" 架构。

* **成功率**：**84% \- 90% (0.84 \- 0.90)** 3。  
  * 有些实验记录显示平均成功率为 0.84，而在某些最佳 checkpoint 上可达 0.90 或更高。  
* **稳定性**：该配置在最后 10 个 checkpoint 的平均成功率也保持在 0.84 左右，显示出训练后期极高的稳定性，没有出现严重的性能崩塌。

#### **3.1.3 状态空间策略 (State-Based)**

如果移除视觉处理部分，直接将物体的真实状态（位置、角度）作为输入，Diffusion Policy 的性能进一步提升。

* **成功率**：接近 **100% (1.0)** 2。  
* **意义**：这证明了扩散策略本身在处理复杂的推移操纵策略上是完美的，主要的性能瓶颈在于视觉感知的精度。

### **3.2 与基线方法的详细对比**

为了直观展示 Diffusion Policy 的优势，下表总结了其与主流基线方法的性能对比 2。

**表 1：Push-T 任务成功率对比分析**

| 方法 (Method) | 架构/配置 | 成功率 (Max) | 成功率 (Avg) | 典型失效模式 |
| :---- | :---- | :---- | :---- | :---- |
| **Diffusion Policy** | **Transformer (CLIP ViT)** | **0.98** | **0.95** | 极其罕见的对齐误差 |
| **Diffusion Policy** | **CNN (ResNet-18)** | **0.90** | **0.84** | 偶尔的接触精度不足 |
| **LSTM-GMM** | RNN Mixture Density | 0.67 | 0.61 | 在 T 块附近“卡死”，无法决策推动方向 |
| **IBC** | Implicit Behavior Cloning | 0.20 | 0.10 | 过早停止，无法推入目标区 |
| **BET** | Behavior Transformer | 低 (Unspecified) | \- | 动作犹豫，缺乏模态承诺 |

### **3.3 真实世界 (Real-World) 实验表现**

在真实世界的 UR5 机械臂实验中，Diffusion Policy 同样展现了强大的鲁棒性。尽管真实世界存在感知噪声、通信延迟和物理模型误差（Sim-to-Real Gap），策略依然保持了极高的成功率。

* **鲁棒性测试**：实验人员在机器人执行任务过程中人为地移动 T 形块（扰动）。Diffusion Policy 能够实时感知状态变化，动态重新规划路径，将块推回正确轨迹 2。  
* **抗干扰能力**：面对视觉遮挡（如手在相机前挥动）或光照变化，策略表现出了惊人的稳定性，这得益于其基于整个分布进行预测的特性，而非单一确定性输出。

## ---

**第 4 章 超参数配置深度解析**

用户关于“超参数分别设置的是多少”的提问触及了复现该算法的核心。Diffusion Policy 在 Push-T 上的成功并非偶然，而是依赖于一组特定的参数配置，这些参数平衡了感知视野、预测未来与执行动作之间的关系。

### **4.1 时间视界参数 (Horizon Parameters)**

这是 Diffusion Policy 中最关键的超参数组，定义了策略的时间结构。

* **动作预测视界 ($T\_p$, Prediction Horizon)**：**16 步**。  
  * **定义**：在每一个推理时间步，扩散模型一次性生成未来 16 个时间步的动作序列。  
  * **作用**：长视界预测迫使模型学习动作之间的连贯性。对于 Push-T 这种需要平滑推移的任务，预测未来的轨迹能有效避免单步预测带来的“锯齿状”抖动路径。  
  * **来源**：4  
* **动作执行视界 ($T\_a$, Action Execution Horizon)**：**8 步**。  
  * **定义**：虽然模型预测了 16 步，但机器人仅执行前 8 步。执行完这 8 步后，模型会再次进行推理，预测新的 16 步。  
  * **作用**：这种“重叠视界控制”（Receding Horizon Control）机制至关重要。$T\_a \< T\_p$ 意味着模型总是在执行其预测序列中置信度最高的前半部分，同时保留了对未来的规划。如果 $T\_a$ 设置过小（如 1），推理计算量会剧增且容易引入高频噪声；如果 $T\_a$ 过大，策略对环境扰动的响应会变慢。8 步是一个在响应速度和平滑度之间的最佳平衡点。  
  * **来源**：6  
* **观测视界 ($T\_o$, Observation Horizon)**：**2 步**。  
  * **定义**：模型输入条件包含当前帧图像和前一帧图像（共 2 帧）。  
  * **作用**：Push-T 任务主要是一个马尔可夫决策过程，但物体的速度信息对于推移至关重要。仅仅 $T\_o=1$ 无法推断速度，$T\_o=2$ 足以通过帧间差分隐式获取速度信息。更有趣的是，消融实验表明，增加 $T\_o$ 到 4 或更多并没有带来性能提升，反而在图像编码器中引入了不必要的计算负担和过拟合风险 7。

**表 2：Push-T 任务关键时间视界参数汇总**

| 参数名称 | 符号 | 设置值 | 备注 |
| :---- | :---- | :---- | :---- |
| **预测视界** | $T\_p$ | **16** | 一次推理生成的动作序列长度 |
| **执行视界** | $T\_a$ | **8** | 实际下发给机器人的动作步数 |
| **观测视界** | $T\_o$ | **2** | 输入的历史图像帧数 |
| **下采样率** | \- | None | 通常不进行时间下采样，保持控制频率 |

### **4.2 优化与训练超参数 (Optimization & Training)**

* **优化器 (Optimizer)**：**AdamW**。  
  * AdamW 相比 Adam 提供了更正确的权重衰减（Weight Decay）实现，对于 Transformer 和 ResNet 等深层网络在小数据集上的泛化至关重要。  
* **基础学习率 (Base Learning Rate)**：**1e-4 (0.0001)**。  
  * 这是用于训练核心扩散网络（UNet 或 Transformer）的学习率。它遵循 **Cosine** 衰减调度策略，并带有预热（Warmup）阶段（CNN 为 500 步，Transformer 为 1000 步）4。  
* **视觉编码器学习率 (Encoder LR)**：**1e-5** (或更低)。  
  * **关键细节**：当微调预训练的视觉模型（如 CLIP-ViT 或 R3M）时，必须使用比基础网络小一个数量级（10x smaller）的学习率。如果使用 1e-4，预训练的特征会被迅速破坏（Catastrophic Forgetting），导致性能大幅下降。这一细节是实现 98% 成功率的秘诀之一 2。  
* **权重衰减 (Weight Decay)**：**1e-6**。  
  * 用于正则化。鉴于 Push-T 数据集较小（约 136-200 条演示），强正则化有助于防止过拟合。  
* **批量大小 (Batch Size)**：  
  * **图像输入 (Image-based)**：**64**。受限于显存（处理序列图像）。  
  * **状态输入 (State-based)**：**256**。  
* **训练时长**：通常设定为固定的 **12 小时** 或约 **3000-4500 epochs**。但实际上，模型往往在前 500 个 epoch 内就已经收敛到较高性能 2。

### **4.3 扩散过程配置 (Diffusion Process)**

* **算法**：**iDDPM** (Improved Denoising Diffusion Probabilistic Models)。  
  * 训练目标是预测加入的噪声残差（noise residual）。  
* **噪声调度器 (Noise Scheduler)**：**Squared Cosine Schedule**。  
  * 相比线性调度，余弦调度能更平滑地破坏信息，特别是在扩散过程的末期保留更多的信号，这对于精细控制任务很有利。  
* **去噪迭代次数 ($K$)**：  
  * **训练 (Training)**：**100 步**。  
  * **推理 (Inference \- Sim)**：**100 步**。在仿真中为了追求极致精度，通常使用全量迭代。  
  * **推理 (Inference \- Real)**：**16 步**。在真实机器人上，为了降低延迟（Latency），采用 **DDIM** (Denoising Diffusion Implicit Models) 采样算法，将 100 步的生成过程压缩至 16 步，从而将推理频率提升至 10Hz 以上，满足实时控制需求 7。

### **4.4 网络架构参数详解**

针对 Push-T 任务，论文对比了两种主要架构：

1. **CNN-Based (DiffusionPolicy-C / Hybrid)**：  
   * **Backbone**：ResNet-18（去除了最后的池化层，改为 Spatial Softmax 或直接展平）。  
   * **Conditioning**：**In-painting (拼接)**。  
     * **特别注意**：大多数任务中使用 FiLM（特征线性调制）来将图像特征注入 UNet。但在 Push-T 任务中，作者发现使用 **In-painting**（将图像特征图直接与噪声动作图在通道维度拼接）效果更好。这是因为 Push-T 是一个强空间相关的 2D 任务，直接的空间对齐（Spatial Alignment）比全局特征调制更有效 7。  
   * **Downsampling**：UNet 包含 3 层下采样，通道数分别为 。  
2. **Transformer-Based**：  
   * **Backbone**：MinGPT 或 DiT 变体。  
   * **Embedding**：扩散步数嵌入维度为 64，特征嵌入维度通常为 256。  
   * 该架构在处理序列建模时表现更佳，尤其是结合 CLIP 时。

## ---

**第 5 章 技术实现细节与复现指南**

### **5.1 数据集特性**

Push-T 的训练数据集规模相对较小，这使得它成为检验算法数据效率（Data Efficiency）的绝佳场景。

* **演示数量**：约 **136 到 205 条** 成功的演示轨迹 8。  
* **数据频率**：通常为 10Hz 或 20Hz。  
* **数据增强**：虽然报告中未详细展开，但在图像任务中，通常会对输入图像进行随机裁剪（Random Crop）或色彩抖动，以增强模型的泛化能力。

### **5.2 模拟到真实的迁移 (Sim-to-Real)**

虽然 Push-T 是一个 2D 任务，但真实世界中的摩擦力分布不均匀、传感器噪声和通信延迟都会带来挑战。

* **延迟处理**：通过 $T\_p=16, T\_a=8$ 的设置，模型实际上是在规划未来。这种前瞻性规划天然地能够抵消感知和计算带来的延迟。即便当前时刻的观测由于延迟滞后了 0.1 秒，模型预测出的未来动作序列依然包含了对当前时刻之后动作的有效估计。  
* **位置控制 vs 速度控制**：实验表明，在 Push-T 任务中，Diffusion Policy 使用**位置控制 (Position Control)** 的效果显著优于速度控制。速度控制更容易受到延迟引起的累积误差影响，导致推移过头或不足 8。

### **5.3 关键代码配置片段**

为了方便技术复现，以下是从官方代码库 (diffusion\_policy/config/task/pusht\_image.yaml) 中提取并整理的关键配置逻辑：

YAML

\# 核心任务配置  
task:  
  name: pusht\_image  
  image\_shape:  \# 仿真通常使用 96x96 分辨率  
  dataset:  
    horizon: 16       \# 对应预测视界 Tp  
    n\_obs\_steps: 2    \# 对应观测视界 To  
    pad\_after: 7      \# 与执行视界 Ta=8 相关 (用于数据加载时的padding)

\# 优化器配置  
optimizer:  
  learning\_rate: 1.0e-4  
  weight\_decay: 1.0e-6  
  betas: \[0.9, 0.95\]

\# 训练配置  
training:  
  batch\_size: 64  
  sample\_every: 50    \# 每50个epoch进行一次评估  
  checkpoint\_every: 50

## ---

**第 6 章 结论**

通过对 Diffusion Policy 在 Push-T 任务上的全面剖析，我们可以得出明确结论：**Diffusion Policy 是目前解决此类高精度、多模态操作任务的最优解**。

1. **成功率结论**：在最优配置（CLIP-ViT）下，其成功率高达 **98%**；在标准 CNN 配置下也能达到 **84%-90%**，远超传统基线。  
2. **超参数结论**：成功的关键在于打破了传统单步预测的范式，采用了 **16 步预测、8 步执行、2 步观测** 的时间结构。这一结构配合 **1e-4** 的学习率和 **1e-5** 的视觉微调策略，使得模型既能捕捉长时序的动作连贯性，又能保持对环境变化的快速响应。  
3. **技术启示**：对于未来的机器人策略开发，Diffusion Policy 在 Push-T 上的成功证明了生成式模型在连续控制领域的巨大潜力。特别是其通过分布建模解决多模态歧义、通过长程预测解决动作平滑性的思路，为解决更复杂的 3D 灵巧操作任务提供了标准范式。

本报告确认，Diffusion Policy 不仅是一个学术上的创新，其实用的超参数配置和稳健的性能使其成为工业界和学术界在视觉运动控制任务上的首选基准。

#### **引用的著作**

1. lerobot/diffusion\_pusht \- Hugging Face, 访问时间为 一月 11, 2026， [https://huggingface.co/lerobot/diffusion\_pusht](https://huggingface.co/lerobot/diffusion_pusht)  
2. Visuomotor Policy Learning via Action Diffusion, 访问时间为 一月 11, 2026， [https://diffusion-policy.cs.columbia.edu/diffusion\_policy\_ijrr.pdf](https://diffusion-policy.cs.columbia.edu/diffusion_policy_ijrr.pdf)  
3. Visuomotor Policy Learning via Action Diffusion \- arXiv, 访问时间为 一月 11, 2026， [https://arxiv.org/html/2303.04137v5](https://arxiv.org/html/2303.04137v5)  
4. Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control \- arXiv, 访问时间为 一月 11, 2026， [https://arxiv.org/html/2502.12724v1](https://arxiv.org/html/2502.12724v1)  
5. Diffusion Policy, 访问时间为 一月 11, 2026， [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)  
6. EVE: A Generator-Verifier System for Generative Policies \- arXiv, 访问时间为 一月 11, 2026， [https://arxiv.org/html/2512.21430v1](https://arxiv.org/html/2512.21430v1)  
7. Diffusion Policy: \- Robotics, 访问时间为 一月 11, 2026， [https://www.roboticsproceedings.org/rss19/p026.pdf](https://www.roboticsproceedings.org/rss19/p026.pdf)  
8. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, 访问时间为 一月 11, 2026， [https://arxiv.org/pdf/2303.04137](https://arxiv.org/pdf/2303.04137)  
9. Diving into Diffusion Policy with LeRobot \- Radek Osmulski, 访问时间为 一月 11, 2026， [https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/](https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/)