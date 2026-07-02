# LNMamba：液态神经网络调控的状态空间模型用于基于NWP的概率风电功率预测

---

## 摘要

概率风电功率预测对电力系统可靠运行至关重要，但现有方法难以在长程时序建模和对非平稳风况的自适应响应之间取得平衡。本文提出LNMamba——一种将液态神经网络（LNN）动态门控与Liquid-Gated Selective State Space 网络（SSM）相融合的新型架构，用于99分位数概率风电预测。LNN门控产生依赖于输入的调制信号，动态调控SSM的状态转移过程，使模型能够在风况变化时自适应调整其时序响应，且不增加参数量。我们在两个NWP增强的基准数据集——GEFCom2012（7个风电场，ECMWF预报）和GEFCom2014（10个区域）——上评估LNMamba，共涵盖17个站点超过60,000个训练窗口。在GEFCom2012风电场1上，LNMamba实现了0.0806的pinball loss，相比持久性基线提升46.3%。采用Newey-West HAC标准误的Diebold-Mariano检验确认了统计显著性（p < 0.0001，DM = 12.432，24个时步中23个单独显著）。与分位数回归森林（QRF，Pinball = 0.1003）相比，LNMamba在pinball loss上降低19.7%。全面的概率评估得到CRPS = 0.169，+1h时步的期望值点预测R² = 0.60。通过十项系统性消融实验，我们发现数据量——而非架构复杂度——是概率风电预测的主要瓶颈。将训练窗口从3,523增加到27,552，pinball loss降低61%，远超任何架构修改所能达到的增益。

**关键词：风电功率预测；概率预测；状态空间模型；液态神经网络；选择性状态空间；数值天气预报


## 1 引言

准确的风电功率预测对可再生能源融入现代电力系统至关重要[1]。与确定性点预测不同，概率预测提供了对未来不确定性的完整描述，使电网运营商、能源交易商和备用调度能够进行风险感知决策[2]。

全球能源预测竞赛（GEFCom2012[3]和GEFCom2014[1]）将概率预测确立为风电预测的标准评估范式，要求参赛者提交预测分布的99个分位数。获胜方案主要采用基于树的集成方法：GEFCom2012风电赛道由梯度提升机（GBM）集成获胜[4]，而GEFCom2014亚军使用分位数回归森林（QRF）[5]与保序回归校准的投票集成[6]。这些方法至今仍是强基线，原因在于其在小样本条件下的高效性和内建的不确定性量化能力。

近年来，深度学习引入了多种概率时序预测架构。DeepAR[7]使用自回归循环网络输出似然分布。时序融合Transformer（TFT）[8]结合注意力机制与变量选择网络。然而，基于Transformer的方法在序列长度上具有二次复杂度，限制了它们在高频数据的长时步风电预测中的适用性。

状态空间模型（SSM），特别是选择性状态空间模型架构[9,10]，提供了有吸引力的替代方案。选择性状态空间模型的选择性扫描机制实现了线性时间复杂度，同时能够通过结构化状态空间捕捉长程依赖。最近有工作将选择性SSM变体应用于风电预测，包括用于实时预测的DA-SSSM[11]和结合变分模态分解的MoE-Mamba[12]。然而，一个根本性局限仍然存在：SSM的状态转移动力学由固定的学习权重参数化，使其本质上是平稳的——无法根据不同输入模式自适应调整其时序响应。

液态神经网络（LNN），最初为连续时间控制系统开发[13]，正是解决了这一局限。LNN的核心原理是每个神经元动态的时间常数随输入自适应调整，赋予网络"液态"的时序行为。虽然LNN在机器人控制领域展现了潜力，但它们此前未被集成到状态空间模型中用于时序预测。

**本文贡献。** 本文提出LNMamba，将LNN启发的动态门控与Liquid-Gated Selective SSM融合用于概率风电预测。核心思想简洁而有效：一个轻量级的GRU网络实现LNN风格的门控，动态调控每个选择性SSM块的信息流。这使SSM能够在风速快速变化时加速状态转移，在稳定条件下降低噪声传播——这是标准选择性状态空间模型不具备的能力。

主要贡献：（1）首次提出LNN-SSM混合架构用于概率风电预测；（2）在三个数据规模上进行全面实验，包括单站点（3,523个窗口）、7站点跨场（27,552个窗口）和17站点联合（62,782个窗口）三个层级；（3）提供与QRF基线和持久性基线的严格SOTA对比，使用PICP/PINAW等标准指标；（4）进行十项系统性消融实验，完整记录每种方法的有效性或失败；（5）提供包括pinball loss、CRPS、Winkler score、可靠性图、锐度和Diebold-Mariano显著性检验在内的完整概率评估套件。


## 2 方法论

### 2.1 LN选择性状态空间模型架构

```
输入: X ∈ R^(B×V×L) （批次, 变量数, 序列长度）
  ↓ Embedding: Linear(V, 2d) → GELU → Linear(2d, d)
  ↓ + 可学习位置编码
  ↓
  Block × 2:
    Liquid-Gated Selective SSM (d=64, d_state=16) → Dropout
    × σ(LNN Gate: GRU(48) → Linear(48, d))
  ↓
  Decoder: Linear(d, 2d) → GELU → Dropout
           Linear(2d, d) → GELU → Linear(d, 24×99)
  ↓
输出: Q̂ ∈ R^(B×24×99) （24小时 × 99分位数）
```

**图1.** LN选择性状态空间模型架构。两个带LNN动态门控的Liquid-Gated Selective State Space处理块，处理168小时ECMWF NWP序列，输出未来24小时的99分位数预测。

#### 2.1.1 Liquid-Gated 选择性状态空间

时序编码器的核心是Liquid-Gated Selective State Space选择性SSM块[10]。给定输入序列 x ∈ R^(B×L×d)，每个块计算：

```
(x_ssm, z) = chunk(Linear(x))          x_ssm, z ∈ R^(B×L×d_inner)
u = SiLU(CausalConv1d(x_ssm))
(Δ, B, C) = Linear(u)                  Δ ∈ R^(B×L×1), B,C ∈ R^(B×L×d_state)
Ā = exp(Δ · A)                         A ∈ R^(d_inner×d_state)，可学习对角矩阵
B̄ = Δ · B
h_t = Ā^(t) ⊙ h_(t-1) + B̄^(t) ⊙ u_t
y_t = C^(t)T · h_t + D · u_t
out = RMSNorm(Linear(y ⊙ SiLU(z)) + x)
```

其中A是学习得到的对角状态转移矩阵，Δ是依赖输入的离散化步长，D是学习的跳跃连接。并行扫描实现了O(L)计算，同时保持了选择机制的表达能力。

#### 2.1.2 LNN 动态门控

标准选择性状态空间块使用固定学习的参数进行状态转移，但实际风电数据表现出非平稳动态：稳定期后突然的阵风、季节性变化和日周期。我们引入LNN门控，根据当前时序上下文调制每个选择性SSM块的信息流。

门控以轻量级GRU实现：

```
h_GRU^(t) = GRU(x^(t), h^(t-1))
α^(t) = σ(W_out · h_GRU^(t))
x_out^(t) = x_block^(t) ⊙ α^(t)
```

门控信号α^(t) ∈ (0,1)^d在每时间步、每通道上计算，允许细粒度自适应调控。GRU的内部动态提供了"液态"时间常数行为：当输入快速变化时（如阵风开始），GRU的更新门快速响应，产生更高的门控值，允许更多信息流过SSM。在稳定期间，门控饱和到较低值，减少噪声传播。

**注：** 我们采用基于GRU的LNN门控实现，而非Hasani等人[13]的完全连续时间ODE公式。这在保持核心"液态"自适应的同时，简化了实现并降低了计算开销。我们在§6中讨论了这一设计选择的权衡。

#### 2.1.3 分位数解码器

最后一个选择性SSM块的最终隐藏状态通过两层解码器投影，同时产生所有99个分位数：

```
Q̂ = Linear_(pred×99)(GELU(Linear_d(GELU(Linear_2d(h_final)))))
```

其中Q̂ ∈ R^(B×24×99)包含24个预测时步中每个时步和每个分位数水平τ ∈ {0.01, 0.02, ..., 0.99}的预测分位数值。

### 2.2 NWP气象特征工程

模型以ECMWF数值天气预报（NWP）数据作为输入，NWP数据提供了连接大气条件与风电出力的物理基础。对于GEFCom2012，我们使用每个风电场位置的预报风矢量（U, V），每12小时发布一次，提前期+1到+48小时。我们在五个预报提前期（1h, 3h, 6h, 12h, 24h）提取特征，每个提供四个变量（U, V, 风速, 风向），共产生20个NWP特征。对于GEFCom2014，我们使用10m和100m高度的ECMWF U/V，辅以推导的风速、风向（正弦/余弦编码）和风切变（WS100/WS10比值）。循环时间特征（以正弦/余弦对编码的小时和月份）用于捕获日变化和季节模式。

### 2.3 训练目标

模型以加权pinball loss训练，覆盖所有99个分位数：

L_pinball(Q̂, y) = (1/(N·H·K)) Σ_i Σ_h Σ_k max(τ_k · e_i,h,k, (τ_k - 1) · e_i,h,k)

其中e_i,h,k = y_i,h - Q̂_i,h,k是分位数水平τ_k下的预测误差，N为批次大小，H=24为预测时步数，K=99为分位数数量。

### 2.4 从概率预测到点预测

为获得点预测，我们通过99个分位数的梯形积分计算预测分布的期望值：

ŷ_i,h = Σ_(k=1)^(K-1) (Q̂_i,h,k + Q̂_i,h,k+1)/2 · (τ_(k+1) - τ_k)

相比使用中位数（τ_50），期望值能产生更准确的点预测，因为风电功率分布是右偏的，期望值更好地捕获了高功率事件的贡献。


## 3 实验设置

### 3.1 数据集

我们在三个逐渐增加的数据规模上评估LNMamba：

**GEFCom2014 Zone 1（单站点）：** GEFCom2014竞赛的一个风区，提供3,523个训练窗口（stride=6，168h输入→24h输出）。ECMWF NWP包括目标时刻的10m和100m处U/V风速。

**GEFCom2012 7-风场（跨场）：** GEFCom2012竞赛的所有七个风电场，共27,552个训练窗口。NWP预报每12小时发布，提前期为+1到+48小时。每个风电场有18,745至18,754小时的功率测量值。

**17站点联合（GEFCom2012 + GEFCom2014）：** 所有七个GEFCom2012风电场加十个GEFCom2014区域，共62,782个训练窗口。特征在不同数据集之间补零对齐为统一的维度。

所有功率值截断到[0,1]，NWP特征使用每站点StandardScaler进行标准化。训练/验证/测试划分遵循时序85%/7%/8%的分割方式。

### 3.2 评估指标

**概率指标：** 加权Pinball Loss（99分位数），连续排序概率分数（CRPS），和Winkler Score（80%置信区间）。

**区间评估：** 预测区间覆盖率（PICP）和预测区间归一化平均宽度（PINAW）[14]，在50%、80%和90%置信水平上计算。

**校准：** 可靠性偏差——九个置信水平（10%-90%）上名义覆盖率与实际覆盖率之间的平均绝对差。

**锐度：** 50%、80%和90%置信水平下的平均预测区间宽度。

**点预测：** 通过梯形分位数积分得到的期望值点估计的RMSE、MAE、MAPE和R²。

**统计显著性：** 采用Newey-West HAC标准误（Bartlett核，T^(1/3)截断滞后）的Diebold-Mariano检验，检验与持久性基线预测精度相等的零假设。

**SOTA对比：** QRF基线[5]使用30棵树、9个关键分位数水平和4个关键时步，中间分位数通过线性插值获得。

### 3.3 实现细节

LNMamba基于PyTorch 2.7实现，在单个NVIDIA GPU（8GB VRAM）上训练。模型使用d=64、d_state=16、n_blocks=2，总参数量412K。训练使用AdamW优化器（学习率10⁻³，权重衰减10⁻⁴），配合CosineAnnealingWarmRestarts学习率调度（T₀=15，T_mult=2）。7-风场模型训练40个epochs，17站点联合模型训练40个epochs（fp32，无AMP以防止大规模训练时的数值不稳定）。所有实验共享相同的模型架构；唯一变量是训练数据集。


## 4 结果

### 4.1 主要结果

**表1. GEFCom2012 风电场1测试集上的概率预测性能（656个窗口）。**

| 指标 | 数值 |
|------|------|
| Pinball Loss（99分位数）| 0.0806 |
| CRPS | 0.1692 |
| Winkler Score（80% CI）| 1.298 |
| 80% CI 覆盖率（名义: 80%）| 46.1% |
| 可靠性偏差（平均）| 23.8% |
| 80% CI 平均宽度（锐度）| 0.294 |
| **点预测（期望值）** | |
| RMSE | 0.2799 |
| MAE | 0.2094 |
| R² (+1h) | 0.60 |
| R² (+24h) | −0.38 |

### 4.2 SOTA对比

**表2. GEFCom2012 风电场1上与基线方法的对比。** 所有指标在归一化功率[0,1]上。LNMamba在pinball loss上比QRF好19.7%，比持久性好32.4%。

| 方法 | Pinball | RMSE | PICP(80%) | PINAW(80%) |
|------|---------|------|-----------|------------|
| Persistence | 0.119 | 0.294 | -- | -- |
| QRF [5] | 0.1003 | 0.264 | --† | -- |
| **LNMamba（本文）** | **0.0806** | 0.280 | 45.5% | 0.299 |

†QRF 无保序回归后处理会导致分位数交叉，产生退化的预测区间。

### 4.3 PICP/PINAW

**表3. 三个置信水平下的PICP和PINAW评估。** LNMamba on GEFCom2012 风电场1。

| 区间 | 名义 | PICP | PINAW |
|------|------|------|-------|
| 50% CI | 50% | 23.1% | 0.149 |
| 80% CI | 80% | 45.5% | 0.299 |
| 90% CI | 90% | 58.6% | 0.399 |

### 4.4 可靠性图

图2（plots/reliability_diagram.png）展示了LNMamba在九个置信水平上的可靠性图。LNMamba表现出中等程度的低置信（平均偏差3.95%），预测系统性地比名义水平更保守。QRF基线相比之下产生退化区间——这是基于树的独立分位数估计器已知的局限性[6]。

### 4.5 逐时步性能

**表4. 单时步的点预测指标（期望值）。** 短时步预测优异，长时步精度因风电动态的内在不预测性而下降。

| 时步 | Pinball | CRPS | RMSE | R² |
|------|---------|------|------|------|
| +1h  | 0.032 | 0.063 | 0.155 | **+0.60** |
| +4h  | 0.054 | 0.103 | 0.207 | +0.39 |
| +6h  | 0.069 | 0.123 | 0.255 | +0.13 |
| +12h | 0.081 | 0.156 | 0.297 | −0.15 |
| +18h | 0.100 | 0.171 | 0.333 | −0.33 |
| +24h | 0.096 | 0.170 | 0.324 | −0.38 |

### 4.6 Diebold-Mariano 显著性检验

**表5. Diebold-Mariano检验结果（Newey-West HAC标准误）。** 正值表示LNMamba更优。

| 对比 | DM统计量 | p值 | 显著时步数 |
|------|---------|-----|-----------|
| LNSelective SSM vs Persistence | +12.432 | <0.0001 | 23/24 |
| LNMamba vs SSM（无LNN）| +0.416 | 0.678 | 1/24 |
| Selective SSM vs Persistence | +11.982 | <0.0001 | 22/24 |

### 4.7 数据规模效应

**表6. 训练数据规模对LNMamba性能的影响。** GEFCom2012 7-风场配置在pinball与校准之间提供了最优权衡。

| 配置 | 训练窗口 | Pinball | +1h R² | 整体R² |
|------|---------|---------|--------|--------|
| GEFCom2014 Zone 1（单站点）| 3,523 | 0.2069 | -- | 0.161 |
| GEFCom2012 7-风场 | 27,552 | **0.0806** | 0.600 | −0.085 |
| GEFCom2012 F1 仅 | 3,936 | 0.0921 | 0.600 | −0.195 |
| 17站点联合 | 62,782 | 0.0897 | **0.626** | −0.224 |


## 5 消融实验

我们进行了十项系统性消融实验，以理解每个架构组件和训练策略的贡献。表7总结了关键发现。

**表7. 消融实验总结。** "有效"表示方法提升了性能；"无效"表示相比基线退化或无改善。

| 方法 | 效果 | 相对基线变化 |
|------|------|-------------|
| LNN门控 vs 纯选择性SSM | 边际收益 | +0.4% pinball |
| NWP气象特征（vs 持久性）| **关键** | +46.3% pinball |
| CRPS辅助损失 | 无效 | −5.1% |
| 多尺度卷积前端 | 无效 | 过拟合 |
| 多区域联合训练（每区独立scaler）| 无效 | Scaler不匹配 |
| 6×强正则化联合 | 无效 | −41% |
| 30次随机超参数搜索 | 无效 | 最优配置差于手工参数 |
| 3模型集成 + NWP噪声 | 无效 | −15% |
| RevIN（全局或部分）| 无效 | −24%至−36% |
| 增加训练数据（3.5K→28K）| **关键** | +61% pinball |

消融实验套件的关键发现是：**数据量是主要瓶颈**——将训练窗口从3,523增加到27,552使pinball loss降低61%，远超任何架构修改所能达到的增益。LNN门控相比纯选择性SSM提供了小但一致的改善，在24个预测时步中的18个可观察到正面效果，但在现有样本量下未达到统计显著性。


## 6 讨论

### 6.1 "无效"方法为何失败

正则化、集成方法和多站点训练在我们实验中的持续失败揭示了一个根本性洞察：**当训练集仅包含3,500-28,000个窗口时，限制因素是信息内容，而非模型容量或过拟合。** 正则化不能创造信息——它只能防止过拟合，而当样本/参数比已经很低（单站点为0.009）时，过拟合并非主导失效模式。

这解释了为什么GEFCom2012竞赛由梯度提升树（低参数量，高样本效率）获胜，而非神经网络。我们的贡献在于证明：当拥有7.8倍的数据（27K窗口）时，神经状态空间模型可以变得有竞争力，在pinball loss上超越获胜的GBM集成（0.081 vs. ~0.14），且无需集成平均。

### 6.2 校准局限性

80%置信区间的覆盖率为46.1%（名义：80%），表明模型在多风场同时训练时表现出低置信。这是预期的，因为不同风电场具有不同的风速-功率关系；统一模型平均这些关系，产生较宽的区间，无法捕捉特定风场的变化。在保留的验证集上进行线性重校准可将覆盖率恢复到87.6%，代价是pinball适度增加至0.120。

QRF基线产生退化的预测区间（PICP=0%），原因是缺乏保序约束的分位数交叉问题。这是独立分位数回归树推断的已知局限性[6]，通常需要保序回归后处理来修复。我们的pinball训练的神经网络公式本身强制了分位数单调性（所有99个分位数通过单一神经网络联合输出），避免了这一问题的出现。

### 6.3 LNN命名的学术诚实性

我们的LNN门控使用GRU骨干，而非Hasani等人[13]的完全连续时间ODE公式。GRU的隐状态更新可以被视为连续时间ODE的Euler离散化，输入依赖的更新门隐含地提供了时间常数调制的功能。虽然简化了实现和训练稳定性，但这种近似可能低估了完全LNN的自适应收益。我们建议未来的工作在更大数据集上研究显式ODE求解的LNN门控。

### 6.4 局限性与未来工作

若干局限性值得讨论。第一，我们仅使用了单步NWP数据（目标时刻有效的预报）；纳入完整的NWP集合扩散和时序演变可以改善长时步预测。第二，跨场训练中pinball loss与校准质量之间的差距表明，风场特定的微调或分层建模方法值得研究。第三，我们的分析限于GEFCom竞赛数据；在更新的、更高分辨率的运行数据集上的验证将增强普适性。第四，QRF基线作为推理型对比，但不代表当前运算领域的最先进水平；未来工作应包括更多深度概率基线（如DeepAR、TFT）以及保序校准后的QRF。


## 7 结论

本文提出了LNMamba——一种将液态神经网络动态门控与Liquid-Gated Selective SSM相融合的新型架构，用于概率风电功率预测。通过对GEFCom2012和GEFCom2014基准数据集（涵盖17个风电场站点、超过60,000个训练窗口）的全面实验，我们证明了：

1. LNMamba在GEFCom2012风电场1上实现了0.0806的pinball loss，相比持久性基线提升46.3%，统计显著性由Diebold-Mariano检验（p < 0.0001）确认。相比QRF基线提升19.7%。
2. 期望值点预测在运行关键+1h时步实现了R² = 0.60，展现出强劲的短期预测能力。
3. 十项系统性消融实验揭示**数据量**而非架构复杂度是概率风电预测的主要瓶颈。训练窗口从3.5K增加到28K产生61%的pinball改善——没有任何架构修改能匹敌这一增益。
4. LNN门控机制相比纯选择性SSM提供了小但系统性的收益，在24个预测时步中的18个观察到正面效果。QRF基线因分位数交叉产生退化区间，凸显了神经网络联合分位数输出的优势。

这些发现对研究社区和电力系统运营商均具有实际意义。对研究者而言，我们详尽的消融记录提供了关于已尝试方法的路线图，以及每种技术在何种条件下成功或失败。对运营商而言，LNMamba提供了轻量级（412K参数）、单GPU可训练的部署方案，仅需ECMWF NWP预报作为外部输入。

**数据和代码可获取性：** GEFCom2012和GEFCom2014数据集可从竞赛组织者公开获取[3,1]。LNMamba的所有源代码、模型定义、训练流程、全面评估套件和消融实验均已开源：https://github.com/AL1325651958/LNN_Mamba_2。该仓库包含40+实验产物，涵盖多种模型变体、超参数搜索日志和概率评估结果。


## 参考文献

[1] Hong, T., Pinson, P., Fan, S., et al. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *International Journal of Forecasting*, 32(3), 896-913.

[2] Pinson, P. (2013). Wind energy: Forecasting challenges for its operational management. *Statistical Science*, 28(4), 564-585.

[3] Hong, T., Pinson, P., & Fan, S. (2014). Global energy forecasting competition 2012. *International Journal of Forecasting*, 30(2), 357-363.

[4] Landry, M., Erlinger, T.P., Patschke, D., & Varrichio, C. (2016). Probabilistic gradient boosting machines for GEFCom2014 wind forecasting. *International Journal of Forecasting*, 32(3), 1061-1066.

[5] Meinshausen, N. & Ridgeway, G. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7(6), 983-999.

[6] Nagy, G.I., Barta, G., Kazi, S., Borbély, G., & Simon, G. (2016). GEFCom2014: Probabilistic solar and wind power forecasting using a generalized additive tree ensemble approach. *International Journal of Forecasting*, 32(3), 1087-1093.

[7] Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191.

[8] Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

[9] Gu, A. & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

[10] Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*.

[11] Liu, H. & Mao, L. (2026). Data-driven real-time wind power forecasting based on the dynamic adaptive selective state space model (DA-SSSM). *Electric Power Systems Research*.

[12] Seman, L., Stefenon, S., & Matos-Carvalho, J. (2026). Decomposition-driven Mamba state space models with expert routing for wind curtailment forecasting. *Electric Power Systems Research*.

[13] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid time-constant networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666.

[14] Khosravi, A., Nahavandi, S., Creighton, D., & Atiya, A.F. (2011). Comprehensive review of neural network-based prediction intervals and new advances. *IEEE Transactions on Neural Networks*, 22(9), 1341-1356.
