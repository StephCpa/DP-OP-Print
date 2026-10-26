# OPSD 成员泄露的事前预测：一页实验设计（v0，2026-10-26）

**读过的材料**：OP-Print（arXiv 2610.31616，PDF 全文含附录 A/B）；2505.11837（PDF 全文）；DuoLearn 2502.19726（摘要和方法部分）。标 **[未核实]** 的内容是推断或只看了摘要，不是读原文核实的结论。

## 0. OP-Print 实际做了什么（与我们之前假设的差别）

- **核心量**：Δ_t(x,y) = KL( p_θ(·|p_t(q,r), y<t) ‖ p_θ(·|p_s(q), y<t) )。两边都用**被攻击的已训练模型 θ**：p_t 带参考解 r，p_s 不带。y 是从 p_θ(·|p_s(q)) 采样的 rollout。训练时的 teacher 是固定的“θ0 + 特权上下文”（附录 A）。
- **灰盒分数**：OP-Print-G 为 S = −(1/K)Σ_k (1/T_k)Σ_t Δ_t，按 64-token chunk 计算，K=32。OP-Print-G-Max 为 S = −(1/K)Σ_k max_t Δ_t。符号含义：**member 的残余修正更小**，分数更高。
- **黑盒分数 OP-Print-B**：采 M 条完整生成，在 |P|=10 个前缀比例 ρ~U(0,1) 处截断，各重启 R 次，求两两“归一化字符串相似度”的均值，再对 m 和 ρ 取平均。附录 B 只给了单调替代假设 g(C) 和方差论证，没有可辨识性定理。
- **实验设置**：Qwen3-1.7B / 4B / 8B（**包含 1.7B**）。数据是 OpenThoughts 数学子集，随机切分后取 500 member / 500 non-member。全参训练，3 epoch，lr 5e-6，max grad norm 0.1，“JSD token clip 0.05”（具体含义 **[未核实]**），8×A6000。只训练**一个目标模型**，不用 shadow/reference model（文中明确只做 metric-based）。
- **报告指标是 AUC、Acc 和 TPR@5%FPR，没有 TPR@1%FPR。** JSD 目标下（表 1）：
  - G-Max 的 AUC：1.7B 0.594，4B 0.611，8B 0.587；最强非 OP-Print 基线约 0.52。
  - G-Max 的 TPR@5%FPR：0.110 / 0.127 / 0.102，随机水平是 0.05。
  - OP-Print-B 的 AUC：0.554 / 0.575 / 0.557。
  - tinker 风格目标（表 2）下 G-Max 的 AUC 为 0.583 / 0.575 / 0.576。在 tinker 4B 上，G-Max 的 TPR 低于 G。
- **“distillation variants”只指 JSD 对比 tinker 风格**（采样 token 的 teacher−student logprob 优势）。**没有 top-k teacher，没有 DP，也没有任何防御**，Limitations 里写明把防御留作未来工作。
- “印记稀疏、集中在不确定 token”这一说法的证据有两条：
  1. 图 2 中散度随学生不确定性上升。但这描述的是**训练信号本身**的分布，不是 member 与 non-member 之差。
  2. G-Max 比 G 高 0.01–0.03 AUC。每类 n=500 时，AUC 的标准误约 0.018 **[估算]**，所以这个差距只是边缘显著。
- 因此，“成员信号稀疏”在原文里是**间接证据**，正好可以拿来做我们的可证伪检验。

**循环性问题：确认。** 先看用 θ 算的情况：集中度（top-k 份额 / Gini）和 G-Max 都是同一个 Δ 向量的函数，相关是机械的。
即使集中度改在 θ0 下计算，问题也没有消失。lr 5e-6、3 个 epoch，Δ_θ 与 Δ_θ0 在逐样本层面应高度相关 **[推断，待测]**，原始 G-Max 分数里的主要方差来自样本难度，不是成员身份。原文 AUC 只有约 0.6，与这一点一致。所以结果变量必须换成 in/out 差。

## 1. 已有工作的重叠

- **2505.11837**（Cui, Zhang, Pei）：研究**off-policy、混合监督 KD**（CE + λ·KL），师生是不同模型（Pythia→DistilPythia 等）。
  - 他们的“只在 non-vulnerable 点上蒸馏”是这样做的：用 MIA 打**已训练 teacher** 在其训练集上的分数，按分数把点分成 D_v 和 D_nv。这是**事后**的、基于攻击结果的筛选，而且有明显的 PPL 代价。
  - 另外两个方法是 Bottleneck Projection 和 NoNorm。
  - 他们的机制解释是：vulnerable 点上 teacher 与 ground truth **一致**（KL 小），因此被强化记忆。这和 OP-Print 的“高差异位置携带信号”方向**相反**，可以当作一个有意思的张力点。
  - 没做 DP，也没做 on-policy。
  - **与我们的区别**：我们要在 θ0、训练前预测，并且针对 on-policy 自蒸馏。
- **DuoLearn**（ACL'25 Findings）：用 reference model 校准的 token loss，把 token 分成 hard（学习）和 memorized（梯度上升遗忘），是 token 级经验防御。逐 token 截断类防御必须和它对比，或者至少说清差别。
- **还需要查的**（只看过搜索摘要 **[未核实]**）：
  - 2602.15919，*Assessing Per-Sample MI Vulnerability without Retraining*：不重训就估计逐样本风险，**与 H1 最接近，必须先读**。
  - 2505.18773：LiRA 在 LLM 上逐样本决策很不稳定，会直接影响 H1 结果变量的噪声。

## 2. 假设（可证伪）

- **H0（复现门槛）**：在 Qwen3-1.7B + LoRA 下，OP-Print-G-Max 的 AUC 和 TPR@5%FPR 显著高于 Min-K% 等基线（bootstrap 95% CI 不含基线值）。
- **H1（主线，事前预测）**：在 θ0 下算的逐样本 Δ 集中度 c(x)（top-10% chunk 的 Δ 份额，或 max/mean），在控制难度 d(x) 之后，能预测逐样本脆弱度 v(x) = mean_in S(x) − mean_out S(x)。d(x) 用 θ0 的 pass@k 和 θ0 下的平均 Δ 衡量。
  - 判据：回归 v ~ c + d 中 c 的系数显著，且偏相关 ρ(c, v | d) ≥ 0.1。
  - **难度零假设对照**：c 与 mean_out S 的相关若和 c 与 v 一样强，就说明测到的是难度，H1 不成立。
- **H1'（副产品，可能最容易出结果）**：参考校准的攻击 S_θ(x) − S_θ0(x)（θ0 就是公开的 Qwen3 底座）应明显强于原版 OP-Print，因为它直接去掉了难度。原文没有做这个基线。
- **H2（DP 审计紧度）**：DP-SGD 下，OP-Print 给出的经验 ε 下界 ε̂ 远小于理论 ε。判断依据是 TPR ≤ e^ε·FPR + δ。
- **H3（经验防御）**：逐 token 截断 Δ 能降低 G-Max，但在自适应攻击（sum、top-k 均值、学习型聚合器）下，泄露会部分恢复。
- **H4（最低优先级）**：在相同效用下，top-k teacher 的泄露 ≥ full teacher；无偏 top-k（TT-OPD 式）尾部采样的泄露介于两者之间。

## 3. 实验设置

- **模型**：Qwen3-1.7B，bf16，LoRA（r=16, α=32，作用在全部 attn+MLP 线性层）。
  - teacher 就是关掉 LoRA 的底座加特权 prompt，与原文“固定 teacher = θ0 + 特权信息”一致，不占额外显存。
  - 显存预算：权重约 3.4 GB，加 vLLM colocate（gpu_util 约 0.35–0.4）和梯度检查点，16 GB 应可行 **[未实测]**。
  - **与原文的偏差**：原文是全参训练。LoRA 可能降低泄露，需要在报告中注明。
- **数据**：OpenThoughts 数学子集，取 N=1000 的样本池，与原文同源。
- **影子模型**：训练 8 个 LoRA 模型，每个随机取 500 条作为 in，于是每个样本约 4 次 in、4 次 out。训练超参照抄表 3（3 epoch、max completion 1024、T=1.1、top-p 0.95、top-k 20）。
- **算力估计** **[粗估，需用 1 个模型实测校准]**：
  - 训练：每个模型约 1–2 h，8 个约 8–16 GPU·h。
  - 打分：K=8 条 rollout（原文图 5a 显示 K 从 32 降到 8，AUC 只小幅下降），共 8 个模型 × 1000 条 × 8 条 rollout，约 6.4 万条 ≤1024-token 的 rollout，外加两次 prefill，约 10 h。
  - θ0 特征只需算一次。
  - 合计约 **1–1.5 个 GPU·天**，适合考试后集中跑。
- **指标**：
  - AUC、TPR@5%FPR，以及汇总 8 个模型的 out 分数后的 TPR@1%FPR。单个模型只有 500 个 non-member，1%FPR 只对应 5 个负例，估计不可靠。
  - 逐样本 v(x) 的 split-half 信度。如果两组 4 模型估出的 v 之间相关很低，H1 本来就无法检验。
- **基线**：
  - 攻击基线：Min-K%/Min-K%++、loss、zlib、θ0 参考校准 loss，以及原文的 RC-Rate / R-Std。
  - 预测器基线（H1）：θ0 loss、pass@k、平均 Δ、2602.15919 的估计量（如果可复现）。
- **DP 阶段**：
  - 隐私单元是“一条 (q, r) 记录”。同一步内这条 q 的全部 rollout 先求和成一个梯度，再做 per-example clip。
  - 采样用 Poisson subsampling，用 RDP/PRV accountant 记账，ε ∈ {1, 4, 8, ∞}，δ = 1/N。
  - 审计时插入 canary，用 one-run auditing（Steinke et al. 2023）或 in/out 模型计算 ε̂，并报告 Clopper-Pearson 置信区间。

## 4. 顺序与止损（kill criteria）

1. **读文献，约 0.5 天**：精读 2602.15919，确认 H1 没有被做过；查清 “JSD token clip” 的定义。
   - **Kill**：如果 2602.15919 已经在 LLM 微调或蒸馏上做了 θ0 事前预测，就把 H1 改成“on-policy 特权信号作为预测器”，或者放弃 H1。
2. **单模型复现**：
   - **Kill**：G-Max 的 AUC < 0.55，或者 TPR@5%FPR 的 CI 与基线重叠，就停止。可以先试一次全参或更高 lr，确认问题不是 LoRA 造成的。
3. **8 个模型 + H1/H1'**：
   - **Kill**：v(x) 的 split-half 信度 < 0.3，或者控制难度后 c 没有增量解释力，就把 H1 当作负结果记录（“稀疏印记”这一解释不成立，本身也是一个发现），然后转做 H1'/H2。
4. **H2 → H3 → H4**：每个干预都在相同效用下比较（相同的 pass@1，或相同的训练 KL 下降量）。每个干预都必须配一个自适应攻击。

## 5. 风险

- 信号弱：原文 AUC 只有约 0.6，逐样本 v(x) 可能被训练随机性淹没（见 2505.18773）。缓解办法是增加模型数，或者把目标改成“风险分层”而不是逐样本排序。
- LoRA 与全参的差异可能让复现失败。
- 特权上下文 r 很长时，p_t 的 prefill 开销会变大。
- 黑盒分数依赖采样温度，T=1.1 的设置需要固定。
- H2 在本地算力下只能给出较松的 ε̂ 下界。论文的叙述重点应放在“相同 ε 下不同机制之间的经验差异”上，而不是声称达到了紧审计。
- 如果 DP-IER-E 的查询选择使用了私有数据却没有计入预算，那就是一个隐私漏洞，不是研究发现（这一点需要先核对 DP-OPD-004 的隐私单元）。
