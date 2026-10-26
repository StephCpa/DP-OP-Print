# OPSD 成员泄露的事前预测：实验设计 v1（2026-10-26）

**本版依据**：
- OP-Print（2610.31616）全文。
- OPSD 官方代码 `siyan-zhao/OPSD`：`opsd_trainer.py`、`opsd_train.py`、`scripts/run_opsd_{1b,4b,8b}.sh`、README。
- 已通读：2505.11837 全文。
- 读了摘要和关键段落：2602.15919、2605.25819、2505.18773、2609.36331、LT-IQR（arXiv 2411.05743 / USENIX Sec '25）。
- 标 **[未核实]** 或 **[推断]** 的内容不是读原文得到的事实。v0 保留不动，见 `op-print-design.md`。

## 变更记录 v0 → v1

1. **纠正 v0 的事实错误**：v0 写“原文是全参训练”，这是我推断错了。证据如下：
   - OP-Print 附录 A 说所有运行都用 fixed teacher。OPSD 代码中 `fixed_teacher=True` **必须配 PEFT**，否则直接报错。
   - 表 3 的超参（β=0，λ=1，T=1.1，top-p 0.95，top-k 20，max grad norm 0.1，lr 5e-6，8B 的 token clip 0.06 而 1.7B/4B 是 0.05）与 OPSD 官方脚本**逐项吻合**。官方脚本用的是 **LoRA r=64、α=128、作用于全部线性层，lr 5e-6**。
   - 因此 OP-Print **很可能就是 LoRA r=64 + lr 5e-6，训 3 epoch**（官方脚本是 30 epoch）。OP-Print 正文没写 LoRA，这是 **[强推断]**。
2. **H0 之前的修正**：
   - LR 按“训练集上去掉了 θ0 的 Δ 的比例”来选，不看攻击 AUC。记录 max-grad-norm 0.1 的裁剪率。
   - 处理 bf16 merge/unmerge 的舍入问题。
   - LoRA 与全参的对照改用 Qwen3-0.6B 全参，并附显存账。
3. **H1 重写**：
   - 主结果变量改为 log 比值；ε 和 winsorize 规则写死。
   - 难度零假设对照换成 placebo-membership 置换。
   - 协变量加 log T 和 θ0 leverage。
   - 训练设计改为 4 个随机切分加各自的补集，每个样本恰好 4 in / 4 out。
   - 写清信度门槛（原始值还是 Spearman-Brown 校正值）。
   - 合并多个模型前做 out 校准。
   - 加入 FPC 偏差处理。
   - kill 条件改为“c 在 θ0 leverage + loss 之上没有增量”。
4. **H0 统计**：AUC 差用 paired bootstrap；G 与 G-Max 用配对 DeLong。
5. **H2 改为比较式**：两个机制在相同 ε、相同效用下比较 OP-Print 优势；δ = 1e-5。DP-OPD-004 / DP-IER-E 作为条件臂。
6. **H3 改为剂量轴**：调的是现成的 `jsd_token_clip`，定义已经从代码钉死，剂量里包含关闭。
7. **小项**：
   - 澄清 K 的含义。
   - KL 按 chunk 计算以控制显存。
   - H1′ 用公共随机数（CRN），并加 θ0-teacher 臂。
   - 撤回 v0 里说 2505.11837 与 OP-Print 方向“张力”的说法。
   - 加 LT-IQR 式的 Δ 轨迹作为训练中的上限对照。
   - 把“风险分层”从退路提升为可能的主产出。
8. **新发现**：
   - OPSD 代码自带 `--top_k_loss`，H4 不用自己实现。
   - token clip 是**逐词表元素**截断，不是逐 token 截断，详见 §3。

## 0. 已钉死的原文与代码事实

- **核心量**：Δ_t = KL(p_θ(·|p_t(q,r), y<t) ‖ p_θ(·|p_s(q), y<t))。两侧都用被攻击的 θ，y ~ p_θ(·|p_s(q))。
- **两个灰盒分数**：
  - G = −(1/K)Σ_k (1/T_k)Σ_t Δ_t
  - G-Max = −(1/K)Σ_k max_t Δ_t
  - 实现上按 64-token chunk 计算 Δ。
- **K 的含义**：K 是**完整 rollout 的条数**（默认 32，图 5a 消融了条数和 chunk 大小）。chunk 是在每条完整 rollout（最长 1024 token）**内部**切出的 64-token 段。Algorithm 1 只描述黑盒，不涉及灰盒 K。v0 的 §0（K=32）和 §3（K=8）说的都是 rollout 条数，没有矛盾。
- **训练目标（β=0）**：代码中 `kl_div(student_logp, teacher_logp)` 即 KL(teacher‖student)，方向与 Δ 一致。teacher 是关掉 adapter 的 base 加特权上下文。
- **`jsd_token_clip` 的定义（已读代码）**：`jsd` 的形状是 `[B,T,V]`，在**对词表求和之前**逐元素执行 `clamp(max=clip)`，然后对 mask 后的元素求 sum / token 数。clip=0 表示关闭。README 也写明是 point-wise，不是对 token 级总和截断，并提示 loss 可能为负。
- **原文报告的指标只有 AUC、Acc、TPR@5%FPR**。G-Max 的 AUC：1.7B 0.594，4B 0.611，8B 0.587。

## 1. H0 之前必须先修的三件事

- **LR 选择**：
  - 定义 f_rm = 1 − mean_{x∈train} Δ̄_θ(x) / Δ̄_θ0(x)，在固定的 CRN rollout 上计算，见 §3。
  - 网格为 {5e-6（官方锚点）, 2e-5, 5e-5, 1e-4}，r=64，3 epoch。
  - 预注册目标：f_rm 落在 [0.2, 0.5]，且 θ0 的 pass@1 不下降超过 2 个百分点。在这两个条件下取最小的 LR。
  - **全程不看任何攻击指标。**
  - 记录 max_grad_norm=0.1 的裁剪率，即裁剪前范数大于 0.1 的步数占比，以及范数中位数。如果接近 100%，训练实际上是“归一化梯度 + LR”，在论文里注明这一点。
- **bf16 merge 问题**：
  - 已读代码：每次同步 vLLM 时，代码先 `merge_adapter()`，再 `load_weights`，再 `unmerge_adapter()`。
  - [推断] 这会带来两个后果：
    1. rollout 来自 bf16 舍入后的合并权重，与训练中的策略略有不同。
    2. bf16 的原地加减不可逆，base 权重会累积漂移。而 fixed teacher 正是“关 adapter 的 base”，所以 **teacher 本身也会漂**。
  - 对策：
    - 打分时一律用**不合并的 adapter**（HF 前向，base bf16 + adapter），或在 fp32 里合并。
    - 训练前后对 base 权重做哈希或 L∞ 比较，量化漂移。
    - 如果有漂移，改用 vLLM 原生 LoRA（`enable_lora`）加载 adapter，不做 merge。
    - 在论文中注明 rollout 策略与训练策略的差异。
- **全参对照**：
  - 1.7B 全参放不下 16 GB：AdamW 的 fp32 m、v 约 13.6 GB，批评意见中这个数正确。
  - 改用 **Qwen3-0.6B 全参** [估算]：
    - bf16 权重 1.2 GB，bf16 梯度 1.2 GB，AdamW fp32 m、v 4.8 GB（8-bit Adam 则约 1.2 GB）。
    - 冻结 teacher 副本 1.2 GB。全参模式下 OPSD 的 fixed teacher 不可用，需要自己加一个冻结副本。
    - 激活（开梯度检查点）加 chunk 化 logits 约 2–3 GB，vLLM 0.3 的显存份额约 4.8 GB。
    - 合计：fp32 Adam 约 15–16 GB，太紧；**8-bit Adam 约 12 GB，可行**。需要实测。
  - **我的立场**：既然原文目标模型很可能就是 LoRA，这一项只在 H0 失败时用来区分“是不是 LoRA 容量导致的”，**降为可选**。

## 2. 假设

- **H0（复现）**：Qwen3-1.7B + LoRA r=64 用 §1 选出的 LR。
  - 检验：G-Max 与最佳基线的 AUC 差，做 **paired bootstrap**（按样本重采样，2000 次）。
  - “最佳基线”在另一半样本上选定，避免挑选偏差。
  - G 与 G-Max 在每个模型上做**配对 DeLong** 检验，并报告 8 个模型之间的 seed 波动。原文“G-Max 更好”目前既不能算显著，也不能算边缘。
- **H1（主线）**：θ0 下的集中度 c(x)（Δ_θ0 在 chunk 上的 top-10% 份额，以及 max/mean）预测 v(x)。
  - **主结果变量**：v(x) = log((Δ̄_out + ε) / (Δ̄_in + ε))，即训练去掉的修正比例。**次结果变量**：差值 Δ̄_out − Δ̄_in。
  - ε 预注册为所有（模型, 样本）Δ̄ 的第 1 百分位数。v 在 1% 和 99% 分位处 winsorize。G-Max 版本作为次要分析。
  - **合并前的校准**：每个模型的 Δ̄ 先除以该模型 out 样本 Δ̄ 的中位数。这是乘性的 out 校准，保持 Δ̄ 为正，比值仍然有意义。差值变量和 ROC 合并则用每个模型的 z-score。注意 z-score 会破坏比值，所以两种结果变量分开处理。
  - **协变量 d**：θ0 pass@k、θ0 下的 Δ̄、log T（θ0 rollout 的平均长度）、θ0 loss，以及 θ0 last-layer leverage（2602.15919 的方法移植到 θ0 hidden states，见 §4）。
  - **零假设对照（placebo-membership）**：对每个样本，在其 8 个模型中随机重标 4 个为 in（计数固定），重算 v_placebo 并重新拟合，重复 200 次。c 对 v 的系数必须超过 placebo 分布的 95% 分位，同时 c 不应预测 v_placebo。
  - **Kill**：在 d 之上，c 的偏系数 CI 包含 0，或增量 R² 不超过 placebo，即 **c 在 θ0 leverage + loss 之上没有增量**。
- **H1′（参考校准攻击）**：用 CRN，在同一批 rollout、同一批 chunk 上计算 Δ_θ 和 Δ_θ0。
  - 分数 ①：S_θ − S_θ0。
  - 分数 ②：剩余修正比例 ρ(x) = ΣΔ_θ / ΣΔ_θ0，即 ρ_S 的逐样本类比。
  - 分数 ③，**θ0-teacher 臂**：KL(p_θ0(·|p_t) ‖ p_θ(·|p_s))，也就是训练目标本身。可以再加一个按 clip 截断后的版本。
  - 这三个分数只需要公开的 base 和 r，原文都没做。额外成本只有一次 θ0 prefill。
- **H2（比较式 DP）**：在相同 ε（δ = 1e-5，远小于 1/N = 2e-3）和相同效用（held-out pass@1 相差不超过 1 个百分点）下，比较两个机制的 OP-Print 优势：TPR@1%FPR 和 AUC − 0.5，都用 out 校准后的合并 ROC。
  - **臂 A**：DP-SGD，按记录（(q, r)）裁剪，同一条 q 的所有 rollout 梯度先合并成一个再裁剪，Poisson 采样，PRV accountant。
  - **臂 B**：DP-OPD-004 / DP-IER-E。**这一臂需要 Alex 确认其隐私单元是蒸馏记录 (q, r)，而不是 teacher 的训练数据。** 确认之前标为 out-of-scope，用臂 B′ 代替：DP-SGD + token clip 开启，与臂 A 的 clip 关闭对比。
  - 预测（可证伪）：B 或 B′ 的 OP-Print 优势显著低于 A，或者两者无差别，两个方向都写进预注册。
  - 不再把“ε̂ 远小于 ε”当作结论，因为这是默认结果。
- **H3（剂量轴）**：`jsd_token_clip` ∈ {0（关闭）, 0.2, 0.05（原文）, 0.01}。
  - 因为截断逐元素作用在高差异位置，预测 G-Max 的优势随剂量单调下降。
  - 必须配自适应攻击：G（求和）、top-q chunk 均值，以及用 2 个模型训练、6 个模型测试的学习型聚合器（chunk Δ 向量输入逻辑回归）。
  - 每个剂量都要报告效用。对照 DuoLearn（token 级经验防御）。
- **H4（最低优先级）**：用 OPSD 自带的 `--top_k_loss` ∈ {0, 20, 100} 对比 TT-OPD 式无偏 top-k，在相同效用下比较。

## 3. 设置与算力

- **数据池**：OpenThoughts 数学子集，N+ = 1000。**4 个随机切分加各自补集，共 8 个模型**，每个模型训练 500 条，每个样本恰好 4 in / 4 out。
- **Split-half**：取 2 对（切分, 补集）为一半，另外 2 对为另一半，每半恰好 2 in / 2 out。共有 3 种分法，取平均。
- **信度门槛**：**原始** split-half r ≥ 0.3，对应 Spearman-Brown 校正后的 8 模型信度约 0.46。
- **FPC 偏差（2605.25819 已读确认）**：从固定超集抽 N = 0.5·N+ 时，FPC = 1 − N/N+ = 0.5，in/out 方差被低估，脆弱度被高估。
  - 补集设计**恰好把 N/N+ 固定在 0.5**，这正是 FPC 影响最大的情形，两条批评意见在这里有冲突。
  - 我的处理：H1 关心的是相对排序，FPC 近似是一个公共因子，所以保留补集设计。凡报告绝对 TPR 的地方，按该文的 FPC 修正处理。该文的 decomposed FPC 细节只读了一部分 **[未核实]**。
  - 拼接 ROC 会使每个样本的 FPR 不一致，所以报告 out 校准后的结果，并注明这一点。
- **打分**：K=8 条完整 rollout（图 5a 显示在 4B 上，K 从 32 降到 8，G-Max 的 AUC 约降 0.02）。
  - 在 200 个样本的子集上用 K=32 检查排序稳定性。
  - “只取前 64 token”只作为廉价消融。原文的 max 是对整条 rollout 取的，截断会改变统计量，所以**我不采纳“rollout 可以停在 64 token”**。
  - KL 按 chunk 算：每个 chunk 64 × 151,936 × 4 B ≈ 39 MB/侧，而整条 1024 token 序列约 0.62 GB/侧。
  - 特权 prompt 的 KV 在 K 条 rollout 之间复用。Qwen3-1.7B 每 token 的 KV 约 115 KB [估算]，8k token 的 r 约 0.9 GB。
- **算力** [粗估，需用 1 个模型实测校准]：
  - 训练：8 个模型，约 8–16 GPU·h。
  - 打分：约 6.4 万条 rollout 加两次 prefill，约 10 h。
  - H1′ 要多算一次 θ0 前向，约 +3 h。
  - LR 网格：4 个 LR × 1 个模型，约 4–8 h。
- **训练中的上限对照（LT-IQR 式）**：记录每个 in 样本每次被训练时的 Δ 损失轨迹，以及 save_steps 检查点上 CRN rollout 的 Δ。用 IQR 排序作为“训练中可得信息”的上限，与 c（θ0，训练前）比较。
  - 注意：3 个 epoch 下每个样本**只有 3 个免费点**，加检查点评估就不再免费。

## 4. 已有工作（核实结果）

- **2602.15919**（Dorseuil, Atif, Cappé）：已核实。
  - 方法是用**单个已训练模型**，以 loss 加 last-layer 表示的 leverage 分数评估逐样本风险。**不是事前预测**。
  - 实验只覆盖分类和回归：CIFAR、BloodMNIST、SST-2 BERT、表格数据。没有生成式 LLM。
  - 我们的做法是把 leverage 移植到 θ0 的 hidden states（rollout token 平均后，对数据池做 ridge leverage），这是**我们自己的改编**，迁移效果 **[未核实]**。
- **LT-IQR**（Pollock 等，USENIX Sec '25，arXiv 2411.05743）：已核实。
  - 用逐样本的逐 epoch loss 轨迹的 IQR 排序高风险样本，不需要额外模型。CIFAR 上 precision@1% 为 92%。
  - 只在图像分类上验证过 [摘要]。
- **2605.25819**（Jälkö 等）：已核实两点。① 从固定超集抽 shadow 训练集会因有限样本偏差高估脆弱度；② 拼接 TPR 时每个样本的 FPR 不一致。两者都给了修正方法。
- **2505.18773**（Hayes 等，NeurIPS '25）：已核实。
  - 即使 AUC 高于随机，很多逐样本判定受训练随机性影响，与抛硬币无异。
  - “脆弱度规律被样本长度复杂化”这一点在正文 §6.1 和附录 F 中写明：越长的样本越脆弱。
- **2609.36331**（Rita, Zhang, Tramèr，AISec '26）：已核实，原文写明“per-example calibration 是 one-round 攻击和 one-run DP 审计的共同瓶颈”，方法是用邻居样本做校准。实验只在图像分类上。H2 如果走 one-run 路线，需要参考这篇。
- **2505.11837**：v0 说它与 OP-Print 方向存在“张力”，**撤回**。它比较的是“teacher 对 ground truth 的 KL”，在固定数据集上算；OP-Print 的 Δ 是“teacher 对 student”在 student rollout 上算。两者不是同一个量，在同一设置里都算出来之前不作为论文主线。
- **DuoLearn**：H3 的对照。

## 5. 执行顺序与止损

1. 打通 OPSD 代码 → 做 bf16 漂移检查 → 跑 LR 网格（只看 f_rm 和 pass@1）。
2. **H0**：1 个模型。**Kill**：paired bootstrap 的 AUC 差 CI 包含 0。之后可选做 0.6B 全参，判断失败原因是不是 LoRA。
3. 8 个模型 → 信度门槛。**原始 r < 0.3 就不做逐样本 H1**，直接转到风险分层（见 §6）。
4. H1，H1′ 同时进行 → LT-IQR 上限对照。
5. H3（剂量轴，不需要 DP，成本低）→ H2（等 Alex 确认臂 B 的隐私单元）→ H4。

## 6. 风险

- **风险分层很可能成为主产出**，而不只是退路，因为 2505.18773 显示逐样本判定极不稳定。分层方法：按 c 把样本分成 5 档，检验各档合并后的 TPR@1%FPR 是否单调。这比逐样本排序稳健，也已经满足“事前可算的风险指标”这一目标。
- 原文效应很弱（AUC 约 0.6）。
- LoRA 配置的推断如果有误，复现可能失败。
- bf16 漂移会污染 teacher。
- 长 r 的 prefill 开销大。
- 采样温度要固定在 T=1.1。
- 臂 B 的隐私单元如果不是蒸馏记录，H2 只能做 A 与 B′ 的比较。

## 7. 与批评意见不同的地方

1. **“LoRA 照抄 5e-6 会让 adapter 几乎不动”**：前提基本不成立。OPSD 官方配置本身就是 LoRA r=64 加 5e-6，OP-Print 很可能也是，并且在这个配置下测出了 AUC 0.59。我仍然采纳“按 f_rm 选 LR”这个流程，因为它是对的，但 5e-6 保留为锚点。
2. **K 和 64-token 截断**：K 是 rollout 条数，chunk 在完整 rollout 内部切分，所以截断只能作为消融。
3. **补集平衡设计与 2605.25819 的 FPC 问题相互冲突**：处理见 §3。
4. **z-score 与比值结果变量不兼容**：主结果变量改用乘性 out 校准。
5. **0.6B 全参降为可选**，理由见 §1。
6. **LT-IQR 在 OPSD 里并非完全免费**：3 个 epoch 只有 3 个点。

其余各点全部采纳。
