# 针对模型八个缺点（W1–W8）的文献检索（2026-09-30）

这份文档按 DEFECTS.md M8–M11 里测出的缺点逐条查了文献。表里只列搜索代理打开过页面、核对过作者、年份、发表处和链接的论文；没核对上的三篇放在文末一行。和以前两份检索（RELATED_WORK.md、LESSONS_FROM_RELATED_WORK.md）重复的论文，只在某个缺点确实需要时才列，并注明"见 RELATED_WORK.md"。

要先说清楚证据的性质：下面每条"证据"都是论文在它自己的平台上（机械臂、四足、无人机、天气、视频、玩具问题）测到的结果，没有一篇是在我们的船或类似的喷水滑行艇上验证的。"对我们有用"一栏是我们的推断，要靠我们自己的实验确认。

## 总结：文献最改变我们下一步的四件事

1. **推演卡住、漂走（W2），最便宜的办法是训练时给"推演中会被喂回去的历史"加随机噪声**。GameNGen、Diffusion Forcing 和湍流基准（Kohl）都表明：训练时把作为条件的历史故意弄脏（噪声强度随机，并把强度告诉网络），目标仍是真实的下一步，这样模型就不会过分相信自己喂回来的样本；湍流基准里去掉这一招，生成模型退化到和普通网络差不多。这和我们 M9 失败的"在自己推演上用能量得分训练"是两回事：它不需要推演、不偏，而且直接针对"喂回来的喷口滞后迟迟不消"。同时值得便宜地试两件事：让 flow 从上一步误差出发而不是从纯噪声出发，以及把 24 步确定性欧拉积分换成带可调噪声的随机采样。
2. **MPPI 里所有候选开法应该用同一组随机数（W6）**。我们的 flow 头是"固定的随机输入 z → 误差样本"的确定映射，所以可以在每个控制步只抽一次 S 组 z，192 条开法全部共用。这样比较开法优劣时，差别不再被各自独立的随机性淹没，每条开法需要的样本数很可能从 8 降到 2–4（要实测）。这几乎不花钱，而且能让推演对开法可求导，为后面的加速方法铺路。再配合"保留上一步最好的几条开法"（iCEM）和"先用一次便宜推演筛出前几名，再对前几名做完整随机评估"，候选数也可能大减。
3. **在线学习时，把"按验证回合选的 1.25 倍放大系数"换成一个在线自动调节的规则（W3）**。Conformal PID 的做法是：每个预测步长 h、每个通道各有一个放大系数，h 步前做的预测现在能核对了，真值落在 90% 区间外就放大一点，落在里面就缩小一点。它对相关的、会漂移的数据也保证长期平均覆盖率。更进一步，Laplace-LoRA 给出了"全量微调准但过度自信、只调头稳但不够准"之间的中间路线：只调插在各层旁边的小矩阵，再给这些小矩阵一个"还可能是多少"的高斯分布，每条推演抽一组、整条用到底。
4. **先验不对（W1）既要"加宽关系的种类"，也要"用目标数据校准"，而且应该先能自动发现"已经出了先验"**。Raventós 的受控实验说明：预训练任务种类不够多时，上下文学习模型只会在"见过的几类关系"里挑，不会现场回归新关系；种类超过一个门槛后，它才像通用学习器。M11 的纵摇现象（目标上纵摇误差和纵摇角、航速明显相关，训练数据里几乎无关）正是先验缺少"随状态变化的关系"。Panda 的缩放结果说：总数据量固定时，增加"不同系统的个数"远比增加每个系统的回合数有效。DROPO / NPDR 给出用少量目标回合直接算似然、调整先验随机范围的办法，Schmitt 给出"部署时检测出了先验"的监视器。

探测（W8）的奖励设计也有一条明确结论：不能用"预测意外程度"当奖励（会奖励去浪大、噪声大的地方），要用"有这段探测和没有这段探测，模型对另一组后续数据的预测好了多少"。

---

## W1 先验和目标不符

**我们的缺点**：只用目标数据微调同一个模型、不加新传感器，艏摇误差从约 0.94 降到约 0.3（占误差方差的比例），纵摇从约 0.91 降到约 0.47。信息在观测里，是训练世界里随机生成的"误差规律"（浪力、滑行纵摇的回复和阻尼、执行机构的形状）不像目标的。M11 还测到：目标纵摇误差和当时纵摇角、航速有关，训练数据里的纵摇误差几乎和状态无关；模型的技能随历史变长而变好，说明它确实在从历史里现场推断。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| DROPO: Sim-to-Real Transfer with Offline Domain Randomization | Tiboni, 2023 | Robotics and Autonomous Systems | https://arxiv.org/abs/2201.08434 | 用一小段真实记录直接调仿真参数的随机范围：把仿真重置到记录的状态，重放指令，看真实下一步在仿真抽样分布下的可能性，用不求导的搜索（CMA-ES）调范围 |
| Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression | Raventós, 2023 | NeurIPS | https://arxiv.org/abs/2306.15063 | 预训练关系的种类要超过一个门槛，模型才会在历史里现场回归新关系；解释了我们为何学不到目标的纵摇关系 |
| Bridging the Sim-to-Real Gap with Bayesian Inference | Rothfuss, 2024 | IROS | https://arxiv.org/abs/2403.16644 | 先验 = "随机参数的粗仿真" + "一般的平滑未知残差"；在目标上微调时，在没数据的地方把网络拉向先验的预测，而不是简单地拉向原权重 |
| Panda: A pretrained forecast model for chaotic dynamics | Lai, 2026 | ICLR | https://arxiv.org/abs/2505.13755 | 纯合成训练的动力学预测模型，系统族靠"扰动参数 + 两两耦合 + 合理性筛选"造出来；数据量固定时，多造不同的系统比多跑同一系统有效得多 |
| Neural Posterior Domain Randomization | Muratore, 2021 | CoRL | https://proceedings.mlr.press/v164/muratore22a.html | 从少量真实回合推断仿真参数的完整分布（含参数间相关）；每隔 K 步把仿真重置到真实状态，防止快系统很快分叉 |
| On the adaptation of in-context learners for system identification（见 RELATED_WORK.md） | Piga, 2024 | IFAC SYSID | https://arxiv.org/abs/2312.04083 | 和我们结构最近：合成先验上的上下文学习模型遇到先验外系统时，以它为起点继续训练，混入原合成数据，用验证集及时停止 |
| Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models | Zhang, 2025 | NeurIPS | https://arxiv.org/abs/2510.21204 | 决定先验由哪几块组成、比例多少的流程：每块单独训一个模型，做"在 i 上训、在 j 上测"的表，优先加入别的块预测不了的块 |
| Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks | Schmitt, 2023 | GCPR | https://arxiv.org/abs/2112.08866 | "出了先验"监视器：训练时把合成数据的内部特征推向标准高斯，部署时比较目标特征和合成特征的分布差距，超阈值就报警 |
| Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data | Garg, 2025 | arXiv 预印本 | https://arxiv.org/abs/2507.03971 | 有真实数据后，把适应做成"续训"：每个真实回合切成许多随机长度的上下文任务，小学习率并惩罚偏离原权重；少而干净的数据好过多而杂的 |
| Understanding Domain Randomization for Sim-to-real Transfer | Chen, 2022 | ICLR | https://arxiv.org/abs/2110.03239 | 理论：带记忆的模型能边用边辨识；先验加宽只有在"目标附近多放了概率"时才有用，盲目加宽反而稀释 |

**可以借鉴的（按优先级）**

1. **加宽"关系的种类"，而不是只加宽已有关系的参数范围**（Raventós、Panda）。现在的随机误差算子只作用在 5 个速度通道上（M10 刚给 2 个执行机构通道加了随机族）。可以在先验里加一个通用成分：随机挑任意观测通道（包括纵摇角、升沉、航速）作输入，经随机滤波和随机非线性映射到任意误差通道。这是不针对具体机制的做法，正好补上 M11 里"训练数据的纵摇误差与状态无关"这个缺口。检验办法我们已经有：看目标上误差是否随历史变长而继续下降（M11 已做过这种按历史长度的比较）。
2. **训练预算优先花在"更多不同的船、海、执行机构"，而不是同一设置下更多回合**（Panda 的缩放结论）。这和以前 LESSONS 文档里"训练集太小会被记住"的风险是同一方向。
3. **用目标回合直接校准先验的随机范围**（DROPO、NPDR）。把低保真仿真重置到目标记录的状态，重放指令若干步，从先验抽 K 组，看目标的一步误差在这些抽样下的可能性，用不求导的搜索调整随机范围。只需要现有的 4–16 个目标回合。NPDR 的"每隔 K 步重置到真实状态"对滑行艇这种容易分叉的动力学应该照搬。DROPO 的教训也值得记下：只用仿真训练的推断网络，遇到落在它训练范围外的真实数据时推断会很差，所以校准要直接用仿真器算可能性，不要指望网络自己推断。
4. **加一个"出了先验"监视器**（Schmitt）：取 Transformer 的内部状态作特征，训练时加一项让合成数据的特征接近标准高斯；在线时对最近几段目标数据算分布差距，超过阈值就报"在先验外"，触发在线学习或放大预测分布。可以和第 W3 节的在线覆盖率监控并用。
5. **先验的组成用"训 i 测 j"的表来决定**（Mitra），不凭感觉加。受 16 GB 内存限制，只能用小模型串行跑，放在后面。
6. **目标数据到来后的适应做成"续训"**（Piga、Real-TabPFN、Rothfuss）：混入原合成数据、随机切历史长度、小学习率，保住模型"从历史现场推断"的能力，而不是把它拟合成只认识一条船。

**不适用或要注意的**

- DROPO、NPDR 只能在已选定的参数族里调宽窄和位置，造不出先验里根本没有的关系；而我们缺的恰恰主要是关系的形式。所以第 1 条（加关系种类）优先于第 3 条（调范围）。
- Rothfuss 的"粗仿真 + 物理结构随机族"如果做成"在简化模型里加一个有名字的浪力项或纵摇回复项"，就违背了我们已定的"方法里不放针对具体机制的项"的原则。可接受的形式是：这些结构只出现在训练世界的随机族里（像 M10 的执行机构族那样），方法本身仍是通用的误差模型。
- Raventós 只是线性回归的玩具实验，门槛的具体数值不能搬过来；加多样性可能降低先验内的精度，要实测权衡。Mitra、Real-TabPFN 是表格数据，不是时间序列。
- Chen 2022 的理论假设真实系统在族内或其附近，我们的问题恰恰是不在族内。
- 我们自己的一个想法（文献里没找到检验）：上下文学习模型在目标数据上的逐步预测可能性之和，近似于"该先验下看到这批数据的可能性"，可用来比较候选先验。需要自己验证。

---

## W2 随机推演卡住、漂走

**我们的缺点**：推演时模型把自己抽的误差喂回去。M8 测到：用真实历史时喷口滞后约 5 步归零，推演中第 5 步还剩约四分之一；持续的假滞后让艏摇误差保持同号，积分进航向，1 秒后越推越偏。M9 在自己推演上用整段能量得分（一种给"一群样本"打分的方法，既奖励接近真值也奖励有散布）训练，起点附近不动；原因是训练世界里推演本来就没问题，毛病来自训练族外的执行机构。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Diffusion Models Are Real-Time Game Engines (GameNGen) | Valevski, 2025 | ICLR | https://arxiv.org/abs/2408.14837 | 训练时给作为条件的历史帧加随机强度的噪声，并把强度告诉网络，目标仍是真实下一帧；没有这一招 20–30 步后明显退化 |
| Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion | Chen, 2024 | NeurIPS | https://arxiv.org/abs/2407.01392 | 上一招的一般形式：序列里每一步有自己的噪声程度；推演时把自己生成的历史标成"略带噪声"再喂回去 |
| Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation | Kohl, 2023 | arXiv（期刊版未能打开） | https://arxiv.org/abs/2309.01745 | 物理系统上的证据：给条件加噪声是生成式推演长期稳定的关键；提供"每步变化率"指标，能区分"卡住"（变化塌缩）和"漂走"（变化增长） |
| Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion | Huang, 2025 | NeurIPS | https://arxiv.org/abs/2506.08009 | 和我们 M9 最像：在自己推演上用整段分布打分训练；它的收益出现在模型"在分布内自己输出上就错"的情况，这解释了 M9 为何无效 |
| Skillful joint probabilistic weather forecasting from marginals (FGN) | Alet, 2025 | arXiv 预印本（DeepMind） | https://arxiv.org/abs/2506.10772 | 在自己推演上用恰当打分训练确实能得到长时校准的集合预报；关键细节：推演长度从 1 步逐渐加到 8 步、逐通道逐步打分、低维噪声跨通道共享 |
| ArchesWeather & ArchesWeatherGen | Couairon, 2024 | arXiv 预印本 | https://arxiv.org/abs/2412.12971 | 和我们结构最像（确定性基础模型 + 学残差的 flow 模型）；推演散布偏窄，原因是测试时残差比训练时大；修法是在基础模型没见过的数据上训残差，并按多步覆盖率放大初始噪声 |
| Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes（见 RELATED_WORK.md） | Chen, 2024 | ICML | https://arxiv.org/abs/2403.13724 | 生成从当前状态出发而不是从纯噪声出发；采样噪声大小训练后可调；2026 年的 StocBench 报告确定性 ODE 采样会压低随机推演的统计量 |
| Diffusion Model Predictive Control（见 RELATED_WORK.md） | Zhou, 2025 | TMLR | https://arxiv.org/abs/2410.05364 | 候选开法事先定好时，可以一次生成整段 24 步修正，视界内不喂回样本，没有暴露偏差 |
| On Rollouts in Model-Based Reinforcement Learning (Infoprop) | Frauenknecht, 2025 | ICLR | https://arxiv.org/abs/2501.16918 | 把"模型不懂"的不确定和"本身随机"的不确定分开；沿推演累计"模型不懂"的量，超预算就不再信任后面的推演 |
| Deep RL in a Handful of Trials using Probabilistic Dynamics Models (PETS)（见 RELATED_WORK.md） | Chua, 2018 | NeurIPS | https://arxiv.org/abs/1805.12114 | 我们已有"随机模型 + 抽样推演"的一半；缺的另一半是用几个模型的分歧表示"模型不懂"，每个样本整条推演固定用同一个模型 |

**可以借鉴的（按优先级）**

1. **训练时给会被喂回去的历史加随机噪声，并把噪声强度作为输入**（GameNGen、Diffusion Forcing、Kohl）。对象是推演时来自模型自己的那部分历史：过去的误差通道，特别是油门、喷口的"实际减指令"，以及由它们推出的简化模型状态。目标仍是真实的下一步误差，所以学到的是一个正当的条件分布，不会有 M9 设计时指出的"逐步纠偏目标对随机系统有偏"的问题。推演时噪声强度按"没参与训练、专门拿来检验的仿真回合"上的多步覆盖率来选。效果是模型学会少信自己喂回来的样本，多信指令和物理状态，这正是出族执行机构让喂回来的误差路径变陌生时所需要的。
2. **把 Kohl 的"每步变化率"指标加进推演评估**，按通道算。喷口卡住会表现为变化率塌缩，航向漂走表现为变化率增长，用它能快速检查上一条是否有效。
3. **便宜地试两处 flow 采样改动**（Chen 2024、StocBench）：对有持续性的通道（执行机构滞后、升沉、纵摇），让 flow 从上一步误差出发而不是从纯噪声出发，"保持在原处附近"成为默认；把 24 步确定性欧拉积分换成噪声可调的随机采样器。StocBench 报告确定性采样会压低随机推演的统计量，这可能是"卡住"的一个原因。
4. **如果再做在自己推演上训练，照 FGN 的清单来**：推演长度从 1 逐步加到 8；逐通道、逐步的边缘打分放在整段联合打分旁边；并且在目标数据上（模型真正在分布外的地方）做，而不是只在训练世界里做（Self Forcing 的教训）。另一个降噪办法：用低保真仿真对同一起点、同一开法抽很多条真实路径，拿模型的路径群和仿真的路径群比，而不是和一条观测路径比。
5. **按多步覆盖率校准推演散布**（ArchesWeatherGen），逐通道放大 flow 的初始噪声作为事后可调的旋钮，这和 W3 里已发现有用的放大系数是同一件事。
6. **加入"模型不懂"的信号**（Infoprop、PETS）：3–5 个 flow 头，或在目标微调后的几个头；每个推演样本整条用同一个头；沿推演累计头之间的分歧，超预算就降低视界后段代价的权重或缩短可信视界。这个量也是 W8 探测的天然信号。
7. **整段生成 24 步修正**（D-MPC）：以历史、整段计划和简化模型在该计划下的开环轨迹为条件，一次 flow 生成整条"真实轨迹减简化模型开环轨迹"。视界内不再喂回样本，还把 24 次顺序推理合成一次（也帮 W6、W7）。

**不适用或要注意的**

- 所有加噪声的论文都没测推演散布是否校准，只测了画面质量；加噪声会让分布内的预测变宽，锐度换稳健性。对执行机构通道加太多噪声，可能洗掉模型需要的延迟信息。
- 加噪声不告诉模型"已出了族"，它只是让模型不那么依赖喂回来的量。M9 的判断仍成立：剩下的偏差来自训练族外的执行机构行为，根本的补法是先验（M10）或在线学习。
- Self Forcing 用的分布匹配损失倾向于压缩多样性，对校准有风险；而且需要一个强的"老师"模型。
- D-MPC 的整段生成可能在族外泛化更差（FlowTime 2025 预印本发现逐步版本外推更好），要直接比较。
- Infoprop 和 PETS 的集成成员如果都只在同一个合成先验上训练，可能一起错而彼此同意；"模型不懂"的信号可能需要在少量目标回合上训出的成员。
- DIAMOND（NeurIPS 2024）原本被认为用了历史加噪声，打开原文后确认没有：它保持历史干净，长期稳定归功于输出参数化方式。

---

## W3 少量目标数据的在线学习要保持校准

**我们的缺点**：M11 在 4–16 个目标回合上全量微调，准确度提高，但 90% 区间只盖住 30–70% 的真值（覆盖率，即真值落进模型给的 90% 区间的比例）；只调 flow 输出头（冻结 Transformer）覆盖率约 0.8，再按验证回合把抽样噪声放大 1.25 倍后约 0.9，准确度和全量相当（纵摇还更好）。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Bayesian Low-rank Adaptation for Large Language Models (Laplace-LoRA) | Yang, 2024 | ICLR | https://arxiv.org/abs/2308.13111 | 只调插在冻结权重旁边的小矩阵，事后给这些小矩阵一个高斯分布（"还可能是多少"），宽度用训练数据本身选，不用拿出验证回合；小数据下的过度自信大幅下降 |
| Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving | Levy, 2025 | RSS | https://arxiv.org/abs/2504.16923 | 和我们最像（MPPI + 物理模型 + 学习残差 + 在线适应，真车）；在线适应器的噪声水平和先验不手调，而是把适应器展开、按多步预测误差反向训练出来 |
| Meta-Learning Priors for Efficient Online Bayesian Regression (ALPaCA)（见 RELATED_WORK.md） | Harrison, 2018 | WAFR | https://arxiv.org/abs/1807.08912 | 训练时就模拟"看了前 k 步、预测后面"，让少数据下的散布按构造校准；没有这种训练时散布收得太快 |
| Conformal PID Control for Time Series Prediction | Angelopoulos, 2023 | NeurIPS | https://arxiv.org/abs/2307.16895 | 区间宽度当作反馈控制量：漏掉真值就放大、盖住就缩小；对相关、漂移的数据保证长期平均覆盖率 |
| Low-rank extended Kalman filtering for online learning of neural networks from streaming data (LoFi) | Chang, 2023 | CoLLAs | https://arxiv.org/abs/2305.19535 | 在线逐步更新全部（或某一块）权重，以预训练权重为起点，权重的不确定用"对角 + 低秩"存储，成本和权重数成正比；带遗忘 |
| ARCADE: Adaptive Robot Control with Online Changepoint-Aware Bayesian Dynamics Learning | Yadav, 2025 | arXiv 预印本 | https://arxiv.org/abs/2512.14331 | 用模型自己最近的预测可能性检测"工况变了"，检测到就把积累的证据放松回先验，不确定性自动变宽 |
| Probabilistic Conformal Prediction Using Conditional Random Samples (PCP) | Wang, 2023 | AISTATS | https://proceedings.mlr.press/v206/wang23n.html | 只需要样本的区间校准：真值到最近样本的距离作为分数，适合我们 10 维、非高斯、会多峰的误差 |
| Staggered Integral Online Conformal Prediction ... with Multi-Step Coverage Guarantees (SI-OCP) | Cherenson, 2026 | arXiv 预印本（投 CDC 2026） | https://arxiv.org/abs/2604.06058 | 对整条未来视界（而不是每一步分别）做在线覆盖保证，用于 MPC；分数取视界内累计误差的最大值 |
| Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds（见 RELATED_WORK.md） | O'Connell, 2022 | Science Robotics | https://arxiv.org/abs/2205.06908 | 支持"只调头"的设计原则：共享特征不含工况信息，工况相关的都放进小的可调部分；训练时用一个对手网络逼特征不带工况 |
| Surgical Fine-Tuning Improves Adaptation to Distribution Shifts | Lee, 2023 | ICLR | https://arxiv.org/abs/2210.11466 | 数据少时该调哪几层取决于差异类型；按"梯度大小除以权重大小"在目标数据上给各层排序，只调前一两名 |

**可以借鉴的（按优先级）**

1. **用在线自动调节替换固定的 1.25 倍放大系数**（Conformal PID，配合"延迟核对"）。每个通道、每个预测步长 h 各一个系数；h 步前的预测此刻可以核对，看真值是否落在放大后样本的 90% 带里，然后系数加上"步长 × (是否漏掉 − 0.1)"。它套在任何适应方式上面（只调头、LoRA、卡尔曼式更新都行），能跟上工况变化，而且保证不依赖数据独立。
2. **对整条推演做覆盖校准**（SI-OCP、PCP）：MPC 需要整条 24 步路径被覆盖，不只是每步分别被覆盖。分数可以取"真实路径到最近一条样本路径的归一化距离"，或"视界内累计误差相对预测带的最大值"，再用第 1 条的在线规则调。
3. **在只调头和全量微调之间加一条中间路线**（Laplace-LoRA、Surgical）：按梯度与权重之比在 4–16 个目标回合上给 6 层排序，只在前一两层插小矩阵；训练后给小矩阵一个高斯分布，宽度用训练数据自己选，不花掉宝贵的目标回合做验证。推演时每条序列抽一组小矩阵、整条 24 步用到底，再每步抽 flow 噪声。这样得到的"模型本身不确定"沿视界是一致的，固定放大系数做不到。
4. **在线逐步更新时用 LoFi 式的权重不确定**：以预训练权重为中心，不确定在数据少时保持宽，数据多时自然收窄；遗忘速度决定旧海况多快被忘掉。
5. **把"看了前 k 步、预测后面"的训练方式扩到适应器本身**（Levy、ALPaCA）：在合成先验的回合上展开在线适应器，按 24 步推演误差训练它的噪声水平和先验宽度，替代手选放大系数。
6. **检测到工况变化时自动放宽**（ARCADE）：用模型自己最近的预测可能性做变化检测（例如进入滑行、换海况），检测到就把适应过的部分往预训练权重拉回一些。

**不适用或要注意的**

- 已定原则：线性回归一类的经典估计（岭回归、贝叶斯线性回归）只作评估里的参照，不作方法的组件。ALPaCA、ARCADE、Neural-Fly 的核心都是"冻结特征上的线性头 + 闭式贝叶斯更新"，直接照搬会违背这条原则，而且冻结特征在目标需要新方向时（M11 显示全量微调比只调头准）会"校准但不准"。它们可借鉴的是训练方式（模拟少数据适应来训练）和变化检测，不是线性头本身。Laplace-LoRA 和 LoFi 作用在网络权重上，保留 flow 头，更符合原则。
- Laplace-LoRA 在分类任务上验证，没测区间覆盖率，也没测多步推演；我们的 flow 头没有闭式似然，曲率要用近似（例如 flow 回归损失的高斯-牛顿近似）。
- Conformal PID 只保证长期平均：可能在滑行速度下盖不住、低速时盖太多，平均仍是 90%。按工况分开的覆盖在少量回合下没有找到可靠方法。放大样本带也假设带的形状本身是对的。
- LoFi 只在小网络上试过，没试过带缓存推演的 Transformer。SI-OCP 是仿真、最坏情况的有界扰动设定，结果偏保守（目标 90%、实际 98.8%）。
- 我们 M11 的"全量微调覆盖 30–70%、只调头约 0.8"在文献里没找到同类对比（动力学模型、按区间覆盖评判），可能本身就是新的证据。

---

## W4 浪的信息用不上

**我们的缺点**：M11 在目标数据上微调时，船身下 15 点的完美浪高让艏摇从约 0.3 降到约 0.1、纵摇从约 0.47 降到约 0.35，半步预见更好；但只在训练世界训练的加浪高模型学不会用（艏摇 0.94→0.90，纵摇无改善），因为训练世界的"浪→误差"随机关系不像真实浪力。另外，用户已定 D6：误差预测器不输入浪高，实船也没有 15 点浪高测量，所以这一节是"以后若要用浪"的准备。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Predicting ship responses in different seaways using a generalizable force correcting machine learning method（见 RELATED_WORK.md） | Marlantes, 2024 | Ocean Engineering | https://arxiv.org/abs/2405.08033 | 保留一个会对浪起反应的简单线性模型，小网络只学剩余的力；基础模型保留的物理越多，换海况泛化越好；有用的浪历史只需约 10–20% 主波周期 |
| Data-driven system identification of 6-DoF ship motion in waves with neural networks | Silva, 2022 | Applied Ocean Research | https://arxiv.org/abs/2111.01773 | 浪高时间历史到 6 自由度运动的网络；测点越多越好但 3 个以上收益小；测点跨约一个波长 |
| The influence of seaway parameters on the generalizability of a force-correcting machine learning method | Marlantes, 2026 | Journal of Ocean Engineering and Marine Energy | https://link.springer.com/article/10.1007/s40722-026-00541-x | 训练海况怎么选：中等周期、中到大浪高最好；陡浪里砰击主导、性质不同，在陡浪训练泛化差，砰击整个落进学习修正里 |
| Limits to the extent of the spatio-temporal domain for deterministic wave prediction | Naaijen, 2014 | International Shipbuilding Progress | https://journals.sagepub.com/doi/10.3233/ISP-140113 | 从一块观测海面能确定性预测哪片时空区域，由群速度决定；给"预见能有多远、多准"定规则 |
| Machine learning for phase-resolved reconstruction of nonlinear ocean wave surface elevations from sparse remote sensing data | Ehlers, 2023 | Ocean Engineering | https://arxiv.org/abs/2305.11913 | 同样是"合成数据训练再迁移"：用仿真海面加雷达成像模型训练网络，从稀疏雷达图重建海面，能泛化到别的海况 |
| Dynamic positioning using model predictive control with short-term wave prediction | Øveraas, 2023 | IEEE Journal of Oceanic Engineering | https://ieeexplore.ieee.org/document/10217185 | 小型无人艇 MPC 用短期浪预测（步长 0.25 s，和我们几乎一样）；预见只有在执行机构跟得上时才有用 |
| Data-driven uncertainty-aware seakeeping prediction of the Delft 372 catamaran using ensemble Hankel DMD | Palma, 2026 | Journal of Hydrodynamics | https://link.springer.com/article/10.1007/s42241-026-0015-z | 真实水池数据：高速双体船旁一个浪高仪就包含大部分升沉纵摇信息；按不同数据子集训练的集成校准得好，按超参数抽样的不好 |

**可以借鉴的（按优先级）**

1. **训练世界里"浪→误差"的随机关系要像真实浪力**（Marlantes 2024、2026）。在不违背"方法里不放有名字的物理项"的前提下，可以只在训练世界的随机族里加入这类结构：由船身各点的相对浸没得到升沉力和纵摇、艏摇力矩，配随机增益、随机滞后、随机饱和，再加一般的随机修正，而不是完全一般的随机滤波。和 W1 第 1 条"加关系种类"是同一件事。
2. **训练海况的分布**（Marlantes 2026）：偏重中等周期、中到大浪高，不要让陡浪、砰击主导的海况占主导。砰击需要单独的随机族（例如船身某点重新入水时的冲击，大小随相对垂向速度变化），因为在别的海况学到的修正覆盖不了它。
3. **浪的输入要"不完美"**（Naaijen 2014、Ehlers）：在低保真仿真里加一个合成的测量模型（遮挡、远处稀疏、噪声、随机标定误差），让模型学会该多信浪的估计。按我们自己的估算（不是论文的数）：25 m/s 时 5.76 s 视界约 150 m，深水群速度约 0.78×周期 m/s（6 s 浪约 5 m/s），远小于船速，所以预见主要是"测量前方约 150 m 的海面再往前推一点"；预见区外的浪高误差应当随距离变大。
4. **MPC 推演里的未来浪高依赖每条候选路径**（Silva）：候选开法不同，船到的位置不同，遇到的浪也不同。未来浪输入必须沿每条候选路径从浪场预报中取，不能重放记录路径上的浪。文献里没有人处理过这一点。
5. **少量回合的校准可以用"不同数据子集训练的几个头"**（Palma）：这是 W3 的一个便宜做法。

**不适用或要注意的**

- 所有用浪输入的模型都是按单条船、用该船的高保真或水池数据训练的，没有一篇是"合成先验训练、零样本换船"。
- Marlantes 的做法是在基础模型里加浪力项，这属于方法里的有名字物理项，和已定原则冲突；只能借鉴到训练世界的随机族里。
- 滑行艇只有 Marlantes 2021/2022 一条线（只有升沉纵摇、迎浪、仿真），而我们浪的最大收益在艏摇；40–50 节滑行艇用实测浪的预测或控制没有找到。
- 实船真正能拿到的是"雷达反演的浪高，带反演误差"，不是完美的 15 点浪高。5.8 m 小艇上装什么浪传感器（雷达量程、分辨率，或激光雷达）是未解决的问题。
- Øveraas 是低速定点保持、只有水平面，浪预测来自运动历史而不是浪传感器。

---

## W5 执行机构的动态

**我们的缺点**：目标上的执行机构（延迟、一阶平滑、速率限制、死区、泵启动）在训练族外；M8 测到真实喷口多了约 0.1 s 延迟和一点平滑，M9 显示这是推演卡住的主因。M10 已在低保真仿真里加了一个宽的随机执行机构族（延迟、滞后、速率、死区、间隙、增益、二阶），数据 meta3。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Learning agile and dynamic motor skills for legged robots | Hwangbo, 2019 | Science Robotics | https://arxiv.org/abs/1901.08652 | "执行器网络"：小网络从指令与实际之差的历史学实际输出；历史要长于延迟加响应时间；采数据时激励频段要宽，否则学出的模型自己会振荡 |
| A Multi-step Dynamics Modeling Framework For Autonomous Driving In Multiple Environments | Gibson, 2023 | ICRA | https://arxiv.org/abs/2305.02241 | 真车 + MPPI：转向、油门、刹车用"带速率限幅的一阶滞后 + 小网络修正"，整链按 5 秒轨迹误差多步训练 |
| ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills | He, 2025 | RSS | https://arxiv.org/abs/2502.01143 | 修正放在执行器端（改动作）比放在状态残差上长时推演更稳；状态残差模型过拟合、推演时误差放大 |
| Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks | Schneider, 2026 | arXiv 预印本 | https://arxiv.org/abs/2604.09487 | 没有力传感器时，把执行器网络放在可求导的物理一步前面，用下一步位置误差训练，不需要力的标签 |
| Data-Driven System Identification of Quadrotors Subject to Motor Delays | Eschmann, 2024 | IROS | https://arxiv.org/abs/2404.07837 | 只用指令和惯导：对候选时间常数把指令过滤后拟合，扫描找最好的；约一分钟数据，估得和厂商值一致 |
| RAPTOR: A Foundation Policy for Quadrotor Control | Eschmann, 2026 | Science Robotics | https://arxiv.org/abs/2509.11481 | "合成先验 + 历史推断 + 零样本"在执行器上成功（10 架真机）；参数按物理关系联合采样；失败处正是训练里没有的传感器通信延迟 |
| Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays | Sandha, 2021 | CoRL | https://proceedings.mlr.press/v155/sandha21a.html | 每步随机的采样间隔和执行延迟，并把本步实际时间关系作为输入交给网络 |
| Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion | Xie, 2021 | ICRA | https://arxiv.org/abs/2011.02404 | 盲目随机化让结果变保守；先找出真正敏感的因素，只对它们宽随机化 |
| Sim-to-Real Transfer and Robustness Evaluation of RL Control ... on an ASV for Floating Waste Capture | Batista, 2026 | IEEE Transactions on Field Robotics（已接收） | https://arxiv.org/abs/2605.02529 | 船上的直接证据：控制板固件里一个未公开的速率限幅是最大的迁移差距（不含它成功率 73%，含正确限幅时 28 种条件中 23 种 100%） |
| Modeling and Analysis of Actuators in Multi-Pump Waterjet Propulsion Systems | Jia, 2025 | Journal of Marine Science and Engineering | https://www.mdpi.com/2077-1312/13/1/154 | 喷水推进喷口和倒车斗液压伺服的结构：阀死区（磨损变大）、增益下降（漏油）、流量饱和（速率上限）、二阶滞后、左右不对称；倒车斗中位时推力为零 |

**可以借鉴的（按优先级）**

1. **执行机构参数按物理关系联合采样，而不是各自独立宽范围**（RAPTOR、Jia）。例如推力的升速和降速时间常数不同并和发动机、泵惯量挂钩；喷口速率上限和伺服大小挂钩；死区和增益同时随"磨损"变化；左右可以不对称。这仍是训练世界的随机族，不是方法里的组件，符合原则。
2. **观测通道也随机化延迟和时间抖动，而不只在执行机构上**（RAPTOR 的失败、Sandha）。GNSS/IMU 时间戳偏移、每步不同的计算延迟都应进先验；我们一步 MPC 计算很久，"指令基于旧状态算出、晚一点执行"本身就是延迟的一部分。可以把每个指令和测量的真实时间戳作为 Transformer 的输入，让模型不必从历史里猜时间关系。
3. **检查 M10 的族是不是太宽**（Xie、Chen 2022）：在低保真仿真里先看哪些执行机构参数真正改变误差模型的预测，只对这些宽随机化，其余收窄到物理可信范围；比较"宽族"与"窄族"模型在目标上的锐度和覆盖率。
4. **VM18 下水前，在岸上把整条指令链逐段测清楚**（Batista）：控制器输出→伺服或液压→喷口角；油门信号→发动机控制单元→转速→泵。找出固件级限速、死区、延迟、倒车斗切换时间。已打开的一篇博士论文（Sonnenburg 2012，弗吉尼亚理工，滑行充气艇、舷外机、液压转向）实测过：转向速率约 ±15°/s 饱和，油门到进气压力约 0.5 s 延迟，换挡再加约 1.5 s，油门死区约 −20% 到 35%，可作 VM18 执行机构族初始范围的参考（注意是舷外机，不是喷泵）。
5. **一个便宜的"是否在族内"检查**（Eschmann 2024）：只用指令和 IMU/GNSS，把指令过一个"延迟 + 一阶滞后 + 速率限幅"滤波，扫描这几个参数看简化模型的加速度拟合何时最好。这是诊断工具，用来判断目标执行机构是否落在合成族范围内、族应以哪附近为主，不作方法的组件。
6. **试航时做宽频段的喷口、油门激励**（Hwangbo），也可以作为 W8 探测策略的第一版目标。

**不适用或要注意的**

- Gibson 在简化模型里放显式的执行机构块、ASAP 只适配一个执行器修正模块——这属于针对具体机制的结构，和已定原则冲突。可借鉴的是证据本身：执行器端的偏差对长时推演影响最大，所以先验和在线学习应确保这两个通道被覆盖好。
- 这里的执行器模型几乎都是确定性的，没有预测分布；执行机构不确定时概率模型的推演表现没有找到直接研究。
- 泵启动、进水口吸气（通风）对推力的影响只找到 CFD 和水池研究，没有能放进控制模型的形式；用试航数据学喷水推进执行机构的机器学习论文一篇都没有。
- Jia 是大型多泵系统，没有定量误差数字可用于定范围；VM18 原车转向多是机械拉索，自主改装后的伺服参数要以改装件实测为准。

---

## W6 带随机学习模型的抽样 MPC 太慢

**我们的缺点**：MPPI（抽样式 MPC：随机生成许多候选开法，用模型推演打分，按分数加权平均）每条候选开法要推演好几条随机样本，例如 192 × 8 ≈ 1500 条序列，而每个控制步只有 0.24 s；带缓存的推演约 1.35 s 跑 1000 条。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| PEGASUS: A Policy Search Method for Large MDPs and POMDPs | Ng, 2000 | UAI | https://arxiv.org/abs/1301.3878 | 公共随机数的经典形式：事先抽好随机数，所有候选都在同一组随机数上评估，比较不再被独立噪声模糊 |
| Probabilistic Traversability Model for Risk-Aware Motion Planning in Off-Road Environments | Cai, 2023 | IROS | https://arxiv.org/abs/2210.00153 | MPPI 里把不确定样本在所有候选间共用是常规做法；8 个样本估尾部风险不可靠，用"均值 + c×标准差"更稳 |
| Risk-Aware Model Predictive Path Integral Control Using Conditional Value-at-Risk | Yin, 2023 | ICRA | https://arxiv.org/abs/2209.12842 | 性能代价用每条开法一次推演算，风险项才用多条随机推演，风险超阈值才惩罚 |
| Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation | Schramm, 2026 | RSS | https://arxiv.org/abs/2602.03639 | 用代价的二次近似（梯度和曲率）构造更好的抽样分布，样本只按剩余部分加权；倒立摆 2 个样本就接近最优 |
| Sample-efficient Cross-Entropy Method for Real-time Planning (iCEM) | Pinneri, 2020 | CoRL | https://arxiv.org/abs/2008.06389 | 保留上一步最好的几条开法（平移一步）作为候选；同样效果所需样本少 2.7–22 倍 |
| Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing (DIAL-MPC) | Xue, 2025 | ICRA | https://arxiv.org/abs/2409.15610 | 视界近处噪声小、远处大，因为近处已被前几步反复优化过 |
| TD-MPC2: Scalable, Robust World Models for Continuous Control | Hansen, 2024 | ICLR | https://arxiv.org/abs/2310.16828 | 一部分候选来自学到的策略；视界末端用学到的"之后还会花多少代价"，视界可大大缩短 |
| ProxPI: Proximal Prior Injection for Sampling-Based MPC under Learned-Prior Mismatch | Im, 2026 | arXiv 预印本 | https://arxiv.org/abs/2609.00941 | 学到的策略在分布外时，以它为抽样中心会毁掉规划（得分 0）；改成"加一项靠近它的代价 + 作为一条候选"就安全 |
| Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation | Belvedere, 2026 | IEEE RA-L | https://arxiv.org/abs/2506.14855 | 对推演求导得到线性反馈增益，两次重新规划之间用高频反馈修正，接受更慢的规划 |
| Deep RL in a Handful of Trials using Probabilistic Dynamics Models (PETS)（见 RELATED_WORK.md） | Chua, 2018 | NeurIPS | https://arxiv.org/abs/1805.12114 | 每条候选保留少量随机样本比只推均值好；加入多个模型时每个样本固定用一个，不增加序列数 |

**可以借鉴的（按优先级）**

1. **公共随机数**（PEGASUS、Cai）。flow 头是"随机输入 z → 误差"的确定映射，所以每个控制步抽一次 S 组 z 序列，192 条开法的第 s 个样本都用同一组 z。然后测开法排名的稳定性，看 S 能否从 8 降到 2–4。可选：下一步复用平移一步的随机数，让热启动前后的比较一致。副作用是固定 z 后推演对开法和起始状态可求导，是第 4、6 条的前提。
2. **候选数—代价曲线**（iCEM）：把上一步最好的几条开法平移 0.24 s 作为显式候选（不只是均值热启动），再加几条固定的安全开法（保持、收油门）；然后用真实的学习模型测闭环代价随候选数（32、64、96、192）的变化。论文显示有了记忆后这条曲线可能很平，这是对我们最关键的一个实验。
3. **两级评估**（Cai、Yin）：先对每条开法用一个固定的"代表性"随机输入（例如每步 z = 0）推演一次，算航迹、航向、速度损失、指令变化这些性能代价；再只对前 k 名做完整的 S 样本随机评估，得到砰击、峰值加速度这类风险项。序列数从约 192 × 8 降到约 192 + k × S。风险项用"均值 + c×标准差"或平滑的指数型风险，不用 8 个样本估硬尾部平均。
4. **视界上的噪声安排**（DIAL-MPC）：前几个样条节点给小噪声，后面给大噪声，不花一分钱。
5. **用简化模型的曲率引导抽样**（Schramm）：用简化物理模型（加固定的公共随机误差样本）算代价对 7 个样条节点的梯度和曲率，构造引导的高斯抽样分布，昂贵的随机学习模型只按剩余部分重新加权；可能让候选数降到几十。
6. **规划慢时用反馈补**（Feedback-MPPI）：如果完整随机 MPPI 只能每 2–4 个控制步跑一次，用求导得到的反馈增益在两次规划之间修正航速、艏摇、纵摇偏差。
7. **以后有了 RL 探测策略**（TD-MPC2、ProxPI）：让它提供 10–20 条候选，但只作为"靠近它的代价 + 一条候选"，不作抽样中心，因为它在低保真仿真里训练，在目标上会在分布外。学到的视界末端价值可让 24 步视界缩到 8–12 步。

**不适用或要注意的**

- 所有速度数字都来自便宜的解析模型或小网络；没有一篇在像我们这样重的模型（Transformer + flow）上展示过这些手段的组合，我们的加速要自己测。
- 公共随机数只有在候选相近、代价正相关时才有帮助（热启动的 MPPI 通常是这样）；固定的小随机数集可能偏向恰好利用这几组随机数的开法，所以每步都要重抽。
- 二次近似来自简化模型，在学习误差最要紧的地方（纵摇、砰击）恰恰有偏，收益取决于简化模型能抓住多少代价曲率。
- 砰击这类不光滑代价会让反馈增益变噪。
- "固定预算下候选数与每条样本数如何分配"没有找到针对 MPPI 的研究；一条开法只抽一个动力学样本时，指数权重会偏向"抽到好运"的开法，这个乐观偏差也没找到明确分析。

---

## W7 flow 采样步数太多

**我们的缺点**：flow 头每生成一个样本要走 24 个欧拉小步，每个控制步、每条序列都要做。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Distributional Diffusion Models with Scoring Rules | De Bortoli, 2025 | ICML | https://arxiv.org/abs/2502.02483 | 唯一直接针对"步数少时保持散布"的方法：网络多一个噪声输入、输出一个样本，用能量得分训练；2–4 步即可，散布和真后验一致 |
| Tyche: One Step Flow for Efficient Probabilistic Weather Forecasting | Xu, 2026 | arXiv 预印本 | https://arxiv.org/abs/2605.06916 | 两阶段：先训一步到位的平均速度 flow，再用短推演上的概率打分微调来恢复散布；散布与误差之比接近 1 |
| Swift: An Autoregressive Consistency Model for Efficient Weather Forecasting | Stock, 2025 | NeurIPS 2025 研讨会 | https://arxiv.org/abs/2509.25631 | 诚实的警告：一步模型即使经推演打分微调，散布仍偏窄 |
| Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) | Alonso, 2024 | NeurIPS | https://arxiv.org/abs/2405.12399 | 和我们最像（少步随机模型喂回自己输出）；1 步时随机转移被平均成模糊画面，所以用 3 步 |
| Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation | Prasad, 2024 | RSS | https://arxiv.org/abs/2405.07503 | 控制回路里 1–3 步、约快 10 倍；但沿确定性路径蒸馏（用慢模型的输出训练快模型）会丢掉多峰，缩小初始噪声会直接压窄区间 |
| How to build a consistency model: Learning flow maps via self-distillation | Boffi, 2025 | NeurIPS | https://arxiv.org/abs/2505.18825 | 学"从时刻 s 直接跳到 t"的映射；原来的 flow 损失保留为 s = t 的特例，另加自蒸馏损失，不需要老师；约 4 步后质量饱和 |
| One Step Diffusion via Shortcut Models | Frans, 2025 | ICLR | https://arxiv.org/abs/2410.12557 | 网络多一个"步长"输入，一个网络一次训练支持 1 到多步；有机器人控制测试 |
| Mean Flows for One-step Generative Modeling | Geng, 2025 | NeurIPS | https://arxiv.org/abs/2505.13447 | 学区间上的平均速度，一步生成接近多步质量，训练只慢约 16% |
| Bespoke Solvers for Generative Flow Models | Shaul, 2024 | ICLR | https://arxiv.org/abs/2310.19075 | 不重训模型，只学一个约 80 个参数的专用积分器，让少步结果贴近多步结果；成本约为原训练的 1% |
| Improving the Training of Rectified Flows | Lee, 2024 | NeurIPS | https://arxiv.org/abs/2405.20320 | 用现有模型生成（噪声，样本）配对，再按直线路径重训一次，一轮就够 |

**可以借鉴的（按优先级）**

1. **先不重训：给 10 维 flow 头学一个专用积分器**（Bespoke），在存下的 Transformer 特征上让 4–8 步贴近 24 步（或更细）对同一噪声的结果。因为复现的是同一个"噪声→样本"映射，校准基本继承原模型。同时顺手比较普通的中点法、Heun 法。
2. **再做"可选步数"的头**（Boffi 的 flow map，或更简单的 Shortcut）：给头加 (s, t) 输入和自蒸馏损失，原 flow 损失保留，24 步采样仍可作为后备和参照；运行时选 1、2 或 4 步，例如 MPPI 给 1500 条序列打分用 1–2 步，评估和最终执行的开法用更多步。Transformer 特征不依赖 s、t，所以求导只经过小的头，成本低。
3. **想要少步又保散布，用能量得分训练的头**（De Bortoli）：我们 W2 已有能量得分代码；头多一个噪声输入，直接输出误差样本，每个训练例抽 4–8 个样本，采样 2–4 步。
4. **如果走一步生成**（MeanFlow + Tyche 的第二阶段）：之后要用短推演（1–2 步）上的概率打分在合成先验上微调来恢复散布，并按每通道、每步长的散布误差比和覆盖率验收。
5. **验收标准统一**：步数按"整条推演的覆盖率和散布误差比"选，不按一步均方误差选（DIAMOND 的教训）；输出参数化保持在高噪声下表现好的形式（预测干净目标或速度，不预测噪声）；蒸馏时绝不缩小初始噪声（Consistency Policy 的陷阱）。

**不适用或要注意的**

- 除天气（Tyche、Swift）和二维玩具（De Bortoli、Boffi）外，没有论文报告低维条件分布的区间覆盖率随步数的变化；控制方面的论文全是策略，只按任务成功率评判。
- 一步模型容易向条件均值靠拢，散布偏窄（DIAMOND、Swift），这会加重 W2 的卡住和漂移；所以必须保留一个散布修正（W3 的在线放大）。
- 蒸馏出来的学生不会比老师好，还会复制老师的错；配对数据只覆盖合成先验的特征分布，不覆盖目标。
- 头在目标上微调（W3）后，flow map 或蒸馏头是否要重训、在先验外是否仍校准，都没有文献证据。

---

## W8 探测（以后做）

**我们的缺点**：还没开始。计划是 RL 策略学会主动激励船来辨识模型，奖励是预测器的信息增益；已知的难点是"先有鸡还是先有蛋"：探测只有在后续能用上信息时才有回报，而利用又要先有探测来的信息。

| 论文 | 第一作者，年 | 发表处 | 链接 | 对我们有什么用 |
|---|---|---|---|---|
| Optimizing Sequential Experimental Design with Deep Reinforcement Learning | Blau, 2022 | ICML | https://arxiv.org/abs/2202.00821 | 奖励拆成每一步的信息增益贡献（真参数对比从先验抽的若干参数），逐步相加正好等于整段信息增益；只给末尾奖励学得慢、结果差 |
| Can In-Context Learning Support Intrinsic Curiosity? | Elmoznino, 2026 | arXiv 预印本 | https://arxiv.org/abs/2606.19476 | 正是我们的设定（用上下文学习模型算奖励）：证明"意外程度"= 信息增益 + 不可减的噪声，会诱导去噪声大的地方；提出"有无这一步时对后续数据预测的差"作奖励 |
| Environment Probing Interaction Policies | Zhou, 2019 | ICLR | https://arxiv.org/abs/1907.11740 | 奖励 = 探测后预测模型在同一环境另一批数据上改进了多少，而不是任务回报；预测模型要定期重训，防止探测去钻它的空子 |
| Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices (DREAM) | Liu, 2021 | ICML | https://arxiv.org/abs/2008.02790 | 直接解决"先有鸡还是先有蛋"：利用一侧训练时知道真实任务，只保留对任务有用的那部分信息；探测一侧的奖励是"帮助还原这部分信息"的多少 |
| ASID: Active Exploration for System Identification in Robotic Manipulation | Memmel, 2024 | ICLR | https://arxiv.org/abs/2404.12308 | 仿真里训探测策略，真机只跑一次探测，据此调仿真再做任务；真机成功率明显高于域随机化 |
| Overcoming the Sim-to-Real Gap: Leveraging Simulation to Learn to Explore for Real-World RL | Wagenmaker, 2024 | NeurIPS | https://arxiv.org/abs/2410.20254 | 仿真不准时，学"怎么探索"比学任务更能迁移（真机 6/6 对 0/6）；训一组不同的探测策略，之后加随机激励 |
| Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design | Hedman, 2025 | ICML | https://arxiv.org/abs/2507.14057 | 先验不对时，实验进行到一半用已有数据修正、再短暂微调探测策略；先验偏移下原方法信息增益降到约 0，修正后仍为正 |
| Optimistic Active Exploration of Dynamical Systems (OPAX) | Sukhija, 2023 | NeurIPS | https://arxiv.org/abs/2306.12371 | 随机模型下信息奖励的正确形式：可减少的不确定与不可减的噪声之比，不是总散布 |
| Task-Oriented Active Learning of Residual Dynamics for MPPI | Aoki, 2026 | arXiv 预印本 | https://arxiv.org/abs/2609.19378 | 和我们的控制器最像（MPPI + 学习的误差模型）：在每条候选推演的代价里减去"早期观测能让后面预测好多少"，按任务相关性加权；可作 RL 探测要打败的基线 |
| Safe Active Dynamics Learning and Control | Lew, 2022 | IEEE Transactions on Robotics | https://arxiv.org/abs/2008.11700 | 安全探测：整条轨迹以高概率满足约束，并且每次探测都要回到安全状态 |

**可以借鉴的（按优先级）**

1. **奖励不用"意外程度"或 flow 样本的散布**（Elmoznino、OPAX），否则浪和砰击噪声都会被奖励。改用"有这段探测和没有这段探测，Transformer 对另一组后续数据的预测可能性差多少"。为避免状态前后耦合带来的偏差，后续数据要和探测"在给定船的规律下互不相关"：在低保真仿真里用同一个误差算子、执行机构和海况，从新的起点和噪声重新模拟一组后续机动来打分。去掉一段历史会破坏缓存，所以按 2–5 秒的探测段算奖励，不按 0.24 s 的每一步。
2. **奖励要逐段、稠密**（Blau）：只给末尾奖励学得慢、结果差。用带多个价值网络的离策略 actor-critic。
3. **用 DREAM 的分工破解"先有鸡还是先有蛋"**：仿真里知道真实的误差算子、执行机构和海况；利用一侧 = 带误差模型的 MPPI，只需要"会改变 MPPI 代价的那部分规律"；探测奖励 = 获得了多少这部分信息。这样探测不会把力气花在 MPC 不关心的通道上，也不需要先有一个好的端到端策略。这和已定的"MPC 负责响应、RL 因提供可用信息而得奖"一致。
4. **探测轨迹要回灌给预测器**（Zhou 2019）：探测会产生预测器没见过的指令分布，冻结的预测器可能被钻空子；交替训练，把探测历史混进 Transformer 的训练数据。
5. **训练多个不同的探测策略，之后加随机激励**（Wagenmaker），让数据还能覆盖先验没预料到的方向（例如 W5 的出族执行机构）。
6. **部署流程**（ASID、Step-DAD）：目标上先跑一段短探测，适应误差模型（W3），再交给 MPPI；探测进行到一半，用目标数据重新加权先验或用微调后的预测器作新仿真，短暂微调探测策略。
7. **基线和安全**（Aoki、Lew）：ToIA 式的"推演代价里减去信息项"是 RL 探测必须打败的基线；探测只在预测的纵摇、升沉加速度和砰击以高概率不超限时进行，并且每次探测要回到安全状态（直航、稳定滑行）。

**不适用或要注意的**

- Elmoznino 很新、未经同行评审，实验是小的非时间序列玩具；它的正面结论需要"互不相关"的条件，我们的船天然不满足，要靠另外生成的后续数据来构造。
- ASID、Blau 都假设真实系统在先验族内，而 W1 说明我们的目标不在族内，按合成参数算的信息可能指向错误的激励；只有 Step-DAD 和 Wagenmaker 部分涉及这一点。
- Lew、Aoki、OPAX 的闭式信息量都依赖高斯线性头或 GP；对我们的 flow 头要换成"有无探测的预测可能性差"这类只用模型本身算的量，不能引入线性回归组件（已定原则）。
- 没有找到滑行艇或喷水艇上的学习型探测；也没有找到砰击、海豚跳、舭部侧滑时的激励限度。船舶方向只有传统的试验设计（例如 Ljungberg 等从 Z 形、螺旋等标准操纵中选组合，而且在 30 m 船上随机设计有时更好，因为有风天的噪声）。

---

## 综合优先级：先做什么、为什么

按"便宜、直接针对已测到的问题、不违背已定原则"排序。每条都要在我们自己的数据上测，文献数字不能代替。

1. **MPPI 里用公共随机数，测候选数—代价曲线**（W6）。几乎不花训练成本，只改推演调用；决定实时化还差多少，也是后面求导类方法的前提。同时加上"保留上一步最好几条开法"和视界上的噪声安排。
2. **训练时给喂回去的历史加随机噪声（强度作输入）**（W2），在新 C 组推演上看喷口滞后是否按真值速度归零、航向是否不再漂；评估加"每步变化率"指标。顺带便宜地试"flow 从上一步误差出发"和"随机采样器代替确定性欧拉"。这是 M8、M9 之后最直接的修法，而且和 M10 的执行机构族互补（一个治推演，一个治先验）。
3. **在线学习用"只调头 + 在线覆盖率调节"**（W3），把固定 1.25 倍换成按通道、按步长、按延迟核对自动调的系数；之后再试"按层排序只调一两层的小矩阵 + 高斯分布"作为比只调头更准、又不崩校准的中间路线。M11 已证明全量微调崩校准，这一条让在线学习可以直接用。
4. **等 M10（meta3）结果出来后，按 W1 改先验**：先加"任意观测通道经随机滤波和非线性影响任意误差通道"的通用成分（补目标纵摇随状态变化这类关系），训练预算优先给"更多不同的系统"；再用 4–16 个目标回合、以仿真重置加似然的方式校准随机范围；并加"出了先验"监视器。执行机构族参数改为按物理关系联合采样，观测通道也加延迟和时间抖动（W5）。这一步要重生成数据，比前三条贵。
5. **减少 flow 步数**（W7）：先学专用积分器（不重训），再做可选步数的头；一律按整条推演覆盖率验收，并保留在线散布修正。和第 1 条一起决定能否在 0.24 s 内跑完。
6. **探测**（W8）：奖励用"有无探测段对另一组后续数据的预测改进"，逐段稠密；用 DREAM 式分工处理"先有鸡还是先有蛋"；ToIA 式信息项作基线；探测轨迹回灌预测器。前提是第 2、3 条让预测器的推演和不确定性可信，否则奖励本身不可靠。
7. **浪**（W4）：在 D6 下暂不做。若以后重新考虑，训练世界的浪族要像真实浪力并有砰击族，浪输入要带真实传感器式误差（约 150 m 预见区），推演里的未来浪要沿每条候选路径取。

**未核实、已从表中去掉的论文**：Marlantes & Maki 2022（Ocean Engineering 262，滑行艇升沉纵摇的神经修正）、Lee 等 2023（Physics of Fluids 35，时空浪场加运动历史预测船舶运动）、Naaijen 等 2018（OMAE2018-78037，导航雷达的确定性浪和运动预测实船验证）——出版社页面有人机验证，只核对了 DOI 元数据。
