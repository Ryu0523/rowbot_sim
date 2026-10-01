# 误差估计在控制里的其他用法：MPC 之外、更省的执行方案（文献检索，2026-09-30）

这份文档回答一个问题："接入 MPC 时是不是只查了 MPC 的论文？有没有更高效地利用误差估计的执行方案？"

回答是：之前的检索（RELATED_WORK_WEAKNESSES.md 的 W6）只查了"让抽样 MPC 本身更快"的办法，例如所有候选共用同一组随机数、保留上一步的好开法、先粗筛再细评、两次规划之间用反馈增益补。那里列过的论文这里不再重复。这次补查了五个方向：

- E1：学习误差只沿一条计划推一次，再用便宜的近似给所有候选；
- E2：扰动观测器和前馈，即在规划之下每步直接抵消预测的误差；
- E3：把学习模型"蒸馏"回控制器自己的简单模型，也就是在线调简单模型的参数；
- E4：只学代价需要的东西，用学到的"剩余代价"缩短视界；
- E5：把规划摊到训练里，训练一个策略网络，运行时不再推演。

表里只列搜索时打开过页面、核对过作者、年份、发表处和链接的论文；没核对上的放在文末一行。每条"证据"都是论文在它自己的平台上（无人机、小车、机械臂、慢速小船、模拟基准）测到的，没有一篇是在喷水滑行艇上做的。"对我们有什么用"是我们的推断，要靠自己的实验确认。凡标"算术"的数字是按序列数和步数算出来的，没有实测。

---

## 先说清：我们现在的做法慢在哪里

**设计中的做法**（DESIGN_MPC.md，`learn/meta/mpc_learned.py` 已写，闭环比较还没跑）：

- 每个控制步（0.24 s），MPPI（抽样式 MPC：随机生成许多候选开法，用模型往前推演、打分，按分数加权平均）生成 128 条油门开法。任务实际用 128 条，不是 192 条。
- 每条开法用学习模型抽 4 条随机样本，往前推 24 步（5.8 s），共 512 条序列。
- 推一步 = Transformer 读一个新位置 + flow 输出头做 24 步欧拉积分（把一个随机输入一步步变成一个误差样本）。
- 喷口不规划。推演里由自动舵的 torch 版本（`SteerT`）按推演出的状态每步设定。
- 代价只看四项：沿航迹速度、冲击（由相邻两步升沉速度之差算出）、航迹偏差、油门变化。冲击项只在 `horizon_check` 测出可信的前 `link.j_imp` 步计价。

**实测的时间**（设计 7 节。当时 GPU 被别的任务占满，数字偏大）：

- 一艘船一次调用约 2 s。从 128 条序列到约 1000 条序列，时间几乎不变。
- 这说明时间不是花在序列多上，而是花在"必须一步接一步做的小计算"上：24 个视界步，每步 6 层 Transformer 加 24 步欧拉积分。GPU 每次只干很少的活，大部分时间在等下一次启动。

**由此得到的一个关键区分**：

- 减少序列数（候选数或样本数），省的是算力和显存，可以让更多船同时算。但一艘船的等待时间不变。
- 要缩短一艘船的等待时间，必须减少串行步数：缩短视界、只算当前一步、换成策略网络。另一条路是把整段推演录成 CUDA 图（把几千次小的 GPU 启动合成一次重放），设计 7 节已列为选项。
- 真船只有一艘，所以对真船来说，串行步数比序列数要紧得多。

**模型在多远以内可信**（DEFECTS M7–M10，设计风险 R2、R3）：

- 在目标上，升沉、纵摇的推演 1 秒以后不比不修正好；航向、艏摇的推演 1 秒以后比不修正还差。
- 所以 24 步推演的后约 19 步，贡献的主要是随机散布，而不是信息。这一点和"省时间"指向同一个方向。

---

## 总结

1. **不只有 MPC 这一种用法。** 对我们最有希望的三种执行方式是：
   - (a) 视界缩到模型可信的约 1 秒，之后的部分交给一个学到的"剩余代价"网络（末端价值）；
   - (b) 学习模型只沿当前最好的那条开法（上一步的加权平均，平移一步）推一次，再用它"对开法有多敏感"给每条候选一个近似误差；
   - (c) 以后把慢的完整随机 MPPI 当老师，离线跑，蒸馏（让一个小网络模仿它的输出）成一个读历史的策略网络。
   
   (a) 同时解决"慢"和"1 秒后不可信"两件事。(b) 不需要任何新训练，是检验"每条候选是否真需要自己的随机样本"最干净的实验。(c) 和项目以后要做的 RL 探测策略是同一个网络。
2. **最便宜的对照是老办法：把当前一步的误差估计沿视界保持不变或逐步衰减**（无偏移 MPC）。三篇文献都建议把它当作必须打败的基线。在我们的代价上，它很可能和不修正的 m0 一样几乎看不见冲击，因为冲击来自浪驱动、来回变化的升沉误差。但只有跑了它，才能回答"逐候选推演值不值"。
3. **内环前馈**（规划之下每步抵消预测的误差）在无人机上效果很大，前提是执行机构能直接抵消误差。我们的冲击来自升沉、纵摇，没有执行机构能直接抵消；侧移也没有直接的执行机构。一篇海上试验显示：仿真里的收益到了海上缩水一半以上，能耗还增加了。所以排在最后。
4. **把学习模型蒸馏成简单模型的参数**（DuSt-MPC、APHYNITY 一类），和本项目"不能换成上限更低的模型"的原则冲突，所以不列入候选。只保留一种用法：把学习模型当作虚拟对象，离线调 MPPI 的权重。
5. **文献的空白：** 没有一篇论文把"读历史、随机抽样的序列模型"放进控制器，再做上面这些近似。它们用的都是没有记忆的残差（高斯过程、小网络、傅里叶特征），而且几乎都只用均值。我们的做法要自己验证，比较时看闭环得分，不看一步误差。

---

## E1 学习误差只沿一条计划推一次，再给所有候选一个便宜的近似

**思路**：昂贵的学习模型每个控制步只在"当前最可能执行的计划"上算一次，得到误差序列和它对状态、指令的敏感度（导数）；优化器内部只用这份近似，不再调用网络。

| 论文 | 第一作者，年 | 发表处 | 链接 | 运行时代价 | 对我们有什么用 |
|---|---|---|---|---|---|
| Real-Time Neural MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms | Salzmann, 2023 | IEEE RA-L 8(4) | https://arxiv.org/abs/2203.07747 | 每步沿上一步的解，在 10 个节点上批量算一次网络值和导数，再解一个与网络大小无关的二次规划；嵌入式板上 50 Hz | 核心模板：网络只沿热启动计划算一次，候选用"误差 + 导数 × 偏离量"；真实四旋翼上跟踪误差比无学习模型低 82% |
| Cautious Model Predictive Control Using Gaussian Process Regression | Hewing, 2020 | IEEE TCST 28(6) | https://arxiv.org/abs/1705.10702 | 均值和方差在上一步的解上预先算好，优化时固定；求解时间和不用学习模型时差不多 | 不抽样也能传递"均值 + 散布"；它丢掉不可靠的横向速度均值、只在前 20/30 步用散布收紧约束，是"按通道、按步数决定信多远"的先例 |
| Zero-Order Optimization for Gaussian Process-based Model Predictive Control | Lahr, 2023 | European Journal of Control | https://arxiv.org/abs/2211.15522 | 每轮迭代一次前向传递散布（不对散布求导），再解一个名义大小的二次规划；每轮复杂度从 nx^6 降到 nx^3 | 散布每步沿当前计划重算，只当固定惩罚，不参与求导；正好对应我们"热启动、每步一轮"的 MPPI |
| L4acados: Learning-based models for acados, applied to Gaussian process-based predictive control | Lahr, 2024 | arXiv 预印本 | https://arxiv.org/abs/2411.19258 | 残差值和导数在求解器外批量计算；全尺寸汽车 50 Hz，8.2 ms | 现成工具：PyTorch 模型可以直接包成残差模块；以后把代价弄光滑，可以和 MPPI 同代价对比 |
| Robust Trajectory Tracking of Autonomous Surface Vehicle via Lie Algebraic Online MPC | Dong, 2025 | arXiv 预印本 | https://arxiv.org/abs/2511.18683 | 30 个特征沿 30 步预测状态算一次，加一个小梯度步和一次二次规划；单核 CPU 50 Hz | 最接近的船上例子：残差沿预测状态算、在线更新线性系数；28 kg 小船河上试验 RMSE 比名义 MPC 低约 35%；无明显浪 |
| Learning disturbance models for offset-free reference tracking | Krupa, 2025 | IEEE TAC | https://arxiv.org/abs/2312.11409 | 每步一次扩展卡尔曼滤波更新参数，再沿视界算小网络 | 视界上保持的是"参数"而不是"误差"：把 Transformer 当固定特征，只在线更新最后一层，更新自带不确定度；仅有仿真 |
| Data-Driven Model Predictive Control for Trajectory Tracking With a Robotic Arm | Carron, 2019 | IEEE RA-L 4(4) | https://doi.org/10.1109/LRA.2019.2929987 | 沿视界算高斯过程均值，加一个小的在线偏差估计 | 两层用法：离线学到的模型给有结构的误差，在线估的慢变偏差补它在目标上的系统偏差；只核对了摘要，没读全文 |
| Disturbance models for offset-free model-predictive control | Pannocchia, 2003 | AIChE Journal 49(2) | https://doi.org/10.1002/aic.690490213 | 每步一次观测器更新，可忽略 | 无偏移 MPC 的奠基文献：当前误差估计在整个视界上保持不变；我们的基线（变体 C）；只核对了题录 |

**要注意的**：

- 这些论文的残差都没有记忆，只是当前状态和指令的函数，而且是确定的。我们的模型读历史、会抽样、有事件和继电器那样的跳变。对一条抽出的路径求导会很吵，所以更适合"固定随机输入，对开法做有限差分"：把某个样条节点加一点点再推一次，看误差变多少。
- 近似只在热启动计划附近准。我们的 MPPI 候选散得很开（样条节点噪声 0.25，油门可以偏 ±0.5）。
- Salzmann 的作者说明：用状态和指令历史的模型，在他们的设置里只能在仿真中跑。没有人验证过带历史的模型。
- 另外核对过、未进表的：Pan & Theodorou 的 Probabilistic Differential Dynamic Programming（NeurIPS 2014），给出重新规划之间可用的局部反馈增益；Bharadhwaj 等把梯度步和交叉熵法交替（L4DC 2020）。

---

## E2 扰动观测器和前馈：规划之下每步直接抵消预测的误差

**思路**：规划器用简单模型（或加学习误差均值）规划，下面一层每步把"模型预测的误差"或"测得的误差"直接从指令里扣掉，让真船表现得像规划器以为的那样。

| 论文 | 第一作者，年 | 发表处 | 链接 | 运行时代价 | 对我们有什么用 |
|---|---|---|---|---|---|
| Neural Lander: Stable Drone Landing Control Using Learned Dynamics | Shi, 2019 | ICRA | https://arxiv.org/abs/1811.08027 | 每步一次小网络前向加一次不动点迭代 | 指出前馈的关键难点：误差依赖本步指令，"指令 = 规划指令 − 预测误差"是个隐式方程；用上一步指令起算一次迭代，并限制网络对指令的敏感度，才能保证收敛 |
| Neural-ESO: A Dual-Pathway Architecture for Provably Robust Learning-Based Control | Zhang, 2026 | arXiv 预印本 | https://arxiv.org/abs/2607.06535 | 小网络加几个观测器状态，可忽略 | 学习模型给前馈，一个低带宽的扩展状态观测器（把未知合力当作额外状态来跟踪的小滤波器）只估剩下的部分；分布外时更稳 |
| L1-Adaptive MPPI Architecture for Robust and Agile Control of Multirotors | Pravitra, 2020 | IROS | https://arxiv.org/abs/2004.00152 | MPPI 不变；内环每 2.5 ms 几次小矩阵运算 | 分层：MPPI 按名义模型规划，快速内环估计失配并抵消；仅有仿真，要求执行机构能直接抵消误差 |
| DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control | Huang, 2023 | CoRL | https://arxiv.org/abs/2310.09053 | 策略推理 3.2 ms 以内，加一个轻量估计器 | 训练时给策略真实扰动，部署时换成估计值；对应我们"策略读误差模型的均值和历史特征"（见 E5） |
| DOB-Net: Actively Rejecting Unknown Excessive Time-Varying Disturbances | Wang, 2020 | ICRA | https://arxiv.org/abs/1907.04514 | 每步两个 GRU 步加小网络 | 水下机器人的仿真先例：浪力超过执行机构能力时，只能靠提前预判；仅有仿真，扰动只随时间变 |
| Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments | van der Saag, 2025 | IROS | https://arxiv.org/abs/2506.21265 | 每个自由度几个观测器状态 | 海上警告：仿真里横向误差降 30–40%，海上只降 10–20%，大扰动下能耗多约 50%；把浪频运动也抵消掉，费力而收益小 |

**已在 RELATED_WORK_WEAKNESSES.md 列过**：Neural-Fly（W3，用于"只调头"的原则）。这里补它在执行上的两点：一是它的在线系数更新本质上是对最后一层的卡尔曼滤波，每步只要几微秒，可以在一个回合之内更新，补上 `adapt_head` 只在回合之间更新的空档；二是预测误差只做一次网络前向、不推演，就直接进控制律。

**要注意的**：

- 这些方法都假设执行机构能直接抵消误差（四旋翼的推力向量、带艏侧推的船）。我们只有油门（纵向）和喷口（转艏）能直接作用；冲击来自升沉和纵摇，侧移也没有直接的执行机构。
- 喷水推进的延迟、速率限制和泵启动，会限制任何内环的带宽。
- 都只用均值，散布（砰击风险）用不上。
- 另外核对过、未进表的：Walker 等的浪中定点保持前馈（OCEANS 2023，arXiv 2304.05222），扰动估计准时改善约 48%，不准时只剩约 17%；Gu 等的船舶扰动观测器综述（Control Engineering Practice 2022）；Tal 与 Karaman 用测得加速度补未建模力（IEEE TCST，arXiv 1809.04048）。

---

## E3 把学习模型蒸馏回控制器的简单模型（在线调参数）

**思路**：不在规划里调用学习模型，而是用最近的数据或学习模型，随时更新简单模型的几个物理参数（阻尼、喷口效率、延迟），规划器照旧只推简单模型。

| 论文 | 第一作者，年 | 发表处 | 链接 | 运行时代价 | 对我们有什么用 |
|---|---|---|---|---|---|
| Dual Online Stein Variational Inference for Control and Dynamics | Barcelos, 2021 | RSS | https://arxiv.org/abs/2103.12890 | 每步对一组参数粒子做几步更新；规划推演仍是确定的简单模型 | 每条候选抽一个参数粒子推演；真实地面机器人中途加载 5.3 kg 后，固定参数版发散，它没有；它自己承认把真实差距当作不变的高斯噪声是个假设 |
| BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators | Ramos, 2019 | RSS | https://arxiv.org/abs/1906.01728 | 一个密度网络的前向；仿真全在离线 | 一次前向就给出参数的后验分布；先验不对时会给出"自信而错误"的参数 |
| Robust adaptive model predictive control: Performance and parameter estimation | Lu, 2021 | Int. J. Robust Nonlinear Control | https://arxiv.org/abs/1911.00865 | 每步一次参数集合更新（约占总计算 2%） | "名义参数追求性能 + 鲁棒余量保安全"的结构；指出跟踪和激励辨识之间的取舍，是探测（W8）的正式说法 |
| Performance-oriented model learning for data-driven MPC design | Piga, 2019 | IEEE L-CSS 3(3) | https://arxiv.org/abs/1904.10839 | 全部离线；运行时是固定的 MPC | 模型参数和 MPC 权重按闭环代价选，而不是按拟合误差；需要几百次闭环试验 |
| Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting (APHYNITY) | Yin, 2021 | ICLR | https://arxiv.org/abs/2010.04456 | 训练期方法 | 物理部分尽量多解释，网络只管剩下的；物理结构不对时，参数会被推到不合物理的值 |

**和本项目原则的冲突**：

- 运行时只推简单模型，等于换成一个上限更低的模型。项目规定：学习模型效果不好时要找原因、改进，不能退回弱模型。
- 选"调哪几个参数"也是按机理手工定的，这和"通用、不做针对具体机理的修补"冲突。
- 所以这一方向不列入候选。只保留 Piga 的一种用法：用学习模型加简单模型当虚拟对象，离线调 MPPI 的几个权重（温度、平滑项）。运行时模型不变，也不降级。
- 另外核对过、未进表的：Richards 等的 Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems（RSS 2021），按闭环跟踪误差而不是按拟合误差学特征。

---

## E4 只学代价需要的东西；用学到的剩余代价缩短视界

**思路**：规划只需要"这条开法总共花多少代价"，不需要把每个状态都推准。所以可以在视界末端接一个学到的"之后还会花多少代价"（末端价值），只推模型可信的前几步；也可以在训练模型时按代价加权。

| 论文 | 第一作者，年 | 发表处 | 链接 | 运行时代价 | 对我们有什么用 |
|---|---|---|---|---|---|
| Value-Aware Loss Function for Model-based Reinforcement Learning | Farahmand, 2017 | AISTATS | https://proceedings.mlr.press/v54/farahmand17a.html | 只是训练损失，运行时不变 | 奠基：模型只需在价值敏感的方向上准；可作辅助损失，不能替代似然损失（模型还要当 RL 的仿真器） |
| The Value Equivalence Principle for Model-Based Reinforcement Learning | Grimm, 2020 | NeurIPS | https://arxiv.org/abs/2011.03506 | 训练和检验准则 | 给出不同于逐通道均方误差的检验：模型能否复现各候选开法的总代价和排名；可以直接回答"1 秒后的艏摇误差到底改不改变选哪条开法" |
| Calibrated Value-Aware Model Learning with Probabilistic Environment Models | Voelcker, 2025 | ICML | https://proceedings.mlr.press/v267/voelcker25a.html | 训练时每次更新至少抽 2 个样本 | 按代价加权的训练会把随机模型推向过窄；它减去样本方差来修正，适合我们的 flow 头；要对样本求导，和 24 步欧拉冲突（W7） |
| Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control (POLO) | Lowrey, 2019 | ICLR | https://arxiv.org/abs/1811.01848 | 每条候选末端多算 6 个很小的网络，可忽略 | MPPI 加学到的末端价值：有限视界表现得像长视界；价值网络组的分歧可作探测信号 |
| Blending MPC & Value Function Approximation for Efficient Reinforcement Learning (MPQ(λ)) | Bhardwaj, 2021 | ICLR | https://arxiv.org/abs/2012.05909 | MPPI 推演不变，每步多算一个小网络；视界可缩短 | 最像我们：MPC 的模型故意有偏，用一个权重 λ 把"模型推演"和"学到的价值"混合；模型误差随视界变大，价值误差随视界变小 |
| Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion (STEVE) | Buckman, 2018 | NeurIPS | https://proceedings.neurips.cc/paper/2018/hash/f02208a057804ee16ac72ff4d3cec53b-Abstract.html | 训练期 | 训练价值时，按各推演长度的散布给权重：模型只在可信的长度内被使用；前提是散布可信（我们在目标上还不校准，W3） |
| Data-driven Economic NMPC using Reinforcement Learning | Gros, 2020 | IEEE TAC 65(2) | https://arxiv.org/abs/1904.04152 | 运行时不变，只改 MPC 参数 | 模型不对时，也可以用 RL 按闭环回报调 MPC 的代价、末端代价参数，让它给出好策略；为梯度型 MPC 写的 |
| Practical Reinforcement Learning For MPC: Learning from sparse objectives in under an hour on a real robot | Karnchanachari, 2020 | L4DC | https://proceedings.mlr.press/v120/karnchanachari20a.html | 价值网络及其导数进每次求解 | 真实地面车上 45 分钟内学出放进 MPC 的价值；价值要光滑，否则 MPPI 的权重会被噪声带偏 |
| Multistep Belief Space Dynamics Learning For Risk-Aware Control | Gibson, 2026 | arXiv 预印本 | https://arxiv.org/abs/2605.12628 | 每条候选推一次"均值 + 协方差"，代价在 2n+1 个代表点上算；全尺寸越野车上 1.8 万条候选实时 | 用一次"分布推演"替代多次抽样推演，风险用条件风险值算；高斯分布会抹掉砰击这类事件型误差 |

**已在 RELATED_WORK_WEAKNESSES.md 列过**：TD-MPC2（W6：一部分候选来自策略，末端用学到的价值）。本节补的是：末端价值可以用真实回合的结果来学，从而吸收模型 1 秒后的错误（MPQ(λ)、POLO），不只是为了省时间。

**要注意的**：

- 没有论文处理"只在合成先验上训练、再搬到另一个系统"的情形。在低保真仿真里学的价值带着先验不符（W1），要用目标数据重拟合。
- 按散布决定信多远（STEVE），前提是散布校准。我们在目标上 90% 区间只盖住 30–70%（全量微调时）。
- 按代价训练模型会让模型只适合这一种代价，和"通用、换船可用的仿真器"冲突，所以只能作辅助项。
- 另外核对过、未进表的：Lambert 等的 Objective Mismatch in MBRL（L4DC 2020），一步准确度和控制效果不一致；Sikchi 等的 LOOP（CoRL 2021）；Farshidian 等的 Deep Value MPC（CoRL 2019）。

---

## E5 把规划摊到训练里：训练一个策略网络，运行时不推演

**思路**：昂贵的随机推演只在训练时用，可以是在学习模型里训练策略，也可以是让慢的 MPC 当老师。运行时只跑一次网络。

| 论文 | 第一作者，年 | 发表处 | 链接 | 运行时代价 | 对我们有什么用 |
|---|---|---|---|---|---|
| When to Trust Your Model: Model-Based Policy Optimization (MBPO) | Janner, 2019 | NeurIPS | https://arxiv.org/abs/1906.08253 | 只跑策略网络 | 在学到的随机模型里只推 1–25 步短推演，从真实状态出发，长远部分交给价值网络；200、500 步长推演反而更差 |
| Mastering Diverse Domains through World Models (DreamerV3；Nature 版题为 Mastering diverse control tasks through world models) | Hafner, 2025 | Nature 640 | https://arxiv.org/abs/2301.04104 | 每步一次模型状态更新加一次策略前向，不做前瞻 | 最干净的摊销方式：策略读模型的历史状态，在想象的推演上训练；我们的想象推演要比 16 步短 |
| Learning Deep Control Policies for Autonomous Aerial Vehicles with MPC-Guided Policy Search | Zhang, 2016 | ICRA | https://arxiv.org/abs/1509.06791 | 一个 2×40 的小网络 | MPC 只在训练时用，而且加一项"别离策略太远"，让老师的数据策略学得会；仅有仿真 |
| Agile Autonomous Driving using End-to-End Deep Imitation Learning | Pan, 2018 | RSS | https://arxiv.org/abs/1709.07174 | 一次 CNN 前向，50 Hz | 真实越野小车：用学到的概率模型做 MPC 的老师，可以蒸馏成网络；标签必须在学生自己走到的状态上补打（DAgger：学生开，老师在学生到过的状态上给答案，再合并训练），在线补打完成率 100%，批量模仿只有 51–69% |
| Efficient Deep Learning of Robust Policies from MPC using Imitation and Tube-Guided Data Augmentation | Tagliabue, 2024 | IEEE T-RO | https://arxiv.org/abs/2306.00286 | 平均 15 微秒，最高 500 Hz | 在每个老师解周围按扰动范围抽状态，用局部反馈律便宜地打标签，少调老师；我们可以用误差模型自己的散布当范围 |
| Exploring Model-based Planning with Policy Networks (POPLIN) | Wang, 2020 | ICLR | https://openreview.net/forum?id=H1exf64KwH | 仍是完整规划，除非只用策略 | 策略给候选、加噪声，同样质量需要的候选更少；只用策略时在难任务上失败 |
| Temporal Difference Learning for Model Predictive Control (TD-MPC) | Hansen, 2022 | ICML | https://proceedings.mlr.press/v162/hansen22a.html | 视界 5 步加末端价值；规划约 20 ms，只用策略约 3–4 ms | 短视界加末端价值加少量策略候选的原始版本；只用策略通常比规划差 |
| Bootstrapped Model Predictive Control (BMPC) | Wang, 2025 | ICLR | https://arxiv.org/abs/2503.18871 | 策略几乎追上 MPC，部署可只用策略 | 策略模仿 MPC 的动作分布，MPC 用策略当先验；只对不到 1% 的存储状态重新规划来更新标签，训练只多花 10–20% |
| On the role of planning in model-based deep reinforcement learning | Hamrick, 2021 | ICLR | https://arxiv.org/abs/2011.04021 | 不是方法 | 受控实验：测试时搜索多数环境只多几个百分点，规划的价值主要在训练时；模型会累积误差时，搜得越深反而越差 |
| Deep Model Predictive Optimization (DMPO) | Sacks, 2024 | ICRA | https://arxiv.org/abs/2310.04590 | 仍要推演，但候选少 8–16 倍 | 学出 MPPI 的"一轮更新"：512 条候选达到普通 MPPI 4096 条的效果；真实四旋翼；保留显式代价项 |

**要注意的**：

- 策略会利用训练它的模型的错误。在合成先验上训练，先验不符（W1）会原样带进策略。
- 没有论文证明蒸馏出的策略保住了 MPPI 的风险处理（砰击、冲击）。
- 训练期的想象推演对我们仍然贵（每步 24 步欧拉），所以少步采样（W7）即使在策略方案里也还重要。
- 另外核对过、未进表的：Li 等的 Accelerating and Scaling MPC-Guided RL for Humanoid Locomotion and Manipulation（arXiv 2606.05687）；Hwangbo 等 2019 年 Science Robotics 的腿足机器人工作，是"仿真里加学到的部件、训练策略、单独部署"的经典例子。

---

## 候选控制器变体（按对我们的比较价值排序）

**记号**：完整随机推演 F = 128 条候选 × 4 条样本 × 24 步，即 512 条序列。每一步要做一次 Transformer 新位置加 24 步欧拉积分，一次调用要依次做 24 个这样的步。每次调用都有一次读历史（最多 232 个位置、一次并行完成），各变体都要做，下面不计。

**排序依据**：在我们的代价（速度、冲击、航迹）上预计能保住多少质量，能省多少串行步数（一艘船的等待时间），以及要写多少新东西。

| 变体 | 每步序列数 | 依次要算的步数 | 相对 F（算术，未实测） |
|---|---|---|---|
| F 完整随机推演 | 512 | 24 | 1 |
| B 短视界 + 末端价值 | 512 | 5 | 算量约 1/5，等待约 1/5 |
| A 沿一条计划推一次 + 敏感度 | 32 | 24 | 算量 1/16；一艘船等待时间基本不变（小调用约 2 s 的下限，实测） |
| A + B 组合 | 32 | 5 | 算量约 1/80，等待约 1/5 |
| C 当前误差沿视界保持 | 约 16 | 1 | 算量不到 1/500，等待约 1/24 |
| D 蒸馏出的策略 | 0（只读历史） | 0 | 只剩读历史和一个小网络 |
| E 内环前馈 | 在规划之外再加约 16 条 × 1 步 | 1 | 在所用规划器之上略增 |

### 1. B：短视界完整推演 + 学到的末端价值

依据：POLO、MPQ(λ)、TD-MPC、STEVE、Karnchanachari。

- **怎样用学习模型**：完整随机推演只推前 5 步（1.2 s），也就是目标上升沉、纵摇、航向推演还比不修正好的范围。第 5 步之后的代价由一个小网络 V 给出。V 的输入是第 5 步的推演状态（在航迹坐标里）、参考航速，以及 Transformer 对历史的编码。历史编码描述"这是什么船、什么海"，读历史时本来就算出来了，不多花钱。
- **V 怎么来**：用真实闭环回合的结果拟合，不用模型推演的结果。目标值是"之后 19 步（到原视界末）真实发生的任务代价"，这样可以和 F 直接比；以后再试更长的剩余时间。先用已有的 hand、m0、prior 目标回合拟合，再用新控制器自己的回合重拟合一两轮，因为 V 取决于产生数据的控制器。用真实结果拟合的好处是：V 会吸收模型 1 秒以后的错误，这是 MPQ(λ) 的思路。
- **运行代价**：算量和等待时间都约为 F 的 1/5（算术）。
- **能抓住的**：前 1.2 s 里随开法变化的误差和完整的随机散布，冲击的尾部也按样本计价；1.2 s 以后的平均代价，包括"这个航速在这片海里之后会有多少冲击"。航向、艏摇的不可靠推演基本被截掉。
- **抓不住的**：1.2 s 以后随开法变化的细节，V 只知道到达的状态；V 带着拟合数据的偏向，包括产生数据的控制器和世界；多了一个要训练、要检验的部件。
- **在代码上怎么做**：
  - `LearnedMPPI` 加参数 `h_roll=5`：`__call__` 把 `pl[:, :, :5]` 和 `base[..., :5, :]` 交给 `M.rollout_core`，它本来就接受短于 24 步的计划（`assert Hh <= HB`）；
  - `task_cost` 算前 5 步，再加 `V(states[..., -1, :], ...)`；
  - 样条节点和热启动平移仍用 24 步的 `knot_matrices()`，所以 F 和 B 的开法完全一样；
  - 历史编码：`M.kv_history` 的第二个返回值（最后一层输出）现在在 `_rollout_core` 里被丢掉，需要让 `rollout_core` 把最后一个历史位置的输出一起返回；
  - V 的训练放在一个新文件里，数据来自 `run_group` 记下的 `StepLog`。

### 2. A：学习模型只沿当前最好的开法推一次，再按敏感度给所有候选近似误差

依据：Salzmann、Hewing、Lahr、Dong；和 W6 里的"两级评估"不同，这里每条候选都得到一个随开法变化的学习误差，而不只是 m0。

- **怎样用学习模型**：
  - 每步只调一次 `rollout_core`，计划共 8 条：热启动的平均开法，加上"7 个样条节点各加一点点 δ（例如 0.05）"的 7 条；每条 4 个样本，8 条共用同一组随机输入（公共随机数）。
  - 得到 8 × 4 条误差序列。对每个样本路径，算误差对每个节点的敏感度：(第 i 条的误差 − 平均开法的误差) / δ。
  - 每条候选的误差 = 平均开法的误差 + Σ 敏感度 × (候选节点 − 平均节点)，逐样本算，并把偏离量限制在一个范围内。
  - 然后在简单模型上推 128 条候选（每条 4 个样本），代价函数 `task_cost` 不变。
  - 因为 4 条样本路径各自保留，冲击这种"超过门槛才计价"的尾部项仍按样本算，而不是在均值上算。
- **运行代价**：序列数 32 条，约为 F 的 1/16。但仍要依次推 24 步，一艘船的等待时间停在约 2 s 的下限（实测：小调用都在这个下限上）。它省下的是显存和算力：可以让更多船同时算（设计里 16 船一组时显存不够），或者免费增加候选数，因为候选只推简单模型（128 条 × 24 步单样本在 CPU 上实测 0.09 s，4 样本未测）。要省等待时间，需要和 B 组合，或者用 CUDA 图。
- **能抓住的**：开法变化对误差的一阶影响，既包括"指令不同 → 误差不同"，也包括"状态偏了 → 误差不同"。因为网络在每条扰动开法上都读了自己推出的历史，这一点正是现有论文做不到、要我们自己验证的；随机散布由共用的样本路径带着。
- **抓不住的**：
  - 离平均开法很远的候选。节点噪声 0.25，油门可以偏 ±0.5，线性近似可能失效。
  - 只在某条候选上触发的事件型误差。
  - 散布对所有候选相同，不随候选变。
  - 有限差分碰上跳变会很吵。
  - 艏摇、航向：喷口在两种推演里都由 `SteerT` 设，影响只通过沿航迹速度和航迹项进来。冲击项已经只在 `horizon_check` 测出的可信步数（`link.j_imp`）内计价；对艏摇、航向误差也可以用同样的测量决定截断步数，不手定。
- **在代码上怎么做**：
  - 在 `learn/meta/mpc_learned.py` 新建 `SharedErrorMPPI(LearnedMPPI)`，重写 `__call__`：用 `self.nominal` 和 `self.E` 生成 8 条计划；`base_tensor` 加一个候选数参数（8 而不是 128）；取 `M.rollout_core` 的第一个返回值（归一化误差样本）。
  - 新函数 `rollout_with_e(plans, xs_k, en, e_add, e_sd, steer)`：在 `rollout_zero` 上加进 `_rollout_core` 里的两行状态更新（`sr = m0(sr, Uj) + (e_raw[..., :n_st] * dtc) @ sel`，以及执行机构位置的截断）。
  - 检验：把 `rollout_core` 自己抽出的误差喂给 `rollout_with_e`，状态应一致到 1e-9；所有候选等于平均开法时，代价应等于平均开法的代价。

### 3. C：当前一步的误差估计沿视界保持或衰减（无偏移 MPC 基线）

依据：Pannocchia、Carron、Krupa。

- **怎样用学习模型**：
  - 每步只让学习模型预测当前这一步：计划为平均开法的第一个指令，推 1 步，16 个样本，取均值 ē。
  - 所有 128 条候选在简单模型上推 24 步，每步都加上 ē × 衰减系数。
  - 衰减系数按通道由数据决定：用本回合 `LiveData.E` 里各通道误差"隔 j 步的自相关"，不手定。
- **运行代价**：依次只推 1 步，算量不到 F 的 1/500（算术）；等待时间主要是读历史，未实测。
- **能抓住的**：慢变的偏差，例如纵荡的系统偏差、稳定的艏摇偏差。
- **抓不住的**：
  - 随开法变化的误差：所有候选得到同样的修正，猛加油门和巡航得到同一个修正。
  - 来回变化的浪致升沉、纵摇误差：设计第 0 节第 12 条已测出，误差为 0 时推演出的升沉加速度约为 0，冲击几乎看不见。所以在冲击项上，它预计和 m0 差不多。
  - 散布：全部丢掉。
- **为什么还要跑它**：它是回答"逐候选随机推演值不值"的对照行。三篇文献都把它当作必须打败的基线。如果 F 在闭环得分上打不过 C，说明我们花的算力没有换来决策质量。有了 A 的 `rollout_with_e` 以后，它只要几十行代码。

### 4. D：把完整随机 MPPI 蒸馏成读历史的策略网络

依据：MPC-GPS、Pan、BMPC、Tagliabue、MBPO、DreamerV3、Hamrick。

- **怎样用学习模型**：
  - 学习模型只在训练时用：完整随机 MPPI（F，可以更多样本、更多轮，不必实时）当老师。
  - 学生策略输入 Transformer 的历史编码、当前状态和参考，输出油门样条节点。
  - 按 DAgger 训练：学生在仿真里开船，老师在学生到过的状态上给答案，合并数据再训练。
  - 像 BMPC 那样，每轮只对一小部分存储状态重新请老师，控制训练成本。
- **运行代价**：读一次历史加一个小网络，不需要 flow 采样。
- **能抓住的**：老师做到的一切，都摊销进一个网络；历史编码让它能按"这是什么船、什么海"调整行为。
- **抓不住的**：
  - 老师自己的偏差，包括模型的错误，加上模仿误差；
  - 训练里没到过的状态；
  - 冲击风险没有保证；
  - 老师每步约 2 s（实测、有争用），训练数据量受限。
- **与项目路线的关系**：项目以后要一个会做探测的 RL 策略，D 训练出的网络可以作为它的起点。部署时保留一个小的短视界 MPPI 作检查，按 W6 里 ProxPI 的方式："加一项靠近策略的代价 + 作为一条候选"，而不是以策略为抽样中心。
- **在代码上怎么做**：
  - 老师：`LearnedMPPI` 设 `capture=True`，它会记下 `cand`、`cost`、`cmds`；
  - 学生开船：`run_group` 里把控制器调用换成策略；
  - 历史编码：同 B，从 `rollout_core` 返回。

### 5. E：规划之下的内环前馈（最后）

依据：Neural Lander、Neural-ESO、L1-MPPI、Neural-Fly；海上警告来自 van der Saag。

- **怎样用学习模型**：
  - 规划器给出油门之后、`m.advance` 之前，用学习模型预测"这个指令下一步的纵荡误差均值"（推 1 步，约 16 个样本）；
  - 按简单模型的油门→纵向加速度关系换算成油门修正，扣掉；
  - 因为误差依赖指令本身，用 Neural Lander 的做法：从上一步指令起算一次迭代，并打折（例如乘 0.5）以保证收敛；
  - 喷口仍由自动舵设。
- **运行代价**：在所用规划器之上加一次 1 步调用。
- **能抓住的**：纵向速度的偏差，这对应任务得分里的速度项。
- **抓不住的**：
  - 冲击：升沉、纵摇没有执行机构能直接抵消，而冲击恰恰是决定油门的关键项；
  - 侧移；
  - 浪频的来回误差：去追它会费油门，海上试验里能耗多约 50%。
- **结论**：对我们的得分，预计收益只在速度项，而且规划器本来就在调油门。所以放最后；更适合以后真船上作为慢规划之下的补充。

### 不列入的

- **E3 的参数蒸馏**：原因见 E3 节，会降级到上限更低的模型。
- **E4 的按代价训练模型**：它改的是训练，不是执行，只能作辅助损失。
- **Gibson 的"均值 + 协方差"推演**：用高斯分布替代 flow 样本，会抹掉事件型误差，而那正是选 flow 头的理由；而且需要另训一个传播网络。
- **值得做的一个检验**（来自 Grimm 的价值等价原则）：在同一状态出发的多条开法上，比较"模型推演出的代价排名"和"真实代价排名"。这需要在目标仿真里从同一状态复制出多个分支。它能直接回答"1 秒后的艏摇误差到底改不改变选哪条开法"，比逐通道误差更贴近控制。

### 比较怎么做

- 同一套 16 个评估任务（`task.EVAL_SEEDS` × 两条航段），同样的海况、节点噪声和样本随机数。
- 行：m0、F（设计里的 prior）、C、A、B、A + B；D 和 E 以后再做。
- 看闭环任务得分及其三项分量，不看一步误差。
- 同时报每个控制步的等待时间，分两种情况："一艘船单独算"（真船的情形）和"8 船一组同时算"。
- 这是文献里明确的空白：没有一篇在同一任务上测过"完整随机模型 → 只用均值 → 线性近似 → 短视界加价值"各损失多少闭环质量。

---

**未核对（页面打不开或只见到题录，未进表）**：Kjerstad 2016 Disturbance Rejection by Acceleration Feedforward for Marine Surface Vessels（IEEE Access）；Fossen 1999 Passive nonlinear observer design for ships using Lyapunov methods（Automatica）；Ocean Engineering 2026 Efficient Data-Driven Model Predictive Control for Unmanned Surface Vehicles Under Model Uncertainties via Recursive Updates of a Sparse Variational Gaussian Process；Ocean Engineering 的 Data-driven model predictive control for ships with Gaussian process（pii S0029801822027032）；Ocean Engineering 2022 Online adaptive parameter identification of an unmanned surface vehicle without persistency of excitation；Control Engineering Practice 2025 的一篇 USV 参数辨识与实时运动预测（pii S0967066125002709）；Zhang 等 2025 Ocean Engineering 的 USV 艏向数据驱动扩展状态观测器（有实船试验）；Kumar 等 RMA（RSS 2021）；Possas 等 Online BayesSim（IROS 2020）；两篇 Ocean Engineering 的 USV 深度网络实时最优控制与学习型路径跟踪（2025、2026）。
