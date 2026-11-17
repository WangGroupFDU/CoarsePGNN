# PU 全数据重训练

本流程将“保留全部可追溯实验观察”和“评价未参与当前模型训练的样本”同时实现为数据与训练约束。新代码位于 `src/coarsepgnn_retrain`，入口为 `python -m coarsepgnn_retrain`。历史 `pu-inverse-20261111` 运行、成功合同、模型和负结果保持原字节。

## 1. 数据与评价对象

当前完整公开 PU 入口有 3,657 条任务记录。恢复父均值中明确列出的 561 条原子 Tg 测量后，共保留 4,218 条记录；246 条父均值保留来源谱系，不与其成员重复监督。其余 **3,972 条观察全部进入正向和逆向的同一监督母表**，没有要求一条记录同时具备四个性质，也不因多组分、缺测试条件或缺 Mn 而删除整行。

这覆盖完整公开 PU 入口；原始 SQLite 尚未取得。原库若包含更多可追溯记录，应通过本流程的只读 SQLite 入口重新导入并生成新的数据审计和分组清单。不能把公开归档的完整性表述为原始数据库已穷尽。

详细来源、任务计数、字段级处理和端点修复见 [数据合同](pu_retrain_data_contract.md)。

## 2. 两条原则如何同时满足

每个外层模型只使用约 80% 的来源闭包训练，在其余约 20% 的闭包上评价。五个外折合起来，每条观察恰好得到一次来自“没有训练过该观察及其关联闭包”的预测。各折训练出的不同模型也让全部可用记录参与了训练。

最后按内层验证结果确定模型配置和训练 epoch，在 **全部 3,972 条观察**上另行拟合部署模型。该部署模型没有在这同一母表上的独立测试成绩；泛化证据来自前述折外预测，不能用部署模型对训练数据的拟合替代。

来源、规范化出版物、已核实专利家族、材料、配方、原图以及派生父子关系的传递闭包必须整体进入同一集合。父均值即使不监督，仍参与闭包连接。新版还加入源无关的可观测原料结构、规格与配比身份，保守隔离相同可观测配方；性质结果不参与分组。

旧 200 闭包方案遗漏三组已确认专利家族，现已退役为诊断。修复后真实母表形成 173 个闭包，最大闭包 603 行；5 个外层折的监督数为 775/795/774/772/856，每外折再分 3 个内层折。全部 56 个专利已查询，其中 54 个家族字段已核读；`US3746665A` 和 `CN118165222A` 仍明确标为未解析，不能称全部潜在复录已排除。原 `tensor_hash` 有值条数为 0，不宣称实际完成了全部模型输入张量去重。完整证据见 [来源家族审计](pu_retrain_source_family_audit.md)。

历史 R10 测试已经查看过。本次属于 **回顾性分组嵌套交叉验证**，不声称取得了一份从未被研究者接触的新盲测集。后续新增来源或预先封存的新实验才可形成新的外部验证。

## 3. 实际模型与合理改进

正向候选包括 Ridge、ExtraTrees 和可训练的多组分集合网络。集合网络使用分子 Morgan 指纹、原料角色、Mn/分子量/官能度、连续配比、测试条件和缺失掩码；配方及条件分支均参与训练。现阶段实际运行的是集合网络，不能将预留的 `graph_encoder` 接口表述为已经运行的 CoarsePGNN 图主干。

逆向候选使用性质值与任务掩码、测试条件，输出当前训练折原料目录的组分存在分数，以及连续组成的 logistic-normal 混合分布。变长组分集合可以表示多 D/P/E；缺失字段只屏蔽相应损失。已知零投料代表未加入，不转成“存在的 E”。目录外的留出原料保留在评价分母并明确标记，不能当作完整恢复成功。

训练使用以下有界改进：

- 主任务等权，任务内材料等权，避免重复协议和多条 Tg 观察使一个材料支配训练。
- 辅助任务仍参加每个 epoch；其总训练权重相对于四主任务设为 0.25，避免只有两条记录的辅助任务获得过高权重。
- 预先列出线性目标变换、力学性质 `log1p` 变换、较强正则化与条件 dropout 三套神经配置。
- AdamW、梯度裁剪和内层早停；每个活跃 epoch 无放回遍历所有分配给该模型的观察。
- 每折的尺度、条件变换和逆向目录只在该折训练记录上拟合。
- 神经模型选出的 checkpoint 写入 SHA 清单，重建新对象后验证预测和损失与保存前一致。

监督损失只使用实测标签；没有把前向模型的伪标签计作实验数据。辅助“未明确端点强度”与 UTS、辅助热事件与 Tm 分开建头和报告。

## 4. 配置和选择规则

实际候选定义在 `configs/retrain/pu_all_data_v2.json`：三套神经配置、两套经典配置，神经上限 120 epoch、patience 20、batch 128，最终种子为 20261117、20261118、20261119。GPU 请求失败时停止，不静默转成 CPU。

每个内折先选择 checkpoint。各候选再汇合三个内折的留出观察：误差按各内折训练集的性质 IQR 归一化，先在每个主任务内平均，再对四个任务等权。正向配置和家族都按这一内层指标选择；完全平局按既定候选顺序，家族平局优先经典基线。外层 MAE、图形和逐样本残差不进入选择。

逆向按内层完整目录可评分记录的组合监督损失选择，并同时报告全部记录的目录覆盖率。该损失是组分存在 BCE 与比例 ALR 密度损失的组合，不是实验性质成功率，也不是任意配方空间下已经校准的成功概率。

外折训练使用内层已选 epoch 的中位数，固定 epoch 后不再传入外层标签进行早停。三种子分别报告，不能选择测试成绩最好的种子。若一个组合管线复用了同一个经典模型的预测，必须标明该预测被复用，不能称为三个独立经典模型。

## 5. 环境

在 Python 3.11 或以上环境安装匹配设备的 PyTorch，再安装新流程依赖。当前 CPU 实测环境记录在运行回执中，包括 PyTorch 2.5.1+cpu、RDKit 2025.3.2 和 scikit-learn 1.8.0。服务器上的 CUDA 版本应与其驱动相容；本代码通过实际 `torch.cuda.is_available()` 检查设备。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-retrain.txt
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_retrain --help
```

## 6. 固定公开入口与母表构建

```bash
python -m coarsepgnn_retrain fetch --output downloads/pu-all-data

python -m coarsepgnn_retrain prepare \
  --source downloads/pu-all-data/assets/pu/dataset_manifest.csv \
  --provenance downloads/pu-all-data/assets/pu/provenance \
  --membership downloads/pu-all-data/tg_mean_membership_r4.csv \
  --evidence downloads/pu-all-data/training_evidence_r4.json \
  --graph-root downloads/pu-all-data/assets/pu \
  --source-family-map configs/retrain/pu_source_family_keys.json \
  --output outputs/pu-all-data/dataset
```

`fetch` 固定 GitHub 版本并核对下载及解包原字节。`prepare` 输出完整 `dataset.jsonl`、`audit.json` 和 `source_inventory.json`。来源家族映射须与公开 `sources.csv` 的 SHA-256 及完整来源集合一致；源表改变时拒绝套用。当前增强母表 SHA-256 为 `40a2f8aa8f05a104c05ece27d56066fc10ff2c8259032149ab854168a4c1942b`。文件清单、父子关系及正反向 ID 清单都可审查。任何既有非空输出目录都不会被覆盖。

若服务器上可以读取完整原库，改用实际 SQLite 路径，避免把旧 `accepted` 子集当成入口：

```bash
python -m coarsepgnn_retrain prepare \
  --sqlite /absolute/path/to/database_v24.sqlite \
  --output outputs/pu-database-all/dataset
```

这会读取全部 `measurements` 及关联表并保留原文。SQLite 内嵌 `sources` 的 DOI、专利公开身份和明确给出的家族键会保留用于分组；未经另行审查的家族状态是 `not_reviewed`，不会猜测归属。当前 202 来源映射不能强行套用到不同的新库。无法转换单位、无有效标签或缺少可确认配方身份的记录保留在母表并给出逐行原因；不编造标签或配比。新原库会产生新的母表及划分，必须重新做全部数据审计，不能套用当前 3,972 条记录的结果。

## 7. 一键训练

```bash
python scripts/train_pu_all_data.py \
  --dataset outputs/pu-all-data/dataset/dataset.jsonl \
  --config configs/retrain/pu_all_data_v2.json \
  --output outputs/pu-all-data-training \
  --device cuda --threads 4 --max-seconds 86400
```

脚本顺序执行分组、经典基线、神经嵌套评价、仅基于内层的正向家族选择，以及全量部署拟合。所有阶段共用一个截止时间；失败保留已经生成的原始产物，不自动换一个更有利的方案重试。`--evaluation-only` 可只执行评价与家族选择。

当前只有 CPU 时显式使用 `--device cpu`。实际耗时取决于设备、原库规模和早停 epoch；预算不足会留下失败/超时回执，不把未完成的折或种子标成通过。

## 8. 分阶段执行与产物

```bash
python -m coarsepgnn_retrain split \
  --dataset outputs/pu-all-data/dataset/dataset.jsonl \
  --output outputs/pu-all-data/splits.json

python -m coarsepgnn_retrain smoke \
  --dataset outputs/pu-all-data/dataset/dataset.jsonl \
  --splits outputs/pu-all-data/splits.json \
  --output outputs/pu-all-data/smoke --device cpu --max-seconds 300

python -m coarsepgnn_retrain train \
  --dataset outputs/pu-all-data/dataset/dataset.jsonl \
  --splits outputs/pu-all-data/splits.json \
  --config configs/retrain/pu_all_data_v2.json \
  --output outputs/pu-all-data/neural_nested --device cuda --max-seconds 86400

python -m coarsepgnn_retrain refit \
  --dataset outputs/pu-all-data/dataset/dataset.jsonl \
  --splits outputs/pu-all-data/splits.json \
  --config configs/retrain/pu_all_data_v2.json \
  --nested-run outputs/pu-all-data/neural_nested \
  --output outputs/pu-all-data/deployment_neural --device cuda --max-seconds 14400
```

| 产物 | 用途 |
|---|---|
| `dataset/audit.json` | 每条记录保留、任务路由、缺失字段和正反向共享 ID |
| `splits.json` | 最终已核实来源/家族/可观测配方闭包、全部外/内折 ID 和实际零重合检查 |
| `run_lock.json` | 开始拟合前固定的配置、母表、划分和源码散列 |
| 每次拟合的 `history.jsonl` | 每个 epoch 的正反向真实覆盖、损失与更新数 |
| `selection_lock.json` | 在外层预测或全量部署拟合之前记录的选择结果 |
| `checkpoint_manifest.json` / `checkpoint_reload.json` | 权重/特征原字节及重载一致性 |
| `oof_predictions.csv` / `selected_oof.csv` | 每条留出观察的原单位预测与完整评价分母 |
| `metrics.json` | 主/辅助任务分开、来源闭包区间、三种子与目录支持率 |
| 部署模型目录 | 用全部共同记录拟合的可交付权重与变换 |
| `failure.json` / 执行回执 | 真实失败边界、部分产物和实际耗时 |

误差区间通过来源闭包重采样计算，条件于已有折外预测；它们没有包含重新训练和重新调参带来的全部不确定度。只有一个闭包的辅助结晶温度任务不会伪造来源泛化区间。

## 9. 逆向结论的范围

当前逆向评价包含目录覆盖、已知组分的恢复和比例损失。`component_set_at_true_count` 使用真实组分数，是诊断指标；它不等于自由生成的 Hit@K。候选采样接口的角色限制和组成归一化，也不能替代完整化学计量、反应构图、逐协议性质复核及实验验证。

现有训练基础是单性质原子观察。模型可以接收多个目标掩码，但在没有共同状态联合训练证据时，多目标调用必须标记为未验证的组合查询；不能把不同材料、不同状态的性质拼成一条“四性质实测”。

后续优化应从内层误差、数据质量和适用域分析出发。外层结果保留为评价事实；不能通过删除高误差样本、选择有利种子或重新命名旧测试集来改善报告数字。
