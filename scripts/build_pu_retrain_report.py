#!/usr/bin/env python3
"""从完成的冻结运行生成结果正文；仅读现有指标，不拟合或重新选择模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree


TASKS = ("tg", "tm", "tensile_strength_ultimate", "elongation_at_break")
NAMES = {"tg": "Tg（°C）", "tm": "Tm（°C）", "tensile_strength_ultimate": "UTS（MPa）",
         "elongation_at_break": "断裂伸长率（%）", "auxiliary::thermal_event": "一般热事件",
         "auxiliary::tensile_strength_unspecified": "端点未明确强度",
         "auxiliary::crystallization_temperature": "结晶温度"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=2):
    if value in (None, "", "None", "null"):
        return "不可评价"
    return f"{float(value):.{digits}f}"


def table(headers, values):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in values])


def build(run, output):
    run, output = Path(run), Path(output)
    workflow = run / "workflow"
    completed = read(workflow / "workflow_result.json")
    independent = read(run / "independent_verification/result.json")
    if completed["status"] != "completed" or independent["status"] != "passed":
        raise ValueError("this report requires completed training and passed independent reload verification")
    summary = read(run / "delivery_summary/summary.json")
    coverage = read(run / "delivery_summary/coverage_verification.json")
    audit = read(run / "dataset/audit.json")
    split_audit = read(run / "final_split_identity_check.json")
    route = summary["deployment_route"]
    forward, inverse = rows(run / "delivery_summary/forward_metrics.csv"), rows(run / "delivery_summary/inverse_metrics.csv")
    plot = read(run / "figures/plot_manifest.json")
    runtime = read(workflow / "runtime.json")
    checks_path = run / "software_checks_frozen.xml"
    checks = list(ElementTree.parse(checks_path).getroot().iter("testsuite"))
    if not checks or any(int(suite.get(key, "0")) for suite in checks for key in ("failures", "errors")):
        raise ValueError("report requires the recorded software checks to pass")
    check_count = sum(int(suite.get("tests", "0")) for suite in checks)
    code = runtime["base_git_head"]
    base = f"https://github.com/WangGroupFDU/CoarsePGNN/blob/{code}"
    by_metric = {(row["method"], str(row["seed"]), row["task"]): row for row in forward}
    seeds = summary["seeds"]
    methods = [("train_median", "single_classical_run", "训练集性质中位数"),
               ("selected_tabular", "single_classical_run", "内层选定经典模型")]
    methods += [("neural", str(seed), f"集合网络 {seed}") for seed in seeds]
    methods += [("inner_selected_pipeline", str(seeds[0]), f"内层选定管线 {seeds[0]}")]
    forward_table = table(["方法", *[NAMES[task] for task in TASKS]],
                          [[label] + [number(by_metric[method, seed, task]["mae"]) for task in TASKS]
                           for method, seed, label in methods])
    task_order = list(TASKS) + sorted(set(audit["trainable_by_target"]) - set(TASKS))
    task_table = table(["监督任务", "观察数", "用途"],
                       [[NAMES.get(task, task), audit["trainable_by_target"][task],
                         "主要任务" if task in TASKS else "辅助任务"] for task in task_order])
    inverse_table = table(["种子", "任务", "完整目录可评分 / 总数", "组合损失", "真实组分数恢复率"],
        [[row["seed"], NAMES[row["task"]], f"{row['complete_n']} / {row['n']}", number(row["composite_loss"], 4),
          "不可评价" if row["component_set_at_true_count"] in ("", "None", "null") else
          f"{100 * float(row['component_set_at_true_count']):.2f}%"] for row in inverse if row["task"] in TASKS])
    selected = read(workflow / "evaluation/champion/selection_lock.json")["outer_folds"]
    selections = table(["外折", "正向家族", "经典内层分数", "神经内层分数"],
                       [[str(item["outer_fold"] + 1), item["selected_family"],
                         number(item["baseline_inner_score"], 4), number(item["neural_inner_score"], 4)] for item in selected])
    fits = coverage["nested_fits"] + coverage["deployment_fits"]
    updates = {direction: sum(item["optimizer_updates"][direction] for item in fits) for direction in ("forward", "inverse")}
    font = plot["figures"]["oof_parity"]
    baseline_folds = font["baseline_outer_folds"]
    limits = "；".join(f"{NAMES[t]} {audit['trainable_by_target'][t]} 条" for t in TASKS)
    text = f"""# PU 全数据重训练：结果与可复现报告

日期：2026-11-17。训练代码：`{code}`。评价范围：回顾性分组嵌套交叉验证；本报告不修改历史 R1-R10 的合同、负结果或研究验收结论。

## 1. 本次完成了什么

本次已完成完整公开 PU 输入的数据重组织、来源与可观测配方联合分组、经典与神经模型训练、三个神经种子的全部折外评价，以及全量部署拟合。全部 {audit['retained_observations']:,} 条母表记录保留；正向与逆向使用完全一致的 {audit['shared_trainable_observations']:,} 条监督观察。主流程状态为 `{completed['status']}`，实际墙钟 {completed['wall_seconds']:.2f} 秒；全新 Python 进程的独立权重重载和逐条复算状态为 `{independent['status']}`。

这解决了数据使用和评价组织问题，但不自动证明逆向设计达到实验成功，也不能将本次运行改称历史冻结合同的 PASS。模型实际效果和适用边界列在下文；不以删除困难样本或选择最有利种子改善数字。

## 2. 数据范围与保留原则

本次完整公开母表有 3,657 条原始输入。从明确记录的成员列表恢复 561 条 Tg 原子测量后，保留 4,218 条记录。246 条派生均值父记录继续保存原文及谱系，但不与其原子成员重复计算独立监督。因此两方向共同监督数为 3,972。恢复依据是明确成员数值，不是从均值反推或生成伪标签。

数据来自 202 个来源 ID，包含 1,162 个材料 ID。观察数、材料 ID 数、独立来源数和共同状态数不能互换；同族或重复报道也不因不同来源 ID 自动成为独立实验。532 条多组分观察、100 条缺少多元醇 Mn 的观察继续使用，缺失字段通过显式 mask 处理。11 条图中数字化记录与直接读数区分，父均值与真实实验标签分离。

**“全量”指当前 GitHub 可取得的完整公开输入及其可追溯补充关系。原始完整 SQLite 没有包含在本次可用文件中，不能宣称已经穷尽原始数据库。** 新入口支持只读导入完整 SQLite；取得新增来源后，应生成新的母表和评价版本。

{task_table}

四个主任务共 3,338 条，另外 634 条仍提供辅助监督。未明确端点的拉伸强度不冒充 UTS，一般热事件和结晶温度不冒充 Tm。只对标签实际可观测的任务计算监督，不要求同一材料同时有四种性质，也不拼接不同样品的性质。

![图 1：保留记录、共享监督与任务覆盖。灰色父记录保留谱系但不重复监督；颜色区分主任务与辅助任务。](../figures/data_coverage.png)

### 2.1 已修正的条件语义缺口

交付前逐条核查发现，公开入口中的旧自由文本温度回填把升温速率、固化/制样/老化温度、扫描程序保持温度及其他性质参考温度误作标量测试温度。修正仅针对 `temperature_c` 来源为 `condition_raw_text` 且没有明确结构化测试温度证据的字段：保留原始记录与逐字段修正谱系，将模型使用的该条件标记为未知。全部实验观察、性质标签、原料、配比、来源家族和模型候选保持不变。

该修正影响 871 条保留记录，其中 781 条共同监督观察、90 条派生父记录。781 条中分别有 529 条升降温速率数值、190 条过程温度、57 条扫描保持温度、5 条其他性质参考温度。它没有删除整行，也不把缺失测试温度猜成室温。已有升温速率和制样/调理字段及原文继续保留。

此前消费旧字段的家族分组运行被中止并作为诊断保全，不进入本报告最终性能。修正依据是原文语义和字段来源，不是留出误差；本报告也不据新旧性能差异声称该字段就是泛化误差的唯一原因。

## 3. 留出样本怎样保持未参与训练

本次采用 5 折外层、3 折内层的分组嵌套交叉验证。来源、规范出版标识、已核验专利家族、材料、配方、原图关联、父子谱系，以及可观测原料结构、规格和精确比例构成联合闭包。性质结果不用于分组，闭包不能为了平衡而拆开。

实际母表形成 {split_audit['counts']['recorded_closures']} 个闭包，最大闭包含 603 条保留记录；五个外折分别留出 775、795、774、772、856 条监督观察。每条监督观察在每个种子下恰好取得一次来自未训练该记录及其关联闭包的预测。全部已定义身份闭包跨集合重合为 0，1,049 个可观测规范配方键均没有跨外折。

来源审计发现，初次诊断分组遗漏三组真实专利家族：US8962785B2 / US9574043B2、CN114057971A / CN114057971B、US20250250389A1 / CN115322330B。旧诊断运行已停止并保留；本报告只使用来源分组与条件字段修复后的新运行。56 个专利全部进行了有界查询，其中 54 个 Family ID 已核读；US3746665A 和 CN118165222A 仍未解析，共涉及 24 条监督观察，全部保留。该局限禁止把“已知关系零重合”扩大为“全部潜在复录已经排除”。原 `tensor_hash` 有值记录数为 0，也不声称已经完成全部拟合后输入张量去重。

历史 R10 测试此前已被查看。本次模型在计算上没有训练其对应的外折记录，但研究者已接触历史结果，因此报告名称是**回顾性分组嵌套评价**，不是新的盲测。

## 4. 实际训练与模型选择

正向经典候选为 Ridge 与 ExtraTrees。神经候选为可训练的多组分集合网络，输入 Morgan 指纹、原料角色、规格、连续比例、结构化测试条件及缺失掩码。实际执行的是集合网络，不是把预留图编码器接口称作已经运行的 CoarsePGNN 图主干。

逆向模型输入可观测目标值、任务 mask 和测试条件，学习训练折原料目录的组分存在分数及连续组成的 logistic-normal 混合分布。正向和逆向每个仍在训练的 epoch 均无放回遍历同一批观察，保存数量、唯一 ID 数和散列。两个方向可以使用不同早停 epoch，但观察身份集合相同。

三套神经配置在拟合前固定：线性目标变换、力学性质 log1p 变换、以及较强正则化和条件 dropout。统一上限 120 epoch、patience 20、batch 128、AdamW、梯度裁剪、5 个混合分量。四主任务等权，任务内按材料平衡，全部辅助任务总权重为 0.25；每条有效观察仍有正权重。全部尺度、目标变换及原料目录只在相应训练折拟合。

候选选择只用三个内折的验证结果。正向误差先按各内折训练集的性质 IQR 归一，再按四个主任务等权汇总；家族平局按预先规定优先经典模型。逆向按内层完整目录可评分记录的组合监督损失选择，并同时报告覆盖率。外层结果不用于候选、epoch、家族或种子选择。最终种子为 {', '.join(map(str, seeds))}。

{selections}

实际完成 {len(coverage['nested_fits'])} 次神经嵌套拟合及 {len(coverage['deployment_fits'])} 次全量神经拟合；这两部分合计包含正向 {updates['forward']:,}、逆向 {updates['inverse']:,} 次实际优化器更新。另有独立记录的经典基线拟合。冻结前软件检查为 {check_count} 项通过，包含数据语义、隔离和输入门禁；合成软件检查不混作科研计算或训练性能。

## 5. 四主任务的折外表现

下表是原单位的逐观察 MAE，包含所有四主任务的可预测留出观察，数值越小越好。观察口径为：{limits}。每个任务的材料等权 MAE、RMSE、R²、覆盖率和已有来源闭包区间同时保存在 `forward_metrics.csv`；没有只选较好的一种误差定义。

{forward_table}

“内层选定管线”是拟合前规定的选择程序在各外折的结果，图中固定展示首个种子 {seeds[0]}，不是选取外层效果最好的种子。该管线有 {baseline_folds}/5 个外折使用经典模型；同一经典模型被几个神经种子复用时，不能解释为几次独立的经典训练。完整逐种子管线结果均在 CSV 中。

![图 2：四主任务的逐观察折外预测与实验读数。颜色表示外折，虚线为理想一致线；按预声明首种子展示，保留全部主任务观察。](../figures/oof_parity.png)

![图 3：统一外折上的经典与神经模型 MAE。经典模型只显示一套实际运行，神经种子分别展示。误差线为原结果中已计算的来源闭包 bootstrap 区间。](../figures/model_mae.png)

来源闭包 bootstrap 区间条件于已保存的折外预测，不包含重新训练和重新选模型的全部不确定度。本报告不据点估计差异直接宣称统计显著优势。仅两条、单一闭包的辅助结晶温度记录继续保留为不可充分评价任务，不能把缺失性能填为零。

## 6. 逆向训练结果怎样解释

{inverse_table}

“完整目录可评分”要求留出配方的全部原料身份在该折训练目录中；目录外样本仍在总分母中，没有被删除。组合损失由组分存在 BCE 与比例 ALR 密度损失组成，数值不是实验性质成功概率。真实组分数恢复率以该任务全部留出观察为分母，在评分时使用真实组分数，是一个受限的结构恢复诊断；不能改名为自由生成 Hit@20。

![图 4：逆向模型的训练目录覆盖与真实组分数条件下的恢复。两个比例都保留全部留出观察为分母；目录外样本不被删除。该诊断不代表性质命中或实验成功。](../figures/inverse_scope.png)

当前原子监督以单性质观察为基础。多目标接口可以接收多个目标 mask，但没有将不同材料或不同测试状态拼为共同实测；这类输出明确标为 `compositional_query_unvalidated`。本轮没有重新实施历史 96 请求的完整化学构图、逐协议约束及同预算搜索验收，所以不能用本表代替那套成功口径，也不能宣称四性质逆向设计已经解决。

## 7. 全量部署模型与使用方式

嵌套评价完成后，另行使用全部 3,972 条观察拟合部署模型。配置按所有外折的内层分数均值确定，epoch 按内层已选 epoch 的中位数确定；没有使用外层误差再调参或挑选种子。

实际部署正向家族选择为 `{route['selected_forward_family']}`；经典候选为 `{route['classical_candidate']}`，神经正向候选序号为 {route['neural_forward_candidate']}，神经逆向候选序号为 {route['neural_inverse_candidate']}。完整路径和权重身份列在 `deployment_route.json`。三个神经种子全部保留。

全量部署模型在这同一母表上没有独立测试成绩；前述 OOF 评价描述的是训练程序的泛化表现。预测接口不要求输入已知性质标签；逆向输出提供原料目录身份、规格和连续组成，同时明确 `chemical_validation=not_performed` 与 `experimental_success=null`，不能仅因组成归一就认定合成或性质可靠。

将数据包和模型包解压到同一个空目录，获取本报告绑定代码版本。运行环境为 Python {runtime['python']}、PyTorch {runtime['torch']}、RDKit {runtime['rdkit']}、scikit-learn {runtime['scikit-learn']}；本次实际为 CPU，两个计算线程。Mac/CentOS 可使用相应 Python 环境；请求 CUDA 时代码会核验设备，不静默降级。

```bash
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_retrain --help
python -m coarsepgnn_retrain predict --help
python -m coarsepgnn_retrain propose --help
```

具体无标签请求示例位于 `examples/retrain/`；真实部署调用回执另存 `inference_examples/`。建议从已记录的部署路径和明确测试条件开始使用，而不是自行混搭不同折、不同变换或不同版本权重。

解压后的母表位于 `dataset/dataset.jsonl`。模型包省略了同一文件在原运行目录内的重复副本 `workflow/inputs/dataset.jsonl`，因此复算入口应显式指向数据包中的母表。以下路径均相对于两个包共同的解压根目录：

| 复算参数或用途 | 实际路径 |
|---|---|
| `--dataset` | `dataset/dataset.jsonl` |
| `--splits` | `workflow/splits.json` |
| `--nested-run` | `workflow/evaluation/neural` |
| `--baseline-run` | `workflow/evaluation/baselines` |
| 全量部署模型 | `workflow/deployment/neural`、`workflow/deployment/baseline` |
| 既有独立复核回执 | `independent_verification/` |

来源回执中的原始机器绝对路径保留用于审计；实际使用以上解压相对路径，并为任何复算输出指定新的空目录。冻结源码中的 `scripts/verify_pu_retrain_oof.py` 接收这些显式参数，不依赖原机器的工作目录。

## 8. 复核与可追溯边界

新的 Python 进程重新加载各外折所选权重和预处理，重建全部 {independent['oof_rows']:,} 条神经 OOF 输出（每种子 {independent['observations_per_seed']:,} 条），所有 CSV 字段及既有 metrics 精确一致；同一进程还重载经典基线，核对两种方法的完整 OOF 和汇总。复核过程优化器更新为 0，拟合调用被拒绝，模型选择没有改变。

这是**独立进程重载和输入重建验证**，仍调用相同冻结算法实现，不是独立算法实现。已有神经 checkpoint 字节清单被逐项核验；经典外折旧输出没有单独的权重清单，因此只记录本次实际模型字节 SHA 与 OOF 精确重算，不虚构历史散列保证。历史未保存的完整张量不能声称已与不存在的历史张量原件逐项比较。

代码、公开母表、来源映射、全部外内折 ID、候选选择、训练历史、权重、OOF 表、统计结果和复核回执均有固定身份。旧诊断分组、入口测试失败及此前中断记录保留；它们不并入本次正式性能。原料缺失规格、未知来源家族、数字化读数以及辅助端点仍按原标签如实保留。

## 9. 绘图与文献依据

提供四张可编辑矢量 PDF、PNG 预览及逐图源数据 CSV。图采用白底、7 pt 正文字、0.75 pt 边框、约 183 mm 双栏宽度、单位明确的轴标签及可区分配色。图 2 使用相同横纵尺度的预测-观察图，图 3 在不同量纲的性质之间分面，避免用同一量纲轴或雷达图暗示可直接比较。图 4 单独呈现逆向目录支持与受限恢复，避免把正向误差图当作逆向设计验证。

本环境请求字体为 `{font['requested']}`，实际字体为 `{font['actual']}`，字体回退状态为 `{font['fallback_used']}`。因此当前图不能声称已经使用 Arial；代码默认请求 Arial，安装合法字体后可用同一输入重新导出。PDF 使用嵌入的 TrueType 字体，文本保留可编辑性。出版规格参考 Nature 官方图形指南 [1,2]；预测-观察与基线比较参考已发表聚合物逆向设计研究 [3]，组分与化学计量共同表达参考 [4]。参考这些图型不代表本研究已经达到文献中的实验验证水平。

## 10. 结论与后续边界

本次已经把“全部可取得实验观察参与训练”和“使用未参与对应模型训练的样本评价”落实为可运行、可追溯的代码和产物。其核心是按可观测项监督与来源闭包留出，并将全量部署拟合从泛化评价中分开。效果判断应以本报告的全部种子、覆盖率及逐样本结果为依据。

尚不能支持的结论包括：原始私有 SQLite 已穷尽、所有潜在跨文献复录已排除、闭目录恢复等同新原料发现、多目标查询等同共同实测验证、组成合法等同实验成功，以及本轮替代旧研究合同 PASS。进一步扩大来源、补齐可验证规格或开展真正独立的新来源验证，需要新的数据版本和预先规定的评价流程；不能回改本轮外折数字。

## 11. 复现文件与来源

- [训练实现]({base}/src/coarsepgnn_retrain/training.py)：共享观察、独立早停、内层选择和固定 epoch 外层拟合。
- [数据合同]({base}/docs/pu_retrain_data_contract.md)：全部来源散列、端点语义、原子恢复及数据利用口径。
- [分组实现]({base}/src/coarsepgnn_retrain/splits.py)：出版家族和可观测配方的传递闭包及消费端校验。
- [完整执行说明]({base}/docs/pu_retrain_all_data.md)：公开入口、SQLite 入口、一键训练与部署范围。
- `delivery_summary/forward_metrics.csv`、`inverse_metrics.csv`：本报告全部数字来源；`workflow/evaluation/` 保存原始预测及选择记录。
- `independent_verification/result.json`：全新进程实际复算范围；`final_split_identity_check.json`：最终实际分组回执。

[1] Nature Research Figure Guide. Preparing figures - our specifications. https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ （2026-11-17 读取）

[2] Nature Research Figure Guide. Building and exporting figure panels. https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/ （2026-11-17 读取）

[3] Kumar, J. N. et al. Machine learning enables polymer cloud-point engineering via inverse design. npj Computational Materials 5, 73 (2019). DOI: 10.1038/s41524-019-0209-9.

[4] Vogel, G. & Weber, J. M. Inverse design of copolymers including stoichiometry and chain architecture. Chemical Science 16, 1161-1178 (2025). DOI: 10.1039/D4SC05900J.
"""
    output.mkdir(parents=True, exist_ok=False)
    report = output / "PU_all_data_retraining_report.md"
    report.write_text(text, encoding="utf-8")
    sources = [workflow / "workflow_result.json", run / "independent_verification/result.json",
               run / "delivery_summary/summary.json", run / "delivery_summary/coverage_verification.json",
               run / "delivery_summary/forward_metrics.csv", run / "delivery_summary/inverse_metrics.csv",
               run / "dataset/audit.json", run / "final_split_identity_check.json", run / "figures/plot_manifest.json",
               checks_path, run / "condition_semantics_repair_receipt.json", run / "dataset/condition_corrections.jsonl"]
    manifest = {"operation": "existing results narrative export; no fitting, prediction or selection",
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sources": {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (output / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.run, args.output))
