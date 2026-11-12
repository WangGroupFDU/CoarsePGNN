## configs/inverse/round009_plan.json lines 321-435

```text
321:     "scoring_and_acceptance_policy": {
322:       "authority": "success_contract.json原字节及SHA-256保持不变。以下为执行展开，不能覆盖合同。",
323:       "physical_tolerances": {
324:         "tg": {
325:           "delta": 10,
326:           "error_unit": "K",
327:           "stored_unit": "degC"
328:         },
329:         "tm": {
330:           "delta": 15,
331:           "error_unit": "K",
332:           "stored_unit": "degC"
333:         },
334:         "tensile_strength_ultimate": {
335:           "delta": 5,
336:           "error_unit": "MPa"
337:         },
338:         "elongation_at_break": {
339:           "delta": 100,
340:           "error_unit": "percentage_points"
341:         }
342:       },
343:       "tolerance_meaning": "这些是预注册设计分辨率，不是实测噪声或重复性估计；不得用历史无资格重复差值重解释。",
344:       "forward": {
345:         "aggregation": "每个目标和种子，先在每种材料的合格观察内平均绝对误差与命中指示，再在材料间等权；四任务归一化MAE最后等权。",
346:         "baseline": "固定train中相同target/method/criterion/phase的材料等权中位数，类不存在时回退该目标全train中位数。只依据训练侧信息生成并冻结，不使用测试值选择回退层次。",
347:         "each_target_each_seed_MAE_over_delta_max": 1,
348:         "each_target_each_seed_hit_rate_min": 0.6,
349:         "each_seed_normalized_score_over_baseline_max": 0.9,
350:         "each_target_seed_MAE_range_over_delta_max": 0.25,
351:         "unseen_grade_three_seed_mean_MAE_over_delta_max": 1.25,
352:         "unseen_definition": "配方至少一个规格不在该目标的真实train监督规格集合；不能用全局逆向监督可见性替代逐目标前向可见性。",
353:         "missing_prediction": "不得删观察；未命中且MAE验收不通过，若由计算或实现错误造成同时标记工程失败。"
354:       },
355:       "inverse": {
356:         "anchor_request": "每个实测观察以真实值±该目标delta形成请求，保留真实协议和历史R。真实D/P/E/v仅供独立评分端teacher forcing及窗口核对，不泄漏给生成候选接口。",
357:         "source_precision": "使用原始float64 v_reference，不能先经float32训练张量再转double。窗口为[max(0,v-0.005),min(1,v+0.005)]。",
358:         "probability": "P(D|q,c,R)乘P(P|D,q,c,R)乘P(E|D,P,q,c,R)乘混合高斯在logit窗口内的概率质量；NONE最后因子精确为1。端点0和1分别采用负无穷和正无穷的logit极限。",
359:         "numerics": "稳定float64对数CDF差；每个概率因子评分下限1e-12，分别记录触底次数。NLL使用自然对数，单位nat，不用z空间点密度替代窗口概率。",
360:         "paired_aggregation": "同观察ID、材料、闭包、目标、协议、R和真实配方逐项配对；材料内平均后材料间平均，最后四任务等权。Delta=NLL_no_target_values-NLL_main。",
361:         "each_seed_four_target_mean_Delta_min_nat": 0.1,
362:         "each_target_three_seed_mean_Delta_min_nat": 0,
363:         "joint_anchors": "对106个核准双目标状态全部评分并逐材料聚合，报告配对改进及完整分母；不增加优势门槛，不代替单目标门槛，不构造三或四目标实测标签。"
364:       },
365:       "computational": {
366:         "each_seed_overall_Hit_at_20_min": 0.6,
367:         "each_seed_single_target_Hit_at_20_min": 0.75,
368:         "each_seed_two_target_Hit_at_20_min": 0.5,
369:         "each_seed_three_target_Hit_at_20_min": 0.25,
370:         "each_seed_four_target_Hit_at_20_min": 0.25,
371:         "each_seed_raw_proposal_chemical_legality_min": 0.99,
372:         "each_seed_returned_candidate_legality_min": 1,
373:         "each_seed_median_distinct_passing_DPE_over_successful_requests_min": 2,
374:         "three_seed_mean_Hit_margin_over_strongest_baseline_min": 0.05,
375:         "each_seed_Hit_margin_over_strongest_same_seed_baseline_min": 0,
376:         "overall_Hit_max_minus_min_across_seeds_max": 0.15,
377:         "denominators": "主指标固定96，目标数分层固定24/48/8/16；无输出、搜索未命中及预算耗尽而未找到候选均记0，不能删除困难请求。",
378:         "strongest_baseline": "先分别计算三个基线各自三种子平均Hit，再取最大值；逐种子比较另取该种子最强基线，不挑选对主方法有利的基线。",
379:         "diversity": "只在三代理共同支持的候选中计不同D/P/E组合；成功请求集合为空时多样性中位数不能记为通过。",
380:         "strata": "完整报告目标数、协议A/B、NONE、目录/逆向监督/逐目标前向监督可见性、R/v外推。候选属性分层须明确候选分母；如报告属性限定的请求命中，使用完整96请求为分母并注明分层可能重叠，不能据输出重新定义主分母。"
381:       },
382:       "statistics": {
383:         "dependency_unit": "原来源传递闭包",
384:         "groups_at_least_8": "10000次按闭包重采样的95%百分位区间，固定种子20261114；区间辅助报告，不替代固定集合门槛。",
385:         "groups_2_to_7": "逐闭包结果和留一闭包敏感性范围。",
386:         "groups_1": "置信区间为null，原因not_identifiable_from_one_cluster。",
387:         "seeds": "三个模型种子分别报告并检验，不当作三份独立实验数据，不以最好种子代替全部种子。",
388:         "epsilon": 1e-06,
389:         "epsilon_scope": "仅用于原合同允许的归一化浮点比较，不扩大物理性质区间。",
390:         "claim_boundary": "最终结果支持冻结有限留出集合和固定计算面板上的结论；三代理一致不等于实验成功，Tg一个测试闭包不支持总体泛化区间，三/四目标共同实测仍为0。"
391:       },
392:       "interface_outputs": "保留D/P/E规格、结构表示、Mn或摩尔质量、有效位点资格、R/v、摩尔比、质量分数、兼容图、每协议每模型预测与逐约束结果。计量绝对误差容差仍为1e-8，NONE的v精确为0。来源不足字段保持null；新配方只能返回有出处的类似反应建议，不能补写成已验证合成步骤。"
393:     },
394:     "independent_recomputation_authorization": {
395:       "status": "CONDITIONALLY_AUTHORIZED",
396:       "round": 10,
397:       "maximum_wall_seconds": 14400,
398:       "precondition": "第九轮固定输出完整、身份及工程不变量成立，且使用开窗前已冻结的同一C9复算实现。第九轮科学门槛失败不阻止复算；代码错误不能通过第十轮修改实现后重新选择结果解决。",
399:       "implementation": "独立进程重建模型对象、重载归档权重和冻结变换，读取同一final_window数据及候选锁；先独立计算，再与第九轮结果比较，不能仅重读汇总表。",
400:       "coverage": [
401:         "三个主种子的全部250条前向测试观察。",
402:         "三个主逆模型、对应三个无目标数值对照及20261111的K=1、无测试条件模型的全部250条单目标和106个共同状态评分。",
403:         "所有正式主方法、基线和消融提案池中的候选身份、合法性、图构造、预算与搜索模型评分；失败和重复槽位也保留核对。",
404:         "全部已锁定top20在另外两个主种子上的复核预测、逐约束判断、三代理共同命中及结果合并。",
405:         "从归档提案池独立重算排序、去重、每D/P/E上限和top20身份，不能新增提案或用复算结果重新选择候选。",
406:         "按冻结规则独立重建全部新增测试prompt的编码与条件张量，核对基础资产未变及覆盖层身份。",
407:         "全部材料等权指标、配对改进、闭包统计、分层、基线比较、消融、多样性、稳定性和最终门槛表。"
408:       ],
409:       "comparison_rules": {
410:         "forward_prediction": "abs(p_recomputed-p_archived)/delta_t <= 1e-5。",
411:         "inverse_and_condition_tensors": "沿用已使用的F4浮点比较：abs(a-b)/max(1,abs(a),abs(b)) <= 1e-5，记录实际最大差异；禁止在看到最终差异后放宽。",
412:         "exact_fields": "观察及状态ID、协议与mask、原料规格、源float64配比和窗口、候选顺序、合法性、概率触底标记、逐约束布尔值及最终门槛结论必须一致。",
413:         "raw_log_probabilities": "合法的负无穷按冻结概率下限规则解释；NaN、非法正无穷或与规则不一致的窗口计算属于工程错误，不能用近似比较掩盖。"
414:       },
415:       "deliverables": [
416:         "逐观察和逐候选差异文件、最大误差、身份和布尔一致性统计",
417:         "完整模型与输入下载校验、实际独立加载日志、运行环境与确定性设置",
418:         "新增缓存独立重建回执及零训练更新记录",
419:         "原合同每项required门槛的通过、失败或未完成状态及证据定位",
420:         "C/A/合同/data_lock/权重/候选锁/最终窗口绑定回执和必读清单",
421:         "全部负结果、工程失败和预算未完成项，不自动生成研究PASS"
422:       ],
423:       "failure_boundary": "若第九轮无法在原实现与原预算内完成，第十轮只能核对已有证据并如实交付未完成状态；不补训、不新增搜索、不改选模型、不另开测试窗口。"
424:     }
425:   },
426:   "tests": [
427:     "tests/test_inverse_final.py",
428:     "tests/test_inverse_access.py",
429:     "tests/test_inverse_metrics.py",
430:     "tests/test_inverse_request_api.py",
431:     "tests/test_inverse_search.py"
432:   ],
433:   "output": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_009_final",
434:   "recompute_output": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_recompute"
435: }
```
## docs/pu_final_evaluation.md lines 1-70

```text
1: # PU 最终评价与独立复算
2: 
3: 本入口消费已选定 D0 的三主种子及五个逆向控制权重，不进行训练或重新拟合。
4: 原会话 R8 授权、成功合同、数据锁、96 请求原字节均由
5: `configs/inverse/round009_plan.json` 绑定。模型生产 C6/C8 与消费者 C9 分开记录。
6: 
7: ## 执行位置与预算
8: 
9: 所有真实检查和计算仅在 `codex-cn2` 的现有 Anaconda base 执行：
10: 
11: ```text
12: /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository
13: ```
14: 
15: 每轮最多 14400 秒；监督进程使用同一单调时钟期限，预留 10 秒写终止回执。
16: 已存在输出目录拒绝覆盖。科学门槛失败正常记录；身份、计算或实现错误立即停止。
17: 开窗后不能修改源码再重跑测试，也不能利用剩余轮次补训或另选模型。
18: 
19: ## 冻结与门禁
20: 
21: 先提交完整源码 C9，包括两个入口和合成门禁测试，再生成外部授权回执中的真实
22: `consumer_code_sha`、全 `src/tests/configs` 文件清单和计划 SHA。未来提交 SHA 不预填。
23: 外部回执不改变权重内部生产者身份。运行时核对实际字节、全部权重和14项基础资产。
24: 
25: R9 内先运行指定合成测试，再用全部八套既有模型对原 validation 逐观察结果做兼容性核对。
26: 只有门禁通过后才创建不可覆盖 `final_lock.json` 和唯一 `final_window.json`。
27: 测试的每次物理打开写入 `test_access.jsonl`；R10使用同一逻辑窗口。
28: 
29: 缺失条件只按固定 Qwen revision、batch 4 和原提示规则做推理。输出独立缓存覆盖层，
30: 不覆盖基础键，不拟合 PLE、词表或变换。未知 assay 仍映射训练词表的0索引。
31: 
32: ## 第九轮
33: 
34: 以下命令由 loopctl 正式计算执行器调度，不能在预算外额外运行：
35: 
36: ```bash
37: CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
38:   /opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
39:   --final-plan configs/inverse/round009_plan.json \
40:   --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_009_final
41: ```
42: 
43: 分母为250单目标观察、106双目标共同状态，以及1152个主方法/基线单元和288个消融单元。
44: 每个单元先保存 `search.json` 和 `top20_lock.json`，随后逐种子复核。复核结果不能回流搜索。
45: 单元函数可恢复已持久化的复核状态；部分搜索尚无完整搜索文件时明确拒绝隐式重采样。
46: 整轮入口不自动重试或延长期限。
47: 
48: ## 第十轮
49: 
50: R9完整归档并经评审后，使用其 `scientific_inventory.json` 的实际 SHA 调度：
51: 
52: ```bash
53: CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
54:   /opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
55:   --final-plan configs/inverse/round009_plan.json \
56:   --recompute-reference /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_009_final \
57:   --reference-sha256 ACTUAL_ARCHIVED_INVENTORY_SHA256 \
58:   --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_recompute
59: ```
60: 
61: 独立进程重新编码新增条件、重载权重、复算所有锚点和全部提案池，包含非法、失败及重复槽。
62: 仅从原池重算排序/去重/组合上限，并核对原 top20 身份，不新增提案。
63: 再复算全部 top20 的跨种子预测以及材料等权统计、闭包区间、基线、消融和门槛。
64: 
65: ## 结果含义
66: 
67: `anchor_summary.json` 是实测有限留出核对；`panel_summary.json` 是三个初始化代理模型
68: 对固定96请求的共同支持，不能称新配方实验成功。`acceptance.json` 保留所有通过和失败项。
69: Tg 单闭包总体区间为 null，三/四性质共同实测为0。任何脚本均不会自动写研究 PASS；
70: 只有冻结合同全部满足、独立复算成立并获原会话版本匹配 PASS 后才完成研究。
```
## src/coarsepgnn_inverse/__main__.py lines 1-86

```text
1: """PU 逆向设计的审计、请求校验和化学计量命令入口。"""
2: from __future__ import annotations
3: 
4: import argparse
5: import json
6: from pathlib import Path
7: 
8: from .audit import audit
9: from .chemistry import decode_ratios
10: from .io import read_json, write_json
11: from .schema import validate_request
12: 
13: 
14: def main() -> None:
15:     """运行准备步骤或已获Pro授权的真实训练；训练入口再次核验冻结合同。"""
16:     parser = argparse.ArgumentParser(description=__doc__)
17:     sub = parser.add_subparsers(dest="command", required=True)
18:     command = sub.add_parser("audit", help="核验来源、完整图张量和无泄漏划分")
19:     command.add_argument("--config", required=True, type=Path)
20:     command.add_argument("--output", required=True, type=Path)
21:     command = sub.add_parser("validate-request", help="校验目标与测试协议，不调用模型")
22:     command.add_argument("--input", required=True, type=Path)
23:     command = sub.add_parser("stoichiometry", help="将双官能 PU 的 R/v 换算为投料比")
24:     command.add_argument("--input", required=True, type=Path)
25:     command.add_argument("--output", required=True, type=Path)
26:     command = sub.add_parser("train", help="依据冻结合同及授权，在远程训练四任务与逆模型")
27:     command.add_argument("--config", required=True, type=Path)
28:     command.add_argument("--output", required=True, type=Path)
29:     command.add_argument("--resume", action="store_true", help="仅接续同一代码、合同和配置的未完成轮")
30:     command.add_argument("--deadline-monotonic", type=float, help="外层整轮共享单调时钟期限")
31:     command = sub.add_parser("design", help="从冻结权重生成配方、粗图和逐协议预测")
32:     command.add_argument("--model-dir", required=True, type=Path)
33:     command.add_argument("--manifest-sha256", required=True)
34:     command.add_argument("--contract", required=True, type=Path)
35:     command.add_argument("--data-lock", required=True, type=Path)
36:     command.add_argument("--input", required=True, type=Path)
37:     command.add_argument("--output", required=True, type=Path)
38:     command = sub.add_parser("evaluate", help="开发锚点评价或显式授权的最终评价/独立复算")
39:     command.add_argument("--config", type=Path)
40:     command.add_argument("--model-dir", type=Path)
41:     command.add_argument("--manifest-sha256")
42:     command.add_argument("--output", required=True, type=Path)
43:     command.add_argument("--readonly-producer", type=Path)
44:     command.add_argument("--final-plan", type=Path, help="冻结的R9/R10无训练计划")
45:     command.add_argument("--recompute-reference", type=Path)
46:     command.add_argument("--reference-sha256")
47:     args = parser.parse_args()
48:     if args.command == "audit":
49:         result = audit(read_json(args.config), args.output)
50:     elif args.command == "validate-request":
51:         request = validate_request(read_json(args.input))
52:         result = {"status": "valid_request", "constraints": len(request["constraints"]),
53:                   "predictions_computed": False}
54:     elif args.command == "train":
55:         from .training import train
56:         result = train(read_json(args.config), args.output,resume=args.resume,deadline_monotonic=args.deadline_monotonic)
57:     elif args.command == "design":
58:         from .inference import FrozenDesignModels
59:         from .design import run_design
60:         model=FrozenDesignModels(args.model_dir,manifest_sha256=args.manifest_sha256,
61:                                  contract_path=args.contract,data_lock_path=args.data_lock)
62:         result=run_design(read_json(args.input),model,args.output)
63:     elif args.command == "evaluate":
64:         if args.final_plan:
65:             if any((args.config, args.model_dir, args.manifest_sha256, args.readonly_producer)):
66:                 parser.error('最终计划不能混入开发评价参数')
67:             if bool(args.recompute_reference) != bool(args.reference_sha256):
68:                 parser.error('独立复算须同时给原目录和归档清单SHA')
69:             from .final_round import run
70:             result=run(args.final_plan,args.output,'recompute' if args.recompute_reference else 'evaluate',
71:                        reference=args.recompute_reference,reference_sha256=args.reference_sha256)
72:         else:
73:             if not all((args.config,args.model_dir,args.manifest_sha256)) or args.recompute_reference or args.reference_sha256:
74:                 parser.error('开发评价要求config/model-dir/manifest-sha256且不接受复算参数')
75:             from .evaluation import evaluate_configuration
76:             result=evaluate_configuration(read_json(args.config),args.model_dir,args.manifest_sha256,args.output,
77:                 readonly_producer=read_json(args.readonly_producer) if args.readonly_producer else None)
78:     else:
79:         result = decode_ratios(**read_json(args.input))
80:         write_json(args.output, result)
81:     print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
82: 
83: 
84: if __name__ == "__main__":
85:     main()
86: 
```
## src/coarsepgnn_inverse/anchor_evaluation.py lines 1-160

```text
1: """实测锚点评价：原始完整图正向预测、真实配方窗口概率及配对消融。"""
2: from __future__ import annotations
3: 
4: from pathlib import Path
5: 
6: import torch
7: from torch.utils.data import DataLoader, Dataset
8: 
9: from coarsepgnn.schema import package_schema
10: from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset
11: 
12: from . import TARGET_UNITS
13: from .io import write_json
14: from .metrics import closure_statistics, forward_baseline, inverse_anchor_score, material_mean
15: from .networks import device_batch
16: from .qualification import protocol_definition
17: from .training_data import ConditionInputs, InverseCollator
18: 
19: 
20: class FixedAnchorDataset(Dataset):
21:     """只构建固定单目标锚点，不包含随机状态/子集采样或优化器入口。"""
22: 
23:     def __init__(self, rows: list[dict], formulas: dict, grade_ids: list[str], transforms: dict,
24:                  contract: dict, *, split: str) -> None:
25:         """验证调用方已加载的单个评价集合；test仅能由外层最终锁加载器传入。
26: 
27:         真实配方只进入概率评分与审计，不传给候选搜索。不能把测试配方当逆模型输入。
28:         """
29:         if split not in ('validation','test') or not rows or any(r['split']!=split for r in rows):
30:             raise ValueError('实测锚点必须属于明确的validation或已开窗test')
31:         self.rows=sorted(rows,key=lambda r:r['observation_id']);self.formulas=formulas
32:         self.grades={g:i for i,g in enumerate(grade_ids)};self.transforms=transforms;self.contract=contract
33: 
34:     def __len__(self):
35:         """全部合格单目标实测进入分母，不删掉未见规格或困难观察。"""
36:         return len(self.rows)
37: 
38:     def __getitem__(self,index):
39:         """构造y±delta固定请求与真实配方评分因子；保存float64配比真值。"""
40:         row=self.rows[index];formula=self.formulas[row['material_id']]
41:         transform=self.transforms[row['target']];delta=self.contract['design_tolerances'][row['target']]['delta']
42:         pair=[(float(row['value'])+sign*delta-transform['mean'])/transform['std'] for sign in (-1,1)]
43:         return {'observation_ids':[row['observation_id']],
44:             'property_ids':[list(TARGET_UNITS).index(row['target'])],'bounds':[pair],'bound_mask':[[True,True]],
45:             'nco_oh_ratio':formula['nco_oh_ratio'],'choices':[self.grades[g] for g in formula['grade_ids']],
46:             'v':formula['extender_oh_fraction'],'audit':{'material_id':row['material_id'],'split':row['split'],
47:                 'observation_ids':[row['observation_id']],'state_ids':[row['state_id']]}}
48: 
49: 
50: class AnchorCollator(InverseCollator):
51:     """保留训练接口的掩码张量，但评分v中心使用真实float64，不经float32再转回。"""
52: 
53:     def __init__(self,conditions,formulas):
54:         """独立保留源配方映射，用材料身份核对评分中心而非相信预处理item。"""
55:         super().__init__(conditions);self.formulas=formulas
56: 
57:     def __call__(self,items):
58:         """未改变神经网络输入精度；v只用于独立窗口质量评分。"""
59:         batch=super().__call__(items)
60:         batch['v_reference']=torch.tensor([item['v'] for item in items],dtype=torch.float64)
61:         batch['v_source']=torch.tensor([self.formulas[item['audit']['material_id']]['extender_oh_fraction']
62:                                         for item in items],dtype=torch.float64)
63:         return batch
64: 
65: 
66: def prepare_anchor_conditions(model, rows: list[dict]) -> ConditionInputs:
67:     """使用与训练完全相同的固定文本/PLE资产编码评价协议，不从标签拟合任何参数。
68: 
69:     此处目标边界仅为组织请求，传入文本编码器的字段不包含value、来源或材料ID。
70:     所有新prompt预先合并编码一次；实现须由FrozenDesignModels提供批量准备接口。
71:     """
72:     requests=[]
73:     for row in rows:
74:         definition=protocol_definition(row);delta=model.contract['design_tolerances'][row['target']]['delta']
75:         requests.append({'schema_version':1,'constraints':[{**definition,'unit':row['unit'],
76:             'protocol_id':row['condition_input_sha256'],'lower':float(row['value'])-delta,'upper':float(row['value'])+delta}],
77:             'nco_oh_ratio':1.0})
78:     model.prepare_requests(requests)
79:     parts=[model.encode_request(request)['conditions'] for request in requests]
80:     payload={key:torch.cat([part[key] for part in parts],dim=0) for key in ('raw','keys','values','mask')}
81:     payload['observation_ids']=[r['observation_id'] for r in rows]
82:     payload['input_sha256']=[r['condition_input_sha256'] for r in rows]
83:     return ConditionInputs(payload,rows,model.vocab)
84: 
85: 
86: @torch.no_grad()
87: def real_anchor_predictions(model, rows: list[dict], training: list[dict], formulas: dict,
88:                             *, split: str, include_forward: bool = True) -> tuple[list[dict],list[dict]]:
89:     """原始完整图预测与真实配方概率分别计算，禁止以重构候选图替换留出来源图。"""
90:     if split not in ('validation','test') or any(r['split']!=split for r in rows):
91:         raise ValueError('锚点集合归属不一致')
92:     conditions=prepare_anchor_conditions(model,rows);forward=[]
93:     for target in (TARGET_UNITS if include_forward else ()):
94:         current=[r for r in rows if r['target']==target];by_id={r['observation_id']:r for r in current}
95:         dataset=PUObservationDataset(current,split_map={r['observation_id']:split for r in current},
96:             split=split,target_transform=model.transforms[target],
97:             graph_store=GraphStore(package_schema(),max_cache_size=len({r['graph_relpath'] for r in current})),mode='structure_only')
98:         loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
99:             generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=conditions.forward_collate)
100:         network=model.load_model(target+'_full');predictions={}
101:         for batch in loader:
102:             raw=model.transforms[target].inverse(network(device_batch(batch,model.device)).cpu().numpy())
103:             for audit,value in zip(batch['audit'],raw):predictions[audit['observation_id']]=float(value)
104:         if set(predictions)!=set(by_id):raise ValueError('正向实测锚点分母不闭合')
105:         baselines=forward_baseline([r for r in training if r['target']==target],current)
106:         delta=model.contract['design_tolerances'][target]['delta']
107:         for row,baseline in zip(current,baselines):
108:             value=predictions[row['observation_id']];error=abs(value-float(row['value']))
109:             forward.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target','unit','forward_supervised_seen')},
110:                 'value':float(row['value']),'prediction':value,'absolute_error':error,'hit_delta':float(error<=delta),
111:                 'baseline_prediction':baseline,'baseline_absolute_error':abs(baseline-float(row['value'])),
112:                 'graph_file_sha256':row['graph_file_sha256']})
113:         del network
114:     dataset=FixedAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
115:     loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
116:         generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions,formulas))
117:     inverse=model.load_model('inverse_full');by_id={r['observation_id']:r for r in rows};scored=[]
118:     for batch in loader:
119:         score=inverse_anchor_score(inverse,device_batch(batch,model.device))
120:         for i,audit in enumerate(batch['audit']):
121:             row=by_id[audit['observation_ids'][0]]
122:             scored.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target')},
123:                 'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[row['material_id']])})
124:     if len(scored)!=len(rows) or {r['observation_id'] for r in scored}!=set(by_id):
125:         raise ValueError('逆向实测锚点分母不闭合')
126:     return forward,scored
127: 
128: 
129: def forward_summary(rows: list[dict], contract: dict) -> dict:
130:     """按冻结材料等权口径汇总四目标及训练中位数基线，独立闭包用于区间。"""
131:     targets={}
132:     for target in TARGET_UNITS:
133:         current=[r for r in rows if r['target']==target];delta=contract['design_tolerances'][target]['delta']
134:         targets[target]={'mae':material_mean(current,'absolute_error'),'hit_delta':material_mean(current,'hit_delta'),
135:             'baseline_mae':material_mean(current,'baseline_absolute_error'),'observations':len(current),'delta':delta,
136:             'statistics':closure_statistics(current,'absolute_error'),
137:             'unseen_specifications':closure_statistics([r for r in current if not r['forward_supervised_seen']],'absolute_error')}
138:     score=sum(r['mae']/r['delta'] for r in targets.values())/4
139:     baseline=sum(r['baseline_mae']/r['delta'] for r in targets.values())/4
140:     return {'targets':targets,'normalized_score':score,'normalized_baseline_score':baseline,
141:             'ratio_to_baseline':score/baseline if baseline>0 else None}
142: 
143: 
144: def paired_inverse_summary(main: list[dict], control: list[dict]) -> dict:
145:     """逐观察配对无目标值消融，再在材料、性质之间等权；不把不同分母强行配对。"""
146:     lookup={r['observation_id']:r for r in control}
147:     if len(lookup)!=len(control) or set(lookup)!={r['observation_id'] for r in main}:
148:         raise ValueError('逆向主模型与对照观察分母不同')
149:     paired=[]
150:     for row in main:
151:         other=lookup[row['observation_id']]
152:         if any(other[k]!=row[k] for k in ('material_id','group_id','target')):
153:             raise ValueError('逆向配对身份不同')
154:         paired.append({**row,'control_nll':other['nll'],'improvement_nats':other['nll']-row['nll']})
155:     targets={target:{'main_nll':material_mean([r for r in paired if r['target']==target],'nll'),
156:                       'control_nll':material_mean([r for r in paired if r['target']==target],'control_nll'),
157:                       'improvement_nats':material_mean([r for r in paired if r['target']==target],'improvement_nats'),
158:                       'statistics':closure_statistics([r for r in paired if r['target']==target],'improvement_nats')}
159:              for target in TARGET_UNITS}
160:     return {'targets':targets,'four_target_mean_improvement_nats':sum(r['improvement_nats'] for r in targets.values())/4,
```
