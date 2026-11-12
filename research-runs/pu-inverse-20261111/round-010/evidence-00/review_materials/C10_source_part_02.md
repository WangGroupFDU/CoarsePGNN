## configs/inverse/round010_plan.json lines 361-480

```text
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
431:     "tests/test_inverse_search.py",
432:     "tests/test_inverse_final_recovery.py"
433:   ],
434:   "output": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle",
435:   "recompute_output": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle/recompute",
436:   "recovery_review": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-8bc39ca44eed.md",
437:   "recovery_policy": {
438:     "status": "CONDITIONALLY_AUTHORIZED",
439:     "round": 10,
440:     "authorization_scope": "一次最后正式轮次中的最小工程修复验证、唯一最终评价和独立全量复算；不包含新增拟合或模型选择。",
441:     "scientific_policy_source": {
442:       "repository": "WangGroupFDU/CoarsePGNN",
443:       "read_at_commit": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
444:       "path": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-5f082242266a.md",
445:       "request_id": "pu-inverse-20261111-5f082242266a",
446:       "bytes": 80670,
447:       "sha256": "950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2",
448:       "inherited_policy_dictionaries": [
449:         "final_evaluation_authorization",
450:         "test_condition_inference_policy",
451:         "final_window_policy",
452:         "final96_ablation_policy",
453:         "scoring_and_acceptance_policy",
454:         "independent_recomputation_authorization"
455:       ],
456:       "inheritance_rule": "以上六个原字典完整保留为科学政策基底。仅本对象supersedes中的明确编排覆盖项具有优先效力；未列项目全部沿用原值、公式、分母、容差和失败解释。禁止通过任意字典合并新增科学选择自由度。"
457:     },
458:     "recovery_review_binding": {
459:       "request_id": "pu-inverse-20261111-8bc39ca44eed",
460:       "reviewed_code_sha": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
461:       "reviewed_artifact_sha": "212cb94afe4ac26274588c7a233d7ab4bcae43ad",
462:       "complete_reply_sha256": null,
463:       "binding_rule": "将本完整回复归档后计算其实际SHA，并在C10外部授权回执中填写和验证。null表示尚未产生的真实散列，不是允许运行时保留的占位值。"
464:     },
465:     "supersedes": [
466:       {
467:         "policy_location": "final_window_policy.recovery中关于R9未完成时不得将R10用于新增搜索的分支",
468:         "replacement": "仅针对本次已证实在开窗前终止的R9，允许把尚未发生的唯一最终评价及其原定搜索移入R10同一个合并包。没有第二次最终测试窗口，不适用于已经读取测试结果后的模型或策略修订。"
469:       },
470:       {
471:         "policy_location": "independent_recomputation_authorization.failure_boundary",
472:         "replacement": "本次不再要求R10仅审计R9的四个失败产物；允许先完成尚未发生的最终评价。只有R10评价阶段工程完整并锁定产物后，才能在同一剩余预算内启动独立复算。若R10评价不完整，后续仍只保全和审计已产生证据，不补做新搜索。"
473:       },
474:       {
475:         "policy_location": "final_evaluation_authorization.round及相关C9消费者引用",
476:         "replacement": "最终评价在round=10执行，使用预开窗冻结的实际C10消费者；所有模型继续保留原C6或C8生产者。原R9失败及其4.40927419392392秒保留。"
477:       },
478:       {
479:         "policy_location": "independent_recomputation_authorization.precondition、implementation中的R9输出和C9实现引用",
480:         "replacement": "独立复算消费R10评价阶段的完整不可覆盖产物、实际scientific_inventory SHA和同一final_window；两个入口均属于首次测试读取前冻结的同一C10。独立性由另一个全新Python进程、重新加载权重和重新计算保证。"
```

## configs/inverse/round010_plan.json lines 481-600

```text
481:       },
482:       {
483:         "policy_location": "两个阶段各自maximum_wall_seconds的执行解释",
484:         "replacement": "R10整体最多14400秒，评价与复算共享唯一绝对截止；不是评价14400秒加复算14400秒。"
485:       }
486:     ],
487:     "budget": {
488:       "maximum_wall_seconds_total": 14400,
489:       "maximum_total_rounds": 10,
490:       "maximum_total_compute_seconds": 259200,
491:       "formal_rounds_before_start": 9,
492:       "cumulative_compute_seconds_before_start": 1848.0574975546915,
493:       "clock_rule": "正式外层任务只建立一次单调时钟起点t0。硬截止为t0加14400秒与累计总预算剩余期限中的较早者；可在该期限内部预留10秒收尾。所有子进程只接收这一绝对截止和剩余时间，不重新计时。",
494:       "included_work": [
495:         "实际清单、源码、权重和输入校验",
496:         "全部合成和真实开发输入门禁",
497:         "八模型validation兼容性核对",
498:         "测试条件推理编码",
499:         "实测锚点评分",
500:         "候选搜索和跨种子复核",
501:         "统计与门槛计算",
502:         "独立全量复算及其必要输入校验",
503:         "同一执行内的日志、产物锁和失败收尾"
504:       ],
505:       "free_preflight_allowed": false,
506:       "eleventh_round_allowed": false
507:     },
508:     "training_authorization": {
509:       "status": "PROHIBITED",
510:       "new_supervised_stages": 0,
511:       "optimizer_updates": 0,
512:       "new_distillation": false,
513:       "new_configuration_selection": false,
514:       "model_seeds": [
515:         20261111,
516:         20261112,
517:         20261113
518:       ],
519:       "selected_configuration": "D0",
520:       "model_count": 8
521:     },
522:     "immutable_inputs": {
523:       "success_contract_sha256": "e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef",
524:       "data_lock_sha256": "d6ea2d43d60ba12784e91f03d37515210e300b0144597fa12d0412162b06f52d",
525:       "selection_lock_sha256": "af2022579871ef286c67e6641b22bb97fdf921ff47be94cad372d99a6d5ad41d",
526:       "base_inputs_ready_sha256": "efc85171ce01657bba50e4d362a52707666e6b0197eb462d0876bffb13d88dfd",
527:       "fixed96_requests_sha256": "39c55cbc533acaa51b634fd7c47257f98d6b5720dd6721c850d92003c29815ec",
528:       "main_20261111_checkpoint_manifest_sha256": "0a5d20560f7a11317cf179dbe13ebbd70999380e910aa13b828ef341edcaff78",
529:       "main_20261112_checkpoint_manifest_sha256": "12dc8314ea3d5b63cabf55fe3344a1472ef481830a7e0857c1aabb619a781768",
530:       "main_20261113_checkpoint_manifest_sha256": "1e3fe9edcb6e622ce69d41231baea5a8fd6f5d369b6005aefc8105475c0c4c24",
531:       "no_target_20261111_checkpoint_manifest_sha256": "7c55e1c518161ea060681059f9977054a59eeb25e26bdc39ade916e455bd0553",
532:       "no_target_20261112_checkpoint_manifest_sha256": "cbbe30ba61aca49008d66f94f027dd6acdd4335f4de7e4d39dd83e76de5842a4",
533:       "no_target_20261113_checkpoint_manifest_sha256": "f1374936b4feddfdc68b9ddcef3a0ffce74978dca14fa48f871fe5726272fd1d",
534:       "k1_20261111_checkpoint_manifest_sha256": "3ae66fd98d1950900ec0ebb6614dca692cadbd0e8aebd3cdf9d0eac2c03fc4d8",
535:       "no_test_conditions_20261111_checkpoint_manifest_sha256": "37ef781bae3794ab169ea863f8b0688e196be05c88f3c519eee43441f902c65c",
536:       "identity_rule": "C4/A4保留资格生产者身份；主种子20261111保留C6/A6生产者身份，另外两个主种子及五个逆向消融保留C8/A8生产者身份。C9/A9是本次被评审失败版本，C10只是新的评价消费者。清单中的权重、共享前向生产者、14项资产和内部checkpoint身份全部保持。"
537:     },
538:     "allowed_implementation_changes": [
539:       "兼容原checkpoint清单是否包含bytes，强化实际SHA和大小回执，不改历史清单。",
540:       "增加一个共享绝对截止的合并监督器，顺序启动评价和独立复算两个全新进程。",
541:       "增加本次恢复授权绑定、C10计划及外部源码回执；更新对应CLI和执行文档中的阶段与目录。",
542:       "补充实际清单格式、错误SHA和大小、共享截止、阶段失败、子进程测试隔离和失败审计的必要门禁。",
543:       "修正未进入审计、强制终止和引用窗口状态的回执表达；不改变任何预测或评分。",
544:       "可补充复算列表长度与ID集合的显式覆盖断言，但不得改变原评分分母、数值公式或比较容差。"
545:     ],
546:     "forbidden_changes": [
547:       "修改任何原监督权重、模型架构、优化器训练结果、数据资格、划分或原料库。",
548:       "增加第五配置、重选D0、选择较好种子、重新拟合变换、词表或PLE。",
549:       "改变最终目标、250观察或106状态分母、固定96请求、1440任务矩阵。",
550:       "改变采样、比例细化、检索罚值、排序、去重、每组合上限、512搜索或40复核预算。",
551:       "改变任何原合同门槛、概率下限、源float64窗口、物理容差、统计规则或1e-5复算规则。",
552:       "在首次测试读取后修改代码再继续评价，或者将工程失败改写为科学通过。"
553:     ],
554:     "preopening_freeze": [
555:       "静态准备中一次性完成限定修改，提交实际C10；不预填未来提交SHA。",
556:       "同时冻结评价入口、复算入口、合并监督器、调用依赖、计划和测试代码。",
557:       "生成外部回执，绑定C10、完整源码清单、计划、原R8评审、本次R9评审和全部不可变输入；原R8六个政策字典保留。",
558:       "正式R10启动后验证实际部署HEAD、源码字节、确定性环境、八套模型和14项资产；执行合成门禁及八套模型真实validation核对。",
559:       "全部门禁通过后建立不可覆盖的task_manifest和final_lock，再原子创建唯一final_window。任何门禁失败都消耗最后一轮，不能改称免费预检重跑。"
560:     ],
561:     "execution_sequence": [
562:       {
563:         "phase": "PREOPENING",
564:         "action": "验证限定修复、双重授权、全部实际输入和模型兼容性；测试标签仍封存，正式96性能仍不计算。",
565:         "completion_condition": "所有门禁通过，真实零拟合和隔离证据完整，250观察ID、106状态ID及1440任务清单锁定。"
566:       },
567:       {
568:         "phase": "FINAL_EVALUATION",
569:         "action": "在第一个工作进程中打开唯一窗口，按原规则生成缺失prompt覆盖层，完成全部锚点、主方法、基线、消融和逐门槛报告。",
570:         "completion_condition": "产生完整final_result、全部任务产物及不可覆盖scientific_inventory；科学门槛失败正常记录，不阻止完成其余固定任务。"
571:       },
572:       {
573:         "phase": "EVALUATION_SEAL",
574:         "action": "评价工作进程退出后，由监督器核对退出状态、完整任务集合、科学产物清单及实际SHA，记录剩余时间。",
575:         "completion_condition": "评价工程完整且清单闭合才启动复算；不以numerical_contract_satisfied=true作为启动条件。"
576:       },
577:       {
578:         "phase": "INDEPENDENT_RECOMPUTATION",
579:         "action": "启动不同PID的全新Python解释器，重新加载同一C10、八套模型和冻结资产，消费同一窗口与评价清单，独立重编码并全量重算。",
580:         "completion_condition": "全部锚点、候选池、top20、图、计量、来源、预算、排名、统计和门槛比较完成；新提案数和优化器更新数均为0。"
581:       },
582:       {
583:         "phase": "CLOSURE",
584:         "action": "输出原合同逐项PASSED、FAILED或INCOMPLETE表，归档全部差异、负结果、未完成项及C/A/R绑定。",
585:         "completion_condition": "状态与真实产物一致；无论科学失败、工程失败还是超时，均不启动第11轮，不自动写研究PASS。"
586:       }
587:     ],
588:     "evaluation_and_recomputation_coverage": {
589:       "test_observations": 250,
590:       "test_target_counts": {
591:         "tg": 6,
592:         "tm": 13,
593:         "tensile_strength_ultimate": 110,
594:         "elongation_at_break": 121
595:       },
596:       "joint_two_target_states": 106,
597:       "joint_two_target_materials": 99,
598:       "joint_two_target_closure_groups": 4,
599:       "joint_three_or_four_target_truth_count": 0,
600:       "main_forward_observation_predictions": 750,
```

## configs/inverse/round010_plan.json lines 601-654

```text
601:       "eight_inverse_models_single_target_scores": 2000,
602:       "joint_state_scores_per_inverse_model": 106,
603:       "main_and_baseline_units": 1152,
604:       "ablation_units": 288,
605:       "total_units": 1440,
606:       "fixed_requests": 96,
607:       "request_target_count_partition": [
608:         24,
609:         48,
610:         8,
611:         16
612:       ],
613:       "protocol_partition": {
614:         "A": 60,
615:         "B": 36
616:       },
617:       "search_budget_per_unit": 512,
618:       "independent_recheck_budget_per_unit": 40,
619:       "top_k": 20,
620:       "full_pool_rule": "复算全部原始提案槽位，包括非法、重复和失败槽，不能仅重读汇总或只复算已返回top20。共享池末端排序消融复用主方法原池，不增加新的搜索提案。",
621:       "interpretation": "三代理一致命中属于固定计算筛选证据。三或四性质联合实测仍为0；Tg测试仍只有一个来源闭包，不能生成总体bootstrap区间或把三个种子当作独立实验。"
622:     },
623:     "condition_overlay_rule": "完整继承原test_condition_inference_policy。只使用原固定Qwen权重和既定推理配置，为最终窗口缺失prompt建立独立追加缓存；保持原14资产、词表、PLE、标签变换不变，未知assay使用原0索引。新增缓存和条件张量在窗口内锁定并由复算进程独立重建；不得删除缓存外测试样本，也不得改变prompt规则或重新拟合。",
624:     "recomputation_rule": "完整继承原independent_recomputation_authorization.coverage和comparison_rules。前向预测按abs差除以原delta不超过1e-5；逆向及条件张量使用原F4规范化1e-5规则。观察、状态、协议、mask、原料、源配比窗口、候选次序、合法性、概率触底、逐约束布尔值和最终门槛结论严格一致。",
625:     "failure_and_recovery_boundary": [
626:       "R10采用新且不可覆盖的输出目录，R9四个失败产物和原A9保持不变。",
627:       "本授权不增加外层自动重跑。评价阶段工程失败或超时后，不启动独立复算以补做评价，也不修改代码后重新启动科学计算。",
628:       "复算只消费完整评价产物；复算失败保留实际差异和已完成范围，不能选取较有利输出替换原结果。",
629:       "既有单元幂等逻辑不得改变原候选、顺序或账本。没有完整持久状态的部分搜索仍明确停止，不隐式重采样。",
630:       "科学门槛失败时，在剩余同一预算内继续完成其余原定任务与独立复算；不能调门槛、改分母或换模型。",
631:       "到达绝对截止时终止活动进程组，记录完成任务ID、缺失任务ID、已开窗状态和审计完整性；不能为凑齐结果延长期限。",
632:       "可以只读重取和归档已经产生的证据；不得将这种传输恢复变成第11轮科学计算。"
633:     ],
634:     "feasibility_assessment": {
635:       "evidence_path": "research-runs/pu-inverse-20261111/round-008/evidence-00/search/development_search_summary.json",
636:       "observed_development_comparisons": 96,
637:       "observed_elapsed_seconds": 328.2951095489552,
638:       "simple_linear_estimate_one_phase_seconds": 4924.426643234328,
639:       "simple_linear_estimate_two_phases_seconds": 9848.853286468657,
640:       "judgment": "14400秒可作为合理的有界尝试，但现有证据不足以保证完成。简单线性外推尚未充分计入最终任务组成、新条件编码、八模型锚点、跨种子40复核、完整候选图输出及额外I/O。接近超时不能成为减少候选、删除请求或跳过复算的理由。"
641:     },
642:     "required_final_deliverables": [
643:       "C10和实际A10、两份完整评审正文、冻结合同、data_lock、selection_lock、96面板及八模型生产者的外部绑定。",
644:       "单一外层起点和截止、两个独立工作进程身份、完整计费及停止回执。",
645:       "旧新清单兼容、实际权重SHA和大小、八模型真实validation兼容性及测试隔离门禁。",
646:       "唯一final_window、完整task_manifest、最终输入锁和条件缓存覆盖层。",
647:       "全部逐观察、逐状态、逐候选、逐协议和逐约束结果以及搜索与复核账本。",
648:       "独立复算的逐项差异、完整覆盖分母、排名和布尔一致性、源码及原评价产物未变回执。",
649:       "原合同每项必需门槛的结果、原R8负结果、所有工程错误和准确未完成清单。",
650:       "可实际读取的manifest及必读材料；大文本提供完整原字节读取方式，二进制提供实际校验和可解释的重载回执。"
651:     ]
652:   },
653:   "evaluation_output": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle/evaluation"
654: }
```

## docs/pu_final_evaluation.md lines 1-40

```text
1: # PU 最后一轮评价与独立复算
2: 
3: ## 版本与科学边界
4: 
5: R9 在读取旧 C6 checkpoint 清单时因缺少 `bytes` 失败，未打开测试窗口。原失败 C9/A9 与正式计费保持不变。原 Pro 的 R9 完整回复批准 R10 为唯一最后恢复包；它只覆盖执行编排，原 R8 六个科学政策字典、八套模型生产者、D0、成功合同、划分和固定 96 请求不变。
6: 
7: 实际 C10 提交后才生成外部 `round010_authorization.json` 与源码清单，分别绑定原 R8、本次 R9 的全文散列、计划散列与实际消费者 C10。不得预填未来 A10；结果归档时另建立 C10/A10/R 的绑定。
8: 
9: ## 远程执行
10: 
11: 仅在 cn2 的独立 repository 内、既有 `/opt/anaconda2023/bin/python` 执行。以下命令必须纳入唯一正式 `loopctl run`，不能作为不计费预检或失败后的重跑：
12: 
13: ```bash
14: PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
15: /opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
16:   --final-plan configs/inverse/round010_plan.json --combined-final \
17:   --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle
18: ```
19: 
20: 外层只建立一次起点和 14400 秒硬截止；内部保留 10 秒终止回执时间。评价和独立复算为两个不同 PID 的新 Python 进程，测试进程也继承同一期限。第二阶段只能使用剩余时间；失败、超时均无自动重跑。总预算仍为最多 10 轮或 72 小时，R10 前累计 1848.0574975546915 秒。
21: 
22: ## 开窗前检查与审计
23: 
24: 所有权重始终按原声明 SHA 核验并记录实际大小；C6 无大小声明时 `declared_bytes=null`，仅 C8 已声明时额外比较大小。历史清单和权重不修改。
25: 
26: 正式门禁覆盖实际旧新清单、错误 SHA/大小/缺文件、共享截止、阶段故障及进程终止、唯一输出目录、两份授权与不可变对象。pytest 在自己的进程内拒绝真实封存路径，保存独立 open 和拟合审计，不以父进程补丁冒充子进程保护。
27: 
28: 在真实 validation 集核对全部八套模型：三套主模型前向、八套逆向的 239 观察与 63 双目标状态，逐项比较原 R8 输出。只有实际权重加载、内部身份、数值和工程门禁全部通过，才锁住任务与输入并创建唯一 `final_window.json`。
29: 
30: 审计记录 ENTERED、COMPLETED、FAILED；进程被强杀后未完成的审计按 INCOMPLETE 报告，未进入按 NOT_ENTERED 报告。固定的零字段不能替代真实计数。复算失败必须引用评价目录的窗口状态。
31: 
32: ## 完整评价与复算
33: 
34: 评价输出到 `round_010_final_bundle/evaluation`，覆盖 250 个测试观察、106 个核准双目标状态、96 请求及 1440 比较单元。每单元 512 搜索逻辑预算、最多 40 个其他种子复核，top_k=20；候选先落盘锁定，再做跨种子复核。原始提案池保留非法、重复和失败槽。
35: 
36: 仅使用固定 Qwen 做缺失测试条件的推理编码；不修改基础 14 项资产、PLE、词表或标签变换。未知 assay 使用原索引 0。无新增训练或优化器更新。
37: 
38: 评价工程完整后，监督器独立核验任务全集及科学产物清单，才启动 `recompute` 子目录中的新进程。它重新加载八套模型、重新编码条件、重算所有锚点与完整候选池，并比较图、计量、来源、排名、预算、统计和门槛。科学门槛失败不阻止固定复算；工程不完整时禁止靠复算补做新搜索。
39: 
40: 结果区分工程完成、科学门槛失败、工程未完成。三代理命中属于计算筛选，三/四性质联合实测仍为 0，Tg 测试只有一个来源闭包。只有原合同全部满足、独立复算一致并得到同一 Pro 会话的版本匹配 PASS，才能标记研究完成。
```

## src/coarsepgnn_inverse/__main__.py lines 1-92

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
45:     command.add_argument("--combined-final", action="store_true", help="R10单一共享截止的评价加独立复算")
46:     command.add_argument("--recompute-reference", type=Path)
47:     command.add_argument("--reference-sha256")
48:     args = parser.parse_args()
49:     if args.command == "audit":
50:         result = audit(read_json(args.config), args.output)
51:     elif args.command == "validate-request":
52:         request = validate_request(read_json(args.input))
53:         result = {"status": "valid_request", "constraints": len(request["constraints"]),
54:                   "predictions_computed": False}
55:     elif args.command == "train":
56:         from .training import train
57:         result = train(read_json(args.config), args.output,resume=args.resume,deadline_monotonic=args.deadline_monotonic)
58:     elif args.command == "design":
59:         from .inference import FrozenDesignModels
60:         from .design import run_design
61:         model=FrozenDesignModels(args.model_dir,manifest_sha256=args.manifest_sha256,
62:                                  contract_path=args.contract,data_lock_path=args.data_lock)
63:         result=run_design(read_json(args.input),model,args.output)
64:     elif args.command == "evaluate":
65:         if args.final_plan:
66:             if any((args.config, args.model_dir, args.manifest_sha256, args.readonly_producer)):
67:                 parser.error('最终计划不能混入开发评价参数')
68:             if bool(args.recompute_reference) != bool(args.reference_sha256):
69:                 parser.error('独立复算须同时给原目录和归档清单SHA')
70:             from .final_round import run, run_combined
71:             if args.combined_final:
72:                 if args.recompute_reference or args.reference_sha256:
73:                     parser.error('合并入口自动绑定内部评价清单，不能指定外部复算目录')
74:                 result=run_combined(args.final_plan,args.output)
75:             else:
76:                 result=run(args.final_plan,args.output,'recompute' if args.recompute_reference else 'evaluate',
77:                            reference=args.recompute_reference,reference_sha256=args.reference_sha256)
78:         else:
79:             if not all((args.config,args.model_dir,args.manifest_sha256)) or args.recompute_reference or args.reference_sha256 or args.combined_final:
80:                 parser.error('开发评价要求config/model-dir/manifest-sha256且不接受复算参数')
81:             from .evaluation import evaluate_configuration
82:             result=evaluate_configuration(read_json(args.config),args.model_dir,args.manifest_sha256,args.output,
83:                 readonly_producer=read_json(args.readonly_producer) if args.readonly_producer else None)
84:     else:
85:         result = decode_ratios(**read_json(args.input))
86:         write_json(args.output, result)
87:     print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
88: 
89: 
90: if __name__ == "__main__":
91:     main()
92: 
```

