```json
{
  "type": "REVIEW_CONTROL",
  "protocol_version": 2,
  "run_id": "pu-inverse-20261111",
  "request_id": "pu-inverse-20261111-b50e8c578ea7",
  "round": 10,
  "goal_sha256": "6b7d7be413ada058174c9d6682e4350b746fd181944061764cc9fc3fc80dddd",
  "code_sha": "3d9348a92865528d31a3dc769c93ad0dc7441dae",
  "artifact_sha": "983b0eeae4b62ae4209af89155fdaa8b7ef62150",
  "decision": "BLOCKED",
  "primary_blocker": "原冻结评价已经完成，但71项科学门槛中54项失败，研究成功目标未达到；独立进程完成了锚点和候选逐项比较，却因final_recompute.py遗漏向anchors字典写入结果而未完成独立汇总、门槛比较及成功封存。正式轮次已用尽10/10，本次运行没有获授权的继续计算恢复路径。只能完成现有证据补读、终审归档和未完成状态交付。",
  "next_action": {
    "action": "将本次终审回复按实际完整内容计算散列并归档为独立R记录，绑定本请求C10/A10及原冻结合同；保留原评价的54项失败、复算FAILED状态和八个未生成文件，交付现有代码、冻结权重、逐样本结果、候选原件、运行回执与缺失项清单。只允许读取和整理已有证据，不修改C10，不补算缺失汇总，不重新打开测试，不开展新拟合或新搜索。",
    "done_when": "终审R记录具有真实提交与内容散列并与C10/A10对应；交付索引明确区分原评价完成、逐项复算完成、独立汇总未完成、研究目标未达到；八个真正缺失的复算文件保持缺失并逐项登记；运行以10/10轮和FAILED/未完成状态关闭，不产生第11轮或新的科学结果。该收尾完成不改变本次BLOCKED裁定，也不构成研究PASS。"
  },
  "final_disposition": {
    "goal_achieved": false,
    "research_pass": false,
    "original_evaluation": "COMPLETE_SCIENTIFIC_CRITERIA_NOT_MET",
    "independent_itemwise_verification": "已有八套锚点比较和1440个候选单元逐项比较回执，支持限定范围的一致性结论。",
    "independent_aggregate_verification": "INCOMPLETE_ENGINEERING_FAILURE",
    "formal_execution_status": "FAILED",
    "deliverable_status": "可交付可追溯的研究代码、冻结模型和负结果证据包；不可标为达到原成功目标或完成全量独立核验。",
    "remaining_authorized_scientific_rounds": 0
  },
  "reading_verification": {
    "repository": "WangGroupFDU/CoarsePGNN",
    "artifact_read_revision": "983b0eeae4b62ae4209af89155fdaa8b7ef62150",
    "source_read_revision": "3d9348a92865528d31a3dc769c93ad0dc7441dae",
    "manifest": {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/manifest.json",
      "bytes": 36356198,
      "sha256": "a1cbf5f955291e3b3845f7e0002a4faae181efd41a1d8a4ec8711f54470778ba",
      "file_records": 79324,
      "distinct_file_paths": 79324,
      "required_read_records": 73,
      "git_files_including_manifest": 79325,
      "full_original_json_parsed": true,
      "progress_reference_hashes_match": true
    },
    "required_reads": {
      "read": 73,
      "unread": 0,
      "size_and_sha256_matches": 73,
      "mismatches": []
    },
    "source_export": {
      "changed_or_added_text_files": 11,
      "numbered_export_parts": 6,
      "direct_C10_sources_match_export_hashes_and_numbered_text": true
    },
    "large_originals": {
      "execution_json_bytes": 7893324,
      "experiment_json_bytes": 4131159,
      "anchor_summary_json_bytes": 3136850,
      "panel_rows_json_bytes": 13667230,
      "reading_method": "读取完整原字节并解析JSON；索引、CSV投影和工具截断输出没有替代原件。"
    },
    "remote_identity": {
      "C10_commit_verified": true,
      "A10_commit_verified": true,
      "A10_parent_is_C10": true,
      "C10_tree": "60d7a967e12c8e02d2d913e31e008e3e7c22b5bc",
      "A10_tree": "3280f5f03c0a71c321318b408abd950e81aa245c",
      "current_review_R_commit": null,
      "current_review_R_note": "本回复尚待按实际内容独立归档，不能预先声称已有R10提交或散列。"
    },
    "scope_limitations": [
      "没有声称逐份读取全部79324个产物正文。全量归档回读结论来自已读取的收集回执；本评审独立核验的是本读取清单中的文件。",
      "1440份候选差异回执的完整清点和覆盖信息已读取，并直接读取了三个原始单元差异回执；没有重新执行1440个候选任务。",
      "32个checkpoint的清单、散列验证与装载回执已读，权重二进制本体没有由评审者反序列化或执行，不能标为已解释的模型内容。",
      "两份condition_inputs.pt已取得原字节并核验大小及SHA，但没有反序列化解释张量，以下按unread标注其语义内容。",
      "本次没有训练、模型推理、条件重编码、候选搜索或缺失科学汇总补算。"
    ]
  },
  "immutable_bindings": {
    "success_contract_sha256": "e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef",
    "data_lock_sha256": "d6ea2d43d60ba12784e91f03d37515210e300b0144597fa12d0412162b06f52d",
    "selection_lock_sha256": "af2022579871ef286c67e6641b22bb97fdf921ff47be94cad372d99a6d5ad41d",
    "inputs_ready_sha256": "efc85171ce01657bba50e4d362a52707666e6b0197eb462d0876bffb13d88dfd",
    "fixed_96_requests_sha256": "39c55cbc533acaa51b634fd7c47257f98d6b5720dd6721c850d92003c29815ec",
    "task_manifest_sha256": "2512b7647be0409ced42a7315fcf54091180a88af96da0e5e3287732c99b0bf3",
    "round010_plan_sha256": "b525ab3f732be67ef90b3995ee82a5e2d9579049fa293da3a2218d2d27a19932",
    "producer_separation": {
      "main_20261111_D0": {
        "code_sha": "94f8f785d5c3bfb4197903d5588e7a5490d3e80d",
        "artifact_sha": "cab04775f48555995ed448deffe0e0c1bfa57029"
      },
      "other_two_main_seeds_and_five_inverse_ablations": {
        "code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
        "artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704"
      },
      "consumer_code_sha": "3d9348a92865528d31a3dc769c93ad0dc7441dae"
    },
    "finding": "保留R4数据资格批准和原冻结数值合同。本轮清单、授权、装载及最终锁证据没有支持重选模型、修改数据锁或放宽门槛的行为。C6/C8生产者身份与C10消费者身份保持分离。"
  },
  "progress_assessment": {
    "reported_status": "advanced",
    "reviewed_status": "advanced",
    "accepted": true,
    "finding": "R10首次形成完整冻结测试评价及正式96面板比较，将研究目标是否达到从未知推进为明确未达到；同时形成独立进程的逐项一致性证据，并定位最终汇总失败的具体代码路径。这是实质性进展，包含有价值的负结果，不要求指标上涨。进展不等于研究成功或独立核验闭合。",
    "evidence": [
      {
        "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/acceptance.json",
        "location": "numerical_contract_satisfied、failed_gates、gates",
        "finding": "numerical_contract_satisfied=false；71项中54失败、17通过。"
      },
      {
        "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/failure.json",
        "location": "error、traceback、test_window_exists",
        "finding": "最终汇总阶段KeyError，测试窗口已经存在；不是未开始评价，也不是缺失模型权重。"
      },
      {
        "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/supervisor/round_result.json",
        "location": "coverage、phases、status、audit_states",
        "finding": "两个阶段各有1440个完成任务ID，评价子进程退出0，复算子进程退出1，整体FAILED。"
      }
    ],
    "progress_correction_required": false
  },
  "criteria": [
    {
      "id": "A1",
      "status": "pass",
      "evidence": "在此前已核准的数据资格范围内沿用通过结论。本轮final_lock、checkpoint_verification、授权和条件覆盖层回执均绑定原data_lock与原14项预处理资产；测试人口为250条观察及106个核准共同状态，没有以派生标签补足三/四性质联合实测。此项表示维持原资格批准，不冒充本轮重新逐条审计全部原始数据。"
    },
    {
      "id": "A2",
      "status": "pass",
      "evidence": "C10的configs/inverse/success_contract.json仍为FROZEN、frozen=true，SHA为e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef，绑定原data_lock。完整R8/R9评审与C10计划中的继承政策一致；未发现根据测试结果调整成功目标、容差、分母或选择规则。"
    },
    {
      "id": "A3",
      "status": "fail",
      "evidence": "原评价71项门槛中54项失败。三主种子四性质MAE全部超过原阈值，三种子的平均条件信息优势均为负，主方法Hit@20为29/96、33/96、33/96，低于各种子均36/96的检索基线。独立逐项比较有证据，但独立汇总与门槛比较未执行完毕。"
    },
    {
      "id": "A4",
      "status": "pass",
      "evidence": "限定于继承已核准接口验证和本轮实际输出的工程要求。原料身份、连续计量、NONE处理、逐协议预测、固定top20后独立复核及合成类比来源具有证据；原始和返回化学合法率均为1.0。实际输出保留experimental_validation=false、property_success_probability=null及SUGGESTED_ANALOGUE_ONLY。该通过不表示性质预测可靠，也不表示新配方实验成功。"
    },
    {
      "id": "A5",
      "status": "fail",
      "evidence": "C10/A10、完整manifest及73项必读材料的版本和散列核验通过，正式运行与双进程失败时间线相符；但执行整体FAILED，独立汇总、最终门槛比较及成功封存缺失，原会话不能给研究PASS。本轮没有Zenodo草稿或秘密分享链接需要撤销；当前终审R仍须按实际回复独立归档。"
    }
  ],
  "scientific_assessment": {
    "evidence_basis": [
      "evaluation/acceptance.json",
      "evaluation/anchor_summary.json",
      "evaluation/panel_summary.json",
      "evaluation/panel_rows.json",
      "evaluation/test_population.json",
      "C10:src/coarsepgnn_inverse/final_metrics.py",
      "C10:src/coarsepgnn_inverse/metrics.py"
    ],
    "gate_counts": {
      "total": 71,
      "passed": 17,
      "failed": 54,
      "failed_forward": 32,
      "failed_inverse": 4,
      "failed_panel": 18
    },
    "forward_MAE": {
      "seed_order": [
        20261111,
        20261112,
        20261113
      ],
      "rows": [
        {
          "property": "Tg",
          "unit": "K温差，数值等同于摄氏温差",
          "maximum_allowed": 10,
          "observed": [
            31.65702988645829,
            33.006650873490685,
            34.205616587913596
          ]
        },
        {
          "property": "Tm",
          "unit": "K温差，数值等同于摄氏温差",
          "maximum_allowed": 15,
          "observed": [
            26.384515051333356,
            51.60584957740364,
            43.29867402284046
          ]
        },
        {
          "property": "极限拉伸强度",
          "unit": "MPa",
          "maximum_allowed": 5,
          "observed": [
            8.852243540918847,
            9.04767786936737,
            10.088638412307972
          ]
        },
        {
          "property": "断裂伸长率",
          "unit": "百分点",
          "maximum_allowed": 100,
          "observed": [
            345.1270959965123,
            348.02925086474596,
            342.67830216364683
          ]
        }
      ],
      "finding": "12项MAE门槛全部失败；12项容差内命中率门槛也全部失败。相对中位数基线的平均归一化MAE比值约为0.820149、0.972476、0.949905，只有20261111达到不高于0.9的相对门槛，不能替代绝对误差要求。Tg和Tm的种子稳定性门槛失败，四性质未见规格子集的平均归一化MAE门槛均失败。"
    },
    "inverse_conditional_information": {
      "definition": "配对优势为无目标数值对照NLL减去主逆模型NLL；正值表示主模型在原窗口概率评分下更好。",
      "unit": "nat",
      "minimum_required_per_seed": 0.1,
      "four_task_mean_by_seed": {
        "20261111": -0.0531102772627966,
        "20261112": -3.8012458011733776,
        "20261113": -0.6275117495424638
      },
      "three_seed_mean_by_property": {
        "tg": -6.852747968478864,
        "tm": 0.21071616206836594,
        "tensile_strength_ultimate": 0.5107981915574997,
        "elongation_at_break": 0.1554098442148147
      },
      "finding": "三个种子的主要优势门槛均失败，Tg的跨种子非负优势门槛也失败。Tm、强度和伸长的跨种子平均优势为正，故不能把结果概括成所有任务均无条件信息。八套模型单目标及双目标评分中的概率下限触发计数均为0，现有负结果不能归因于概率下限裁剪。"
    },
    "final_panel": {
      "requests": 96,
      "request_strata": {
        "single_target": 24,
        "two_targets": 48,
        "three_targets": 8,
        "four_targets": 16,
        "protocol_A": 60,
        "protocol_B": 36
      },
      "comparison_units": 1440,
      "comparison_definition": "96请求×3主种子×4方法，加20261111种子的3项消融各96请求。",
      "search_budget": 512,
      "independent_recheck_budget": 40,
      "returned_top_k": 20,
      "hits_by_seed": {
        "inverse": [
          29,
          33,
          33
        ],
        "retrieval": [
          36,
          36,
          36
        ],
        "legal_random": [
          30,
          31,
          36
        ],
        "combination_ratio_search": [
          25,
          27,
          31
        ]
      },
      "main_mean_hit_at_20": 0.3298611111111111,
      "main_mean_margin_over_strongest_baseline": -0.045138888888888895,
      "required_mean_margin": 0.05,
      "main_hits_by_target_count": {
        "20261111": [
          "16/24",
          "12/48",
          "0/8",
          "1/16"
        ],
        "20261112": [
          "15/24",
          "16/48",
          "2/8",
          "0/16"
        ],
        "20261113": [
          "13/24",
          "17/48",
          "1/8",
          "2/16"
        ]
      },
      "chemical_legality": {
        "raw": 1.0,
        "returned": 1.0
      },
      "finding": "三个主种子均未达到总体Hit@20不低于0.6的门槛，也均未达到相对最强基线非负的要求。12项目标数分层门槛中，只有20261112的三目标2/8达到对应阈值。合法性、多样性和命中率种子范围的局部通过不能抵消性质准确性和基线比较失败。"
    },
    "ablation_interpretation": {
      "seed": 20261111,
      "hit_counts_out_of_96": {
        "main": 29,
        "K1": 32,
        "no_test_conditions": 31,
        "last_ranking_only": 14
      },
      "finding": "K=1和无测试条件消融的计算命中数高于主方法，但逆向NLL呈现不同排序，因此没有证据支持K=3或条件输入在所有评价维度上普遍占优。末端排序消融与主方法共享含前向指导细化的提案池，14/96只能用于解释最后排序步骤的贡献，不能解释为完全移除前向搜索的效果。"
    },
    "denominators_and_statistics": {
      "test_population": {
        "tg": {
          "observations": 6,
          "materials": 3,
          "closures": 1
        },
        "tm": {
          "observations": 13,
          "materials": 11,
          "closures": 3
        },
        "tensile_strength_ultimate": {
          "observations": 110,
          "materials": 99,
          "closures": 4
        },
        "elongation_at_break": {
          "observations": 121,
          "materials": 114,
          "closures": 9
        }
      },
      "joint_population": {
        "states": 106,
        "materials": 99,
        "closures": 4,
        "property_pair": "极限拉伸强度与断裂伸长率",
        "three_or_four_property_measured_joint_states": 0
      },
      "assessment": [
        "原实现按材料等权汇总，并按来源闭包处理不确定性，避免把同材料多观察或同来源记录直接当作独立样本。",
        "Tg只有一个测试闭包，其来源层面置信区间为null；Tm和强度的小闭包数只能支持有限的留一闭包敏感性分析。不能用三种随机种子替代独立材料或独立实验重复。",
        "106个共同状态全部属于强度—伸长双目标，不能解释成覆盖全部六种性质对，更不能作为三/四目标共同实测验证。",
        "正式96面板的命中是冻结代理模型与搜索规则下的计算结果，不是实验成功率。三个前向种子的独立复核增强了计算一致性要求，但这些前向模型本身未达到原误差门槛。",
        "这些统计边界属于原合同已经接受的有限结论范围。本次不以此重开资格审计、不删测试样本，也不改变门槛。"
      ]
    },
    "supported_conclusion": "该固定数据、模型、配置选择和搜索预算下，系统能够生成并核验合法候选，但未达到四性质预测、逆向条件信息优势和正式候选命中的预定成功要求。",
    "unsupported_conclusions": [
      "研究目标已达到或研究PASS。",
      "化学合法率1.0意味着性质预测可靠。",
      "代理模型命中意味着新配方实验成功。",
      "三/四性质联合实测性能已经验证。",
      "补上anchors赋值就能使科学门槛通过。",
      "本次负结果证明所有PU逆向学习方法或所有目标条件信息都无效。"
    ]
  },
  "engineering_assessment": {
    "preopening": {
      "tests": 26,
      "passed": 26,
      "failed": 0,
      "sealed_file_access_rejection_observed": true,
      "validation_compatibility_models": 8,
      "validation_observations_per_model": 239,
      "validation_joint_states_per_model": 63,
      "logical_test_window_openings": 1,
      "window_id": "77fecfb8e7479db78f101cb13add2534b994959934ca2835b3d7e2186216442c",
      "finding": "真实子进程封存文件拒绝成立；八套模型的既有验证输出兼容性通过。一个逻辑最终窗口不等于只有一次底层文件open，评价与授权复算使用同一窗口身份。"
    },
    "condition_inference": {
      "new_prompts": 25,
      "encoding_calls_per_phase": 1,
      "encoding_batches_per_phase": 7,
      "batch_size": 4,
      "base_asset_count": 14,
      "base_assets_unchanged": true,
      "independent_reencoded_condition_differences": 0,
      "finding": "新增测试协议使用冻结通用Qwen进行授权推理编码，保存在独立覆盖层；原词表、PLE、标签变换和原资产未重新拟合。条件推理与监督训练在审计中分开。"
    },
    "actual_independent_scope": {
      "evaluation_pid": 267376,
      "recompute_pid": 275752,
      "forward_anchor_predictions_compared": 750,
      "inverse_single_target_requests_compared": 2000,
      "inverse_joint_states_compared": 848,
      "candidate_units_with_existing_receipts": 1440,
      "raw_proposal_slots_in_receipt_inventory": 709344,
      "top20_comparison_entries_in_receipt_inventory": 28800,
      "finding": "锚点差异原件记录数值差异为0；候选清点与监督器覆盖记录支持1440个单元逐项完成，已直接核读的原始候选差异回执与该范围一致。709344是提案槽数，28800是top20比较条目数，均不是独立材料数或实验次数。"
    },
    "failure": {
      "error": "KeyError('main_20261111_D0')",
      "source_locations": [
        "src/coarsepgnn_inverse/final_recompute.py:147",
        "src/coarsepgnn_inverse/final_recompute.py:148-163",
        "src/coarsepgnn_inverse/final_recompute.py:194",
        "src/coarsepgnn_inverse/final_metrics.py:63"
      ],
      "cause": "anchors被初始化为空字典；逐模型循环保存result和差异，但未执行anchors[name] = result。调用anchor_report时索引该空字典而失败。",
      "control_comparison": "原评价入口final_evaluation.py:202有正确赋值，因此原评价汇总已产生。",
      "historical_review_gap": "同类遗漏已存在于C9的final_recompute.py相同循环，前次静态审查没有识别这一阻断。26项预开窗检查没有覆盖完整独立汇总成功路径。",
      "not_supported_as_cause": [
        "模型权重缺失",
        "未执行任何独立候选比较",
        "科学门槛失败导致Python异常"
      ],
      "unexecuted_after_failure": [
        "独立anchor_summary、panel_rows、panel_summary和acceptance的重建及比较",
        "成功路径末端的输入、源码及原评价封存清单复核",
        "zero_fit_audit成功收尾文件",
        "candidate_differences_index总索引",
        "recompute_result成功结果",
        "复算scientific_inventory成功封存"
      ]
    },
    "audits": {
      "evaluation_status": "COMPLETED",
      "recompute_status": "FAILED",
      "observed_supervised_fit_calls": 0,
      "observed_backward_calls": 0,
      "observed_optimizer_step_calls": 0,
      "supervisor_optimizer_updates": null,
      "finding": "FAILED审计保留了不同PID、进入和结束时间及实际零计数，支持本轮没有监督更新。监督器总计保持null并引用阶段审计是正确做法；不能把缺失的成功zero_fit_audit伪装成已生成。"
    },
    "execution_and_budget": {
      "provenance_valid": true,
      "formal_elapsed_seconds": 13329.730552949943,
      "supervisor_internal_elapsed_seconds": 13316.369718706002,
      "cumulative_compute_seconds": 15177.788050504634,
      "maximum_compute_seconds": 259200,
      "formal_rounds_used": 10,
      "maximum_rounds": 10,
      "declared_output_paths": 21682,
      "existing_declared_outputs": 21673,
      "missing_declared_outputs": 9,
      "finding": "两阶段共用原绝对截止，正式外层墙钟与监督器内部墙钟口径不同，不应混用。总秒数尚有余量，但轮次上限已经用尽，不能据剩余秒数授权第11轮。九个声明缺失项中evaluation/failure.json在评价成功时应当不存在，其余八项属于真正未完成的复算收尾。"
    },
    "archive_assessment": "现有收集回执支持评价38933项、复算40350项的独立远端大小/SHA清单及本地回读；相同字节文件的传输缓存复用没有证据表明替代了远端独立计算。ARG_MAX路径清单修复属于归档工具操作，不改变C10科学执行。两份condition_inputs.pt在固定A10中实际存在且原字节SHA匹配。归档完整性不能填补运行时未生成的科学汇总。"
  },
  "required_fixes": [
    {
      "id": "RF1",
      "kind": "unmet_scientific_requirements",
      "status": "UNRESOLVED_NO_FURTHER_SCIENTIFIC_EXECUTION_AUTHORIZED",
      "required_action": "以原冻结71项判据交付研究未达标结论，保留以下54项失败及其原始值、阈值和comparison_epsilon。不得降低门槛、重选种子或模型、删除分层、以合法性替代性质准确性，亦不得把复算工程错误解释为科学失败的替代原因。",
      "evidence": "evaluation/acceptance.json的gates与failed_gates；原始anchor_summary和panel_summary；C10 final_metrics.py:149-196。",
      "failed_gate_ids": [
        "forward/20261111/tg/MAE_over_delta",
        "forward/20261111/tg/hit_delta",
        "forward/20261111/tm/MAE_over_delta",
        "forward/20261111/tm/hit_delta",
        "forward/20261111/tensile_strength_ultimate/MAE_over_delta",
        "forward/20261111/tensile_strength_ultimate/hit_delta",
        "forward/20261111/elongation_at_break/MAE_over_delta",
        "forward/20261111/elongation_at_break/hit_delta",
        "inverse/20261111/paired_delta_nat",
        "panel/20261111/hit_at_20",
        "panel/20261111/targets_1",
        "panel/20261111/targets_2",
        "panel/20261111/targets_3",
        "panel/20261111/targets_4",
        "panel/20261111/strongest_baseline_margin",
        "forward/20261112/tg/MAE_over_delta",
        "forward/20261112/tg/hit_delta",
        "forward/20261112/tm/MAE_over_delta",
        "forward/20261112/tm/hit_delta",
        "forward/20261112/tensile_strength_ultimate/MAE_over_delta",
        "forward/20261112/tensile_strength_ultimate/hit_delta",
        "forward/20261112/elongation_at_break/MAE_over_delta",
        "forward/20261112/elongation_at_break/hit_delta",
        "forward/20261112/baseline_ratio",
        "inverse/20261112/paired_delta_nat",
        "panel/20261112/hit_at_20",
        "panel/20261112/targets_1",
        "panel/20261112/targets_2",
        "panel/20261112/targets_4",
        "panel/20261112/strongest_baseline_margin",
        "forward/20261113/tg/MAE_over_delta",
        "forward/20261113/tg/hit_delta",
        "forward/20261113/tm/MAE_over_delta",
        "forward/20261113/tm/hit_delta",
        "forward/20261113/tensile_strength_ultimate/MAE_over_delta",
        "forward/20261113/tensile_strength_ultimate/hit_delta",
        "forward/20261113/elongation_at_break/MAE_over_delta",
        "forward/20261113/elongation_at_break/hit_delta",
        "forward/20261113/baseline_ratio",
        "inverse/20261113/paired_delta_nat",
        "panel/20261113/hit_at_20",
        "panel/20261113/targets_1",
        "panel/20261113/targets_2",
        "panel/20261113/targets_3",
        "panel/20261113/targets_4",
        "panel/20261113/strongest_baseline_margin",
        "forward/tg/seed_range",
        "forward/tg/unseen_mean_MAE_over_delta",
        "inverse/tg/mean_paired_delta_nat",
        "forward/tm/seed_range",
        "forward/tm/unseen_mean_MAE_over_delta",
        "forward/tensile_strength_ultimate/unseen_mean_MAE_over_delta",
        "forward/elongation_at_break/unseen_mean_MAE_over_delta",
        "panel/mean_strongest_baseline_margin"
      ]
    },
    {
      "id": "RF2",
      "kind": "unresolved_engineering_defect_and_verification_gap",
      "status": "UNRESOLVED_NO_REPAIR_OR_REEXECUTION_AUTHORIZED",
      "required_action": "将final_recompute.py:148-163遗漏anchors赋值登记为未解决缺陷，并登记其在C9已存在、完整汇总成功路径未被预开窗检查覆盖。交付中明确recompute/anchor_summary.json、panel_rows.json、panel_summary.json、acceptance.json、zero_fit_audit.json、scientific_inventory.json、recompute_result.json和candidate_differences_index.json均未生成。不得修改C10后继续运行，不得用evaluation文件、静态推断或existing_unit_receipts_inventory替代这些原始缺失文件。",
      "evidence": "recompute/failure.json回溯；C10 final_recompute.py:147-211；final_metrics.py:63；完整manifest；experiment.json声明输出与execution.json实际输出。",
      "scientific_pass_can_be_inferred_from_itemwise_agreement": false
    },
    {
      "id": "RF3",
      "kind": "terminal_review_archival_and_delivery",
      "status": "PENDING_CURRENT_REVIEW_ARCHIVAL",
      "required_action": "按本回复实际完整内容归档终审R记录并计算真实SHA，绑定C10=3d9348a92865528d31a3dc769c93ad0dc7441dae、A10=983b0eeae4b62ae4209af89155fdaa8b7ef62150及原合同SHA；保持冻结A10不变。交付同时保留原评价完成、复算逐项完成、独立汇总失败、研究目标未达到四个不同状态，并明确Tg单闭包、三/四性质共同实测为0、计算命中不等于实验成功。最终运行状态不得改为COMPLETED或PASS。",
      "evidence": "当前终审回复尚无已生成R提交；supervisor/round_result.json为FAILED；evaluation/acceptance.json为numerical_contract_satisfied=false；review_materials/DELIVERY_GUIDE.md已规定未完成交付边界。"
    }
  ],
  "missing_evidence": [
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/anchor_summary.json",
      "reason": "独立锚点汇总未生成；在anchor_report入口即发生KeyError。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/panel_rows.json",
      "reason": "独立面板行汇总的成功输出未生成，不能用原评价panel_rows替代。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/panel_summary.json",
      "reason": "独立面板统计汇总及与原评价的比较未完成。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/acceptance.json",
      "reason": "独立71项门槛重建与一致性比较未完成。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/zero_fit_audit.json",
      "reason": "成功收尾文件未生成；已有FAILED阶段审计中的实际零计数仍有效，但不是该成功文件。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/scientific_inventory.json",
      "reason": "复算成功路径的科学封存清单未生成。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/recompute_result.json",
      "reason": "独立复算最终成功结果未生成。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/candidate_differences_index.json",
      "reason": "运行时成功收尾总索引未生成；事后已有回执文件清点不能冒充该原始输出。"
    }
  ],
  "expected_absence": {
    "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/failure.json",
    "finding": "评价子进程成功结束，因此该失败文件不存在符合控制流，不作为证据缺口。"
  },
  "round10_closure_authorization": {
    "authorization": "EXISTING_EVIDENCE_READING_ARCHIVAL_AND_INCOMPLETE_DELIVERY_ONLY",
    "additional_formal_rounds": 0,
    "additional_supervised_updates": 0,
    "additional_search_runs": 0,
    "additional_test_windows": 0,
    "missing_scientific_aggregate_recomputation_allowed": false,
    "source_repair_and_resume_allowed": false,
    "allowed_actions": [
      "继续读取已经存在且版本固定的证据，补全真实读取清单。",
      "核对、整理和归档已有文件及当前终审回复，保留原字节和真实散列。",
      "交付失败与未完成状态、已完成范围、八项缺失文件和全部科学负结果。",
      "对交付说明作不改变原研究结果的勘误，并明确其独立版本。"
    ],
    "forbidden_actions": [
      "第11轮或任何改名、拆分、免费预检形式的新增科学执行。",
      "修改C10后补算汇总、重新运行候选比较或重新编码测试条件。",
      "新增训练、重选配置、重选种子、修改权重或搜索策略。",
      "重新打开测试窗口或将测试用于新的迭代选择。",
      "修改成功合同、降低阈值、删除失败样本或分层。",
      "把原评价文件或事后清点文件写成缺失的独立复算原件。",
      "把本次BLOCKED终审或归档完成解释为研究PASS。"
    ],
    "recovery_boundary": "证据交付层面的缺口可以通过已有材料补读和真实R归档关闭；科学目标失败与独立汇总未闭合在本运行内没有获授权的计算恢复路径。"
  },
  "read_files": [
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/manifest.json",
      "status": "read",
      "evidence": "完整36356198字节JSON，SHA匹配a1cbf5f955291e3b3845f7e0002a4faae181efd41a1d8a4ec8711f54470778ba；79324个唯一文件记录、73项required_read；progress三项引用散列匹配，八项缺失复算文件未列入实际产物。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/acceptance.json",
      "status": "read",
      "evidence": "numerical_contract_satisfied=false、research_pass=false；gates共71项，54 FAILED、17 PASSED；failed_gates与逐门槛记录一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/checkpoint_verification.json",
      "status": "read",
      "evidence": "八个模型键下记录生产者、原manifest SHA、基础资产和32个checkpoint核验；C6未声明bytes的历史格式仍有实际大小和强制SHA核验。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/k1_20261111.json",
      "status": "read",
      "evidence": "inverse比较5258个叶项、joint比较1989个叶项；maximum_relative_error均0，status均PASSED。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/main_20261111_D0.json",
      "status": "read",
      "evidence": "forward逐观察误差均0；inverse/joint比较5258/1989个叶项，最大相对差0，支持C6生产权重在C10的只读兼容。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/main_20261112.json",
      "status": "read",
      "evidence": "forward逐观察误差均0；inverse/joint最大相对差0并通过，与既有验证输出兼容。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/main_20261113.json",
      "status": "read",
      "evidence": "forward逐观察误差均0；inverse/joint最大相对差0并通过，与既有验证输出兼容。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/no_target_values_20261111.json",
      "status": "read",
      "evidence": "inverse与joint两块均PASSED，比较叶项5258/1989，最大相对差0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/no_target_values_20261112.json",
      "status": "read",
      "evidence": "inverse与joint两块均PASSED，比较叶项5258/1989，最大相对差0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/no_target_values_20261113.json",
      "status": "read",
      "evidence": "inverse与joint两块均PASSED，比较叶项5258/1989，最大相对差0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/compatibility/no_test_conditions_20261111.json",
      "status": "read",
      "evidence": "inverse与joint两块均PASSED，比较叶项5258/1989，最大相对差0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/condition_overlay/inference_audit.json",
      "status": "read",
      "evidence": "PID267376；text_inference_authorized=true、status=COMPLETED；监督fit、backward及optimizer step计数均0，起止时间独立记录。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/condition_overlay/receipt.json",
      "status": "read",
      "evidence": "missing_prompts为25，encoding_calls=1、encoding_batches=7、batch_size=4；base_asset_count=14且base_assets_unchanged=true；新向量与条件张量有独立SHA。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/evaluation_audit.json",
      "status": "read",
      "evidence": "PID267376、status=COMPLETED；评价区间的fit_calls及update_and_backward_calls均0，13:41:33至15:32:13有真实进入/结束记录。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/final_lock.json",
      "status": "read",
      "evidence": "FINAL_EVALUATION_LOCKED；绑定C10、原合同/数据/模型与任务清单，test_opening_count=1；文件SHA为be6065b358e17248831e47d5705066c649e2a9417d1d26ad907c9276e18a54ab。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/final_result.json",
      "status": "read",
      "evidence": "FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS；250观察、106共同状态、1440单元；independent_recomputation=PENDING是评价结束时快照，不能覆盖后续复算FAILED。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/final_window.json",
      "status": "read",
      "evidence": "实际仓库路径不含反斜杠；13:41:25.058223Z打开唯一逻辑窗口，window_id与最终锁一致，opening_count为1。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/model_load_receipt.json",
      "status": "read",
      "evidence": "八套模型的producer_identity与runtime记录分开；保留C6/C8生产身份，C10执行确定性设置及装载回执，不把消费版本写成生产版本。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/panel_summary.json",
      "status": "read",
      "evidence": "完整原件；主方法三种子命中29/96、33/96、33/96，retrieval均36/96；目标数/协议等分层、合法性、多样性及消融结果保留。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/preopening_audit.json",
      "status": "read",
      "evidence": "status=COMPLETED、PID267376；监督拟合、编码和更新计数均0；结束时间早于final_window的开窗时间。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/preopening_gate.json",
      "status": "read",
      "evidence": "status=PASSED；validation下八模型各239观察、63共同状态通过；包含调用审计、封存访问和固定task_manifest_sha256。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/preopening_tests.log",
      "status": "read",
      "evidence": "真实pytest日志为26 passed，约4.33秒；没有将测试通过解释为完整科学评价或独立汇总通过。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/test_gate/fit_audit.json",
      "status": "read",
      "evidence": "独立pytest子进程PID267493；status=COMPLETED，监督fit、编码、backward和optimizer step均0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/test_gate/result.json",
      "status": "read",
      "evidence": "sealed_access_rejected=true、pytest_exit_code=0、status=PASSED；子进程使用共享worker deadline。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/test_gate/test_access.jsonl",
      "status": "read",
      "evidence": "真实封存测试路径访问尝试被拒绝，allowed=false且window_id=null；是实际open拒绝证据。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/test_gate/tests.xml",
      "status": "read",
      "evidence": "testsuite记录26项，failures=0、errors=0、skipped=0；与preopening_tests.log相符。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/test_population.json",
      "status": "read",
      "evidence": "四任务观察数6/13/110/121，材料数3/11/99/114，闭包数1/3/4/9；106共同状态对应99材料、4闭包，均为强度—伸长组合。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/worker_fit_audit.json",
      "status": "read",
      "evidence": "评价worker全程PID267376、status=COMPLETED；允许单独授权文本推理，监督fit及更新调用均0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/anchor_differences.json",
      "status": "read",
      "evidence": "八模型原始逐项差异；三主模型各250个forward预测差0，八模型各250单目标及106共同状态比较通过、最大相对差0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/checkpoint_verification.json",
      "status": "read",
      "evidence": "八模型32个checkpoint与基础资产验证回执，SHA与评价阶段同名回执相同；该文本一致不能替代权重二进制语义检查。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/condition_overlay/inference_audit.json",
      "status": "read",
      "evidence": "新进程PID275752，条件推理阶段COMPLETED；监督fit、backward和optimizer step计数0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/condition_overlay/receipt.json",
      "status": "read",
      "evidence": "独立重编码25个prompt，原14资产不变；independent_comparison中向量及条件张量差为0，新条件文件SHA与评价阶段相同。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/failure.json",
      "status": "read",
      "evidence": "17:22:52.805067Z记录KeyError('main_20261111_D0')；回溯为final_recompute.py:194至final_metrics.py:63，test_window_exists=true。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/preopening_audit.json",
      "status": "read",
      "evidence": "复算进程的前置审计已完成，具有独立PID和时间；拟合及更新计数0，不代表后续整体复算成功。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/recompute_audit.json",
      "status": "read",
      "evidence": "PID275752、status=FAILED；15:33:07.978165至17:22:52.802345的实际审计保留，fit_calls及update_and_backward_calls均0。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/worker_fit_audit.json",
      "status": "read",
      "evidence": "复算worker最终FAILED并在finally保留实际计数和结束时间；监督更新为0，没有改成成功审计。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/supervisor/evaluation_seal.json",
      "status": "read",
      "evidence": "独立进程启动前封存原评价，scientific inventory SHA为918a11cf3474946d8b6cfedaed33825282600a6439ddb7d49fb3bbe1f3d77075；不等于复算成功封存。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/supervisor/round_result.json",
      "status": "read",
      "evidence": "status=FAILED、research_pass=false；评价PID267376退出0，复算PID275752退出1；coverage两阶段各1440完成ID；optimizer_updates=null并引用真实审计。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/supervisor/round_start.json",
      "status": "read",
      "evidence": "记录单个正式组合任务的单调时钟起点与绝对截止；worker截止比hard deadline提前10秒，没有给两个阶段分别14400秒。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/supervisor/supervisor_events.jsonl",
      "status": "read",
      "evidence": "事件顺序为评价启动与结束、原评价封存、另一PID复算启动及失败；与最终round_result的两阶段状态一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/collection/evaluation_local_readback.json",
      "status": "read",
      "evidence": "记录38933个评价文件、5629602595字节及all_match=true；scientific_computation=false，属于已有文件收集核验。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/collection/recompute_local_readback.json",
      "status": "read",
      "evidence": "记录40350个复算文件、5789167191字节及all_match=true；传输缓存复用以远端大小/SHA匹配为前提，不是补做科学复算。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/collection/supervisor_local_readback.json",
      "status": "read",
      "evidence": "四个监督器原件的本地大小/SHA回读通过，保留FAILED时间线和评价封存回执。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_export_manifest.json",
      "status": "read",
      "evidence": "11个新增/修改文本的原路径、SHA、大小和导出分片映射；与固定C10直接读取源码逐项对应。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_01.md",
      "status": "read",
      "evidence": "round010_plan.json原行1-360；包含固定模型、资产和继承评价政策；编号文本与C10原文一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_02.md",
      "status": "read",
      "evidence": "计划原行361-654、最终评价文档1-40及CLI 1-92；共享截止、唯一窗口和两阶段入口可追溯。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_03.md",
      "status": "read",
      "evidence": "final_authorization、final_common、final_conditions全文；核对授权继承、旧清单可选bytes和固定条件推理覆盖层。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_04.md",
      "status": "read",
      "evidence": "final_evaluation及final_gate全文；评价入口202行正确写入anchors，真实测试子进程有独立拒绝守卫。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_05.md",
      "status": "read",
      "evidence": "final_recompute全文及final_round前120行；147-163行空anchors未回填，194行触发汇总失败。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/C10_source_part_06.md",
      "status": "read",
      "evidence": "final_round剩余部分和恢复测试全文；覆盖共享截止、失败保留及子进程守卫，没有覆盖完整汇总成功路径。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/DELIVERY_GUIDE.md",
      "status": "read",
      "evidence": "交付说明区分研究负结果、工程未完成及可用模型资产，不把已有逐项一致性写成研究PASS。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/archive_pathspec_repair.diff",
      "status": "read",
      "evidence": "归档路径参数改用NUL清单输入，针对ARG_MAX；未修改PU训练、评价、指标或模型源码。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/archive_tool_repair_receipt.json",
      "status": "read",
      "evidence": "记录科学运行后的归档工具修复和测试范围；不能解释成新增科研执行或复算成功回执。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/archive_tool_test_loopctl.log",
      "status": "read",
      "evidence": "归档工具52项测试通过；测试对象属于控制工具，不是PU科学效果测试。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/archive_tool_test_oversized_pathspec.log",
      "status": "read",
      "evidence": "12006路径压力检查通过，证明大路径集合可经清单传递；没有生成缺失科学汇总。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/archive_tool_test_oversized_pathspec.py",
      "status": "read",
      "evidence": "合成路径压力测试实现与NUL路径清单修复一致，验证对象为归档行为。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/candidate_example_index.json",
      "status": "read",
      "evidence": "提供FINAL_001首个单目标和FINAL_081首个四目标原件路径；已进一步读取实际design_result和差异回执。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/evaluation_gate_table.csv",
      "status": "read",
      "evidence": "原acceptance门槛字段投影，71项、54失败/17通过；只作为阅读辅助，原acceptance已全文读取。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/evaluation_summary_reading_index.json",
      "status": "read",
      "evidence": "projection_rule明确仅提取已有汇总字段；source_files绑定acceptance、anchor_summary、panel_summary和final_result原件大小/SHA，原件已补读。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/existing_unit_receipts_inventory.json",
      "status": "read",
      "evidence": "unit_receipts=1440、missing_task_ids为空、raw_slot_entries=709344、top20_entries=28800；scientific_summary_recomputed=false、original_recompute_result_exists=false。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/final_forward_metrics.csv",
      "status": "read",
      "evidence": "三种子四任务MAE的原字段导出；所有12项超过冻结绝对阈值，与原anchor_summary一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/final_panel_metrics.csv",
      "status": "read",
      "evidence": "主方法命中29/96、33/96、33/96，检索均36/96；消融与多样性字段来自原panel_summary。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/model_delivery_index.json",
      "status": "read",
      "evidence": "八套模型、32个checkpoint的交付路径及原C6/C8生产身份；是模型资产索引，不是权重语义读取。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/model_delivery_readback.json",
      "status": "read",
      "evidence": "模型交付资产大小/SHA核验通过；model_loading_or_inference=false，不能把该回读视为新增模型验证。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/round010_authorization.json",
      "status": "read",
      "evidence": "绑定R9恢复裁定及原R8政策，单个最后正式轮次、共享14400秒、禁止新增拟合和测试后改实现。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/round010_deployment_receipt.json",
      "status": "read",
      "evidence": "部署命令从旧C9干净状态快进到C10，回读161个源码散列并确认输出目录未存在；实际部署HEAD为本请求C10。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/round010_failure_diagnosis.json",
      "status": "read",
      "evidence": "cause_supported_by_frozen_source、coverage、missing_declared_outputs与原错误和源码一致；明确9个声明缺失项中1个是成功评价不应生成的failure文件。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/round010_resource_check.json",
      "status": "read",
      "evidence": "属于部署前资源观测，HEAD仍为C9；不能冒充C10运行身份，C10身份由后续部署回执和源码核验支持。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/review_materials/round010_source_manifest.json",
      "status": "read",
      "evidence": "161项源码绑定，包括150项既有实现与11项本轮新增/修改文本；11项导出已与直接C10读取对应。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/report.md",
      "status": "read",
      "evidence": "保留科学门槛失败和复算末端错误；没有把FAILED改成COMPLETED，也没有声称全量独立汇总通过。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/execution.json",
      "status": "read",
      "evidence": "完整7893324字节原件；status=failed、provenance_valid=true、正式墙钟13329.730552949943秒、21673项实际输出，消费者C10一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/experiment.json",
      "status": "read",
      "evidence": "完整4131159字节原件；21682项声明输出与实际输出核对，9项未生成；未以阅读索引替代全文。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/goal.md",
      "status": "read",
      "evidence": "目标仍限四性质、既有实测数据与计算验证，要求原合同独立核准；预算不足或失败不能冒充完成。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/checks/check-000.log",
      "status": "read",
      "evidence": "外层真实执行记录与复算子进程失败一致；没有最终研究成功标志。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/anchor_summary.json",
      "status": "read",
      "evidence": "完整3136850字节原件，SHA22084ce4f5e09de02b4f125fb15d40a8f421ca7c36021fa1a8a84bbd9b56b0d8；直接读取四性质误差、配对NLL、闭包统计和概率下限计数。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/panel_rows.json",
      "status": "read",
      "evidence": "完整13667230字节原件，SHA151a9f2b59f3f0db66196632f39afe97da1a20eb1e6d846d0ac6b67a5f961f4d；1440行完整解析，保留任务、方法、种子、命中和分层字段。"
    },
    {
      "path": "configs/inverse/round010_plan.json",
      "status": "read",
      "evidence": "C10全文654行；六组R8政策和R9恢复授权与完整原评审JSON一致，固定八套模型、任务范围及共享截止。"
    },
    {
      "path": "docs/pu_final_evaluation.md",
      "status": "read",
      "evidence": "C10第5-40行说明生产者/消费者、唯一逻辑窗口、评价与独立复算以及共享时间限制。"
    },
    {
      "path": "src/coarsepgnn_inverse/__main__.py",
      "status": "read",
      "evidence": "C10第64-83行接入最终组合任务与对应入口，未增加训练授权。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_authorization.py",
      "status": "read",
      "evidence": "C10第20-46行核对双评审散列、继承政策与八模型身份，防止用消费版本替代生产版本。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_common.py",
      "status": "read",
      "evidence": "C10第66-96行finally审计；100-116行真实open守卫；128-178行兼容旧manifest可选bytes但强制SHA；191-238行人口和窗口绑定。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_conditions.py",
      "status": "read",
      "evidence": "C10第31-41行固定Qwen身份，90-120行新prompt推理，123-152行独立比较，153-163行核验原资产未变。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_evaluation.py",
      "status": "read",
      "evidence": "C10第169-195行按门禁、兼容、锁定、开窗顺序执行；202行正确写入anchors；223-239行生成原评价汇总与结果。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_gate.py",
      "status": "read",
      "evidence": "C10第31-44行真实子进程封存访问拒绝及pytest；49-51行finally保留状态。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_recompute.py",
      "status": "read",
      "evidence": "C10第147行anchors={}；148-163行未回填；167-192行完成候选逐项比较；194行失败使195-211行汇总、末端复核和成功输出未执行。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_round.py",
      "status": "read",
      "evidence": "C10第87-157行两个worker共用截止并保留FAILED；160-175行封存评价；216-228行传播复算失败，223行成功封存未执行。"
    },
    {
      "path": "tests/test_inverse_final_recovery.py",
      "status": "read",
      "evidence": "C10第19-129行测试旧/新清单、错误SHA/大小、授权、截止、失败审计及真实子进程守卫；缺少完整独立汇总成功路径测试。"
    },
    {
      "path": "configs/inverse/success_contract.json",
      "status": "read",
      "evidence": "C10完整冻结合同；FROZEN、frozen=true、原data_lock及SHA e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef保持不变。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_metrics.py",
      "status": "read",
      "evidence": "C10第43-69行锚点汇总，63行索引空anchors触发错误；87-146行面板汇总；149-196行原71项门槛检查。"
    },
    {
      "path": "src/coarsepgnn_inverse/metrics.py",
      "status": "read",
      "evidence": "C10第13-65行材料等权和中位数基线；68-131行源精度窗口概率及概率下限；134-166行闭包统计，G=1不伪造置信区间。"
    },
    {
      "path": "src/coarsepgnn_inverse/anchor_evaluation.py",
      "status": "read",
      "evidence": "C10全文核对原图输入、float64源v_reference、单目标/共同状态身份和配对评分；未用float32训练张量充当源精度锚点。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_design.py",
      "status": "read",
      "evidence": "C10第31-55行共享池末端排序；58-116行先锁top20再独立复核；185-218行三代理、512+40预算和实验未验证声明。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_compare.py",
      "status": "read",
      "evidence": "C10第6-7行离散身份精确字段，19-48行结构/数值比较，51-69行物理单位误差约束；局部比较不包含后续汇总成功证明。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_recompute.py",
      "revision": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
      "status": "read",
      "evidence": "固定C9第147-158行同样初始化空anchors但未在循环内回填；证明遗漏先于C10，前次静态评审未识别该路径。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-5f082242266a.md",
      "status": "read",
      "evidence": "完整R8评审80670字节，SHA950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2；六组完整政策与C10计划中的继承字典一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-8bc39ca44eed.md",
      "status": "read",
      "evidence": "完整R9评审48265字节，SHA15c3f89f97d9fbfa8fc69fcc3de3538e8bbbb05e4d944a42e596608b6f8dda43；最后一轮恢复授权与C10计划对应，未提供第11轮权限。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/units/inverse_20261111_FINAL_001/design_result.json",
      "status": "read",
      "evidence": "实际单目标输出20/20支持；首候选三代理Tg约-56.782/-56.008/-59.971，均在-60至-40区间；记录512+40预算、连续计量和有页码的类比合成来源，实验验证为false。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/units/inverse_20261111_FINAL_081/design_result.json",
      "status": "read",
      "evidence": "实际四目标输出0/20支持；首候选第三代理Tg约-60.0447及Tm约108.7977越界，正确保留不支持判定，不能解释成物理上无解。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/units/inverse_20261111_FINAL_001/search.json",
      "status": "read",
      "evidence": "完整512提案槽中有8个NONE；槽133为NONE、v=0、两组分计量且连续因子为1，提供实际NONE路径证据。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/units/inverse_20261111_FINAL_001/independent_differences.json",
      "status": "read",
      "evidence": "原始单元回执PASSED；512原始提案和20个返回条目比较差为0，与原design_result范围相符。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/units/inverse_20261111_FINAL_081/independent_differences.json",
      "status": "read",
      "evidence": "原始四目标单元回执PASSED；512原始提案和20个返回条目差为0，证明不支持结果也被一致复现。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/units/retrieval_20261113_FINAL_096/independent_differences.json",
      "status": "read",
      "evidence": "原始检索单元回执PASSED；415原始候选和20个返回条目差为0，检索候选耗尽不以伪候选补足512上限。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/evaluation/condition_overlay/condition_inputs.pt",
      "status": "unread",
      "evidence": "已取得固定A10原字节，750476字节，SHA5ec58188587e6de415775696e1c42015c5bcf02ede8c9d45e0a2e7c462954abe匹配；未反序列化解释张量，故语义内容标unread。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-010/evidence-00/recompute/condition_overlay/condition_inputs.pt",
      "status": "unread",
      "evidence": "已取得固定A10原字节，750476字节，SHA5ec58188587e6de415775696e1c42015c5bcf02ede8c9d45e0a2e7c462954abe匹配；与评价文件字节相同，但未反序列化解释张量，故语义内容标unread。"
    }
  ],
  "optional_suggestions": []
}
```