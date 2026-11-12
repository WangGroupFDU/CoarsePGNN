```json
{
  "type": "PLAN_CONTROL",
  "run_id": "pu-inverse-20261111",
  "request_id": "pu-inverse-20261111-3ca35914035c",
  "round": 0,
  "goal_sha256": "6b7d7be4137ada058174c9d6682e4350b746fd181944061764cc9fc3fc80dddd",
  "decision": "DISCUSS",
  "next_action": "Codex 在现有研究分支完成一次不拟合模型的数据裁定与评价边界修复：恢复或隔离246条跨测量Tg均值，核准共同材料状态，解决规格库中纯2,4-TDI与80/20异构体混合牌号的身份冲突，修复前向训练器自动评估测试集的问题，并导出本回复列出的最小证据包。保持现有闭包和集合归属，保留原始文件及历史回执，不安装包、不更换模型体系。下一次提交只需包含固定代码提交、固定证据提交、审计结果及尚未解决的问题。本回复给出的是明确禁止正式训练的合同草案；原会话补齐待裁决量化字段、批准FROZEN版本并取得该文件实际SHA-256之前，不得进行正式训练、基于模型性能选择方案或打开最终测试。",
  "validation": "原会话实际读取新提交的逐观察裁定表、Tg原子测量谱系、共同状态核准表、规格身份修复表、固定划分重审结果、训练侧测量精度统计和测试访问隔离测试回执。核对每项计数的材料、状态及闭包分母，确认所有恢复记录继承原集合、没有测试标签进入拟合或阈值制定，且自动测试评估路径已关闭。若仍无法确定物理容差、评价分母或独立统计依据，继续标记未冻结或证据不足，不能以工程测试通过代替模型成功。正式训练后的验收必须按冻结合同执行逐样本评分、三种子比较、独立checkpoint重算及逐门槛核对。",
  "review_conclusion": {
    "training_authorized": false,
    "contract_ready_to_freeze": false,
    "reason": "当前问题是数据语义和评价证据不足，不是讨论通道不可用。源码、审计和核验回执已经实际读取；其中出现了不能仅靠补写配置解决的标签、规格身份和测试访问问题。",
    "current_success_claim": "只能确认准备阶段有真实运行回执，神经核心、计量和部分隔离检查已实现；没有读取到真实训练性能，也不能给出研究成功判断。",
    "critical_findings": [
      {
        "id": "F1",
        "finding": "四目标共同specimen候选在train、validation、test中分别为27、19、2组；对应独立闭包组分别为8、1、1。全部候选仍待共同状态语义核准。",
        "implication": "48不是独立测试样本数。即使测试中的两个材料都通过，也不能据此给出跨来源四目标共同达标的统计结论。",
        "evidence_ids": [
          "E05",
          "E06",
          "E08"
        ]
      },
      {
        "id": "F2",
        "finding": "将范围放宽到同材料四性质并集，train、validation、test分别为138、48、60个材料，但对应闭包组只有27、3、2。",
        "implication": "仅核准现有specimen关联不一定能解决独立来源不足。60个测试材料不能当作60个独立来源；恢复既有数据库中被整理流程遗漏的真实测量是否能改善这一点，必须重新审计。",
        "evidence_ids": [
          "E08"
        ]
      },
      {
        "id": "F3",
        "finding": "GRADE_08e02a804b5ad7e62c5a同时关联名称2,4-tolylene diisocyanate和tolylene diisocyanate, 80% 2,4-/20% 2,6-isomer mixture，却使用单一2,4-TDI结构并标记exact_structure。",
        "implication": "存在规格身份冲突。应核查是名称错误、来源解析错误还是混合牌号被合并；核准前不能将该条目视为已经证明身份准确的原料规格。",
        "evidence_ids": [
          "E07",
          "E13"
        ]
      },
      {
        "id": "F4",
        "finding": "当前Tg整理均值谱系提取了aggregation_member_count，但没有导出每个均值的完整原子measurement_id及各自protocol。当前关联的是其中某条原测量的证据。",
        "implication": "不能将该单条测量的条件赋给跨方法或跨状态均值；仅知道平均条数不足以恢复条件监督。",
        "evidence_ids": [
          "E02",
          "E15"
        ]
      },
      {
        "id": "F5",
        "finding": "src/coarsepgnn_hybrid_v3/train.py会建立train、validation、test数据集，并在训练结束后遍历三个集合计算和导出指标。",
        "implication": "直接复用该训练入口会提前访问最终测试。必须增加严格的开发模式和独立最终评价入口。",
        "evidence_ids": [
          "E14"
        ]
      },
      {
        "id": "F6",
        "finding": "当前逆向模型以R=NCO/OH作为输入，只生成v；propose默认beam_width=16，每个含E组合取三个分量代表点，最多约48个初始提案。",
        "implication": "当前实现尚不等于512候选预算下的完整设计系统，也不能表述为模型已经同时反推R和v。",
        "evidence_ids": [
          "E12",
          "E13"
        ]
      }
    ],
    "evidence_scope_decision": "本次研究不新增合成实验。四目标代理筛选、既有实测性质核对和新配方实验成功必须分别报告。当前证据不支持跨来源四目标实测泛化结论，但这不等于四目标计算搜索无法开展；最终合同必须明确哪些必需门槛评价计算搜索，哪些评价既有实测证据，不能在结果出来后变更解释。"
  },
  "evidence_review": {
    "method": "实际通过GitHub连接器读取固定文件，并在工具内使用JavaScript连接已读取的CSV和JSON进行计数。本次没有执行本地shell、服务器训练或仓库修改。",
    "source_code_commit": "b38bf5cd6aa5063637493320429d7b9513f96d41",
    "evidence_commit": "9e05b3c9f16223819fc12e5de73450394b448ac8",
    "evidence_base_url": "https://github.com/WangGroupFDU/CoarsePGNN/blob/9e05b3c9f16223819fc12e5de73450394b448ac8/",
    "source_code_base_url": "https://github.com/WangGroupFDU/CoarsePGNN/blob/b38bf5cd6aa5063637493320429d7b9513f96d41/",
    "read_files": [
      {
        "id": "E01",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/README.md",
        "scope": "全文"
      },
      {
        "id": "E02",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/audit.json",
        "scope": "全文"
      },
      {
        "id": "E03",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/verification_001/result.json",
        "scope": "全文"
      },
      {
        "id": "E04",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/manifest.json",
        "scope": "全文"
      },
      {
        "id": "E05",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/joint_candidates.csv",
        "scope": "完整读取并解析970条候选"
      },
      {
        "id": "E06",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/splits.csv",
        "scope": "完整读取并解析3657条记录"
      },
      {
        "id": "E07",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/catalog.json",
        "scope": "完整获取并解析258条规格，重点核查身份冲突"
      },
      {
        "id": "E08",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/formulations.json",
        "scope": "完整获取并解析945个配方，重算集合及四性质覆盖分母"
      },
      {
        "id": "E09",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/verify.py",
        "scope": "全文"
      },
      {
        "id": "E10",
        "base": "evidence_base_url",
        "path": "configs/inverse/pu_20261111_sources.json",
        "scope": "全文"
      },
      {
        "id": "E11",
        "base": "evidence_base_url",
        "path": "docs/pu_inverse_design.md",
        "scope": "全文"
      },
      {
        "id": "E12",
        "base": "source_code_base_url",
        "path": "src/coarsepgnn_inverse/model.py",
        "scope": "全文"
      },
      {
        "id": "E13",
        "base": "source_code_base_url",
        "path": "src/coarsepgnn_inverse/chemistry.py",
        "scope": "全文"
      },
      {
        "id": "E14",
        "base": "source_code_base_url",
        "path": "src/coarsepgnn_hybrid_v3/train.py",
        "scope": "主评审直接读取360至385及525至565行；代码审查代理另读取完整训练文件"
      },
      {
        "id": "E15",
        "base": "source_code_base_url",
        "path": "src/coarsepgnn_inverse/audit.py",
        "scope": "完整获取，重点核查label_lineage、database_context及导出逻辑"
      },
      {
        "id": "E16",
        "base": "evidence_base_url",
        "path": "research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/result.json",
        "scope": "全文"
      }
    ],
    "directories_read": [
      "https://github.com/WangGroupFDU/CoarsePGNN/tree/9e05b3c9f16223819fc12e5de73450394b448ac8/research-runs/pu-inverse-20261111/preflight",
      "固定提交的根Git树及preflight/artifacts递归Git树"
    ],
    "failed_reads": [
      {
        "url": "https://raw.githubusercontent.com/WangGroupFDU/CoarsePGNN/9e05b3c9f16223819fc12e5de73450394b448ac8/research-runs/pu-inverse-20261111/preflight/artifacts/preflight_002/data/observations.json",
        "result": "GitHub连接器返回404 NOT_FOUND",
        "manifest_declared_bytes": 30832954,
        "manifest_declared_sha256": "73f6d64375f2fff7baa32a8ca6e83debbc7beee854eec75bcd81073693382311"
      }
    ],
    "not_independently_read": [
      "cn2上的原始SQLite数据库及四个完整任务行文件",
      "原始图文件和实际图张量的大文件内容",
      "逐测量原文或补充信息证据全文",
      "既有监督checkpoint及其全部训练来源清单"
    ],
    "hash_verification_boundary": "本次读取了远程核验回执和manifest中的SHA-256；没有独立重新计算服务器原始大文件的SHA-256。回执中的passed仅代表该准备核验，不代表本次研究PASS。",
    "independently_recomputed_counts": {
      "template_eligible_materials": {
        "train": 620,
        "validation": 163,
        "test": 162
      },
      "template_eligible_closure_groups": {
        "train": 142,
        "validation": 10,
        "test": 16
      },
      "four_target_specimen_candidates": {
        "train": {
          "specimens": 27,
          "materials": 23,
          "closure_groups": 8
        },
        "validation": {
          "specimens": 19,
          "materials": 19,
          "closure_groups": 1
        },
        "test": {
          "specimens": 2,
          "materials": 2,
          "closure_groups": 1
        }
      },
      "four_target_material_union_not_joint_supervision": {
        "train": {
          "materials": 138,
          "closure_groups": 27
        },
        "validation": {
          "materials": 48,
          "closure_groups": 3
        },
        "test": {
          "materials": 60,
          "closure_groups": 2
        }
      },
      "joint_candidate_status": "970条全部为linked_specimen_requires_semantic_review",
      "material_split_conflicts_in_splits_csv": 0,
      "qualification": "这些是当前准备文件的计数，尚未经过本回复要求的端点、状态和规格身份裁定，不能直接作为最终训练或验收分母。"
    }
  },
  "hypotheses_and_controls": [
    {
      "id": "H1",
      "hypothesis": "在本次线型双官能PU适用范围内，按新闭包训练的四个CoarsePGNN能够以有用精度预测指定端点和协议，为联合筛选提供可靠基础。",
      "basis": "现有架构能够接受配方、图和测试条件；输入张量对连续比例变化可见。当前没有证据证明新划分下的预测准确性或比例响应准确性。",
      "controls": [
        "按训练材料等权得到的常数中位数预测",
        "具有充分训练支持的协议分层中位数预测；稀疏或未知协议回退到训练总体中位数",
        "具有同规格、相容状态和同协议的实测配比系列，作为比例响应诊断"
      ],
      "evaluation": [
        "四任务分别计算材料等权MAE、RMSE及固定容差内比例",
        "按闭包计算与预先指定基线的配对误差差异区间",
        "配比系列比较实际与预测的局部差值；不假设所有PU性质对比例全局单调"
      ],
      "information_gain": "区分生成器效果不足与前向预测基础不足，判断联合筛选是否具有可信计算依据。",
      "falsification": "某个必需性质未达到冻结的真实留出误差门槛，或主要收益来自定义混杂、来源捷径或泄漏。张量会变化不能抵消此类失败。"
    },
    {
      "id": "H2",
      "hypothesis": "目标条件化的自回归单体选择器加连续配比生成器，在相同前向评价预算下，比训练集检索、合法随机和组合加比例搜索更有效地提出符合目标的候选。",
      "basis": "已实现目标集合编码、自回归D/P/E选择及以最终D/P/E为条件的MDN；这些结构允许学习目标与配方的条件关系，但尚未经过真实训练验证。",
      "controls": [
        "训练集检索",
        "合法随机提案",
        "组合加比例搜索",
        "保留相同计量和用户限制但移除性质目标条件的配方先验消融"
      ],
      "evaluation": [
        "固定请求上的代理Success@20、代理Precision@20和归一化约束偏差",
        "同预算下的候选覆盖、原料组合多样性及首次找到可行候选所用调用数",
        "所有方法采用相同前向模型、化学准入、排序及计费规则"
      ],
      "information_gain": "判断收益来自学习到的条件提案，还是仅来自所有方法共用的前向筛选。",
      "falsification": "在冻结的必需指标上不能达到规定绝对门槛或基线优势；改变目标后候选不发生有效改变；优势依赖额外未计费搜索。"
    },
    {
      "id": "H3",
      "hypothesis": "三分量MDN相对于单分量头，能更好地表示有用的配比多解性。",
      "basis": "同一性质条件可能对应多个配比区间，但当前数据是否足以识别多峰，仍需比较。",
      "controls": [
        "保持编码器、单体选择器、训练样本、优化器和搜索预算不变，将MDN分量数由3改为1"
      ],
      "evaluation": [
        "统一在z=logit(v)空间比较条件负对数似然",
        "比较最终候选命中、比例覆盖及不同分量的有效使用情况"
      ],
      "information_gain": "判断多分量是否有实际收益，避免只因采用MDN就宣称解决多解问题。",
      "falsification": "三分量相对于单分量没有冻结标准所要求的收益，或分量塌缩且候选质量没有改善。",
      "claim_boundary": "该结果可推翻三分量必要性的主张；不能在测试后更换主模型或删除失败消融。"
    },
    {
      "id": "H4",
      "hypothesis": "对单体搭配建立条件依赖，比三个相互独立的单体分类头更适合当前配方分布。",
      "basis": "实际配方涉及原料搭配与配比耦合；当前源码通过D到P到E的条件选择实现这一关系。",
      "controls": [
        "使用相同单体特征、目标编码、MDN及预算，将自回归选择改为三个仅依赖目标的独立分类头"
      ],
      "evaluation": [
        "留出真实配方的单体条件对数似然",
        "最终候选的合法性、代理约束命中和原料组合覆盖",
        "单列训练未见规格的结果"
      ],
      "information_gain": "判断搭配依赖是否被数据支持，以及未见规格的行为是否来自结构特征而非身份或频率泄漏。",
      "falsification": "自回归结构没有可重复收益，或未见规格优势在去除留出组成频率和来源身份后消失。"
    }
  ],
  "success_contract": {
    "filename": "success_contract.json",
    "schema_version": "1.0",
    "contract_version": "0.1.0-draft",
    "status": "DRAFT_NOT_FROZEN",
    "frozen": false,
    "formal_training_allowed": false,
    "run_id": "pu-inverse-20261111",
    "goal_sha256": "6b7d7be4137ada058174c9d6682e4350b746fd181944061764cc9fc3fc80dddd",
    "draft_request_id": "pu-inverse-20261111-3ca35914035c",
    "approval_request_id": null,
    "reviewer_conversation_id": "6af063c3-b6c0-83ea-834e-9a52ab4df547",
    "approved_data_manifest_sha256": null,
    "approved_query_manifest_sha256": null,
    "approved_catalog_sha256": null,
    "required_gate_policy": "所有必需门槛均通过且证据完整才可总体通过。null、未定义、分母不足、证据不足或未完成不能作为通过处理。",
    "scope": {
      "materials": "线型、无化学交联、无填料复合的双官能PU",
      "template": "一种D规格、一种宏二醇P规格、可选一种二醇E规格",
      "properties": [
        "tg",
        "tm",
        "tensile_strength_ultimate",
        "elongation_at_break"
      ],
      "new_synthesis_experiments_required": false,
      "new_model_family_allowed": false,
      "new_package_installation_allowed": false,
      "main_branch_changes_allowed": false,
      "default_top_k": 20,
      "candidate_budget": 512,
      "final_seeds": [
        20261111,
        20261112,
        20261113
      ],
      "maximum_formal_rounds": 10,
      "maximum_compute_seconds": 259200
    },
    "data_admission": {
      "decision_values": [
        "APPROVED_PRIMARY",
        "APPROVED_AUXILIARY",
        "REJECTED",
        "UNRESOLVED"
      ],
      "required_trace_fields": [
        "observation_id",
        "source_measurement_ids",
        "parent_derived_observation_id",
        "material_id",
        "source_id",
        "source_document_hash",
        "evidence_locator",
        "label_origin",
        "endpoint_name",
        "endpoint_definition",
        "method",
        "criterion",
        "phase_assignment",
        "protocol_id",
        "protocol_missing_mask",
        "material_state_id",
        "joint_state_id",
        "split",
        "group_id",
        "inclusion_decision",
        "exclusion_reason",
        "review_version"
      ],
      "source_measurement": {
        "rule": "来源测量并不自动合格；必须核准材料归属、数值单位、端点及证据位置。",
        "author_reported_replicate_mean": "原文在同状态、同端点和同协议下报告的重复实验均值可以使用，保留原文n、SD或SE等已有信息，不将其混同为数据库跨测量整理均值。",
        "unknown_protocol_fields": "保留未知掩码。端点明确但协议部分未知的记录可以进入预先声明的辅助或未知条件层，不得用于证明某个明确特殊协议下的性能。"
      },
      "figure_digitized": {
        "rule": "保存图号、轴校准、原坐标、单位转换及读数误差；未经核准维持UNRESOLVED。",
        "first_round_use": "核准后进入独立辅助层并单列结果，不混入主实测锚点分母。",
        "forbidden": "不得将数字化值伪装为无读数误差的原始表格值。"
      },
      "derived_cross_measurement_mean": {
        "primary_training_allowed": false,
        "primary_acceptance_allowed": false,
        "required_action": "逐条恢复完整成员测量及各自协议，或者隔离该整理值。恢复成员必须去重并继承原闭包归属。",
        "duplicate_policy": "同一原子测量不得既作为独立监督，又通过保留的整理均值重复计权。"
      },
      "pseudo_labels": {
        "first_round_allowed": false,
        "independent_experimental_acceptance_allowed": false,
        "future_use": "只有后续事先批准的蒸馏实验才可使用；按单元格记录教师、版本和预测来源，真实标签不得被覆盖。"
      },
      "endpoint_rules": {
        "tg": {
          "include": "来源明确为玻璃化转变，并可确定当前数值对应的测量定义。",
          "required_distinctions": [
            "DSC与DMA",
            "DSC onset、midpoint或其他明确判据",
            "DMA tan-delta峰、损耗模量峰或其他明确判据",
            "软相、硬相、整体或明确标记的未知相域",
            "扫描次序、升降温过程和已有热历史"
          ],
          "reject_or_quarantine": "存在多个转变却不能确定标签归属，或者跨方法、相域、判据、扫描程序混合成单个条件标签。",
          "unit": "摄氏度；温差容差以K表示，与摄氏温差数值相同。"
        },
        "tm": {
          "include": "来源及测量解释支持熔融转变，并能确定所采用的峰或起始点及相域。",
          "reject_or_quarantine": "仅根据软化、硬相解体、氢键解离、DMA模量下降等现象整理出的tm，未经熔融归属核准不得进入主任务。",
          "missing_states": [
            "NOT_MEASURED",
            "NOT_REPORTED",
            "NOT_DETECTED_IN_WINDOW",
            "AMBIGUOUS_ASSIGNMENT",
            "NOT_APPLICABLE"
          ],
          "forbidden": "不得将上述状态全部补为一个数，也不得对不同相域机械施加统一Tm大于Tg规则。",
          "unit": "摄氏度；温差容差以K表示。"
        },
        "tensile_strength_ultimate": {
          "include": "来源明确为最大工程拉伸应力，或来源标准及曲线足以支持这一解释。",
          "reject_or_quarantine": "未定义的泛称tensile strength、屈服强度、真实应力、指定应变应力，以及不能证明同时为最大应力的断裂应力。",
          "unit": "MPa"
        },
        "elongation_at_break": {
          "include": "来源明确为断裂时工程应变。",
          "conversion": "无量纲工程应变乘100转换为百分数；其他应变定义只有在转换依据完整时才能转换。",
          "reject_or_quarantine": "屈服伸长、最大载荷伸长、拉伸比及定义不清的伸长指标。",
          "unit": "百分数；绝对误差和容差以百分点表示。"
        }
      },
      "joint_supervision_rule": {
        "required": [
          "原料规格及配方身份一致",
          "有证据支持共同制备批次或明确可比的制备背景",
          "加工、预处理、调湿及存储状态相容",
          "每个性质的端点已核准",
          "各测量协议分别保留",
          "关联依据具有可追溯证据位置"
        ],
        "different_physical_specimens": "允许同批材料分取不同试样完成不同测试，不要求四种破坏性或热分析测试作用于同一物理试条。",
        "thermal_history": "共同制备状态与各测试协议引入的热历史分别记录；不同测试协议不能被误写为相同材料后状态。",
        "not_sufficient": [
          "仅material_id相同",
          "仅specimen_id相同",
          "仅材料名称相同",
          "数据库production_batch相同"
        ],
        "unresolved_action": "不能建立联合监督，只保留各自合格的单性质记录。",
        "training_mask_rule": "每个训练目标集合必须是同一个APPROVED共同状态已观测性质的非空子集。"
      }
    },
    "grade_and_stoichiometry_rules": {
      "chemical_eligibility": "每个D、P、E规格分别核准角色、两个相应反应性官能团、结构身份和质量字段；PSMILES星号或RDKit可解析不能替代端基证据。",
      "identity_fields": [
        "role",
        "structure_or_explicit_composition",
        "molar_mass_or_Mn",
        "functionality",
        "end_group_identity",
        "stereochemical_identity",
        "mixture_composition",
        "representation_level"
      ],
      "mixture_policy": "纯异构体与异构体混合牌号必须区分。若现有表示不能忠实表达某个混合规格，将其标为UNRESOLVED或范围外，不以单一纯结构代替。",
      "mass_units": "g/mol；宏二醇采用来源Mn，不用重复单元摩尔质量替代。",
      "definitions": {
        "R": "NCO与OH当量比；在所有参与反应组分均为双官能时等于nD/(nP+nE)。",
        "v": "扩链剂在OH原料中的分子份额，等于nE/(nP+nE)。",
        "nD_nP_nE": "分别为D、P、E的物质的量，使用一致单位。",
        "molecule_ratio": "D:P:E = R:(1-v):v。",
        "mass_fraction_i": "wi = ni*Mi / sum(ni*Mi)，Mi为该规格的摩尔质量或来源Mn。"
      },
      "R_policy": "第一版明确为给定R下的条件逆设计。用户未给定时使用R=1并在规范化请求中显式记录default来源；训练保留每条真实R。若评价请求提供实测R，报告为已给定计量约束的逆设计，不声称仅凭性质恢复全部计量信息。",
      "NONE_policy": "E=NONE时v严格等于0，不计算连续配比损失；有E时要求0<v<1。",
      "numerical_roundtrip_tolerance": {
        "absolute": 1e-10,
        "unit": "无量纲",
        "purpose": "64位计量换算的工程一致性检查，不是把不同配方视为相同的物理容差。"
      }
    },
    "split_and_learning_boundaries": {
      "split_seed": 20261111,
      "preserve_existing_assignments": true,
      "identity_closure_fields": [
        "material",
        "formulation",
        "source",
        "actual_graph_tensor"
      ],
      "maximum_cross_split_overlap_per_field_pair": 0,
      "recovered_observations": "继承原材料及来源闭包的集合归属，不重新按性质数值分配。",
      "new_cross_split_link": "停止并报告新增关联，等待原会话裁定；不得静默挪组、删除困难测试样本或沿用失效散列。",
      "catalog_setting": "允许冻结经过化学审核的完整无性质规格目录，明确命名为已知候选目录中的训练未见标签规格评价。",
      "allowed_catalog_features": [
        "结构",
        "角色",
        "官能度和端基",
        "Mn或摩尔质量",
        "立体与混合组成",
        "表示层级及缺失掩码"
      ],
      "forbidden_catalog_features": [
        "source_material_ids",
        "source_component_ids等身份捷径",
        "留出配方组合及其频率",
        "验证或测试规格出现频率",
        "目标性质值",
        "含性能结果的名称或文本"
      ],
      "train_only_fit": [
        "目标标准化和变换",
        "数值特征标准化",
        "PLE结点或分箱",
        "assay词表",
        "所有任务监督参数",
        "条件校准参数",
        "用于性能判断的不确定性校准"
      ],
      "old_checkpoint_rule": "每个旧监督权重必须提供训练来源与新闭包不相交的证明；不能证明时，按新划分重新训练或拟合。冻结旧权重不消除泄漏。",
      "reusable_fixed_features": "可复用未受本任务标签监督的固定文本基础模型与确定性指纹；记录版本和散列，并排除输入文本中的性能结果。",
      "unseen_grade_rule": "在训练前根据最终审核库确定并冻结子组；最终不能因为表现差而移除。化学不合格造成的排除须在训练前按相同规则执行并报告覆盖损失。"
    },
    "evaluation_tracks": {
      "forward_real_observations": "对未参与拟合的真实配方及其协议预测性质，支持前向可靠性结论。",
      "free_generation": "自由输出D/P/E及连续配比，评价化学合法性、代理约束命中、搜索效率和多样性。新配方没有实测值时只报告代理结果。",
      "inverse_anchor_recovery": "计算已知实测配方的条件似然、单体恢复和配比重构；该轨道评价恢复已有设计的能力，不将未恢复唯一原配方自动判为所有设计均失败。",
      "retrospective_measured_pool": "若核准数据足够，使用事先冻结的实测候选池评价回顾性推荐。所有方法只能看到候选结构和规格，真实性质只由评分器读取。",
      "nearest_recipe_rule": "不能将邻近比例或相近图的实测性质赋给新候选。配方等价仅依据来源报告精度和身份规范，不依据模型表现选择匹配容差。",
      "joint_evidence_limit": "目前四目标测试共同specimen仅一个闭包，不能输出跨闭包总体置信区间；同材料并集也只有两个闭包。审计后仍不足时，只能报告描述性核对，不将其包装为联合统计验证。"
    },
    "metric_definitions": {
      "variables": {
        "m": "一个合格材料；同材料多个状态、观察或请求先按冻结规则在材料内平均。",
        "j": "一个性质及其明确端点、协议约束。",
        "q": "一个冻结评价请求。",
        "k": "一个返回候选，最多20个。",
        "y": "对应真实测量值。",
        "yhat": "前向模型预测值。",
        "L_U": "请求的下界和上界；单侧约束只使用存在的边界。",
        "delta_j": "原会话训练前冻结的物理容差，不能由模型测试误差反推。",
        "G": "评价分母中不同来源闭包组的数量。"
      },
      "MAE_j": {
        "formula": "先计算每个材料在性质j上的mean(abs(yhat-y))，再对材料取等权平均。",
        "units": {
          "tg": "K",
          "tm": "K",
          "tensile_strength_ultimate": "MPa",
          "elongation_at_break": "百分点"
        }
      },
      "within_tolerance_j": {
        "formula": "先计算每个材料中abs(yhat-y)<=delta_j的比例，再对材料等权平均。",
        "denominator": "冻结合格清单中的全部材料；失败预测不能从分母删除。"
      },
      "constraint_violation": {
        "formula": "每项Vj=max(0,L-yhat)/delta_j + max(0,yhat-U)/delta_j，缺失的单侧边界不参与；请求分数V为所有启用Vj之和。",
        "unit": "无量纲",
        "hard_constraint_rule": "所有启用约束分别通过才算命中；不能用其他性质的优势抵消一项失败。"
      },
      "proxy_success_at_20": {
        "formula": "每个请求的前20个返回候选中，只要至少一个合法且所有启用约束的预测同时满足，记1，否则记0。",
        "denominator": "全部冻结请求；空结果和程序失败计0。",
        "interpretation": "仅为代理模型下的搜索成功。"
      },
      "proxy_precision_at_20": {
        "formula": "每个请求中合法且预测满足所有约束的返回候选数除以20；不足20个的空位计0。",
        "interpretation": "同时约束候选质量和有效返回数量，不允许只返回一个候选获得100%的Precision@20。"
      },
      "retrospective_measured_success_at_20": {
        "formula": "在固定实测候选池中，检查前20项是否至少有一个真实标签同时满足请求。",
        "eligibility": "所需真实标签及共同状态均已核准，且性质值未暴露给候选选择器。",
        "interpretation": "既有实测候选的回顾性推荐，不是新连续配方实验验证。"
      },
      "ratio_nll": {
        "training_space": "z=log(v/(1-v))",
        "reporting_rule": "各比较模型使用相同概率空间；若报告v空间NLL，NLL_v=NLL_z+log(v)+log(1-v)。NONE为离散状态，不能用极小正v代替。"
      },
      "diversity": {
        "required_reports": [
          "top20中唯一D/P/E规格组合数",
          "每个组合的连续v覆盖",
          "去重后的配方数",
          "NONE返回比例",
          "各性质约束和未见规格子组的覆盖"
        ],
        "grade_combination_key": "审核后的D、P、E规格ID三元组，NONE单独编码。",
        "warning": "生成概率、混合权重、配比密度均不解释为实验达标概率。"
      }
    },
    "statistical_protocol": {
      "material_weighting": "材料等权；多观察、多请求及三种子不能增加独立材料数。",
      "resampling_unit": "固定来源闭包整组重抽样，保留组内所有材料。",
      "paired_comparison": "主模型与基线使用完全相同的请求及闭包重抽样序列。",
      "bootstrap_replicates": 10000,
      "bootstrap_seed": 20261111,
      "nominal_interval": 0.95,
      "seed_aggregation": "报告每个种子、最差种子和三种子均值；主材料指标先在材料内对种子平均，再进行闭包重抽样。",
      "insufficient_groups": "一个闭包直接返回INSUFFICIENT_INDEPENDENT_GROUPS。少量闭包不能因程序能计算bootstrap就宣称统计可靠；最终最低闭包数及精度要求仍须裁决。",
      "multiplicity": "冻结主终点、必需门槛及比较家族；多重比较处理在正式训练前确定，不能在测试后选择显著项目。",
      "minimum_sample_size_basis": "结合预定最小有意义差异、区间精度、聚类结构和审核后分母制定；不将请求数、重复测量数或种子数当作独立样本数。"
    },
    "quantitative_gates": [
      {
        "id": "G_A1",
        "criterion": "A1",
        "required": true,
        "status": "RULE_FIXED_PENDING_REAUDIT",
        "thresholds": {
          "cross_split_identity_overlap": 0,
          "primary_labels_with_unresolved_endpoint": 0,
          "joint_labels_with_unapproved_state": 0,
          "pseudo_labels_in_real_acceptance": 0,
          "primary_records_without_traceable_source": 0
        }
      },
      {
        "id": "G_A2",
        "criterion": "A2",
        "required": true,
        "status": "NOT_SATISFIED",
        "thresholds": {
          "pending_adjudications": 0,
          "unfilled_required_thresholds": 0,
          "contract_hash_mismatch": 0
        }
      },
      {
        "id": "G_FORWARD",
        "criterion": "A3",
        "required": true,
        "status": "NEEDS_ADJUDICATION",
        "tolerances": {
          "tg_K": null,
          "tm_K": null,
          "tensile_strength_ultimate_MPa": null,
          "elongation_at_break_percentage_points": null
        },
        "maximum_MAE_by_endpoint_protocol": null,
        "minimum_within_tolerance_by_endpoint_protocol": null,
        "minimum_materials_by_endpoint_protocol": null,
        "minimum_independent_groups_by_endpoint_protocol": null,
        "interval_acceptance_rule": "冻结最大误差时使用相应误差上置信界；冻结最低命中率时使用下置信界。具体阈值和比较家族待审计后裁决。",
        "missing_basis": "尚无审核后端点分布、可比重复测量精度和协议分层分母，不能科学解释任意给定的K、MPa或百分点门槛。"
      },
      {
        "id": "G_SEARCH",
        "criterion": "A3",
        "required": true,
        "status": "NEEDS_ADJUDICATION",
        "minimum_proxy_success_at_20": null,
        "minimum_proxy_precision_at_20": null,
        "minimum_advantage_over_locked_baseline": null,
        "maximum_normalized_constraint_violation": null,
        "minimum_query_count_and_strata": null,
        "query_manifest": null,
        "interval_acceptance_rule": "主模型与预先锁定基线的配对差异下置信界达到冻结优势门槛，同时通过绝对质量门槛；不能仅靠优于极弱基线通过。",
        "claim": "代理筛选能力，不是新材料实验成功。"
      },
      {
        "id": "G_REAL_ANCHOR",
        "criterion": "A3",
        "required": true,
        "status": "NEEDS_ADJUDICATION",
        "track_definition": "必须明确采用前向真实锚点核对、逆向配方恢复或固定实测池推荐中的哪一项作为相应必需证据，不得混用名称。",
        "minimum_materials": null,
        "minimum_independent_groups": null,
        "minimum_success_or_recovery": null,
        "ratio_identity_tolerance": null,
        "four_target_population_claim_currently_supported": false,
        "current_four_target_cases_policy": "只能作为待核准的描述性案例，不设置能够通过2/2即可宣称总体成功的门槛。"
      },
      {
        "id": "G_COVERAGE_STABILITY",
        "criterion": "A3",
        "required": true,
        "status": "NEEDS_ADJUDICATION",
        "minimum_candidate_coverage": null,
        "minimum_unique_grade_combinations": null,
        "unseen_grade_required_performance": null,
        "unseen_grade_minimum_independent_groups": null,
        "maximum_seed_performance_spread": null,
        "minimum_worst_seed_performance": null,
        "rule": "冻结前确定困难子组和所有分母；结果出来后不能删去未见规格、NONE或低覆盖协议以获得通过。"
      },
      {
        "id": "G_A4",
        "criterion": "A4",
        "required": true,
        "status": "RULE_FIXED_PENDING_IMPLEMENTATION",
        "thresholds": {
          "invalid_returned_candidates": 0,
          "violated_hard_constraints_marked_as_pass": 0,
          "duplicate_candidates_used_to_fill_top_k": 0,
          "unsupported_synthesis_numbers_presented_as_sourced": 0,
          "incorrect_NONE_stoichiometry": 0,
          "forward_calls_over_budget": 0
        },
        "interface_suite_requirement": "全部冻结接口案例按期望状态通过；合法性通过不能替代非空覆盖和性能门槛。"
      },
      {
        "id": "G_A5",
        "criterion": "A5",
        "required": true,
        "status": "RULE_FIXED_PENDING_EXECUTION",
        "thresholds": {
          "required_manifest_hash_mismatches": 0,
          "missing_required_final_artifacts": 0,
          "unread_required_review_evidence": 0,
          "unresolved_run_version_conflicts": 0
        },
        "checkpoint_recompute_tolerance": {
          "space": "使用冻结训练标准化参数后的预测空间",
          "rtol": 1e-5,
          "atol": 1e-6,
          "purpose": "独立加载相同checkpoint的数值一致性，不是模型物理误差容差。"
        }
      }
    ],
    "pending_adjudications": [
      "审核后数据、规格库和共同状态版本及其散列",
      "四性质各端点和协议的物理容差及绝对预测误差门槛",
      "主评价请求的构造规则、可行性分层和冻结清单",
      "代理命中的绝对门槛与相对基线最小有意义优势",
      "真实锚点轨道、分母、配方等价精度及统计证据等级",
      "最低独立闭包数、区间精度要求和多重比较规则",
      "覆盖、多样性、未见规格和最差种子门槛",
      "三种子完整计算能否在剩余预算内完成的实测工程估计"
    ]
  },
  "minimum_audit_package": [
    {
      "id": "AUDIT_1",
      "action": "建立逐观察数据裁定并恢复Tg原子测量谱系。",
      "outputs": [
        "measurement_adjudication.csv",
        "tg_mean_membership.csv",
        "training_measurement_precision.json"
      ],
      "requirements": [
        "对246条整理均值列出全部成员measurement_id、原数值、单位、方法、判据、相域和protocol；无法恢复时逐条说明并隔离。",
        "逐条裁定Tg、Tm、ultimate强度和断裂伸长端点，不根据模型残差或是否有利于训练做排除。",
        "训练集报告各合格端点的材料数、闭包数、分位数、范围、来源报告精度、可比重复测量差异和数字化误差。",
        "验证和测试在本次规划阶段只导出资格、身份和数量，不提供用于设置阈值的性质分布或模型表现。",
        "提供足以核对裁定的原文位置及必要证据，不只导出approved布尔值。"
      ],
      "acceptance": "每条原记录都有唯一去向；每条恢复记录能追溯到原测量；不存在均值与成员重复监督或协议错配。"
    },
    {
      "id": "AUDIT_2",
      "action": "核准共同材料状态并重新计算联合评价口径。",
      "outputs": [
        "joint_state_review.csv",
        "joint_state_edges.csv",
        "evaluation_denominators.json"
      ],
      "requirements": [
        "审查48个现有四目标候选，并检查同材料四项齐全但specimen不同的记录是否有直接共同状态证据。",
        "不得仅按材料名称或入库批次拼接。",
        "按split、目标集合、端点、协议、标签层级和seen/unseen规格分别报告观察数、材料数、状态数及闭包数。",
        "明确列出共同状态仍未核准、某一端点被排除、只可用于单目标或只可作描述性案例的记录。",
        "如果恢复数据库中的既有测量增加了联合证据，提供完整来源映射并继承原集合归属。"
      ],
      "acceptance": "联合监督只使用APPROVED状态；来源不足不能用增加请求数或三种子重复弥补。"
    },
    {
      "id": "AUDIT_3",
      "action": "修复并审核原料规格身份。",
      "outputs": [
        "grade_identity_review.csv",
        "grade_id_migration.csv",
        "approved_catalog.json",
        "affected_materials.csv"
      ],
      "requirements": [
        "优先核查GRADE_08e02a804b5ad7e62c5a对应的纯2,4-TDI与80/20混合牌号冲突。",
        "全库检查角色、双官能证据、端基、Mn、异构组成、立体信息及表示层级。",
        "身份迁移保留旧ID、来源component、材料关联和理由。",
        "重新报告规格数、训练可用材料数及验证测试未见规格数；82和69不能直接沿用为最终分母。"
      ],
      "acceptance": "所有可返回候选规格具有一致身份；未核准条目不进入保证合法的候选空间。"
    },
    {
      "id": "AUDIT_4",
      "action": "修复训练与最终测试的访问隔离，并核查旧权重来源。",
      "outputs": [
        "checkpoint_lineage.json",
        "split_access_policy.json",
        "test_access_guard_results.json",
        "runtime_estimate.json"
      ],
      "requirements": [
        "开发训练入口只接收train和validation标签，不创建test标签数据集、不导出test预测。",
        "最终评价器单独实现，验证冻结合同、代码、数据、候选策略、三种子checkpoint和授权状态后才能打开test。",
        "用会在读取测试标签时明确报错的测试数据代理验证开发路径，不能只检查没有打印test指标。",
        "列出每个结构权重、条件投影、PLE统计和校准对象的训练来源；不能证明无泄漏的对象标记必须重训。",
        "只进行构图、加载和推理适配的工程耗时测量，不把它们当作模型性能或开始真实训练。"
      ],
      "acceptance": "开发路径测试标签访问次数为0；旧权重来源不明时没有被默认复用；预算估计包含四前向训练、逆模型、基线、消融和最终重算。"
    },
    {
      "id": "AUDIT_5",
      "action": "在不改变既有集合归属的前提下生成审核后完整证据清单。",
      "outputs": [
        "post_adjudication_audit.json",
        "approved_data_manifest.json",
        "audit_readme.md",
        "immutable_evidence_urls.json"
      ],
      "requirements": [
        "重算原始、派生行、规格、图及实际张量散列，重新核对四种身份的跨集合重合。",
        "原文件和旧回执只读，所有修正写入新版本并保留映射。",
        "提供训练侧可读数据投影和测试值封存状态，使原会话能够检查实际内容。",
        "不能只给observations.json的散列而缺少可读取内容或足够的可核对投影。",
        "记录真实计算耗时，不重置正式轮次、预算或历史预检记录。"
      ],
      "acceptance": "上述证据能够通过固定提交读取，且所有数字均能回到具体记录和身份闭包。"
    }
  ],
  "first_training_experiment_after_freeze": {
    "status": "SPECIFIED_BUT_NOT_AUTHORIZED",
    "entry_conditions": [
      "success_contract状态为FROZEN且不存在null必需阈值",
      "全部数据和规格裁定文件已冻结并通过散列核对",
      "开发训练路径测试访问次数为0",
      "剩余计算预算覆盖预定首轮及后续最终验证预留"
    ],
    "round": 0,
    "development_seed": 20261111,
    "maximum_round_compute_seconds": 43200,
    "forward_models": {
      "architecture": "沿用现有CoarsePGNN及混合测试条件框架，四个性质按冻结的新划分训练。",
      "supervised_checkpoint_policy": "没有新划分无泄漏证明时从相应训练阶段重新拟合。",
      "structure_stage": {
        "learning_rate": 0.0002,
        "batch_size": 64,
        "max_epochs": 400,
        "patience": 60,
        "weight_decay": 0.0001,
        "dropout": 0.15,
        "gradient_clip_norm": 5,
        "warmup_fraction": 0.05
      },
      "condition_residual_stage": {
        "learning_rate": 0.0005,
        "batch_size": 64,
        "max_epochs": 250,
        "patience": 40,
        "weight_decay": 0.0001,
        "dropout": 0.15,
        "gradient_clip_norm": 5,
        "warmup_fraction": 0.05
      },
      "development_evaluation": "只计算固定验证清单。未达到冻结的前向可靠性要求时，记录失败原因，不通过放宽目标区间掩盖。"
    },
    "inverse_model": {
      "architecture": "目标集合编码器、自回归D/P/E选择器、完整D/P/E条件化三分量MDN。",
      "hidden_size": 128,
      "mdn_components": 3,
      "optimizer": "AdamW",
      "learning_rate": 0.001,
      "weight_decay": 0.0001,
      "batch_size": 64,
      "max_epochs": 300,
      "patience": 30,
      "gradient_clip_norm": 5,
      "loss": "逐样本三个原料选择交叉熵之和，加非NONE样本的z空间MDN负对数似然；总损失最终按材料等权。",
      "sampling": "每个epoch先对合格训练材料等权遍历或抽样，再选该材料一个核准状态，再选该状态一个非空目标子集。不能把所有测量和全部mask展开后等行抽样。",
      "validation": "使用事先生成并冻结的状态与目标子集清单，先在材料内聚合，再计算材料等权验证负对数似然；不每个epoch随机更换验证mask。",
      "target_interval_augmentation": "只能根据该条真实观测和冻结的容差规则构造包含该观测的请求；不生成新的真实性质标签。",
      "first_round_exclusions": [
        "伪标签蒸馏",
        "强化学习",
        "更换生成模型体系",
        "通过最终测试选择超参数"
      ]
    },
    "candidate_scheduler": {
      "raw_proposal_attempt_limit": 4096,
      "candidate_evaluation_limit": 512,
      "top_k": 20,
      "initial_candidate_slots": 256,
      "refinement_candidate_slots": 256,
      "proposal_rule": "采用概率单体模型的固定束搜索代表路径及固定种子配比探索；对训练中未得到共同状态支持的联合请求，优先从已支持的目标子集提出候选，再统一检查全部要求。",
      "refinement_rule": "在剩余预算内对初始候选中接近满足全部约束的合法组合细化v；新比例属于新候选并计入预算。",
      "counting_rule": "原始提案、重复提案、化学拒绝、构图失败分别计数。进入唯一候选构图或评价槽位后即消耗候选预算，即使失败也不免费补算。",
      "scalar_forward_calls": "另外统计候选、性质协议和预测器成员的实际乘积调用数；四性质单协议、每性质单模型时，512个完整候选最多对应2048次单预测器评价。",
      "cache_rule": "缓存命中仍记录对应逻辑评价成本，不能通过免费离线预筛选获得额外搜索预算。",
      "return_rule": "只返回合法且明确标注逐约束状态的候选；不足20个时返回实际数量，不用重复项或不合格项补齐。"
    },
    "baselines": [
      {
        "name": "training_retrieval",
        "implementation": "仅使用训练集真实标签，按请求端点和协议检索已有配方；多目标缺失时按可支持子目标检索并合并，随后接受与其他方法相同的四性质复核及剩余预算内细化。",
        "forbidden": "不得按验证或测试真实性质直接检索答案。"
      },
      {
        "name": "legal_random",
        "implementation": "在相同审核规格库和请求限制下用固定种子生成合法D/P/E及v；分布只由训练数据和事先冻结规则确定，不使用留出规格或组合频率。"
      },
      {
        "name": "combination_ratio_search",
        "implementation": "使用训练组合信息及固定覆盖序列提出初始组合和比例，再在相同剩余预算内做比例及单体替换搜索；明确记录覆盖规则和每次前向调用，不使用额外预测器。"
      }
    ],
    "necessary_ablations": [
      "去除性质目标条件但保留相同原料限制和R",
      "MDN3替换为MDN1",
      "自回归单体选择替换为三个独立分类头"
    ],
    "first_round_outputs": [
      "四个前向模型与逆模型的真实训练日志、checkpoint和训练来源散列",
      "验证集逐观察前向结果",
      "验证集逐请求候选、拒绝原因、调用预算和逐约束结果",
      "三种基线及必要消融的统一预算结果",
      "材料等权及闭包分组指标",
      "未见规格、NONE和协议覆盖的分层结果",
      "计算耗时、剩余预算、失败原因及下一轮可检验修订"
    ],
    "first_round_information_gain": "判断数据裁定后的四个前向任务是否足以支持筛选，条件提案是否优于简单搜索，以及MDN和自回归依赖是否值得保留。训练损失下降和接口可运行均不能独立回答这些问题。",
    "first_round_failure_handling": "数值失败、数据污染、前向可靠性不足、条件模型不优于对照或预算不足均如实记录。只允许使用验证证据修订实现和已允许的超参数；不更改冻结成功阈值、测试分母或适用范围。"
  },
  "interface_and_synthesis_validation": {
    "required_case_families": [
      "四个单目标分别可解析并返回正确单位",
      "二目标、三目标、四目标逐项约束检查",
      "同一性质在两个测试协议下的请求，以及不受支持协议的明确处理",
      "NONE合法路径和NONE与非零v矛盾的拒绝路径",
      "显式非1的R及默认R的来源记录",
      "审核库中训练未见规格与完全未知规格的不同处理",
      "上下界矛盾、未知单位、NaN、无穷值、非法角色、缺少必要字段及空候选库",
      "预算耗尽但没有候选满足要求",
      "合成依据完整、部分、仅类似路线和无依据的分层输出"
    ],
    "finite_search_no_result_status": "NOT_FOUND_WITHIN_BUDGET",
    "no_result_interpretation": "有限搜索未找到不能表述为物理上无解。",
    "unsupported_request_status": "UNSUPPORTED_ENDPOINT_OR_PROTOCOL",
    "synthesis_evidence": {
      "required_fields": [
        "route_status",
        "source_identifier",
        "evidence_locator",
        "supported_steps",
        "unsupported_or_missing_fields",
        "applicability_to_candidate"
      ],
      "rule": "已有同配方路线、类似原料路线与反应类型依据分别标注。温度、催化剂量、时间、溶剂、产率及后处理缺乏来源时留空，不由LLM补成事实。",
      "scope": "本次评价合成依据的可追溯性和适用说明，不要求新合成实验，也不把可解析结构或路线检索视为已证明可合成。"
    }
  },
  "freeze_and_final_review_protocol": {
    "draft_handling": "本回复中的success_contract可保存为草案以便实现审计，但不得写成FROZEN，不得把它当作训练授权。",
    "freeze_preconditions": [
      "原会话实际读取新的最小证据包",
      "全部必需指标的阈值、单位、分母、最小样本口径和区间判据明确",
      "全部pending_adjudications清空",
      "批准后的合同内容与代码配置不存在冲突",
      "最终测试尚未用于模型或门槛迭代"
    ],
    "serialization": "success_contract.json使用UTF-8，JSON键排序，禁止NaN和Infinity，固定紧凑分隔符并以单个换行结束。SHA-256计算对象为该文件实际字节；散列放在独立sidecar及run manifest中，避免文件自引用散列。",
    "freeze_receipt": "保存原会话完整规划回复、批准请求身份、合同版本、实际SHA-256、数据与规格清单散列及固定Git提交。",
    "final_test_opening": "全部开发选择结束后，先冻结代码、数据、三种子checkpoint、前向权重、候选策略、基线、消融、查询规则和评分器，再一次性打开最终测试评价窗口。三种子和预定比较统一在该窗口执行。",
    "after_test_opening": "不得修改阈值、删样本、选最佳种子、换查询或继续根据测试结果训练。只有能够证明没有改变模型及评价定义的执行故障，才可在保留原失败回执后重跑同一锁定任务；影响结论的实现错误必须报告验收失效。",
    "independent_recompute": "由独立验证入口重新加载锁定checkpoint，重建候选和逐约束预测，并核对逐样本结果、计量、候选预算及数值一致性。",
    "remote_version_evidence": "核验实际C、A、R远端对象及其绑定关系；源码版本、产物清单、合同散列和本会话最终评审必须对应同一次真实运行，不能用工作区未提交状态或另一轮产物替代。",
    "required_final_read_list": [
      "冻结success_contract.json及散列",
      "审核后数据、规格和split清单",
      "全部必需指标的逐项检查结果",
      "逐观察及逐请求留出结果",
      "基线和消融结果",
      "三种子结果",
      "独立checkpoint重算回执",
      "实际运行命令、环境、起止时间、退出码和预算台账",
      "必要大文件的读取方式、字节数和散列核验",
      "合成来源与接口案例结果"
    ],
    "secret_links": "本次只使用公开固定GitHub入口，没有使用秘密预览链接。若后续使用临时秘密链接，先完成必要读取和校验，再由Codex撤销并提供回执；不得把令牌写入公开产物或评审正文。"
  },
  "failure_and_stop_rules": [
    {
      "condition": "审计后仍不能确定端点、共同状态或规格身份",
      "outcome": "相关记录保持隔离；若导致必需评价无法定义，返回EVIDENCE_INSUFFICIENT，不能正式冻结成功合同。"
    },
    {
      "condition": "四目标实测记录仍仅来自一至两个测试闭包",
      "outcome": "不报告跨来源四目标实测总体置信结论。原会话必须在训练前明确最终合同的计算证据与实测证据边界；不得由Codex自行将单项通过解释为联合通过。"
    },
    {
      "condition": "出现跨集合身份关联、旧监督权重污染或开发过程访问最终测试结果",
      "outcome": "停止受影响实验并记录污染范围，不在同一固定测试上继续调参争取通过。"
    },
    {
      "condition": "测试阶段任一必需门槛未通过",
      "outcome": "该冻结合同下研究未达标；提交完整结果和失败原因，不放宽门槛、不删除困难子组。"
    },
    {
      "condition": "达到10个正式轮次或259200秒累计计算预算，任一先到",
      "outcome": "停止新增计算，保存现有checkpoint、日志和未完成项，交付BUDGET_EXHAUSTED或NOT_MET，不减少三种子、删去基线或把准备测试计为完成。"
    },
    {
      "condition": "代理命中很高，但真实前向锚点误差未通过",
      "outcome": "判定联合筛选缺少可靠前向基础，不能以代理自评成功覆盖真实验证失败。"
    },
    {
      "condition": "全部必需门槛通过，但原会话没有读取必读证据或版本不能对应",
      "outcome": "保持REVIEW_INCOMPLETE，不能宣称已获独立PASS。"
    }
  ]
}
```