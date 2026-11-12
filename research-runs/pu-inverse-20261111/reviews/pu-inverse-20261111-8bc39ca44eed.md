```json
{
  "type": "REVIEW_CONTROL",
  "protocol_version": 2,
  "run_id": "pu-inverse-20261111",
  "request_id": "pu-inverse-20261111-8bc39ca44eed",
  "round": 9,
  "goal_sha256": "6b7d7be4137ada058174c9d6682e4350b746fd181944061764cc9fc3fc80dddd",
  "code_sha": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
  "artifact_sha": "212cb94afe4ac26274588c7a233d7ab4bcae43ad",
  "decision": "REVISE",
  "primary_blocker": "R9在最终测试开窗前因checkpoint清单格式不兼容终止：final_common.py第140行无条件访问item['bytes']，而原C6清单的九个checkpoint条目只有path和sha256。最终输入门禁、八套模型的真实兼容性验证、最终评价及独立复算均未执行。现有监督器还会为每次evaluate或recompute调用分别建立14400秒期限，不能直接用于最后一轮的共享预算合并执行。",
  "next_action": {
    "action": "批准一次条件式R10合并恢复包。先在不开测试、不新增训练的前提下完成清单兼容、共享截止监督器、双重授权绑定和失败回执修复，将评价及复算两个入口同时冻结为实际C10并生成外部绑定回执。随后由唯一正式外层任务，在同一个最多14400秒绝对截止内依次完成真实输入门禁、八套模型的原validation兼容性核对、唯一最终开窗、原250观察及106共同状态评价、固定96请求的1440个比较单元，以及另一个全新进程中的完整独立复算。不得重新训练、改选模型、修改科学政策或重置时间预算。",
    "done_when": "R10结束时必须形成两种可核准结果之一：完整评价与完整独立复算均完成，逐项报告原合同门槛的通过或失败；或者工程异常、超时及缺失任务被完整归档，交付未完成状态。只有全部原必需门槛满足、全量复算一致、来源和版本证据完整，并经本会话另行核准，才能研究PASS。任何局部成功、开发负结果被保留或最后一轮预算耗尽均不自动构成研究完成。"
  },
  "stage_review": {
    "classification": "PREOPENING_ENGINEERING_FAILURE",
    "root_cause_status": "CONFIRMED_BY_SOURCE_AND_TRACEBACK",
    "root_cause_evidence": [
      "A9 failure.json和worker.log均定位final_common.py第140行的KeyError('bytes')。",
      "A9 C6_checkpoint_manifest.json的九个checkpoints条目均无bytes；C8主模型和逆向消融清单包含bytes。",
      "C9 final_evaluation.py第164行先执行check_inputs，第170行之后才进入测试访问守卫和no_fitting，第183行进行真实验证兼容性核对，第187行才调用open_test。"
    ],
    "test_window_status": "NOT_OPENED",
    "test_window_evidence": "independent_collection_receipt.json记录final_window_exists=false、test_access_log_exists=false、preopening_tests_started=false、search_units_directory_exists=false，并记录四个实际失败产物回读大小和SHA全部匹配；该回执与实际失败调用链一致。",
    "zero_new_computation_interpretation": "本轮没有新增训练、候选搜索或最终评价，由失败位置及控制流支持。由于no_fitting和测试访问守卫尚未进入，不能将其写成已通过的运行时零调用审计或测试隔离门禁。",
    "not_executed": [
      "preopening_tests",
      "八套模型的真实权重加载与validation兼容性核对",
      "最终条件缓存覆盖层",
      "250条最终测试观察及106个共同状态评分",
      "正式96请求的1440个比较单元",
      "独立全量复算"
    ],
    "additional_static_review": "已核读14个C9新增或修改源码文件及相关依赖。除清单格式错误和合并执行所必需的监督编排修改外，未发现另一个可由当前静态证据证明必然发生的模型加载、统计或候选复算阻断。这不是对尚未运行路径的工程通过声明。",
    "budget_accounting": {
      "round9_elapsed_seconds": 4.40927419392392,
      "completed_formal_rounds": 9,
      "cumulative_compute_seconds": 1848.0574975546915,
      "remaining_formal_rounds": 1,
      "round9_remains_charged": true,
      "round9_may_be_relabelled_as_free_preflight": false
    }
  },
  "progress_assessment": {
    "submitted_status": "none",
    "judgment": "ACCEPTED",
    "scientific_progress": "R9没有产生新的材料预测、条件逆向学习或计算设计效果证据，manifest.progress的判断正确。",
    "engineering_information_gain": "故障已收敛为可解释的旧新清单格式差异；未开窗事实使最后一轮有条件合并评价与复算仍可保持原测试隔离原则。不能把这一诊断或新增源码散列解释为科学性能进展。",
    "evidence_refs_verified": [
      {
        "name": "failure.json",
        "sha256": "8d46553889d6bd8a67e799a1b480df6a720f540a1e4e3a5e9de5814d443488de"
      },
      {
        "name": "independent_collection_receipt.json",
        "sha256": "2cae3bab96905973611d09479c34d1f64397db3e596771bee148767c99162252"
      }
    ],
    "retained_round8_negative_results": {
      "single_target_four_property_mean_conditioning_improvement_nat": {
        "20261111": -0.0007415389344970513,
        "20261112": -0.06558774382059454,
        "20261113": 0.5233393450862655
      },
      "development_single_proxy_hits_out_of_24": {
        "inverse": 23,
        "retrieval": 15,
        "legal_random": 23,
        "combination_ratio_search": 24
      },
      "interpretation": "这些实际开发结果继续保留。它们尚未证明目标数值利用具有稳定优势，也未证明主方法超过最强预算基线；它们不是最终测试结果，不能据此修改合同或重新选择模型。"
    },
    "remaining_scientific_questions": [
      "固定实测留出集上的前向误差、命中率、基线比较和稳定性是否达到原合同。",
      "真实留出配方的条件窗口概率是否相对配对无目标数值模型达到原改进门槛。",
      "固定96请求上，学习得到的提案是否在相同预算下达到覆盖、多样性和最强基线优势门槛。"
    ]
  },
  "criteria": [
    {
      "id": "A1",
      "status": "pass",
      "evidence": "保留已经核准的R4数据资格和固定划分结论。本轮读取的原R8授权、冻结合同及C9绑定继续指向同一data_lock；未发现本轮修改数据、资格或划分的证据。本次没有重做全量资格审计，也没有将未执行的最终测试读取门禁标为通过。"
    },
    {
      "id": "A2",
      "status": "pass",
      "evidence": "实际读取C9中的success_contract.json，SHA-256为e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef，status=FROZEN、frozen=true、unresolved_numeric_thresholds为空。原R8完整回复SHA为950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2；C9计划的六个政策字典与该回复逐字典一致。本次只裁定最后一轮的执行编排，不修改数值合同。"
    },
    {
      "id": "A3",
      "status": "partial",
      "evidence": "原C6/C8模型与开发训练证据保留；R9未产生最终250观察、106状态、1440比较单元、逐项最终门槛或独立复算结果。当前不能证明全部必需科学门槛达到。"
    },
    {
      "id": "A4",
      "status": "partial",
      "evidence": "保留R8已核准的开发接口证据。C9静态实现包含候选锁、512搜索与40复核计费、三代理逐约束判断、图和合成来源输出，但R9未执行这些最终路径或新增门禁，不能提前核准最终端到端行为。"
    },
    {
      "id": "A5",
      "status": "partial",
      "evidence": "A9完整清单、28项必读文本和14个C9源码均已实际读取并核对；失败回执与源码相符。仍缺最终评价、全量独立重载复算、A10版本闭合及本会话最终研究核准。当前zenodo_access为空，没有本轮秘密链接撤销事项。"
    }
  ],
  "required_fixes": [
    {
      "id": "R10-F1",
      "location": "src/coarsepgnn_inverse/final_common.py:model_bindings，现第114至145行",
      "problem": "第140行强制要求所有历史checkpoint条目包含bytes，导致合法的C6旧清单无法消费，并在该条目的SHA核验之前抛错。",
      "required_change": "对每个checkpoint始终读取实际文件、计算并核对原声明SHA，始终记录实际大小；仅在原条目声明bytes时额外核对大小。缺少声明时记录declared_bytes=null，不补写历史清单、不跳过SHA、不更换生产者身份。",
      "validation": "在R10正式预算内同时验证实际C6旧格式、实际C8主模型和逆向单阶段格式；用隔离夹具验证错误SHA、错误已声明大小、文件缺失均被拒绝。随后逐项核对全部八套模型清单中的实际权重和原14项基础资产。"
    },
    {
      "id": "R10-F2",
      "location": "src/coarsepgnn_inverse/final_round.py:run及CLI入口",
      "problem": "现有run每次建立新的started和started+14390期限；分别调用两次会形成两个独立预算。",
      "required_change": "实现一个R10合并监督器，只建立一次计费起点和绝对截止。先启动评价工作进程；评价完整并锁定科学产物清单后，再启动不同PID的全新Python复算进程。两个工作进程、测试子进程和所有实际检查继承同一个截止，不能重新起算14400秒。",
      "validation": "用合成工作进程验证第二阶段只能使用剩余时间、第一阶段失败时不启动第二阶段、到期终止活动进程组、输出目录不可覆盖。真实回执记录外层起点、同一绝对截止、两个不同工作进程PID、开始结束时间及最终总耗时。"
    },
    {
      "id": "R10-F3",
      "location": "src/coarsepgnn_inverse/final_evaluation.py:check_inputs、最终计划及外部授权回执",
      "problem": "C9只绑定原R8最终评价授权；R10合并恢复不能把本次回复冒充包含全部科学政策的R8回复，也不能静默覆盖旧失败分支。",
      "required_change": "保留原R8完整回复及其六个政策字典，另行绑定本次R9评审完整正文及round10_recovery_authorization。仅应用本回复列明的轮次、消费者版本、共享截止、阶段前置条件及失败分支覆盖。冻结实际C10后生成外部回执，记录真实C10、计划SHA、源码清单SHA和两份评审正文SHA；未来A10通过后提交回执绑定，避免自引用。",
      "validation": "运行前同时验证两份评审身份和SHA、六个基础政策字典相等、允许覆盖项精确匹配、八套模型生产者及所有不可变散列未变。篡改政策、模型集合、合同、面板或任一绑定必须在开窗前拒绝。"
    },
    {
      "id": "R10-F4",
      "location": "src/coarsepgnn_inverse/final_evaluation.py:load_models、validate_compatibility及preopening gate",
      "problem": "R9尚未实际加载并验证最终入口所需权重。FrozenDesignModels构造对象时存在惰性加载，仅生成model_load_receipt不能证明实际权重接线通过。",
      "required_change": "保留全部八套模型的原validation锚点兼容性验证：三套主模型的前向、八套逆向的239条单目标观察和63个核准双目标状态，均与原R8源精度逐项输出比较。不得为节省时间改成仅检查文件名、清单或构造空对象。",
      "validation": "实际触发对应checkpoint加载、内部身份与state_dict核验；逐观察ID、协议、配比窗口、概率因子和预测按原比较规则匹配。全部通过后才能生成最终锁和唯一窗口；任一失败即结束最后一轮并保留证据。"
    },
    {
      "id": "R10-F5",
      "location": "final_common.py的no_fitting和test_read_guard、final_evaluation.py的pytest子进程、final_round.py失败回执",
      "problem": "R9尚未进入审计上下文；现有父进程Python补丁不会自动覆盖pytest子进程，现有成功路径审计文件和固定optimizer_updates=0字段也不足以证明任意中途失败的执行状态。",
      "required_change": "对实际评价、覆盖层和复算分别记录审计进入及结束状态；尽可能在finally保存真实调用计数。测试子进程须独立安装封存测试文件拒绝规则并保留自身回执，不能用父上下文计数代替子进程证据。未进入或被强制终止而无法完成审计时明确记录NOT_ENTERED或INCOMPLETE，不伪造通过。复算阶段失败回执应引用评价目录的真实窗口状态，不能因本子目录尚无窗口文件而误报从未开窗。",
      "validation": "实际open正负对照、拟合和更新拒绝夹具、子进程隔离及故障注入均计入R10；真实运行审计与阶段控制流一致。任何零更新声明须说明来自实际拦截计数还是已核实的前置失败控制流。"
    },
    {
      "id": "R10-F6",
      "location": "最终评价与独立复算产物闭合",
      "problem": "最终科学结果和独立复算仍全部缺失，不能以本次静态审查替代。",
      "required_change": "按原冻结政策完成全部250观察、106共同状态和1440比较单元，再由独立新进程重算所有锚点、原始候选池、top20复核、排名、分层、统计和门槛。保留R8负结果，区分最终科学失败与工程未完成。",
      "validation": "逐观察和状态身份闭合；1440任务唯一且完整；候选锁先于跨种子复核；所有实际评分、图、计量和来源可追溯；复算预测满足原1e-5规则，身份、合法性、逐约束布尔和门槛结论完全一致。未完整完成时列出准确缺失任务，不给研究PASS。"
    }
  ],
  "round10_recovery_authorization": {
    "status": "CONDITIONALLY_AUTHORIZED",
    "round": 10,
    "authorization_scope": "一次最后正式轮次中的最小工程修复验证、唯一最终评价和独立全量复算；不包含新增拟合或模型选择。",
    "scientific_policy_source": {
      "repository": "WangGroupFDU/CoarsePGNN",
      "read_at_commit": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
      "path": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-5f082242266a.md",
      "request_id": "pu-inverse-20261111-5f082242266a",
      "bytes": 80670,
      "sha256": "950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2",
      "inherited_policy_dictionaries": [
        "final_evaluation_authorization",
        "test_condition_inference_policy",
        "final_window_policy",
        "final96_ablation_policy",
        "scoring_and_acceptance_policy",
        "independent_recomputation_authorization"
      ],
      "inheritance_rule": "以上六个原字典完整保留为科学政策基底。仅本对象supersedes中的明确编排覆盖项具有优先效力；未列项目全部沿用原值、公式、分母、容差和失败解释。禁止通过任意字典合并新增科学选择自由度。"
    },
    "recovery_review_binding": {
      "request_id": "pu-inverse-20261111-8bc39ca44eed",
      "reviewed_code_sha": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
      "reviewed_artifact_sha": "212cb94afe4ac26274588c7a233d7ab4bcae43ad",
      "complete_reply_sha256": null,
      "binding_rule": "将本完整回复归档后计算其实际SHA，并在C10外部授权回执中填写和验证。null表示尚未产生的真实散列，不是允许运行时保留的占位值。"
    },
    "supersedes": [
      {
        "policy_location": "final_window_policy.recovery中关于R9未完成时不得将R10用于新增搜索的分支",
        "replacement": "仅针对本次已证实在开窗前终止的R9，允许把尚未发生的唯一最终评价及其原定搜索移入R10同一个合并包。没有第二次最终测试窗口，不适用于已经读取测试结果后的模型或策略修订。"
      },
      {
        "policy_location": "independent_recomputation_authorization.failure_boundary",
        "replacement": "本次不再要求R10仅审计R9的四个失败产物；允许先完成尚未发生的最终评价。只有R10评价阶段工程完整并锁定产物后，才能在同一剩余预算内启动独立复算。若R10评价不完整，后续仍只保全和审计已产生证据，不补做新搜索。"
      },
      {
        "policy_location": "final_evaluation_authorization.round及相关C9消费者引用",
        "replacement": "最终评价在round=10执行，使用预开窗冻结的实际C10消费者；所有模型继续保留原C6或C8生产者。原R9失败及其4.40927419392392秒保留。"
      },
      {
        "policy_location": "independent_recomputation_authorization.precondition、implementation中的R9输出和C9实现引用",
        "replacement": "独立复算消费R10评价阶段的完整不可覆盖产物、实际scientific_inventory SHA和同一final_window；两个入口均属于首次测试读取前冻结的同一C10。独立性由另一个全新Python进程、重新加载权重和重新计算保证。"
      },
      {
        "policy_location": "两个阶段各自maximum_wall_seconds的执行解释",
        "replacement": "R10整体最多14400秒，评价与复算共享唯一绝对截止；不是评价14400秒加复算14400秒。"
      }
    ],
    "budget": {
      "maximum_wall_seconds_total": 14400,
      "maximum_total_rounds": 10,
      "maximum_total_compute_seconds": 259200,
      "formal_rounds_before_start": 9,
      "cumulative_compute_seconds_before_start": 1848.0574975546915,
      "clock_rule": "正式外层任务只建立一次单调时钟起点t0。硬截止为t0加14400秒与累计总预算剩余期限中的较早者；可在该期限内部预留10秒收尾。所有子进程只接收这一绝对截止和剩余时间，不重新计时。",
      "included_work": [
        "实际清单、源码、权重和输入校验",
        "全部合成和真实开发输入门禁",
        "八模型validation兼容性核对",
        "测试条件推理编码",
        "实测锚点评分",
        "候选搜索和跨种子复核",
        "统计与门槛计算",
        "独立全量复算及其必要输入校验",
        "同一执行内的日志、产物锁和失败收尾"
      ],
      "free_preflight_allowed": false,
      "eleventh_round_allowed": false
    },
    "training_authorization": {
      "status": "PROHIBITED",
      "new_supervised_stages": 0,
      "optimizer_updates": 0,
      "new_distillation": false,
      "new_configuration_selection": false,
      "model_seeds": [
        20261111,
        20261112,
        20261113
      ],
      "selected_configuration": "D0",
      "model_count": 8
    },
    "immutable_inputs": {
      "success_contract_sha256": "e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef",
      "data_lock_sha256": "d6ea2d43d60ba12784e91f03d37515210e300b0144597fa12d0412162b06f52d",
      "selection_lock_sha256": "af2022579871ef286c67e6641b22bb97fdf921ff47be94cad372d99a6d5ad41d",
      "base_inputs_ready_sha256": "efc85171ce01657bba50e4d362a52707666e6b0197eb462d0876bffb13d88dfd",
      "fixed96_requests_sha256": "39c55cbc533acaa51b634fd7c47257f98d6b5720dd6721c850d92003c29815ec",
      "main_20261111_checkpoint_manifest_sha256": "0a5d20560f7a11317cf179dbe13ebbd70999380e910aa13b828ef341edcaff78",
      "main_20261112_checkpoint_manifest_sha256": "12dc8314ea3d5b63cabf55fe3344a1472ef481830a7e0857c1aabb619a781768",
      "main_20261113_checkpoint_manifest_sha256": "1e3fe9edcb6e622ce69d41231baea5a8fd6f5d369b6005aefc8105475c0c4c24",
      "no_target_20261111_checkpoint_manifest_sha256": "7c55e1c518161ea060681059f9977054a59eeb25e26bdc39ade916e455bd0553",
      "no_target_20261112_checkpoint_manifest_sha256": "cbbe30ba61aca49008d66f94f027dd6acdd4335f4de7e4d39dd83e76de5842a4",
      "no_target_20261113_checkpoint_manifest_sha256": "f1374936b4feddfdc68b9ddcef3a0ffce74978dca14fa48f871fe5726272fd1d",
      "k1_20261111_checkpoint_manifest_sha256": "3ae66fd98d1950900ec0ebb6614dca692cadbd0e8aebd3cdf9d0eac2c03fc4d8",
      "no_test_conditions_20261111_checkpoint_manifest_sha256": "37ef781bae3794ab169ea863f8b0688e196be05c88f3c519eee43441f902c65c",
      "identity_rule": "C4/A4保留资格生产者身份；主种子20261111保留C6/A6生产者身份，另外两个主种子及五个逆向消融保留C8/A8生产者身份。C9/A9是本次被评审失败版本，C10只是新的评价消费者。清单中的权重、共享前向生产者、14项资产和内部checkpoint身份全部保持。"
    },
    "allowed_implementation_changes": [
      "兼容原checkpoint清单是否包含bytes，强化实际SHA和大小回执，不改历史清单。",
      "增加一个共享绝对截止的合并监督器，顺序启动评价和独立复算两个全新进程。",
      "增加本次恢复授权绑定、C10计划及外部源码回执；更新对应CLI和执行文档中的阶段与目录。",
      "补充实际清单格式、错误SHA和大小、共享截止、阶段失败、子进程测试隔离和失败审计的必要门禁。",
      "修正未进入审计、强制终止和引用窗口状态的回执表达；不改变任何预测或评分。",
      "可补充复算列表长度与ID集合的显式覆盖断言，但不得改变原评分分母、数值公式或比较容差。"
    ],
    "forbidden_changes": [
      "修改任何原监督权重、模型架构、优化器训练结果、数据资格、划分或原料库。",
      "增加第五配置、重选D0、选择较好种子、重新拟合变换、词表或PLE。",
      "改变最终目标、250观察或106状态分母、固定96请求、1440任务矩阵。",
      "改变采样、比例细化、检索罚值、排序、去重、每组合上限、512搜索或40复核预算。",
      "改变任何原合同门槛、概率下限、源float64窗口、物理容差、统计规则或1e-5复算规则。",
      "在首次测试读取后修改代码再继续评价，或者将工程失败改写为科学通过。"
    ],
    "preopening_freeze": [
      "静态准备中一次性完成限定修改，提交实际C10；不预填未来提交SHA。",
      "同时冻结评价入口、复算入口、合并监督器、调用依赖、计划和测试代码。",
      "生成外部回执，绑定C10、完整源码清单、计划、原R8评审、本次R9评审和全部不可变输入；原R8六个政策字典保留。",
      "正式R10启动后验证实际部署HEAD、源码字节、确定性环境、八套模型和14项资产；执行合成门禁及八套模型真实validation核对。",
      "全部门禁通过后建立不可覆盖的task_manifest和final_lock，再原子创建唯一final_window。任何门禁失败都消耗最后一轮，不能改称免费预检重跑。"
    ],
    "execution_sequence": [
      {
        "phase": "PREOPENING",
        "action": "验证限定修复、双重授权、全部实际输入和模型兼容性；测试标签仍封存，正式96性能仍不计算。",
        "completion_condition": "所有门禁通过，真实零拟合和隔离证据完整，250观察ID、106状态ID及1440任务清单锁定。"
      },
      {
        "phase": "FINAL_EVALUATION",
        "action": "在第一个工作进程中打开唯一窗口，按原规则生成缺失prompt覆盖层，完成全部锚点、主方法、基线、消融和逐门槛报告。",
        "completion_condition": "产生完整final_result、全部任务产物及不可覆盖scientific_inventory；科学门槛失败正常记录，不阻止完成其余固定任务。"
      },
      {
        "phase": "EVALUATION_SEAL",
        "action": "评价工作进程退出后，由监督器核对退出状态、完整任务集合、科学产物清单及实际SHA，记录剩余时间。",
        "completion_condition": "评价工程完整且清单闭合才启动复算；不以numerical_contract_satisfied=true作为启动条件。"
      },
      {
        "phase": "INDEPENDENT_RECOMPUTATION",
        "action": "启动不同PID的全新Python解释器，重新加载同一C10、八套模型和冻结资产，消费同一窗口与评价清单，独立重编码并全量重算。",
        "completion_condition": "全部锚点、候选池、top20、图、计量、来源、预算、排名、统计和门槛比较完成；新提案数和优化器更新数均为0。"
      },
      {
        "phase": "CLOSURE",
        "action": "输出原合同逐项PASSED、FAILED或INCOMPLETE表，归档全部差异、负结果、未完成项及C/A/R绑定。",
        "completion_condition": "状态与真实产物一致；无论科学失败、工程失败还是超时，均不启动第11轮，不自动写研究PASS。"
      }
    ],
    "evaluation_and_recomputation_coverage": {
      "test_observations": 250,
      "test_target_counts": {
        "tg": 6,
        "tm": 13,
        "tensile_strength_ultimate": 110,
        "elongation_at_break": 121
      },
      "joint_two_target_states": 106,
      "joint_two_target_materials": 99,
      "joint_two_target_closure_groups": 4,
      "joint_three_or_four_target_truth_count": 0,
      "main_forward_observation_predictions": 750,
      "eight_inverse_models_single_target_scores": 2000,
      "joint_state_scores_per_inverse_model": 106,
      "main_and_baseline_units": 1152,
      "ablation_units": 288,
      "total_units": 1440,
      "fixed_requests": 96,
      "request_target_count_partition": [
        24,
        48,
        8,
        16
      ],
      "protocol_partition": {
        "A": 60,
        "B": 36
      },
      "search_budget_per_unit": 512,
      "independent_recheck_budget_per_unit": 40,
      "top_k": 20,
      "full_pool_rule": "复算全部原始提案槽位，包括非法、重复和失败槽，不能仅重读汇总或只复算已返回top20。共享池末端排序消融复用主方法原池，不增加新的搜索提案。",
      "interpretation": "三代理一致命中属于固定计算筛选证据。三或四性质联合实测仍为0；Tg测试仍只有一个来源闭包，不能生成总体bootstrap区间或把三个种子当作独立实验。"
    },
    "condition_overlay_rule": "完整继承原test_condition_inference_policy。只使用原固定Qwen权重和既定推理配置，为最终窗口缺失prompt建立独立追加缓存；保持原14资产、词表、PLE、标签变换不变，未知assay使用原0索引。新增缓存和条件张量在窗口内锁定并由复算进程独立重建；不得删除缓存外测试样本，也不得改变prompt规则或重新拟合。",
    "recomputation_rule": "完整继承原independent_recomputation_authorization.coverage和comparison_rules。前向预测按abs差除以原delta不超过1e-5；逆向及条件张量使用原F4规范化1e-5规则。观察、状态、协议、mask、原料、源配比窗口、候选次序、合法性、概率触底、逐约束布尔值和最终门槛结论严格一致。",
    "failure_and_recovery_boundary": [
      "R10采用新且不可覆盖的输出目录，R9四个失败产物和原A9保持不变。",
      "本授权不增加外层自动重跑。评价阶段工程失败或超时后，不启动独立复算以补做评价，也不修改代码后重新启动科学计算。",
      "复算只消费完整评价产物；复算失败保留实际差异和已完成范围，不能选取较有利输出替换原结果。",
      "既有单元幂等逻辑不得改变原候选、顺序或账本。没有完整持久状态的部分搜索仍明确停止，不隐式重采样。",
      "科学门槛失败时，在剩余同一预算内继续完成其余原定任务与独立复算；不能调门槛、改分母或换模型。",
      "到达绝对截止时终止活动进程组，记录完成任务ID、缺失任务ID、已开窗状态和审计完整性；不能为凑齐结果延长期限。",
      "可以只读重取和归档已经产生的证据；不得将这种传输恢复变成第11轮科学计算。"
    ],
    "feasibility_assessment": {
      "evidence_path": "research-runs/pu-inverse-20261111/round-008/evidence-00/search/development_search_summary.json",
      "observed_development_comparisons": 96,
      "observed_elapsed_seconds": 328.2951095489552,
      "simple_linear_estimate_one_phase_seconds": 4924.426643234328,
      "simple_linear_estimate_two_phases_seconds": 9848.853286468657,
      "judgment": "14400秒可作为合理的有界尝试，但现有证据不足以保证完成。简单线性外推尚未充分计入最终任务组成、新条件编码、八模型锚点、跨种子40复核、完整候选图输出及额外I/O。接近超时不能成为减少候选、删除请求或跳过复算的理由。"
    },
    "required_final_deliverables": [
      "C10和实际A10、两份完整评审正文、冻结合同、data_lock、selection_lock、96面板及八模型生产者的外部绑定。",
      "单一外层起点和截止、两个独立工作进程身份、完整计费及停止回执。",
      "旧新清单兼容、实际权重SHA和大小、八模型真实validation兼容性及测试隔离门禁。",
      "唯一final_window、完整task_manifest、最终输入锁和条件缓存覆盖层。",
      "全部逐观察、逐状态、逐候选、逐协议和逐约束结果以及搜索与复核账本。",
      "独立复算的逐项差异、完整覆盖分母、排名和布尔一致性、源码及原评价产物未变回执。",
      "原合同每项必需门槛的结果、原R8负结果、所有工程错误和准确未完成清单。",
      "可实际读取的manifest及必读材料；大文本提供完整原字节读取方式，二进制提供实际校验和可解释的重载回执。"
    ]
  },
  "reading_verification": {
    "method": "通过GitHub工具按固定完整提交取得正文，解析JSON并计算原UTF-8字节SHA-256；未执行本地命令、模型训练、候选搜索或远端科学计算。cn2运行事实来自已归档原始回执和日志。",
    "artifact_commit": "212cb94afe4ac26274588c7a233d7ab4bcae43ad",
    "source_commit": "90bcf23a0e7f4508ca255f0becaa9be899a538d0",
    "manifest_bytes": 12044,
    "manifest_sha256": "d91cab30b69deea38f83a4f22268b708722e26fa7905088ed27a5f699acabe61",
    "required_files": 28,
    "required_files_read": 28,
    "required_file_hash_and_size_matches": 28,
    "changed_source_files_read_directly": 14,
    "source_export_comparison": "七个C9_source_part文件逐行重组后，与固定C9直接取得的14个原文件完整相同；原行号连续，无省略行，原字节SHA和大小与导出索引匹配。",
    "large_file_reads": [
      {
        "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/execution.json",
        "method": "Git Blob完整正文",
        "bytes": 1720185,
        "sha256": "32c2f58a31c16e974f1c9f3bb7e715f7bc5b39f7c679232320d4734c5df38cec",
        "parsed_completely": true
      },
      {
        "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/experiment.json",
        "method": "Git Blob完整正文",
        "bytes": 1757180,
        "sha256": "151639d84719ad37460b0168ca865b1e1b01facaa44ab2dadb0af332e99e3a06",
        "parsed_completely": true
      }
    ],
    "policy_verification": "实际读取80670字节的原R8完整回复，其SHA匹配；C9计划中的六个政策字典与对应原字典完全相同。",
    "limitations": [
      "本轮未下载、加载或解释PT权重二进制，二进制内容状态为unread；清单读取不能替代R10实际权重校验及模型重载。",
      "round009_source_manifest列有157个文件；本轮直接审查其中14个新改文件及列入read_files的相关依赖，不声称重新逐行审查全部157个历史文件。",
      "未读取封存测试标签，未执行任何最终模型性能或正式96面板计算。",
      "未生成的preopening_tests、final_window、最终评价和独立复算产物不能标为read或passed。"
    ]
  },
  "read_files": [
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/manifest.json",
      "status": "read",
      "evidence": "完整清单含28项required_read；execution_status=failed、provenance_valid=true。progress.status=none的两项引用SHA与实际失败及回读回执匹配。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C6_checkpoint_manifest.json",
      "status": "read",
      "evidence": "identity.code_sha为C6；checkpoints九个条目仅有path和sha256，没有bytes，直接解释C9第140行失败。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C8_main12_checkpoint_manifest.json",
      "status": "read",
      "evidence": "identity对应C8、种子20261112；九个checkpoint含bytes，且显式保留前向及逆向生产者身份。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C8_no_target11_checkpoint_manifest.json",
      "status": "read",
      "evidence": "逆向单阶段checkpoints仅含inverse_full，声明bytes=14552232；forward_producer指向原C6主模型，不能要求它具有九个新checkpoint。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_export_manifest.json",
      "status": "read",
      "evidence": "绑定C9、14个原源码的SHA及大小、七个导出分片SHA；与直接取得的C9原文件和分片重组结果一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_01.md",
      "status": "read",
      "evidence": "round009_plan.json原行1至320，包含固定输入、模型与原R8政策；编号和内容与原文件一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_02.md",
      "status": "read",
      "evidence": "包含计划余部、执行文档、CLI及anchor_evaluation.py原行1至160；确认计划单轮14400秒、最终入口分支和源float64单目标锚点路径。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_03.md",
      "status": "read",
      "evidence": "包含双目标锚点余部、final_common和final_compare；第140行bytes问题、单一窗口创建和精确字段比较均可定位。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_04.md",
      "status": "read",
      "evidence": "完整final_conditions及final_design；新增缓存只追加，top20先锁定再复核，逻辑搜索和复核分别限制为512与40。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_05.md",
      "status": "read",
      "evidence": "完整final_evaluation；check_inputs先于第170行守卫，实际兼容性核对先于开窗，tasks_for固定1440任务。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_06.md",
      "status": "read",
      "evidence": "完整final_metrics及final_recompute原行1至160；包含未见规格Hδ、原门槛矩阵、原池逐槽重算和全部锚点复算。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/C9_source_part_07.md",
      "status": "read",
      "evidence": "包含复算余部、监督器、预处理禁止拟合和最终门禁测试；监督器每次run重新建立期限，测试尚未在R9执行。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/experiment_reading_index.json",
      "status": "read",
      "evidence": "outputs_summary.count=10112，并明确not_a_substitute_for_original_experiment=true。本次另外取得并解析了完整experiment和execution，未把该索引作为原件替代。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/failure.json",
      "status": "read",
      "evidence": "error=KeyError('bytes')；traceback定位final_evaluation.py:164、62和final_common.py:140；test_window_exists=false。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/independent_collection_receipt.json",
      "status": "read",
      "evidence": "记录四个实际失败文件的大小和SHA回读一致；final_window、测试访问日志、preopening tests和搜索单元目录均不存在。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round009_authorization.json",
      "status": "read",
      "evidence": "consumer为实际C9，绑定计划SHA、R8完整回复SHA、157项源码清单SHA及25个静态输入；授权更新数0、墙钟14400。它是原R9授权，不能直接充当R10合并授权。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round009_environment_check.json",
      "status": "read",
      "evidence": "属于部署前环境观测，记录Python、Torch、PyG和CUDA等；其中部分新部署文件当时不存在，不应当作启动时C9身份回执。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round009_failure_diagnosis.json",
      "status": "read",
      "evidence": "保留只读诊断结果：C6条目缺bytes、C8条目有bytes，输出目录仅有四个失败产物。诊断没有提供任何最终模型性能。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round009_resource_check.json",
      "status": "read",
      "evidence": "启动前READY_FOR_FORMAL_R9记录实际HEAD=C9、157源码和25静态输入、clean=true及RTX4090资源；这是部署就绪证据，不是评价通过证据。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round009_source_manifest.json",
      "status": "read",
      "evidence": "consumer_code_sha=C9，files含157项源码或配置散列；导出的14项新改文件与直接取得的原文SHA匹配。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round_result.json",
      "status": "read",
      "evidence": "phase=evaluate、status=FAILED、returncode=1、completed_units=0、test_window_exists=false；内部监督耗时4.173378414008766秒，正式计费以execution的4.40927419392392秒为准。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/round_start.json",
      "status": "read",
      "evidence": "记录evaluate工作进程命令和单调时钟；started=626192.366193936、deadline=640582.366193936，内部预留10秒收尾。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/worker.log",
      "status": "read",
      "evidence": "完整原始traceback与failure.json一致，终止于item['bytes']；没有preopening tests、模型推理或搜索成功日志。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/report_erratum.md",
      "status": "read",
      "evidence": "仅将报告中的开窗文件名更正为final_window.json，与实际源码一致；不改变失败状态和预算。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/report.md",
      "status": "read",
      "evidence": "将本轮记为开窗前工程失败，承认门禁与最终评价未执行，并明确原授权下没有自行继续搜索权限；与原始日志一致。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/execution.json",
      "status": "read",
      "evidence": "完整1720185字节原件经Git Blob取得并核SHA；status=failed、provenance_valid=true，1项检查退出1，16项输入、3项列入预期清单的已生成输出、10109项未生成输出，正式耗时4.40927419392392秒。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/experiment.json",
      "status": "read",
      "evidence": "完整1757180字节原件经Git Blob取得并核SHA；解析10112个唯一预期输出路径、16项输入、固定最终计划及14400秒限制。枚举输出不是已完成结果。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/goal.md",
      "status": "read",
      "evidence": "原目标和A1至A5保持：可追溯实测、无泄漏评价、训练前成功合同、最终独立核准；没有以工程可运行替代科学成功。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-009/evidence-00/checks/check-000.log",
      "status": "read",
      "evidence": "外层CLI收到工作进程退出1后抛CalledProcessError，与worker内部KeyError共同形成完整失败链。"
    },
    {
      "path": "configs/inverse/round009_plan.json",
      "status": "read",
      "evidence": "C9完整原文435行；八套模型、三主种子、五消融、25静态验证输入和六个原R8政策字典均固定；计划SHA为8beb2f4fb0f34b7b9e11447408c3bfc8bf95e0481793a43523a5619f07f5f683。"
    },
    {
      "path": "docs/pu_final_evaluation.md",
      "status": "read",
      "evidence": "C9原行15至17及48至63分别描述逐轮截止和另轮复算。R10合并恢复需要同步更新执行编排说明，科学结果解释仍保留。"
    },
    {
      "path": "src/coarsepgnn_inverse/__main__.py",
      "status": "read",
      "evidence": "C9原行63至71按是否提供recompute-reference选择一个phase并调用run；当前没有单外层合并两个phase的入口。"
    },
    {
      "path": "src/coarsepgnn_inverse/anchor_evaluation.py",
      "status": "read",
      "evidence": "C9原行50至63保留源float64 v；87至126使用完整来源图和真实配方独立评分；164至234仅枚举合格双目标状态并先状态内、再材料内汇总。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_common.py",
      "status": "read",
      "evidence": "C9原行114至145核模型绑定，第140行bytes为确定故障；66至102实现零拟合和真实open守卫；157至203固定250/106分母及唯一窗口。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_compare.py",
      "status": "read",
      "evidence": "C9原行19至48严格比较字段集合、列表长度、布尔及身份，其他有限浮点使用1e-5；51至68按完整观察ID核对前向预测及物理delta误差。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_conditions.py",
      "status": "read",
      "evidence": "C9原行31至42核原Qwen，54至68禁止覆盖旧键；98至114只允许固定文本推理并复用PLE与词表；123至163独立重建比较和原14资产复核。未发现需要重拟合的实现要求。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_design.py",
      "status": "read",
      "evidence": "C9原行58至79先保存搜索和top20锁；82至116逐种子复核并保持幂等；127至175禁止隐式重采样；177至218计算同一候选三模型共同支持并核512+40预算。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_evaluation.py",
      "status": "read",
      "evidence": "C9原行29至63核授权和固定输入；99至118真实validation兼容核对；121至158锁1440任务；164至187明确失败点、门禁和开窗的先后；202至232完成后才写最终结果。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_metrics.py",
      "status": "read",
      "evidence": "C9原行58至59已补未见规格Hδ及统计；125至146核1440任务和96请求分层；149至196按原合同计算逐种子门槛及最强基线比较，不要求指标上涨才报告进展。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_recompute.py",
      "status": "read",
      "evidence": "C9原行24至91逐槽重算整个原池的合法性、评分、计费和排名；122至207重载八模型、重编码、复算全部任务及门槛。只在完整评价产物成立时允许进入。"
    },
    {
      "path": "src/coarsepgnn_inverse/final_round.py",
      "status": "read",
      "evidence": "C9原行27至47每次调用重建截止且只启动一个phase；80至95成功后生成清单、失败后留traceback。R10必须改为一个共享截止的顺序双进程监督。"
    },
    {
      "path": "src/coarsepgnn_inverse/preprocessing_reuse.py",
      "status": "read",
      "evidence": "C9原行13至37禁止变换、词表、PLE及目录特征重新拟合；allow_text_inference只控制文本编码例外，不解除统计拟合禁令。"
    },
    {
      "path": "tests/test_inverse_final.py",
      "status": "read",
      "evidence": "C9原行29至90包含实际open、零更新、缓存和未知assay夹具；121至150覆盖先锁后复核和原池重算；153至194覆盖材料权重、单闭包及1440任务。文件存在不代表R9执行通过。"
    },
    {
      "path": "src/coarsepgnn_inverse/inference.py",
      "status": "read",
      "evidence": "C9原行55至114核原生产者、资产及共享前向来源；116至134才实际加载checkpoint；136至158拒绝缓存外协议；200至261区分前向评分和逆向窗口概率。"
    },
    {
      "path": "src/coarsepgnn_inverse/metrics.py",
      "status": "read",
      "evidence": "C9原行13至65为材料等权与train中位数基线；68至131为源float64稳定窗口质量及1e-12逐因子下限；134至166按闭包提供bootstrap、留一组或单闭包null。"
    },
    {
      "path": "src/coarsepgnn_inverse/access.py",
      "status": "read",
      "evidence": "C9原行10至41在hash前拒绝开发路径误指测试；44至72最终读取要求冻结合同、最终锁、指定对象SHA和测试文件身份一致。"
    },
    {
      "path": "configs/inverse/success_contract.json",
      "status": "read",
      "evidence": "C9完整冻结合同SHA匹配；包含有限集合结论边界、全部数值门槛、512+40预算、三种子、消融、统计和失败规则，本次不修改。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-5f082242266a.md",
      "status": "read",
      "evidence": "从C9实际取得80670字节完整R8回复，SHA=950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2；核实原失败分支确实限制R9未完成时R10只审计，故本回复明确而有限地覆盖该分支。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-008/evidence-00/search/development_search_summary.json",
      "status": "read",
      "evidence": "从固定A8读取并核SHA；24请求、96比较、耗时328.2951095489552秒，主方法/检索/随机/组合搜索分别23/15/23/24命中，formal96_performance_computed=false。用于粗略成本估计，不当作最终指标。"
    },
    {
      "path": "research-runs/pu-inverse-20261111/round-008/evidence-00/no_target_paired_validation.json",
      "status": "read",
      "evidence": "从固定A8读取884236字节完整JSON并核SHA；每种子239单目标观察、63双目标状态，四任务平均条件改进依次为-0.0007415389344970513、-0.06558774382059454和0.5233393450862655 nat。"
    }
  ],
  "missing_evidence": [
    "尚不存在修复后实际C10、完整本回复归档SHA和双重授权外部绑定；必须在正式R10及首次开窗前形成。",
    "R10旧新清单门禁、八套实际模型validation兼容性、子进程测试隔离和真实零拟合回执尚未产生。",
    "唯一最终窗口、完整250观察及106共同状态评分、1440比较单元和原合同逐项最终门槛尚未产生。",
    "独立新进程对全部最终锚点、候选池、top20、缓存、统计及门槛的复算差异证据尚未产生。",
    "实际A10、完整产物清单、最终C/A/R绑定及本会话研究核准尚未产生。"
  ],
  "optional_suggestions": [
    "在final_recompute的逆向逐项zip比较之前增加显式长度和ID集合断言，可提高差异定位清晰度；现有score_anchors已约束250观察及106状态，因此这不是新增科学门槛，也不需要单独计算轮次。",
    "最终报告分别展示工程执行完整性、冻结科学门槛结果和证据完整性，避免把独立复算一致误写为模型性能通过。"
  ]
}
```