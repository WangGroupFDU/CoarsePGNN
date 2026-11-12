## configs/inverse/round010_plan.json lines 1-120

```text
1: {
2:   "schema_version": 1,
3:   "round": 10,
4:   "timeout_seconds": 14400,
5:   "contract": "configs/inverse/success_contract.json",
6:   "data_lock": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_004_qualification/data_lock.json",
7:   "fixed_requests": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_004_qualification/fixed_96_requests.json",
8:   "selection_lock": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/selection_lock.json",
9:   "synthesis_replay": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/search/adapter/adapter_replay.json",
10:   "review": "research-runs/pu-inverse-20261111/reviews/pu-inverse-20261111-5f082242266a.md",
11:   "authorization": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round010_authorization.json",
12:   "models": {
13:     "main_20261111_D0": {
14:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_006_fit",
15:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/D0/evaluation",
16:       "producer_code_sha": "94f8f785d5c3bfb4197903d5588e7a5490d3e80d",
17:       "producer_artifact_sha": "cab04775f48555995ed448deffe0e0c1bfa57029",
18:       "checkpoint_manifest_sha256": "0a5d20560f7a11317cf179dbe13ebbd70999380e910aa13b828ef341edcaff78"
19:     },
20:     "main_20261112": {
21:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/main_20261112/training",
22:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/main_20261112/evaluation",
23:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
24:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
25:       "checkpoint_manifest_sha256": "12dc8314ea3d5b63cabf55fe3344a1472ef481830a7e0857c1aabb619a781768"
26:     },
27:     "main_20261113": {
28:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/main_20261113/training",
29:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/main_20261113/evaluation",
30:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
31:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
32:       "checkpoint_manifest_sha256": "1e3fe9edcb6e622ce69d41231baea5a8fd6f5d369b6005aefc8105475c0c4c24"
33:     },
34:     "no_target_values_20261111": {
35:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261111/training",
36:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261111/evaluation",
37:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
38:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
39:       "checkpoint_manifest_sha256": "7c55e1c518161ea060681059f9977054a59eeb25e26bdc39ade916e455bd0553"
40:     },
41:     "no_target_values_20261112": {
42:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261112/training",
43:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261112/evaluation",
44:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
45:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
46:       "checkpoint_manifest_sha256": "cbbe30ba61aca49008d66f94f027dd6acdd4335f4de7e4d39dd83e76de5842a4"
47:     },
48:     "no_target_values_20261113": {
49:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261113/training",
50:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_target_values_20261113/evaluation",
51:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
52:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
53:       "checkpoint_manifest_sha256": "f1374936b4feddfdc68b9ddcef3a0ffce74978dca14fa48f871fe5726272fd1d"
54:     },
55:     "k1_20261111": {
56:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/k1_20261111/training",
57:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/k1_20261111/evaluation",
58:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
59:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
60:       "checkpoint_manifest_sha256": "3ae66fd98d1950900ec0ebb6614dca692cadbd0e8aebd3cdf9d0eac2c03fc4d8"
61:     },
62:     "no_test_conditions_20261111": {
63:       "root": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_test_conditions_20261111/training",
64:       "validation": "/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated/no_test_conditions_20261111/evaluation",
65:       "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
66:       "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
67:       "checkpoint_manifest_sha256": "37ef781bae3794ab169ea863f8b0688e196be05c88f3c519eee43441f902c65c"
68:     }
69:   },
70:   "policies": {
71:     "final_evaluation_authorization": {
72:       "status": "CONDITIONALLY_AUTHORIZED",
73:       "round": 9,
74:       "maximum_wall_seconds": 14400,
75:       "maximum_total_rounds": 10,
76:       "maximum_total_compute_seconds": 259200,
77:       "budget_accounting": "限定修复后的无拟合检查、缓存推理、所有锚点评价、候选搜索、跨种子复核、统计与恢复计算全部计入同一个第九轮单调时钟预算；不能为静态准备之外的真实计算另设不计费任务。",
78:       "allowed_model_seeds": [
79:         20261111,
80:         20261112,
81:         20261113
82:       ],
83:       "allowed_configuration": "D0，hidden128、主MDN3、既有冻结权重；K=1和无测试条件仅使用已完成的指定消融权重。",
84:       "qualification_identity": {
85:         "code_sha": "93ca43c653a16ff97563c19d3fcc7c629e99d403",
86:         "artifact_sha": "dea09f0642902bbad2b845b112d4c5b0be049484",
87:         "data_lock_sha256": "d6ea2d43d60ba12784e91f03d37515210e300b0144597fa12d0412162b06f52d"
88:       },
89:       "current_review_identity": {
90:         "request_id": "pu-inverse-20261111-5f082242266a",
91:         "code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
92:         "artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704"
93:       },
94:       "consumer_identity": "在读取任何最终测试内容或计算正式96性能前，提交并记录实际C9。C9是冻结评价实现的消费者身份，不替换任何模型的生产者身份；未知的未来提交SHA不得预先填写为虚构值。",
95:       "immutable_inputs": {
96:         "success_contract_sha256": "e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef",
97:         "selection_lock_sha256": "af2022579871ef286c67e6641b22bb97fdf921ff47be94cad372d99a6d5ad41d",
98:         "base_inputs_ready_sha256": "efc85171ce01657bba50e4d362a52707666e6b0197eb462d0876bffb13d88dfd",
99:         "fixed96_requests_sha256": "39c55cbc533acaa51b634fd7c47257f98d6b5720dd6721c850d92003c29815ec",
100:         "policy": "保留原14项资产、候选库、规格特征与可见性、训练范围、原A/B协议、训练侧中位数基线规则和原始96请求字节。最终运行可生成附带种子和执行身份的任务封装，但不能改写原96请求。"
101:       },
102:       "frozen_model_manifests": [
103:         {
104:           "model": "main_20261111_D0",
105:           "producer_code_sha": "94f8f785d5c3bfb4197903d5588e7a5490d3e80d",
106:           "producer_artifact_sha": "cab04775f48555995ed448deffe0e0c1bfa57029",
107:           "checkpoint_manifest_sha256": "0a5d20560f7a11317cf179dbe13ebbd70999380e910aa13b828ef341edcaff78"
108:         },
109:         {
110:           "model": "main_20261112",
111:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
112:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
113:           "checkpoint_manifest_sha256": "12dc8314ea3d5b63cabf55fe3344a1472ef481830a7e0857c1aabb619a781768"
114:         },
115:         {
116:           "model": "main_20261113",
117:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
118:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
119:           "checkpoint_manifest_sha256": "1e3fe9edcb6e622ce69d41231baea5a8fd6f5d369b6005aefc8105475c0c4c24"
120:         },
```

## configs/inverse/round010_plan.json lines 121-240

```text
121:         {
122:           "model": "no_target_values_20261111",
123:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
124:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
125:           "checkpoint_manifest_sha256": "7c55e1c518161ea060681059f9977054a59eeb25e26bdc39ade916e455bd0553"
126:         },
127:         {
128:           "model": "no_target_values_20261112",
129:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
130:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
131:           "checkpoint_manifest_sha256": "cbbe30ba61aca49008d66f94f027dd6acdd4335f4de7e4d39dd83e76de5842a4"
132:         },
133:         {
134:           "model": "no_target_values_20261113",
135:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
136:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
137:           "checkpoint_manifest_sha256": "f1374936b4feddfdc68b9ddcef3a0ffce74978dca14fa48f871fe5726272fd1d"
138:         },
139:         {
140:           "model": "k1_20261111",
141:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
142:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
143:           "checkpoint_manifest_sha256": "3ae66fd98d1950900ec0ebb6614dca692cadbd0e8aebd3cdf9d0eac2c03fc4d8"
144:         },
145:         {
146:           "model": "no_test_conditions_20261111",
147:           "producer_code_sha": "1745621bffb1e7f721d706c7e5d3b6efcc454c51",
148:           "producer_artifact_sha": "025cd99703c63c5002aace37e307ec6905cf9704",
149:           "checkpoint_manifest_sha256": "37ef781bae3794ab169ea863f8b0688e196be05c88f3c519eee43441f902c65c"
150:         }
151:       ],
152:       "preopening_gates": [
153:         "实际代码、模型清单及其中全部权重文件、原预处理资产、合同、data_lock和96请求大小及SHA全部匹配；新旧生产者与C9消费者身份分开验证。",
154:         "新评价入口和第十轮入口均已实现、提交并冻结；开发入口仍拒绝测试访问。使用已知答案和开发样本完成无拟合兼容性检查，不能先读取测试再修复入口。",
155:         "最终任务清单覆盖250条观察、106个共同状态、96×3×4个主方法与基线任务，以及三个20261111正式面板消融；输出目录不可覆盖。",
156:         "验证缓存基础层不变、覆盖层权限受限、未知assay回退0索引、原PLE和数值变换直接重载，所有fit和优化器更新调用计数为0。",
157:         "验证top20先落盘后复核、固定预算、故障恢复幂等性和独立复算的身份核对；不借工程修复修改采样、细化、排序或去重规则。"
158:       ],
159:       "execution_sequence": [
160:         "通过全部开窗前门禁后，建立唯一final_window并按冻结文件身份读取最终测试；核对观察ID、共同状态ID和各层分母。",
161:         "按冻结规则生成缺失条件prompt的独立缓存覆盖层，保存并锁定新增缓存及条件张量身份。",
162:         "重载三个主种子的四目标前向和逆向模型，对全部测试观察完成前向实测核对、逆向窗口评分和与三个无目标数值模型的配对比较；完成106个共同状态及K=1、无测试条件的锚点报告。",
163:         "执行96请求、三个搜索种子和四种方法的1152个固定搜索任务；各自锁定top20后调用另外两个主种子的完整前向模型。",
164:         "完成20261111的K=1、无测试条件和共享提案池末端排序三组96请求消融，记录命中、多样性及全部规定分层。",
165:         "按原合同输出全部门槛矩阵、逐样本与逐候选证据、区间或不可估计原因、失败原因及最终窗口回执；不依据结果再改实现。"
166:       ]
167:     },
168:     "test_condition_inference_policy": {
169:       "authorization": "允许在最终窗口内对缺失测试条件prompt进行固定通用Qwen的推理编码；这属于冻结特征函数的应用，不属于重新拟合。",
170:       "base_model_revision": "959bed3521cc1e8714392c77e15ae213b00aa921",
171:       "base_weight_sha256": {
172:         "model-00001-of-00002.safetensors": "e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8",
173:         "model-00002-of-00002.safetensors": "ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1"
174:       },
175:       "qwen_config_sha256": "b15ef67fce0d65cb2155da30b5e4e072c44df3e12651fcb89a4aced539c20958",
176:       "encoding_rules": [
177:         "继续使用C8已有condition_record、raw_prompt_v2和字段注册规则；条件输入只能来自原合格记录的测试条件与带mask标量。性质真值、真实D/P/E/v、样品或来源ID不得增加为条件特征。",
178:         "先核对基础缓存，再按prompt SHA去重并排序，仅编码确实缺失的prompt；同一SHA必须对应完全相同文本。没有缺失时记录空覆盖层和0次编码调用。",
179:         "使用相同本地revision、分词器和元数据散列，CUDA bfloat16、SDPA、left padding、batch_size=4、truncation=false、最大1024 token；保持last-token pooling、全向量L2归一化、前32维再归一化及float32缓存输出。",
180:         "严格使用eval和inference_mode，无反向传播、无优化器、无监督更新；保留已有向量有限性和范数检查，不允许静默切换模型、精度、后端或截断输入。",
181:         "原14项资产作为只读基础层，新向量只写独立覆盖层。已有键冲突即失败，不覆盖原向量；所有三个主模型和五个逆向消融消费同一基础层与覆盖层。",
182:         "数值条件使用原PLE结点、溢出编码、字段顺序和mask；assay使用原训练词表，未知项沿用0索引。不得根据测试出现频率增加词表或调整变换。",
183:         "合格测试实测锚点通过专用历史条件入口评价，不要求它们属于公共接口的A/B默认协议。公共接口未知协议拒绝规则仍保留，不能为读取测试而静默扩展用户协议注册表。",
184:         "开窗前冻结缓存生成程序和规则；新增缓存本体的散列只能在最终窗口生成后写入不可覆盖的回执，再供第十轮重算使用。不得伪造预先已知的新缓存散列。"
185:       ],
186:       "required_receipt": [
187:         "基础Qwen权重、配置、分词器、提示词程序和运行环境散列",
188:         "新增prompt SHA、文本身份、token数、排序和批次边界",
189:         "覆盖层文件及条件raw/keys/values/mask张量散列",
190:         "原缓存与14项资产前后字节一致的核验",
191:         "0次fit、0次优化器更新及实际推理编码调用数",
192:         "超过长度、键冲突、非法向量或未知字段错误的原始失败记录"
193:       ]
194:     },
195:     "final_window_policy": {
196:       "logical_opening_count": 1,
197:       "definition": "一次性是一个不可覆盖、不可用于再选择模型的逻辑最终评价窗口，不是只允许一次文件系统open调用。第九轮评价与第十轮审计共享同一窗口、同一测试文件身份和同一冻结实现；所有实际读取仍逐次审计。",
198:       "before_open": "冻结C9及两个入口、合同与data_lock、全部模型和变换清单、缓存覆盖层规则、96请求、方法和消融任务表、统计程序、随机种子及恢复状态机。完成门禁后原子记录OPEN时间和窗口ID。",
199:       "test_denominators": {
200:         "total_observations": 250,
201:         "tg": {
202:           "observations": 6,
203:           "materials": 3,
204:           "closure_groups": 1
205:         },
206:         "tm": {
207:           "observations": 13,
208:           "materials": 11,
209:           "closure_groups": 3
210:         },
211:         "tensile_strength_ultimate": {
212:           "observations": 110,
213:           "materials": 99,
214:           "closure_groups": 4
215:         },
216:         "elongation_at_break": {
217:           "observations": 121,
218:           "materials": 114,
219:           "closure_groups": 9
220:         },
221:         "joint_two_target_states": 106,
222:         "joint_two_target_materials": 99,
223:         "joint_two_target_closure_groups": 4,
224:         "joint_three_target_states": 0,
225:         "joint_four_target_states": 0
226:       },
227:       "denominator_rule": "106是状态数，不是106份独立材料。全部冻结观察和共同状态均保留，不能因未见规格、缓存缺失、预测错误、低概率或难请求而排除。新发现的实现错误不能用改写data_lock或删样本解决。",
228:       "per_search_unit_states": [
229:         "SEARCH_STARTED",
230:         "SEARCH_DONE",
231:         "TOP20_LOCKED",
232:         "RECHECK_FIRST_SEED_DONE",
233:         "RECHECK_SECOND_SEED_DONE",
234:         "UNIT_DONE"
235:       ],
236:       "recovery": [
237:         "仅允许相同代码、权重、输入、规则和随机流下的传输重取或执行恢复；已完成单元只读，不重新搜索以选择较好结果。",
238:         "恢复必须核对不可覆盖的候选锁、随机状态、预算计数及已完成复核种子。已写入的复核结果不能重复追加，账本必须区分逻辑额度与实际重放计算。",
239:         "部分搜索只能从已保存的精确状态继续，或按同一预先固定随机流重放并保留原失败记录；所有实际重放耗时计入同轮预算，不能重置失败提案以获得额外探索。",
240:         "开窗后如发现必须修改代码、公式或策略才能继续的错误，停止并记录工程失败；本授权不允许修改后再开一个最终测试选择窗口。",
```

## configs/inverse/round010_plan.json lines 241-360

```text
241:         "第九轮未能在原实现和同轮预算内完成时，交付未完成任务清单；不能把第十轮改为新增训练、候选搜索或参数选择轮。"
242:       ],
243:       "stopping_rules": {
244:         "engineering_failure": "散列冲突、身份不闭合、测试泄漏、非有限预测、评分公式不一致、候选锁改变、计费错误或确定性执行要求无法满足时，立即停止受影响执行并保留证据。概率因子的合法零质量及其按合同触底处理不等同于NaN故障。",
245:         "ordinary_candidate_failure": "按冻结规则产生的非法、重复或不可构图提案保留原槽位与失败原因，按既定合法性和预算口径计入；不得补充免费候选，也不得把正常未命中解释为物理无解。",
246:         "scientific_failure": "实现正确但某个固定门槛未达到时，继续完成预算允许的其余预定评价和独立复算；不得调阈值、改分母、换模型或隐藏失败目标。",
247:         "budget": "单轮14400秒、累计259200秒或第10轮上限任一先到即停止。预算耗尽不构成科学通过，也不授权增加轮次。"
248:       }
249:     },
250:     "final96_ablation_policy": {
251:       "panel": {
252:         "requests": 96,
253:         "single_target": 24,
254:         "two_target": 48,
255:         "three_target": 8,
256:         "four_target": 16,
257:         "protocol_A_requests": 60,
258:         "protocol_B_requests": 36,
259:         "R_unspecified_requests": 96
260:       },
261:       "main_and_baselines": {
262:         "seeds": [
263:           20261111,
264:           20261112,
265:           20261113
266:         ],
267:         "methods": [
268:           "inverse",
269:           "retrieval",
270:           "legal_random",
271:           "combination_ratio_search"
272:         ],
273:         "method_request_seed_units": 1152,
274:         "search_budget_per_unit": 512,
275:         "top_k": 20,
276:         "independent_recheck_budget_per_unit": 40,
277:         "search_model_rule": "种子s的搜索只能读取F_s。保持现有请求采样seed与模型seed一致的冻结策略，不在最终窗口增加新的种子选择或候选初始化方法。",
278:         "consensus_rule": "一个已锁定候选在搜索模型及另外两个主种子的完整前向模型上均满足全部请求约束，才计为三代理计算命中。另两种子不得参与搜索或排序反馈。",
279:         "main_proposals": "192个逆模型提案加64个合法随机提案，再执行256个固定R/v细化提案。",
280:         "random_baseline": "512个合法随机提案；D/P及非NONE的E按原冻结角色域均匀采样，NONE概率0.1，有E时v服从Uniform(0.000001,0.999999)。",
281:         "combination_baseline": "256个合法随机初始提案加256个与主方法完全相同的比例细化提案。",
282:         "retrieval_baseline": "仅用真实train配方，保留真实D/P/E/R/v，沿用完整协议匹配、状态到材料等权聚合、缺失罚10和最大距离/平均距离/配方ID排序。不进行模型预扫、不填伪标签；耗尽可提前停止并如实记账。",
283:         "R_and_refinement": "沿用原32个log(R)分层规则及训练范围。16个不同父配方各16个细化提案，保持D/P/E，v和log(R)扰动及截断完全沿用冻结代码；给定R接口仍保持原值。",
284:         "ranking_and_diversity": "最大归一化区间违反量、平均归一化中心距离、规范配方身份依次排序；R/v去重容差均为1e-6，每个D/P/E组合最多5项，最多20项。",
285:         "accounting": "重复、非法和失败提案占槽；缓存不能增加免费探索。搜索逻辑向量、复核逻辑向量、实际逐任务调用与实际墙钟分别记录。"
286:       },
287:       "single_component_MDN": {
288:         "seed": 20261111,
289:         "requests": 96,
290:         "weights": "k1_20261111既有冻结逆模型",
291:         "search_and_recheck": "保持主方法192+64+256及512搜索、40复核预算，仅将逆向提案模型替换为已训练K=1模型。搜索和复核仍使用三个完整主前向模型。",
292:         "report": "全部真实单目标与双目标锚点评分、正式计算命中、多样性、合法性、预算和规定分层；不新增K=3必须优于K=1的硬门槛。"
293:       },
294:       "no_test_conditions": {
295:         "seed": 20261111,
296:         "requests": 96,
297:         "weights": "no_test_conditions_20261111既有冻结逆模型",
298:         "definition": "只在逆模型中清零48维测试条件表示和assay；保留目标数值、任务ID、上下界mask和R。完整前向模型仍接收真实测试协议。",
299:         "search_and_recheck": "与主方法相同的512搜索和40复核预算，使用三个完整主前向模型，不训练无条件前向替代物。",
300:         "report": "真实锚点评分、正式命中、多样性和全部规定分层；该报告不用于再次选主配置。"
301:       },
302:       "last_ranking_only": {
303:         "seed": 20261111,
304:         "requests": 96,
305:         "pool": "逐请求复用主方法已保存的完整512槽提案池及其散列，保留其中前向指导的细化，不另采样、不重新细化。",
306:         "ranking": "在原合法且可构图候选中，以该候选真实v附近±0.005窗口的逆模型条件概率质量排序，NONE连续因子为1；沿用固定去重、每D/P/E最多5项及身份破平局规则。该分数不包含已学习的R分布，也不是性质成功概率。",
307:         "freeze_and_evaluate": "先独立锁定该排序产生的top20，再读取已保存的搜索模型预测并用另两个完整主前向模型复核；额外搜索调用为0，复核逻辑预算最多40。可复用完全相同候选的已归档复核结果，但必须记录逻辑额度、缓存命中和实际调用。",
308:         "interpretation": "只解释末端前向排序的贡献，不能宣称整个生成和细化过程不使用前向模型。"
309:       },
310:       "no_target_values": {
311:         "seeds": [
312:           20261111,
313:           20261112,
314:           20261113
315:         ],
316:         "required_use": "全部真实单目标与核准双目标锚点的配对对照。",
317:         "final96_search_required": false,
318:         "reason": "冻结合同以该对照检验真实锚点条件信息；本授权不增加未要求的三套正式搜索任务。"
319:       }
320:     },
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
```

