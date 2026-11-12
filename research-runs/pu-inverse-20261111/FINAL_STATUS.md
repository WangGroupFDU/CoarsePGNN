# PU 四性质反向设计：预算上限交付

本研究未达到冻结成功目标。原 Pro 会话终审为 **BLOCKED**；已使用 **10/10 个正式轮次**，累计计算墙钟 **15177.788051 秒（约 4.22 小时）**。代码、训练权重和负结果可交付复核，不能标为验证通过的四性质设计模型。计算已结束，不再启动新训练、搜索、测试开窗或补算。

## 结果与核验边界

- 原评价完成：250 条测试观察、106 个核准双性质状态、1440 个候选比较单元。71 项合同门槛中 **54 项失败、17 项通过**。
- 三主种子的四性质 MAE 全部未达标；固定面板 Hit@20 为 **29/96、33/96、33/96**，检索基线各为 **36/96**。化学合法率 1.0 不代表性质可靠。
- 独立进程完成八套锚点及 1440 个候选单元逐项比较；随后 `final_recompute.py` 漏写 `anchors[name] = result`，在汇总阶段报错。完整独立汇总、门槛比较和成功封存未完成。该遗漏在 C9 中已存在，预开窗测试未覆盖完整汇总成功路径；本次保留未解决缺陷。
- Tg 测试仅一个来源闭包，三/四性质共同实测为 0；新候选的代理预测不能称为实验成功。106 个共同状态全部为极限拉伸强度/断裂伸长率，不覆盖所有性质对。
- Pro 取得并核对完整清单和 73 项必读原件，共列出 104 项读取记录；两份条件张量只核验原始字节，语义反序列化标为 unread，未声称逐份阅读全部产物或执行权重。

## 交付入口

- [完整研究报告](round-010/evidence-00/report.md)：数据资格、实际训练、科学结果、统计分母、失败与预算。
- [使用与复核说明](round-010/evidence-00/review_materials/DELIVERY_GUIDE.md)：CLI 边界、固定候选例子、远程路径和原始文件映射。
- [模型权重清单](round-010/evidence-00/review_materials/model_delivery_index.json)及[32 项权重回读](round-010/evidence-00/review_materials/model_delivery_readback.json)：三主种子与五套逆向消融，沿用 C6/C8 原始产物。
- [原始门槛判断](round-010/evidence-00/evaluation/acceptance.json)、[正向误差表](round-010/evidence-00/review_materials/final_forward_metrics.csv)、[方法比较表](round-010/evidence-00/review_materials/final_panel_metrics.csv)。这些是原评价输出，未冒充独立汇总复算结果。
- [有效终审全文](reviews/pu-inverse-20261111-0cd0a1ad3f62.md)及[收尾记录](closeout/closeout_record.json)。
- [首次终审原文](closeout/round010_original_invalid_review.md)保留一个目标哈希字符遗漏；它不是有效控制记录。[身份修复核对](closeout/round010_identity_repair_comparison.json)证明修复稿仅改变 `goal_sha256` 和新 `request_id`，其他 JSON 字段完全一致。

## 固定版本

| 标识 | 提交 |
|---|---|
| C10：实际评价/复算源码 | `3d9348a92865528d31a3dc769c93ad0dc7441dae` |
| A10：冻结证据 | `983b0eeae4b62ae4209af89155fdaa8b7ef62150` |
| R10：身份修复后有效终审 | `657cd5255136a5f943266c956c1f9acdf6d627cb` |

终审原文 SHA-256：`b7468f8322615e102ecb5d86676498023253d5c3985beeacc2cab6caccb20de9`。A10 manifest SHA-256：`a1cbf5f955291e3b3845f7e0002a4faae181efd41a1d8a4ec8711f54470778ba`，36,356,198 字节；79,324 条文件记录，加清单本身共 79,325 个 Git 文件。本收尾提交仅增加证据说明，不改写 C10、A10 或原始结果。

## 尚未生成的八项复算文件

以下为原执行缺失项，不通过补算或原评价文件替代：

- `recompute/anchor_summary.json`：独立锚点汇总未生成；在anchor_report入口即发生KeyError。
- `recompute/panel_rows.json`：独立面板行汇总的成功输出未生成，不能用原评价panel_rows替代。
- `recompute/panel_summary.json`：独立面板统计汇总及与原评价的比较未完成。
- `recompute/acceptance.json`：独立71项门槛重建与一致性比较未完成。
- `recompute/zero_fit_audit.json`：成功收尾文件未生成；已有FAILED阶段审计中的实际零计数仍有效，但不是该成功文件。
- `recompute/scientific_inventory.json`：复算成功路径的科学封存清单未生成。
- `recompute/recompute_result.json`：独立复算最终成功结果未生成。
- `recompute/candidate_differences_index.json`：运行时成功收尾总索引未生成；事后已有回执文件清点不能冒充该原始输出。

`evaluation/failure.json` 在评价成功时本应不存在，单独列为预期缺失。RF1（科学目标未满足）和 RF2（工程缺陷与核验缺口）仍未解决；RF3 的终审原文及交付索引已归档。归档完成不会改变 BLOCKED 或产生研究 PASS。

## 本地与远程路径

- 本地仓库：`/Users/jifengwang/Desktop/5_ongoing_PolymerGenerator聚合物生成与表示法/work/pu_inverse_20261111/CoarsePGNN`。
- 远程执行工作目录：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository`。
- 远程最终输出：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle`。
- 执行节点 cn2 / codex-cn2；解释器 `/opt/anaconda2023/bin/python`；沿用原环境。

早期准备层文档中的“尚无权重”是历史状态；当前交付以本文件、A10 报告和有效终审为准。未来研究如需继续，必须另行明确授权与独立验证安排，不能把本次已打开的测试集用于调参。
