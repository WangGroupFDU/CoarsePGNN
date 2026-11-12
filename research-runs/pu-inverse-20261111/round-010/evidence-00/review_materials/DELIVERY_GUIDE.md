# PU 模型与证据交付入口

## 当前适用性

已训练的模型和完整候选证据可供复核。最终冻结科学合同未达到，且独立复算在逐项比较完成后因汇总字典遗漏而失败；不能将模型用作已验证的四性质设计工具。当前目录的终审报告和版本匹配评审决定优先于早期准备层文档。

## 路径

- 本地仓库：`/Users/jifengwang/Desktop/5_ongoing_PolymerGenerator聚合物生成与表示法/work/pu_inverse_20261111/CoarsePGNN`。
- 远程执行目录：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository`。
- 远程最终输出：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle`。
- 模型交付清单：`model_delivery_index.json`列出8套模型及32个checkpoint的本地原始归档、远程路径、生产者C/A和SHA；`model_delivery_readback.json`记录这32项本地原始权重实际SHA回读全部匹配。权重已在第6/8轮归档，本轮没有生成替代权重。
- 运行依赖：cn2现有Anaconda base、Python3.11.8、PyTorch2.5.1+cu121、PyG2.6.1、RDKit2025.3.2。没有新增或切换环境。

## 查看结果

本文件位于A10的`review_materials`子目录；以下相对路径以本文件为起点。

1. [最终报告](../report.md)说明真实执行、科学失败、复算和统计范围。
2. [12行正向误差表](final_forward_metrics.csv)、[15行主方法/基线/消融表](final_panel_metrics.csv)、[71项合同门槛表](evaluation_gate_table.csv)均为原始结果字段导出，UTF-8 BOM、逗号、CRLF。表内保留原始JSON字段位置及SHA，不能替代完整原件。
3. [单目标原始示例](../evaluation/units/inverse_20261111_FINAL_001/design_result.json)和[四目标原始示例](../evaluation/units/inverse_20261111_FINAL_081/design_result.json)按预先固定面板各段首ID展示，未按成功结果挑选。其索引及原始SHA见[candidate_example_index.json](candidate_example_index.json)。
4. 全部原始候选位于`../evaluation/units/`，每个比较单元保存`search.json`、候选锁、`design_result.json`、图文件及跨种子复核；对应独立结果在`../recompute/units/`。
5. `design_result.json`中的`top_candidates`是最多20个返回/诊断候选。只有`qualified_candidate_indices`列出的候选通过冻结代理规则；它为空时状态为无模型支持候选。即使非空，也没有新增合成实验证明，且本研究整体前向可靠性门槛失败。
6. 原始结果中的`graph_path`保留远程绝对路径以维持原始字节和哈希；本地归档中将远程`round_010_final_bundle/`前缀对应到本证据目录即可定位同一图。不要为了本地路径便利改写原始JSON或图文件。

## CLI和复现边界

`src/coarsepgnn_inverse/__main__.py`保留`train`、`design`、`evaluate`、`validate-request`、`stoichiometry`入口。`design`需要模型根目录、原manifest SHA、冻结合同、数据锁和带协议的请求；原料与协议的资格限制不能删去。其普通开发入口的单模型输出和本轮三代理复核不是同一验收口径。

C10的正式最终入口与完整命令保存在`../experiment.json`和`docs/pu_final_evaluation.md`。该命令已作为最后一轮正式任务执行，不是可重复提交的待办命令；输出目录不可覆盖。当前最多10轮的研究不再运行训练、额外搜索或最终测试。对新问题继续训练或推理需新的明确授权与验证安排，不能把本次已打开的测试集重新用于调参。
