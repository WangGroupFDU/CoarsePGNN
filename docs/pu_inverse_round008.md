# PU 反向设计第八轮执行约定

本轮落实原 Pro 会话第七轮 `pu-inverse-20261111-8b0e67e7b78a` 的 REVISE。
当前文档描述待执行配置，不表示新训练或评价已通过。

- 本地：`/Users/jifengwang/Desktop/5_ongoing_PolymerGenerator聚合物生成与表示法/work/pu_inverse_20261111/CoarsePGNN`
- 远程执行：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository`
- 输出：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated`
- 节点 cn2，`/opt/anaconda2023/bin/python`，CUDA 单卡，CPU 线程 8，不更改环境。

## 冻结边界

合同 SHA256 `e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef`；data_lock SHA256 `d6ea2d43d60ba12784e91f03d37515210e300b0144597fa12d0412162b06f52d`。本轮不读取测试标签，不计算最终96面板。所有计算属于一次28800秒共享期限，包括工程检查、F3、F4和条件重建。第八轮开始前累计7轮、341.4713731018128秒；任务总上限10轮、259200秒。

`configs/inverse/round008_plan.json` 固定全部分支和派生模板。只读D0保留C6/A6身份；D1至D3按顺序各九阶段。四套全部完成后仅按验证MAE、精确平局NLL及数值lr/wd排序一次。候选主配置完成原24请求四方法、18接口和真实图/计量/来源检查后独占写selection_lock。随后训练所选配置的另外两种子、三个无目标值逆模型及种子11的K1、无测试条件逆模型。最后排序消融沿用完整主提案池，仅改变最后排序。

## 限定修复

1. 修正旧确定性测试参数。新best.pt保存同一最佳epoch的模型和AdamW；旧C6文件只读。
2. 资格、当前评审、执行及资产生产者分开；配置以操作/配置ID/种子/消融四元组授权。
3. 训练、恢复、重载、推理共用逆模型工厂；消融只训练逆模型，组合清单保留两个生产者。
4. 正式更新记录实际请求，控制模型重放；epoch和更新量来自主模型实际完成回执。D0胜出时仅以原源码、原输入、F3身份和终态RNG重建，不能补抽随机数。
5. 正式窗口评分强制源float64；历史float32评分明确为不参与验收的诊断。
6. 超时/失败保留当前阶段和实际更新，终止子进程组；无隐含重试或锁后重选。

## 正式入口

由 `loopctl run` 记录轮次并调用下列入口；不得脱离轮次预算另行执行：

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/opt/anaconda2023/bin/python -m coarsepgnn_inverse.round008 \
  --plan configs/inverse/round008_plan.json \
  --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_008_integrated
```

执行提交C8冻结后生成外部授权回执，绑定实际源码、完整R7回复、计划和配置SHA，避免提交自引用。A8生成后另以回执补齐C8/A8绑定。`final_evaluation_readiness.json`只是待Pro审核的准备包，不是测试开启授权或研究PASS。
