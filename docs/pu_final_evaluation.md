# PU 最终评价与独立复算

本入口消费已选定 D0 的三主种子及五个逆向控制权重，不进行训练或重新拟合。
原会话 R8 授权、成功合同、数据锁、96 请求原字节均由
`configs/inverse/round009_plan.json` 绑定。模型生产 C6/C8 与消费者 C9 分开记录。

## 执行位置与预算

所有真实检查和计算仅在 `codex-cn2` 的现有 Anaconda base 执行：

```text
/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository
```

每轮最多 14400 秒；监督进程使用同一单调时钟期限，预留 10 秒写终止回执。
已存在输出目录拒绝覆盖。科学门槛失败正常记录；身份、计算或实现错误立即停止。
开窗后不能修改源码再重跑测试，也不能利用剩余轮次补训或另选模型。

## 冻结与门禁

先提交完整源码 C9，包括两个入口和合成门禁测试，再生成外部授权回执中的真实
`consumer_code_sha`、全 `src/tests/configs` 文件清单和计划 SHA。未来提交 SHA 不预填。
外部回执不改变权重内部生产者身份。运行时核对实际字节、全部权重和14项基础资产。

R9 内先运行指定合成测试，再用全部八套既有模型对原 validation 逐观察结果做兼容性核对。
只有门禁通过后才创建不可覆盖 `final_lock.json` 和唯一 `final_window.json`。
测试的每次物理打开写入 `test_access.jsonl`；R10使用同一逻辑窗口。

缺失条件只按固定 Qwen revision、batch 4 和原提示规则做推理。输出独立缓存覆盖层，
不覆盖基础键，不拟合 PLE、词表或变换。未知 assay 仍映射训练词表的0索引。

## 第九轮

以下命令由 loopctl 正式计算执行器调度，不能在预算外额外运行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
  --final-plan configs/inverse/round009_plan.json \
  --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_009_final
```

分母为250单目标观察、106双目标共同状态，以及1152个主方法/基线单元和288个消融单元。
每个单元先保存 `search.json` 和 `top20_lock.json`，随后逐种子复核。复核结果不能回流搜索。
单元函数可恢复已持久化的复核状态；部分搜索尚无完整搜索文件时明确拒绝隐式重采样。
整轮入口不自动重试或延长期限。

## 第十轮

R9完整归档并经评审后，使用其 `scientific_inventory.json` 的实际 SHA 调度：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
  --final-plan configs/inverse/round009_plan.json \
  --recompute-reference /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_009_final \
  --reference-sha256 ACTUAL_ARCHIVED_INVENTORY_SHA256 \
  --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_recompute
```

独立进程重新编码新增条件、重载权重、复算所有锚点和全部提案池，包含非法、失败及重复槽。
仅从原池重算排序/去重/组合上限，并核对原 top20 身份，不新增提案。
再复算全部 top20 的跨种子预测以及材料等权统计、闭包区间、基线、消融和门槛。

## 结果含义

`anchor_summary.json` 是实测有限留出核对；`panel_summary.json` 是三个初始化代理模型
对固定96请求的共同支持，不能称新配方实验成功。`acceptance.json` 保留所有通过和失败项。
Tg 单闭包总体区间为 null，三/四性质共同实测为0。任何脚本均不会自动写研究 PASS；
只有冻结合同全部满足、独立复算成立并获原会话版本匹配 PASS 后才完成研究。
