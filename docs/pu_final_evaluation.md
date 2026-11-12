# PU 最后一轮评价与独立复算

## 版本与科学边界

R9 在读取旧 C6 checkpoint 清单时因缺少 `bytes` 失败，未打开测试窗口。原失败 C9/A9 与正式计费保持不变。原 Pro 的 R9 完整回复批准 R10 为唯一最后恢复包；它只覆盖执行编排，原 R8 六个科学政策字典、八套模型生产者、D0、成功合同、划分和固定 96 请求不变。

实际 C10 提交后才生成外部 `round010_authorization.json` 与源码清单，分别绑定原 R8、本次 R9 的全文散列、计划散列与实际消费者 C10。不得预填未来 A10；结果归档时另建立 C10/A10/R 的绑定。

## 远程执行

仅在 cn2 的独立 repository 内、既有 `/opt/anaconda2023/bin/python` 执行。以下命令必须纳入唯一正式 `loopctl run`，不能作为不计费预检或失败后的重跑：

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/opt/anaconda2023/bin/python -m coarsepgnn_inverse evaluate \
  --final-plan configs/inverse/round010_plan.json --combined-final \
  --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_010_final_bundle
```

外层只建立一次起点和 14400 秒硬截止；内部保留 10 秒终止回执时间。评价和独立复算为两个不同 PID 的新 Python 进程，测试进程也继承同一期限。第二阶段只能使用剩余时间；失败、超时均无自动重跑。总预算仍为最多 10 轮或 72 小时，R10 前累计 1848.0574975546915 秒。

## 开窗前检查与审计

所有权重始终按原声明 SHA 核验并记录实际大小；C6 无大小声明时 `declared_bytes=null`，仅 C8 已声明时额外比较大小。历史清单和权重不修改。

正式门禁覆盖实际旧新清单、错误 SHA/大小/缺文件、共享截止、阶段故障及进程终止、唯一输出目录、两份授权与不可变对象。pytest 在自己的进程内拒绝真实封存路径，保存独立 open 和拟合审计，不以父进程补丁冒充子进程保护。

在真实 validation 集核对全部八套模型：三套主模型前向、八套逆向的 239 观察与 63 双目标状态，逐项比较原 R8 输出。只有实际权重加载、内部身份、数值和工程门禁全部通过，才锁住任务与输入并创建唯一 `final_window.json`。

审计记录 ENTERED、COMPLETED、FAILED；进程被强杀后未完成的审计按 INCOMPLETE 报告，未进入按 NOT_ENTERED 报告。固定的零字段不能替代真实计数。复算失败必须引用评价目录的窗口状态。

## 完整评价与复算

评价输出到 `round_010_final_bundle/evaluation`，覆盖 250 个测试观察、106 个核准双目标状态、96 请求及 1440 比较单元。每单元 512 搜索逻辑预算、最多 40 个其他种子复核，top_k=20；候选先落盘锁定，再做跨种子复核。原始提案池保留非法、重复和失败槽。

仅使用固定 Qwen 做缺失测试条件的推理编码；不修改基础 14 项资产、PLE、词表或标签变换。未知 assay 使用原索引 0。无新增训练或优化器更新。

评价工程完整后，监督器独立核验任务全集及科学产物清单，才启动 `recompute` 子目录中的新进程。它重新加载八套模型、重新编码条件、重算所有锚点与完整候选池，并比较图、计量、来源、排名、预算、统计和门槛。科学门槛失败不阻止固定复算；工程不完整时禁止靠复算补做新搜索。

结果区分工程完成、科学门槛失败、工程未完成。三代理命中属于计算筛选，三/四性质联合实测仍为 0，Tg 测试只有一个来源闭包。只有原合同全部满足、独立复算一致并得到同一 Pro 会话的版本匹配 PASS，才能标记研究完成。
