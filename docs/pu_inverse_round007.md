# PU 第七轮执行说明

本轮由原 Pro 会话第六轮完整评审授权，研究仍为 REVISE。合同 SHA-256 为 `e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef`，数据锁与最终测试保持不变。

## 本地与远程

- 本地代码及审阅：`/Users/jifengwang/Desktop/5_ongoing_PolymerGenerator聚合物生成与表示法/work/pu_inverse_20261111/CoarsePGNN`
- cn2 执行目录：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/repository`
- 输出目录：`/home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_007_development`
- 解释器：`/opt/anaconda2023/bin/python`；现有 base 环境，无新增依赖。

## 冻结输入与执行

`configs/inverse/round007_plan.json` 记录完整授权范围、D0–D3 清单及停止条件。D0 是第六轮完整权重的只读重评；D1–D3 依次新训。主配置按四任务平均归一化材料等权 MAE 选择整套模型，只有精确平局才比较源精度窗口 NLL，再按 lr/wd 数值顺序。

`round007_development_panel.json` 固定16条单目标及8条双目标请求，按原评审给定哈希排序选择。请求中的源精度历史 R 是明确条件，真实 D/P/E/v 不进入生成。协议扩展仅登记这些冻结验证观察已有的完整协议，不覆盖数据锁中 A/B 注册表，不重新编码文本。

`round007_interface_cases.json` 包含14类功能要求及子例，共18个调用。NaN 使用显式夹具指令注入，避免产生非标准 JSON。固定无命中预测桩单独标为工程夹具，不进入模型性能。

从远程执行目录启动单个持久任务：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
/opt/anaconda2023/bin/python -m coarsepgnn_inverse.round007 \
  --plan configs/inverse/round007_plan.json \
  --output /home/jifengwang/data/45_polyermization_prediction/inverse_design/runs/pu-inverse-20261111/round_007_development
```

部署后外部授权回执绑定实际代码提交、完整评审原文、三个配置、计划及静态输入散列。全部子进程共享14400秒单调时钟截止时间；失败不自动重试，不删除失败配置。单 batch、两 epoch、F3、F4、所有基线和接口计算全部计入同一正式轮。

## 输出与解释

四配置均产生239条单目标和63个双目标状态的独立评分。源 float64 的 `v_reference` 只供窗口评分，训练张量及损失不变；NONE 连续因子严格为1。双目标枚举全部合格观察对，先状态内、再材料内、最后材料等权。

选中模型运行24请求×4方法，每方法至多512提案。非法、重复、缓存命中都占名额；检索只从真实训练配方排序，缺失距离固定10，先状态、再材料等权，穷尽不补留出配方。输出 top20、每个 D/P/E 最多5项；仅报告单模型开发支持，尚无三种子一致性或实验成功结论。

`train`、`design`、`evaluate` 的具体参数可通过 `python -m coarsepgnn_inverse <command> --help` 查看。`design` 需要固定模型清单 SHA、合同和数据锁；`evaluate` 仍受当前评审及代码身份约束，不能打开测试。

本轮代码在正式任务前仅进行本地 AST/JSON 静态检查。真实测试与 CUDA 执行结果以远程日志、独立重载和本轮结果回执为准。
