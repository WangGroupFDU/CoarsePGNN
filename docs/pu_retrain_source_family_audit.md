# 公开 PU 数据的来源家族审计

审计日期：2026-11-17。目的为修复训练/评价分组中的来源关联遗漏，全部原始记录、性质标签和监督 ID 保留。相同数值不作为同族证据，也不据此删除实验。

## 1. 实际读取范围

从固定 `sources.csv` 读取全部 202 个来源：142 篇期刊文章、56 个专利公开/授权文献及 4 个学位论文。该文件 SHA-256 为 `65e03519ec923e431770e346d7009698dbb1fd73834f3fb924e9d68536ee0d75`。

对全部 56 个专利查询目标文献的家族字段。54 个成功读取明确的 Family ID，2 个仍未解析：`US3746665A` 和 `CN118165222A`。两项尝试中，Google Patents 目标页返回不可访问；搜索返回的引用文献和相似文献不能代替目标家族。`US3746665A` 原授权 PDF 可确认申请日期和序列号，但这不能证明其家族只有一个成员。

两项未解析来源分别有 4 和 20 条监督观察，合计 24 条，全部继续保留并按自身来源及可观测配方身份分组。其家族未知限制不通过删掉这些记录来规避。

已保存证据是检索工具返回的带行号文本及其结构化响应，散列校验绑定这些文本字节，**不冒称原始网页 HTML 的散列**。期刊 DOI 按原 `doi_norm` 规范化；未通过相同标题或相同测试协议自动合并不同论文。

## 2. 已确认的跨源家族

| 家族 ID | 已公开输入中的成员 | 关系依据 | 旧分组中的实际记录 |
|---|---|---|---|
| 42102758 | `US8962785B2`、`US9574043B2` | 家族表一致；申请 13/681,811 是 12/685,858 的 continuation，原 PDF 首页及 Related Applications 明示 | 62 条，旧外折 2/3 |
| 80233289 | `CN114057971A`、`CN114057971B` | 同一申请 `CN202110779028.XA` 的公开/授权版本，家族表一致 | 26 条，旧外折 2/4 |
| 83916461 | `US20250250389A1`、`CN115322330B` | 美国目标页 Country Status 直接列中国成员，原始来源关联同一 `PCT/CN2022/085521` | 36 条，旧外折 0/3 |

可核对的文献页面和原件：

- [US8962785B2](https://patents.google.com/patent/US8962785B2/en)；[US9574043B2](https://patents.google.com/patent/US9574043B2/en)；[US8962785 原始 PDF](https://patentimages.storage.googleapis.com/ba/49/6e/71b90bf5a50bf5/US8962785.pdf)；[US9574043 原始 PDF](https://patentimages.storage.googleapis.com/48/9d/f8/f1d45196ceb6fd/US9574043.pdf)。
- [CN114057971A](https://patents.google.com/patent/CN114057971A/zh)；[CN114057971B](https://patents.google.com/patent/CN114057971B/en)。
- [US20250250389A1](https://patents.google.com/patent/US20250250389A1/en)；[CN115322330B](https://patents.google.com/patent/CN115322330B/zh)。

同一家族可能有不同实施例或追加实验，家族相同不等于每一条值都重复。这里将整个家族作为更保守的划分单位，不对这些记录去重或删除。

## 3. 旧划分发现的具体缺口

只读脚本 `scripts/audit_pu_retrain_inputs.py` 对既有母表和划分回读得到：3657 条原 CSV 记录与各自 `raw_source_record` 全字段一致；561 个恢复子观察与 246 个父均值的谱系有效；父均值均不参与损失；3972 条共同监督记录均有有限性质值和有效双向监督掩码。

旧 200 闭包的已记录身份键跨折数为 0，但新增核实家族跨折为上表 3 组。这说明原记录 ID 闭包没有覆盖全部出版物关系。源无关的直接可观测配方比较还发现 23 组、176 条观察跨折；其中同配方、同任务和完全相同数值的疑点为 14 组、28 条。它们用于查找来源证据，不按数值直接合并。加入完整允许条件后的相同配方/任务签名跨折为 0，也不能据此否认同一实验被不同描述重录。

全部记录的 `tensor_hash` 均为空；原图身份不能代替新模型完整输入张量身份。新版划分进一步按源无关可观测配方做保守分组，其定义和测试与家族字段共同冻结。未知配方细节、哈希碰撞以及未取得家族信息的两项仍需在解释中限定。

旧审计结果：`outputs/pu-source-family-audit-20261117/old_split_identity_audit.json`，SHA-256 `ceb11078100b5dce6fe2a49529ca287f8d94ab6f1013ba7c51cedd9932730afd`。

## 4. 固定映射和增强母表

完整 202 来源映射为 `configs/retrain/pu_source_family_keys.json`，SHA-256 `b06f0fe7cb32ef24346fb79d0493c4c18a3ee5f29e2bf0b5b2cb2d145f925437`。每项保存规范化出版物身份、已核实家族、审查状态、来源行身份以及正文位置与散列。未解析两项的 `source_family_ids` 为空、状态为 `family_unresolved`，不会凭空生成家族。

映射编译入口为 `scripts/build_pu_source_family_map.py`。它只读取保存的元数据证据，不读取性质数值、不联网或拟合模型。前 28 个文献正文在 `outputs/pu-source-family-audit-20261117/publication_evidence/first28_batch_*.json` 及重试文件；后 28 项正文和索引分别为同目录 `second28_raw_retrieval.json` 和 `second28_family_scan.json`。独立成员关系核查为 `independent_lineage_audit.json`。

`build_dataset(..., source_family_path=...)` 要求映射绑定原 `sources.csv` 的完整 SHA-256，覆盖来源集合精确一致，并逐项核对出版物身份；发生不一致即拒绝。它只增加来源元数据字段，模型特征白名单不包含这些字段。

增强母表为 `outputs/pu-all-data-family-20261117/dataset/dataset.jsonl`，SHA-256 `40a2f8aa8f05a104c05ece27d56066fc10ff2c8259032149ab854168a4c1942b`。实际逐字段比较确认旧母表全部值、原料、条件、任务、掩码和 ID 不变；4218 行全部保留、3972 行共同监督。回执为 `metadata_annotation_receipt.json`，SHA-256 `1a22353c67faaf13d1e89d6c7f90671c314cd66f4ca0530a682a27d305a4b371`。

旧隔离不足的诊断产物保持原样。新模型按同一候选配置和训练政策在修复后的划分上重新运行，不以旧/新留出表现比较来选择划分。最终成效需读取真实运行结果；本审计不宣称模型已达到科学成功。
