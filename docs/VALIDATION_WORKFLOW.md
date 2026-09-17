# 验证流程与验收门槛

## 0. 数据构建

使用前复权行情；按季度缓存并生成 manifest。股票池必须覆盖研究期内上市和退市证券。停牌、涨跌停、行业历史成员和财报发布日期需要作为后续全市场研究的独立数据质量表。

```powershell
wave-build-dataset --all-market --start 20150101 --end 20261231 --adjustment qfq --cache data/a_share_2015_2026
wave-build-bulk-market-data --start 20150101 --end 20241231 --output data/a_share_2015_2024/bulk_core --universe data/a_share_2015_2024/point_in_time_universe.parquet --timeout 10
# 可用 --max-dates 20 做可恢复的小批次；只有 manifest.build_complete=true 才算完成
wave-audit-bulk-market-data --output data/a_share_2015_2024/bulk_core --universe data/a_share_2015_2024/point_in_time_universe.parquet --report data/a_share_2015_2024/bulk_core_audit.json
```

命令可中断后重跑，已经成功的股票×季度分区不会重复请求。全市场任务可能受 Tushare 积分、速率和网络状态影响；失败分区会保留在 `manifest.json`，不能把部分成功误报为完整数据。

全市场优先使用第二条按交易日批量下载命令。它保存原始 OHLC + `adj_factor`；每个 walk-forward fold 在自己的截止日锚定 qfq，禁止用最终样本日的因子物化所有历史后再切分。完成判定必须同时满足：全部交易日分区完成、核心端点未触及行数上限、复权因子匹配率达标、时点股票池缺口已解释。

## 1. 边界验证

1. 在 train 窗口内运行 ATR-ZigZag、结构变点与可选 ruptures。
2. 对 OHLCV 做不少于 100 次小噪声扰动。
3. 保存每个边界在 ±3 根 K 线内的复现频率。
4. 对有人工边界的样本计算一对一容差 Precision、Recall、F1。
5. 分类后用离线 HSMM 做历史描述平滑并合并连续同类段，再比较合并前后的自转移率和持续期分布。用于选股的历史状态必须改用 `causal_duration_label`。

bootstrap 稳定度只代表局部鲁棒性，不等于人工正确率。建议初始研究门槛：人工边界 F1 ≥ 0.75；低于门槛不得进入“可信分类”阶段。

## 2. 人工基准集

目标 2,000–5,000 段，覆盖不同年份、行业、市值和市场状态。每段至少两人独立标注。保存 annotator、label、边界评价、notes、reviewed_at；不覆盖原始审核记录。

UNKNOWN 可富集抽样以提高审核效率，但必须同时保留代表性随机层，并冻结 `sampling_probability`。校准集和测试集的 ECE、Coverage–Risk、Brier、Log Loss、Macro-F1 等总体指标使用入样概率倒数加权，禁止把富集样本的原始均值当作总体风险。

输出一致率和 Cohen's kappa。任何不一致段的共识标签为 UNKNOWN。建议在一致性不足时先修改标注指南，不训练分类器。

## 3. 分类训练与校准

每个 fold 使用全市场统一日历的扩展训练窗，之后是独立校准窗和测试窗；不能按每只股票的相对样本序号各自切分。所有股票的训练区间必须整体早于校准区间，校准区间整体早于测试区间，并清除跨边界波段。监督基线优先逻辑回归与浅树。校准只在 calibration 上进行，最终指标只在 test 上报告。

必须报告：Macro-F1、Balanced Accuracy、逐类 Precision/Recall/Support、Brier、Log Loss、ECE、可靠性图、Coverage–Risk。UNKNOWN 阈值从校准集选择，例如目标错误率 ≤10%，不允许从测试集反调阈值。

建议发布门槛不是固定真理，而是首轮闸门：

- ECE ≤ 0.05；
- 被识别部分风险 ≤ 10%；
- 覆盖率 ≥ 65%；
- 每个发布类别至少 200 个测试样本，且覆盖多个股票和年份；
- 相对简单基线的 Macro-F1 有稳定提升。

APS conformal 输出候选类别集合；它依赖校准样本与测试样本近似可交换，市场制度变化时需分状态重校准。

## 4. 选股研究

冻结第一层模型后，将状态按生效时间 backward join 到每日横截面。仅在第二层构造未来 5/20/60 日目标。使用 walk-forward 产生完全样本外的状态分数；训练行必须满足 `target_available_at < decision_time`。再计算：Rank IC/ICIR、分组单调性、行业/市值中性结果、年度与市场状态稳定性、换手、成本和滑点后收益、最大回撤。多日目标的组合收益必须按预测周期非重叠抽样，不能把重叠的 20 日收益逐日复利。

与同样数据和调仓频率下的动量、波动率基线比较。只有跨年份、跨状态、成本后仍有稳定增量，才能称为“辅助选股信息”；否则只是形态描述。

## 常用命令

```powershell
python -m pytest -q
python scripts/evaluate_real_data.py
wave-validate-classifier --features outputs/segment_features.parquet --annotations outputs/annotations.csv
wave-validate-selection --bars data/bars.parquet --states data/oos_states.parquet --scores state_score momentum_20 volatility_20
wave-validate-selection --bars data/bars.parquet --states data/causal_states.parquet --fit-features recognizability prob_CLUSTER_A prob_CLUSTER_B --min-train-dates 252
```

`wave-validate-classifier` 会拒绝未来字段，并只汇报 test fold；`wave-validate-selection --scores` 假设输入状态和分数已经由外部 walk-forward 过程冻结为样本外结果。`--fit-features` 会在第二层生成严格因果 Ridge 基线，但输入的第一层状态仍必须是因果、样本外生成，不能使用全样本拟合状态替代。

发现层的 `walk_forward_discovery_states` 只接受显式 `available_at`，即分割器确认该段及其全部特征当时已知的时间；不能把事后 pivot/end 日期复制为可用时间。模型训练只使用 `available_at < decision_time` 的段，并只对同一证券内与训练 K 线区间重叠的待评分段标为 `UNKNOWN/purged_overlap`；不同证券同期数据不会被误当作相同 K 线。当前全市场候选由离线回看分割产生且没有可信 `available_at`，因此 walk-forward 会拒绝它们；在在线分割器产出可审计确认时间之前，这些离线标签只用于描述与审核，不用于选股回测。

现在可用 `wave_segments.segmentation.segment_ohlcv_causal` 生成首版在线兼容候选。它使用 ATR 反转阈值确认 pivot，`end` 是历史极值日，`available_at` 是反向变动达到确认阈值的日期；末尾尚未确认的腿不输出。该路径保证追加 K 线不会重写已确认输出（已做多段 prefix-invariance 单测），但它仅覆盖 ATR-ZigZag，不等于离线多算法融合，也尚未在全市场评估边界质量。边界概率字段是强度启发式，不可解释为边界正确率。示例：

```python
from wave_segments.segmentation import segment_ohlcv_causal

causal_segments = segment_ohlcv_causal(bars, config)
# 保留 available_at；禁止以 end 覆盖它，再传入 walk_forward_discovery_states。
```

候选、特征与因果发现状态可由统一命令复现；该命令会在发现状态写盘前验证时点隔离、同股区间 purge、概率归一及 UNKNOWN 一致性，并把配置和审计结果写入 manifest：

```powershell
wave-build-causal-segments --bars data/pilot_8_2021_2024/dataset.parquet --output outputs/causal_pilot_8_2021_2024_v5 --discover-oos --bootstrap-iterations 20 --bootstrap-tolerance 3 --min-train-rows 60 --retrain-every 20 --components 4 --ensemble-size 3 --min-cluster-samples 20 --min-cluster-symbols 2 --min-cluster-years 2
```

8 股 2021–2024 试点共 7,752 根日线、459 段。采用每个候选边界确认时点截断的 20 次 bootstrap 后，平均边界复现频率为 0.871、P10 为 0.55；它表示局部扰动稳定度，不代表人工边界准确率。临时簇识别 112 段（24.4%），347 段为 UNKNOWN。按波段开始年份覆盖率：2021 为 0%（跨年份支持门槛尚未满足），2022 为 35.5%，2023 为 34.4%，2024 为 30.4%。不启用因果 bootstrap 的早期试点覆盖更高，但不与该配置混为同一个结果；稳定度门控及特征改变会影响最终覆盖。所有覆盖数字都不是精度证据，临时簇也未经语义标注/概率校准。

注意：曾有一版 bootstrap 在整段历史上扰动并分割，早期边界稳定度可能包含确认日之后的信息。该目录 `outputs/causal_pilot_8_2021_2024_v3/` 的 manifest 已标记 `invalidated_for_causal_use`，不可引用为因果证据。当前 CLI 的因果 bootstrap 在每个扰动路径上运行前缀不变的在线分割，再按候选自己的确认索引过滤；已由 prefix-invariance 测试覆盖。临时簇名只在各自 `oos_model_trained_at` 拟合版本内可解释；年份/股票覆盖表只汇报覆盖，`coverage_by_model_version.csv` 才按各自版本报告簇计数。输出包括 `coverage_by_symbol.csv`、`coverage_by_start_year.csv`、`coverage_by_model_version.csv`、`unknown_reasons_by_model_version.csv` 及候选、特征、OOS 状态 parquet。manifest 的 `selection_research_eligible=false` 是有意的安全限制。

## 2026-09-17 节点：全市场分桶因果样本与收敛敏感性

从全市场已构建符号桶中，以固定 seed 每桶抽取 1 只且要求至少 1,000 根日线，复现命令：

```powershell
python scripts/build_causal_stratified_sample.py --output outputs/fullmarket_causal_sample_1per_bucket_v2 --sample-per-bucket 1 --minimum-bars 1000 --seed 20260917
wave-build-causal-segments --bars outputs/fullmarket_causal_sample_1per_bucket_v2/bars_hfq.parquet --output outputs/fullmarket_causal_64_sample_2015_2024_v1 --discover-oos --bootstrap-iterations 20 --bootstrap-tolerance 3 --min-train-rows 500 --retrain-every 60 --components 4 --ensemble-size 3 --min-cluster-samples 100 --min-cluster-symbols 10 --min-cluster-years 3
```

样本为 64 只证券、129,486 根 HFQ 日线，覆盖 2015-01-05 至 2024-12-31，每股至少 1,000 根；它按稳定哈希桶分层，但不是严格按行业/市值加权的统计代表样本。因果分割得到 7,596 段，20 次 confirmation-prefix bootstrap 的边界复现频率均值 0.903、P10 0.65。默认 `max_iter=100` 的 walk-forward 在 34 个已训练版本中 7 个未收敛；这 7 个版本共 1,490 个输出样本全部 UNKNOWN（拒识门控生效）。覆盖率为 36.7%，不代表分类正确率。

发现层现可独立重跑，不必重复候选和 bootstrap：

```powershell
wave-discover-causal-states --features outputs/fullmarket_causal_64_sample_2015_2024_v1/causal_segment_features.parquet --output outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200 --min-train-rows 500 --retrain-every 60 --components 4 --ensemble-size 3 --min-cluster-samples 100 --min-cluster-symbols 10 --min-cluster-years 3 --max-iter 200
```

`max_iter=200` 的同特征重跑 34/34 版本收敛，识别覆盖升至 47.0%（3,573/7,596），UNKNOWN 4,023 段；这证明迭代上限影响工程可用覆盖，不证明分类准确度。对比时保持分割样本/特征一致；临时簇编号仍是各版本局部名。诊断 CSV 现包含逐版本收敛列，仍需人工真值评估何种覆盖/拒识质量合适。
## 全市场候选构建的资源与复权约束

`wave-build-fullmarket-segments` 是分类层之前的纯候选构建步骤。原始行情按交易日分区，而每只股票分段须读取完整历史，因此它先以稳定哈希将有限数量的交易日文件转置到 64 个（可配置）股票桶，再一次只加载一个桶进行分段和特征提取。每个分片和桶文件均先写临时文件再原子替换，`manifest.json` 支持续跑；桶内异常证券会记录为 `partial`，不会被误报为完整。`--workers N` 可并行 N 个桶：Windows 使用显式 spawn，worker 只写本桶的三个唯一输出文件并返回状态，父进程独占并原子写入 manifest；改变 workers 不会改变数据配置或阻断续跑。

长任务可分别使用 `--max-shards` 与 `--max-buckets` 有界续跑。所有日期分片未完成前，构建器不会处理任何股票桶，防止用截断历史生成看似完整的波段。

默认 HFQ 是按日可得的 `raw_price * adj_factor`。禁止用样本末端因子物化 QFQ；若研究确需 QFQ，必须显式固定锚点，且只产生锚点当日之后的候选。由此导出的 `candidate_segments/`、`segment_features/` 与 `review_candidates/` 是 parquet 分区数据集；所有候选均为 `UNKNOWN/unreviewed_candidate`，不得当成已校准类别或选股输入。

运行全市场离线发现器：

```powershell
wave-discover-fullmarket --features outputs/fullmarket_candidates_hfq/segment_features --output outputs/fullmarket_discovery_offline --fit-sample-size 50000 --components 5 --ensemble-size 5 --min-cluster-samples 200 --min-cluster-symbols 50 --min-cluster-years 3 --random-state 42
```

训练样本按年份可复现抽取，之后按桶写 `discovery_labels/` 与 `review_candidates/`，保存 `discoverer.joblib`、模型可用性和训练 ID 摘要。该命令使用全历史进行无监督发现，manifest 固定标记 `offline_transductive_review_only`；它只服务人工审核抽样，不能作为因果选股状态。改参数需要新输出目录。HDBSCAN 依赖缺失时会在 summary 中明确报告，不会假报噪声率。

## 2026-09-16 节点记录：全市场候选构建首轮复核

当前工作区 `outputs/fullmarket_candidates_hfq/` 的 manifest 显示核心日线行情覆盖 2015-01-05 至 2024-12-31，共 122 个交易日分片；重新执行 `python -m wave_segments.fullmarket_segments_audit_cli --output outputs/fullmarket_candidates_hfq --require-complete --report outputs/fullmarket_candidates_hfq/audit_recheck_20260916.json`，得到 64/64 个股票桶完成、341,544 个候选波段、0 项结构审计问题。审计报告是结构完整性的证据，不是行情源正确性、股票池无幸存者偏差或分割边界正确性的证明。

`outputs/fullmarket_candidates_hfq/annotation_sample_3000.csv` 有 3,000 个分层抽样候选，覆盖 2015–2024、2,186 只股票和 10 个抽样层；当前 3,000 行 `review_status` 均为空，故正式人工真值仍为零。下一节点应启动小规模双人试标并检查标注一致性、边界容差与工作台回写，再决定是否批量标注；试标意见未形成共识前不得用于监督训练或概率校准。全市场历史股票池快照、停牌/涨跌停、daily_basic、历史行业成员及复权源独立核验仍是选股验证前置风险。

## 2026-09-17 节点记录：元数据缺口审计与可恢复补档

复核发现，核心 `daily` + `adj_factor` 数据覆盖 2015–2024 共 2,431 个交易日、9,402,259 行；核心结构审计通过，复权因子匹配率 100%。但这不代表选股所需元数据完整：`daily_basic`、`stk_limit`、`suspend_d` 在 2,431 日中此前各仅有 1,060 日状态为 `ok`。旧审计默认只要求核心端点，因此会显示通过，容易把核心完整误读成研究数据完整。

补档时发现历史行情分区已含部分合并元数据、但缺端点状态/sidecar 的旧格式；重取元数据会造成重复 Parquet 列。已修复恢复合并逻辑并增加回归测试。通过小批量可恢复续跑补齐了 35 个交易日（每个端点各 35 日），现在三类端点各覆盖 1,095/2,431 日；核心行情未重下。严格端点审计报告写入 `data/a_share_2015_2024/bulk_core_audit_metadata_progress_20260917.json`，仍未通过：每类端点各缺 1,336 日，不能用于声称全市场可交易性/基本面元数据已完整。

之后审计全量元数据时显式运行：

```powershell
wave-audit-bulk-market-data --output data/a_share_2015_2024/bulk_core --universe data/a_share_2015_2024/point_in_time_universe.parquet --require-endpoint daily_basic --require-endpoint stk_limit --require-endpoint suspend_d --report data/a_share_2015_2024/bulk_core_audit.json
```

`--require-endpoint` 可以重复；默认只要求 `daily` 与 `adj_factor`，并在报告中列出每个端点的状态计数。补档继续使用同一构建命令和 `--max-dates N` 小批次；只有 `build_complete=true` 且严格审计通过，才把这些端点纳入完整研究数据声明。人工标签集仍为零，不启动监督训练或概率校准。

## 2026-09-17 节点记录：全市场边界 bootstrap 执行链修复

代码复核发现 `wave-build-fullmarket-segments --bootstrap-iterations N` 虽将参数写入分段配置和 manifest，但桶 worker 未执行 bootstrap；非零设置此前不会影响候选边界置信度。现已在每个股票批次实际执行 OHLCV 小噪声扰动，记录起止边界稳定频率，并将 `boundary_probability` 保守设为原检测器概率与 bootstrap 稳定频率的较小值，同时生成 `boundary_uncertainty`。默认仍为 0，避免全市场构建成本意外放大；需要运行时显式启用该参数。端到端合成测试验证非零参数确实生成稳定度字段且不会抬高检测器置信度。稳定频率只是扰动下可复现性，不是人工边界准确率；尚未在完整全市场上执行高迭代 bootstrap。

随后审查“bootstrap 证据经过同类段合并”时发现：旧实现更新了合并段的终点概率，却可能保留第一段的终点稳定度，形成端点字段不一致。现改为记录 detector 与 bootstrap 的起止端点值，合并时只取外侧起点/终点并重算合并段综合 detector 概率、稳定度及边界概率；连续段内部已消除的边界不再影响最终端点置信度。新增合并证据回归测试和全市场端到端端点上界检查。历史 HSMM 仍是离线全序列平滑，仅 `causal_duration_label` 可作为因果状态候选；二者语义与标签仍需人工样本检验。

复核 3,000 条冻结队列确认全行都是 `UNKNOWN / unreviewed_candidate`，即未运行发现器的待审候选，而非模型弃权。已新增 `candidate_status` 字段，区分 `unreviewed_candidate`、`model_abstention`、`classified`；审核 UI 用醒目提示解释状态，并把状态写入追加事件。旧版队列可根据 `unknown_reason` 进行兼容推断，不会将待审候选误显示成模型弃权。原冻结样本、抽样 ID 与入样概率保持不变；审核队列状态区分测试已覆盖三类状态与旧格式兼容。

对全市场 `segment_features/bucket=000` 做了有界发现器集成试跑：4,565 段、23 个描述性数值特征；默认未来字段拒绝过滤已应用。GMM/DPGMM 临时簇计数为 A=671、B=1,443、C=1,246、D=699、E=303；203 段（4.45%）触发 UNKNOWN，剩余 4,362 段状态为 `classified`。运行时没有安装可选 `hdbscan`，现逐行明确给出 `hdbscan_available=false`、`hdbscan_noise=NA`，而不是误报零噪声。此试跑只覆盖一个哈希桶且训练/打分为同一批样本；簇名是临时标签、`recognizability` 不是正确率，不能作为全市场或样本外证据。Student-t 也尚未进入集成。专项回归测试覆盖状态标签、无 HDBSCAN 语义和旧审核队列兼容。

随后将上述有界验证扩展为全市场离线发现 CLI。首次 50k 拟合出现 sklearn 未收敛警告；该运行被中止并在 `outputs/fullmarket_discovery_offline_20260917/manifest.json` 标为 `interrupted_unvalidated`，其部分分区不得使用。发现器新增 GMM bootstrap 成员与 DPGMM 收敛门控：任一混合模型未收敛即全部弃权；`max_iter` 同时控制两类模型。回归测试验证未收敛强制 UNKNOWN。

在 `max_iter=300` 下重跑 2015–2024 全市场 341,544 段：按年份可复现抽取 49,683 段训练，训练 ID SHA-256 与完整配置写入 summary/manifest；GMM 与 DPGMM 均收敛，HDBSCAN 未安装并显式标记 unavailable。输出 64/64 个 `discovery_labels/` 和 64/64 个 `review_candidates/` 分区，逐分区核验 341,544 个唯一 ID、审核队列 ID/状态/标签对齐、软概率行和为 1、UNKNOWN 均有原因码、HDBSCAN 噪声字段为 NA。最终临时标签计数：`CLUSTER_B` 79,559、`CLUSTER_C` 112,135、`CLUSTER_D` 107,202、`CLUSTER_E` 22,631、UNKNOWN 20,017；`CLUSTER_A` 因跨股票/年份支持门槛未通过，分到该候选的行全部弃权。UNKNOWN 主要触发低密度 13,842、低可识别性 6,786、低共识 4,282、模型分歧 3,558（原因可重叠）。这是全历史 transductive 无监督审核发现，不是因果状态或语义准确率。

从这次带临时簇/弃权区分的审核候选中另建 3,000 条队列：UNKNOWN 1,050、已分类 1,950；覆盖 10 年、2,117 只股票和 50 个 `year × label` strata，入样概率范围 0.00250–0.10490；审核事件仍为 0。队列位于 `outputs/fullmarket_discovery_offline_20260917_v2/annotation_sample_3000.csv`。其 UNKNOWN 富集层用于提升弃权复核效率，任何总体误差估计必须按入样概率倒数加权。全市场 walk-forward 拟合及 HDBSCAN/Student-t 覆盖仍未完成，因此不能以该队列训练或验证因果分类器。

### 分类层独立结构诊断（不含收益目标）

使用 `scripts/evaluate_classifier_structure.py` 检查冻结的因果分类输出，不把第二层收益检验当成分类器质量指标：

```powershell
python scripts/evaluate_classifier_structure.py --states outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/causal_oos_discovery_states.parquet --output outputs/classifier_structure_64sample_iter200_20260917_v3 --min-duration-segments 2 --max-duration-segments 20
```

报告包含每个拟合版本的 UNKNOWN 比例、段长、自转移/转移矩阵、描述性特征画像，以及两组离线 HSMM 时长先验敏感性；簇 ID 始终按 `oos_model_trained_at` 分版本统计，UNKNOWN 作为序列断点，不跨弃权段计算转移。该诊断完全不读取未来收益字段。

在 64 只哈希分层样本（不是全市场统计代表样本）上，7,596 段中 3,573 段被识别、4,023 段 UNKNOWN；34 个已拟合版本加一个 `NO_MODEL` 早期期，按版本统计的 UNKNOWN 比例均值为 53.7%。原始已识别相邻段自转移率的版本中位数为 0.140（P10–P90：0.013–0.658），跨版本差异很大；HSMM `min_duration=2` 与 `3` 时自转移率中位数分别升至 0.968 与 0.796，改变的已识别段比例中位数分别为 40.4% 与 24.2%。这说明离线平滑先验足以显著重塑转移结构，不能因输出更连续就认为更准确，也不能将此离线结果用于选股状态。

报告产物：`model_version_structure.csv`、`transitions_by_model_version.csv`、`hsmm_duration_sensitivity.csv`、`descriptive_profiles_by_version.csv`、`classifier_structure_over_time.png`。这是可复现的内部结构节点，不是语义分类正确性、概率校准或边界 F1 证据。正式 accuracy / Macro-F1 / Brier / ECE / Coverage–Risk 仍需要独立的共识人工标签；你已取消当前人工审核，因此监督训练和正确率声明保持关闭，不以收益结果代替。

### 因果分割阈值扫描（相对稳定性，不是边界真值）

复用同一份 64 只、129,486 根日线的 HFQ 样本，仅改变因果 ATR-ZigZag 的 `atr_reversal`，用选定的 2.2 作为参考、±3 根 K 线容差比较边界：

```powershell
python scripts/evaluate_causal_segmentation_sensitivity.py --bars outputs/fullmarket_causal_sample_1per_bucket_v2/bars_hfq.parquet --output outputs/causal_atr_sensitivity_64sample_20260917_v4 --atr-reversals 1.8 2.2 2.6 --reference 2.2 --tolerance 3 --short-threshold 10
```

结果：ATR 1.8 / 2.2 / 2.6 分别产生 9,580 / 7,596 / 6,104 段；对参考参数输出的 ±3 根边界 F1 agreement 为 0.841 / 1.000 / 0.857；段长中位数为 11 / 14 / 18 根，短于 10 根的比例为 39.0% / 27.5% / 19.2%。低阈值的 recall 较高（0.983）而 precision 较低（0.734）；高阈值相反（precision 0.984，recall 0.759），展示了“边界更密/更疏”的参数权衡。因果段表不包含未确认的开尾，故边界提取排除每股首个数据起点、保留最新已确认 pivot；输出中由 `min_bars` 丢弃的短腿可能造成非连续段端点，因此边界数不必等于段数减股票数。这与离线闭合序列两端都作边缘的惯例不同。

这里的 Precision/Recall/F1 是与 2.2 参数输出的**相对一致性**，2.2 本身不是人工真值；不能称为边界准确率，也不能据此宣称 2.2 最优。`agreement_by_year_symbol.csv` 以边界发生年份切片，`agreement_by_symbol.csv` 列出横截面差异。输出目录还保留各参数的完整段表和图。下一步如要评估边界正确性，必须有独立人工边界共识集；当前不以模型自洽替代该证据。

### 冻结分类器的特征扰动稳健性

对已保存的 34 个因果 walk-forward 拟合版本，使用原训练截止时点重建模型，并先核验每折训练行数、已保存标签及软概率。只有精确复现后才测噪声扰动；执行命令：

```powershell
python scripts/evaluate_classifier_feature_robustness.py --features outputs/fullmarket_causal_64_sample_2015_2024_v1/causal_segment_features.parquet --states outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/causal_oos_discovery_states.parquet --manifest outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/manifest.json --output outputs/classifier_feature_robustness_64sample_2015_2024_20260917_v3 --noise-fractions 0.1 0.25 0.5 --replicates 3
```

用各折训练数据估计特征 MAD 尺度，只对描述性数值特征加入 0.1 / 0.25 / 0.5 倍高斯扰动；持续期、边界不确定度、边界本身及已拟合模型保持固定。5,443 个有保存软概率的 OOS 段、34 个版本、每档 3 次重复均纳入；模型重建后 34/34 版本标签完全相同，软概率最大绝对差异均低于 `1.3e-15`，因此扰动比较以保存版本为基线可复现。

按段数加权，噪声从 0.1、0.25 增至 0.5 倍 MAD 时：标签总体一致率为 97.1%、92.5%、82.6%；原本已识别类别的一致率为 96.9%、91.3%、78.0%；扰动后覆盖为 65.9%、65.4%、62.9%；原已识别样本保留为已识别的比例为 98.9%、96.9%、91.3%；软概率总变差距离均值为 0.021、0.057、0.144。UNKNOWN 和各临时簇的逐类变化保存在 `class_stability.csv`。

这是“固定边界、固定模型条件下的局部特征噪声敏感性”，并未扰动 OHLCV 重新分段或重训模型；噪声幅度是压力测试参数，不代表真实测量误差分布。它不证明语义正确率、概率校准或边界鲁棒性，簇 ID 也只在单个冻结模型版本内比较。详见 `replicate_stability.csv`、`class_stability.csv`、`fold_reproduction.csv`、`summary_by_noise.csv`、`feature_robustness.png` 与 `audit.json`。

### 训练样本构成与重拟合稳定性

随后固定每折 OOS 样本和边界，对该折训练样本按 `symbol` 整簇有放回抽样（每只抽中证券保留其全部历史段），每折重拟合 5 次；重拟合簇通过训练折原空间中心、以参考 RobustScaler 标准化距离后做匈牙利匹配。运行命令：

```powershell
python scripts/evaluate_classifier_resample_robustness.py --features outputs/fullmarket_causal_64_sample_2015_2024_v1/causal_segment_features.parquet --states outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/causal_oos_discovery_states.parquet --manifest outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/manifest.json --output outputs/classifier_symbol_bootstrap_64sample_2015_2024_20260917_v2 --replicates 5
```

34 个历史折都以原截止时点重建：OOS 标签完全一致、概率最大绝对差约 `1.3e-15`；5 次符号簇 bootstrap 共 170 个重拟合模型，GMM/DPGMM 均收敛。5,443 个有软概率的 OOS 段在每个 replicate 中重复评分。按样本加权，重拟合后的总体标签一致率 67.6%，但排除基线 UNKNOWN 后的类别一致率仅 62.6%；覆盖从基线 65.6% 变为 65.4%，基线已识别段在重采样下仍保持已识别为 87.7%，软概率总变差均值 0.304。每折“已识别类一致率”的 P10/中位数/P90 为 40.9% / 60.6% / 79.9%，跨度很大；早期 5 折基线全为 UNKNOWN，其 100% 标签一致只是 UNKNOWN 对 UNKNOWN，不构成分类稳定证据。簇 B/C 的逐类重采样标签一致率约 56.8% / 54.1%，明显弱于 A/D，但簇名依旧只是各折临时 ID。

这项检验暴露出训练股票组成会明显重塑聚类，故目前的软标签/簇身份不宜作为稳定的状态分类直接使用。它仍只是 64 股哈希分层样本上的模型稳定性，不是语义准确率；符号簇 bootstrap 保留个股历史依赖，但没有重采样年份/市场制度。原始逐折/逐类结果在 `bootstrap_replicates.csv`、`class_stability.csv`、`class_summary.csv`、`fold_reproduction.csv`、`summary.csv`、`cluster_bootstrap_robustness.png` 与 `audit.json`。

### 跨年份与市场代理阶段切片

将上述同一折内重采样结果按波段 `available_at` 年份汇总；另外用 64 只样本股票的每日横截面中位数收益构造等权市场代理，按截至确认日的过去 20 个交易日累计收益划分 ±5% 的 `BULL_PROXY / BEAR_PROXY / SIDEWAYS_PROXY`。这不是官方指数，不能外推为全市场牛熊标签。重采样命令增加：

```powershell
python scripts/evaluate_classifier_resample_robustness.py --features outputs/fullmarket_causal_64_sample_2015_2024_v1/causal_segment_features.parquet --states outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/causal_oos_discovery_states.parquet --manifest outputs/fullmarket_causal_64_sample_2015_2024_discovery_iter200/manifest.json --bars outputs/fullmarket_causal_sample_1per_bucket_v2/bars_hfq.parquet --output outputs/classifier_symbol_bootstrap_regime_64sample_2015_2024_20260917 --replicates 5
python scripts/plot_classifier_stability_slices.py --report-dir outputs/classifier_symbol_bootstrap_regime_64sample_2015_2024_20260917
```

按年份，排除 2016 年（该年基线覆盖为 0，100% 总标签一致只是全 UNKNOWN）后，基线已识别段的折内重采样类别一致率：2017–2020 为 54.2%、48.3%、53.2%、47.1%；2021–2024 为 66.8%、70.2%、70.5%、73.5%。这些分数是年份分组的模型稳定性，不是分类正确率；年份间训练样本和拟合版本组成不同，且临时簇只在每个拟合折内对齐，不能解释为某个语义类跨年变好。

市场代理阶段的已识别类一致率为：熊市代理 61.4%（1,977 个独立可评分段）、牛市代理 69.4%（360 段）、震荡代理 62.5%（3,106 段）；阶段覆盖分别为 67.1%、72.6%、63.5%。牛市代理样本尤其少，不能据此比较不同市场阶段的统计显著性或声称牛市分类更可靠。日历年/阶段汇总均对同一段的 5 次重采样加权，簇 ID 不跨模型折比较。

报告和图在 `outputs/classifier_symbol_bootstrap_regime_64sample_2015_2024_20260917/`：`stability_by_year.csv`、`stability_by_market_regime.csv`、逐折重复表、因果市场代理日序列 `sample_market_regime_proxy.csv` 和 `stability_by_year_and_regime.png`。市场代理用当日收盘可知信息，若进入选股研究仍须遵循既有规则：状态从下一可交易日生效；本节只用于分类稳定性切片，不涉及选股收益。

### 全市场离线候选上的广样本重采样压力测试

将范围从 64 股因果样本扩大到 2015–2024 全市场 341,544 条离线候选（64 个桶）。先用原 manifest 配置精确重建发现器的 49,683 条拟合样本，训练 ID SHA-256 与保存摘要一致；保存模型逐桶复现标签 100%，概率最大差为 0。随后对 5,428 只训练股票整簇重采样 3 次（每次约 49.7k 条训练段），用共同训练样本上的软成员重叠做匈牙利对齐，再逐桶评分全候选：

```powershell
python scripts/evaluate_fullmarket_discovery_resample.py --features outputs/fullmarket_candidates_hfq/segment_features --discovery-output outputs/fullmarket_discovery_offline_20260917_v2 --output outputs/fullmarket_discovery_symbol_bootstrap_20260917_v2 --replicates 3
```

在 1,024,632 次段×重采样评估中，基线已识别类一致率为 91.6%、全部标签一致率为 90.7%、扰动后覆盖为 90.8%（基线为 94.1%）、基线已识别类别保持可识别率为 95.0%、软概率 TV 均值为 0.136。按起始年份分层，已识别类一致率介于 85.9%（2015）和 93.6%（2017）之间；临时簇 E 的逐类一致率 81.3%，低于 B/C/D（92.2%/94.1%/90.6%）。基线 UNKNOWN 中约 22.6% 在重采样后转为已识别。

该结果**不能与 64 股因果 OOS 重采样的 62.6% 直接比较**：本节是全历史拟合后回看同一全历史候选的 transductive 描述，不是 OOS、非因果，也没有市场/年份分块 holdout；较高一致率不代表更准确。用户要求的“UNKNOWN 可弃权”仍存在，但 UNKNOWN 本身也有 22.6% 重采样变动，值得优先关注。原模型的中心距离匹配被极端 `atr_mean` 簇中心放大（标准化 RMS 46–229），所以本报告以共同训练样本的软成员余弦重叠作为簇排列依据，中心距离仅保留为警示诊断；三次重采样模型均收敛。HDBSCAN 未安装且未参与这次测试，不能把其原生噪声能力算入结果。

结果在 `outputs/fullmarket_discovery_symbol_bootstrap_20260917_v2/`，含逐桶精确复现、逐桶/逐年重采样统计、逐类稳定性、训练 ID 哈希和审计。全市场扩大了内部结构检查覆盖，但尚未补上严格时间外分类评估或语义真值。

从混合队列按 CLUSTER_B/C/D/E/UNKNOWN 各抽 2 段（合计 10 段、10 只股票）做 AI 视觉可读性试审，图和清单在 `outputs/fullmarket_discovery_offline_20260917_v2/ai_visual_review_pilot_10/`。两个 UNKNOWN 一个呈稀疏阶梯式极端跳升、一个横跨下跌整理与末端急升，保留低密度/多状态弃权是合理的待审意见；各临时簇的两例之间仍观察到明显路径差异（如同一簇分别出现横盘后上冲与急跌后横盘），进一步支持只保留 CLUSTER 临时名。样本太少、意见主观，不能证明簇一致性或任何语义准确率；未回写人工事件。

本轮检查了选股状态构建的拒绝字段：`hsmm_label`、`pre_hsmm_label`、`hsmm_path_score`、`hsmm_changed` 均不能通过 `build_selection_dataset`；`causal_duration_label` 可按 as-of 规则使用，收盘时点状态默认到下一交易日才可见。新增四项回归用例覆盖这些边界。另对 `outputs_real_v2/` 的 8 股票试跑做描述性计算：168 段中 `UNKNOWN` 占 86.3%，相邻已识别转移只有 9 对；原始标签与离线 HSMM 的已识别自转移率均为 1.00，因果 dwell 过滤为 0.11，已识别连续状态 run 的中位长度分别为 20、20、10 根。由于 UNKNOWN 占比高且有效转移对极少，这些数字不能证明因果过滤改善了过分割，也不能评价分类正确率；只说明需同时报告弃权率和转移分母，不能单独解读自转移率。
