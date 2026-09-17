# 分层架构

## 边界原则

系统由两个不可反向依赖的层组成：

```text
时点行情/上下文
    ↓
第一层：分割 → 描述特征 → 形态簇/人工类别 → 校准概率/UNKNOWN
    ↓ 冻结的、截至波段结束时可知的状态快照
第二层：未来目标构造 → 横截面研究 → IC/分组/成本后评估
```

第一层不得导入 `selection.py`，不得读取任何 `future_*`、`forward_*` 或下一期收益字段。第二层可以消费第一层的冻结输出，但不得回写分类标签、边界或分类模型参数。

## 第一层：可信波段分类

- `dataset.py`：按季度分区、断点续传；`point_in_time_universe` 保留研究期内退市股票。
- `segmentation.py`：离线描述路径 `segment_ohlcv` 融合 ATR-ZigZag、自适应结构变点、分形、BOCPD 和可选 ruptures；因果路径 `segment_ohlcv_causal` 仅用在线兼容 ATR-ZigZag，输出端点 `end` 与实际确认时间 `available_at`，不输出未闭合尾段。因果路径的边界分数目前是启发式强度，不是校准正确率。
- `stability.py`：局部噪声 bootstrap 边界频率、人工边界容差 F1、连续同类合并。
- `duration.py`：离线显式时长 HSMM 用于历史描述平滑；另有仅用过去信息的因果 dwell filter，供第二层状态快照使用。二者字段明确分离。
- `features.py`：只使用波段结束时已知的价格、量能、形态和上下文。
- `discovery.py`：融合 GMM、DPGMM 和可选 HDBSCAN，只输出 `CLUSTER_A/B/C`；`hdbscan` 未安装时逐行 `hdbscan_available=false`、`hdbscan_noise=NA`，不会假报“无噪声”。数值画像不等于业务语义，类别需满足最小样本、跨股票、跨年份支持。簇名只在一个拟合版本内按特征锚点排序，不是跨 walk-forward 重训的稳定语义 ID；汇总时必须保留 `oos_model_trained_at` 版本边界。当前实现尚未把 Student-t 原型纳入集成；`model.py` 保留为单一 GMM 基线。
- `causal_diagnostics.py`：按股票、起始年份、拟合版本统计因果状态覆盖和 UNKNOWN 原因，避免把不同模型版本里的同名临时簇误读为同一类别。
- `review.py`：UNKNOWN 优先审核队列和人工标签回写；`candidate_status` 明确区分待发现候选、模型弃权和已有临时簇类，不以 `UNKNOWN` 单字段混淆处理状态。
- `fullmarket_discovery_cli.py`：按年份可复现抽取有界训练样本、按桶推理并写入临时簇结果与审核队列；产物模式固定为 `offline_transductive_review_only`，禁止当作因果选股状态。
- `validation.py`：人工一致性、Purged Walk-forward、独立概率校准、分类指标、Coverage–Risk 与 APS 集合预测。

人工确认的标准类别为：上涨推进、下跌推进、高波动震荡、低波动盘整、反转过渡、跳跃事件冲击、UNKNOWN。任何审核分歧默认归入 UNKNOWN。

## 第二层：选股研究验证

`selection.py` 通过 backward `merge_asof` 接入已生效状态，以 `shift(-h)` 单独构造 5/20/60 日未来超额收益、最大不利波动、收益风险比和横截面排名。它只返回研究指标，不返回订单或交易指令。

选股层只能使用 `causal_duration_label`；`hsmm_label` 使用了完整历史序列做离线 Viterbi，严禁作为过去时点特征。

输出包括行业/市值中性后的 Rank IC、ICIR、分组收益、单调性、换手、成本与滑点后多空收益、最大回撤、年度/市场状态稳定性，以及与动量和波动率基线的同口径比较。

## 数据隔离

- train、calibration、test 按时间顺序排列。
- 选股时只能使用 `segment_ohlcv_causal` 产出的段，经 `available_at` backward join；回看融合路径只作历史描述/审核。
- 同股票的 K 线区间不得在分区间重叠；重叠段从较早分区清除。
- 边界 bootstrap 必须在每个 walk-forward 窗口内部运行，禁止使用全历史稳定度回填过去。
- 股票池、行业、市值、财务数据都必须按当时可得时间对齐；不得用今天的成分股替代历史股票池。
