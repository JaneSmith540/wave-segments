# 人工标注指南

每个波段至少由两名审核者独立判断，不查看波段结束后的行情。界面只展示截至波段终点及少量历史上下文；若展示终点之后的数据，必须明确仅用于“边界事后审查”，不得用于形态真值。

审核队列的 `candidate_status` 与 `predicted_label` 含义不同：`unreviewed_candidate` 表示尚未运行发现器；`model_abstention` 表示发现器运行后触发 UNKNOWN 弃权门控；`classified` 表示已有临时簇类、仍未具备人工语义。仅凭 `predicted_label=UNKNOWN` 不可推断模型覆盖率或拒识率。

类别定义：

- 上涨推进：方向明确、上涨路径占主导，回撤相对受控。
- 下跌推进：方向明确、下跌路径占主导，反弹相对受控。
- 高波动震荡：无稳定净方向且区间振幅、波动显著。
- 低波动盘整：无稳定净方向且波动、振幅较低。
- 反转过渡：方向或波动制度正在切换，难归为稳定推进或盘整。
- 跳跃事件冲击：缺口、异常单日跳跃或事件驱动主导整段表征。
- UNKNOWN：信息不足、多种解释均合理、边界不合理或不属于现有类型。

同时填写边界 `yes/no/uncertain`；若为 `no`，建议新的开始/结束位置。不要为了提高覆盖率而避免 UNKNOWN。共识生成时只接受全体一致标签，任何分歧均进入 UNKNOWN 队列。

AI 可以协助检查图表是否可读、指出路径异质或边界可疑，但 AI 复核不是人工标注，不计入多人一致性、真值、模型校准或最终测试集。AI 观察到的“看起来清楚”也不能覆盖模型的 UNKNOWN；尤其 `purged_overlap`、支持不足等原因代表门控/样本条件，不等同于形态本身不可理解。

## 多人一致性报告

每个 `segment_id` 的每位审核者只取最新一条追加事件。至少两位审核者完全同意时才产生类别共识；单人标注或任意分歧都产生 `consensus_label=UNKNOWN`，同时保留 `vote_counts` 和 `consensus_reason`，不会把多数票硬当真值。

类别报告给出原始 pairwise agreement、按 `sampling_probability` 倒数加权的 Cohen kappa；有至少三位审核者时还给出 Fleiss kappa。边界报告以审核事件中的 `start_idx`/`end_idx`（或数值形式的修正值）在默认 ±3 根 K 线容差内逐审核者对计算 precision、recall、F1。日期文本修正必须先依据该股票交易日历转换为 bar index，不能把自然日差当 K 线差。

## 抽样与防偏差

不能只审核 UNKNOWN，也不能只挑图形“典型”的波段。使用分层抽样同时覆盖年份、股票、临时簇、行业、市场状态和市值桶，并保留一部分 UNKNOWN 富集样本：

```powershell
wave-build-review-sample --segments outputs/merged_labeled_segments.parquet --target-size 3000 --unknown-share 0.35 --output outputs/annotation_sample.csv
```

队列中的 `sampling_stratum` 用于检查覆盖面，`sampling_probability` 是 UNKNOWN/非 UNKNOWN 池内的入样概率。因为审核样本有意富集 UNKNOWN，估计总体错误率时必须使用该概率的倒数加权；训练集可以重采样，但最终测试集的抽样方案和权重必须在看结果前冻结。

AI 视觉检查试点及图表渲染命令见因果样本输出目录下的 `visual_review_pilot_20/AI_VISUAL_REVIEW.md`；它只用于发现展示和样本异质性问题，不能代替正式双人试标。
