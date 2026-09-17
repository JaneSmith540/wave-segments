# 变长波段识别

面向 OHLCV 行情的研究工具：按数据结构切分变长波段，为波段输出软概率标签或 `UNKNOWN`，并提供上下文特征、描述统计、转移分析、图表和人工审核记录。它是描述性分类/归纳工具，不预测未来涨跌，不生成交易信号，也不要求覆盖每一段行情。

## 安装

需要 Python 3.10 或更高版本。

```powershell
python -m pip install -e ".[data,changepoint,dev]"
```

Tushare 是可选行情来源。将令牌设置在当前 shell 的环境变量中，不要写入源码、配置文件或提交记录：

```powershell
$env:TUSHARE_API_TOKEN = "你的 Tushare token"
```

程序也兼容 `TUSHARE_TOKEN` 和 `TS_TOKEN`。本项目不附带行情数据；使用 Tushare 还需遵守其服务条款和权限限制。也可以读取本地 CSV/Parquet 文件。

## 快速运行

在线读取 Tushare 日线并运行描述流水线：

```powershell
wave-segments --symbols 000001.SZ 600000.SH --start 20200101 --end 20241231 --freq D --adj qfq
```

或者先准备包含 `symbol,timestamp,open,high,low,close,volume` 列的本地文件：

```powershell
wave-segments --input data/bars.parquet --config config.example.json --output outputs
```

生成仅含合成行情的演示数据并运行：

```powershell
python scripts/generate_demo.py
wave-segments --input data/demo_bars.parquet --config config.example.json --output outputs/demo
```

输出目录包含候选波段、特征、概率/弃权结果、类别描述统计、转移矩阵、规则说明、人工审核 CSV 和可视化。数据、模型产物和运行结果均默认留在本地，不属于源码发布内容。

## 方法概览

- **变长分割**：ATR 回撤转折，并可组合结构变点、分形和 BOCPD；算法和阈值可在配置中调整。
- **软分类**：聚类/集成模型保留概率分布与候选解释，不把临时簇名冒充固定业务语义。
- **UNKNOWN**：结合概率、熵、模型分歧、密度/似然、边界稳定性和上下文冲突弃权；低可识别样本可留待审核。
- **上下文**：支持大盘、行业/板块和多周期特征的时间对齐；对齐时应只使用该时点可获得的信息。
- **解释与审核**：生成类别画像、持续期和转移统计、浅层规则、波段着色图与审核队列；人工意见可作为后续迭代记录。

更多细节见 [分层架构](docs/ARCHITECTURE.md)、[验证流程](docs/VALIDATION_WORKFLOW.md)、[审核指南](docs/ANNOTATION_GUIDE.md) 和 [风险登记表](docs/RISK_REGISTER.md)。这些统计用于描述和研究，不代表稳定预测能力或投资建议。

仓库中还保留了独立的因果状态/横截面研究实验代码；它不是本项目分类器的目标，不参与默认波段分类流水线，也不产生交易信号。使用者可以忽略该扩展，仅使用分割、波段表征、概率类别、UNKNOWN 和审核功能。

## 安全与数据

不要提交 `.env`、API token、原始/派生行情、个人审核记录、模型文件或运行输出。`.gitignore` 已排除本地 `data/` 与 `outputs*/`。如需报告安全问题，请勿在公开 issue 中粘贴凭据或个人数据。

## 开发与测试

```powershell
python -m pytest -q
ruff check .
```

## 许可证

本项目按 [MIT License](LICENSE) 发布。
