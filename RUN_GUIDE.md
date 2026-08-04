# 选股组合投资模型 - 运行指南

## 核心脚本

| 脚本 | 用途 | 耗时 |
|------|------|------|
| `quick_test.py` | 20 只样本股验证全流程 | ~2–3 分钟 |
| `main_pipeline.py` | S&P 500 完整四阶段选股 | ~30–45 分钟（首次） |
| `auto_rebalance.py` | 手动触发一次重平衡 | 同 main_pipeline |
| `scheduler.py` | 后台定时重平衡 + 相关性监控 | 持续运行 |
| `final_report.py` | 从 `results/` 生成报告 | 秒级 |

## 推荐流程

```bash
# 1. 安装依赖（使用 VS Code 配置的 Anaconda 解释器）
/opt/anaconda3/bin/python -m pip install -r requirements.txt

# 2. 快速验证
python quick_test.py

# 3. 完整运行
python main_pipeline.py
```

## 输出文件（`results/`）

- `selection_results.json` — 选股、权重、动量
- `optimal_weights.csv` — 权重表
- `correlation_matrix.csv` — 相关性矩阵
- `backtest_results.csv` — 滚动样本外回测分段
- `portfolio_metrics.json` — **样本外指标（优先参考）** + 样本内指标
- `risk_analysis.json` — 风险摘要
- `final_recommendation.json` — 由 `final_report.py` 自动生成
- `current_portfolio.json` — 自动化模块当前组合
- `correlation_alerts.json` — 相关性预警历史

## 重要说明

1. **报告不再硬编码**：`final_report.py` / `FINAL_REPORT.md` 均从 `results/` 动态生成。
2. **回测**：Stage 4 在训练窗内重新优化权重，拼接测试期得到 `walk_forward_metrics`（比样本内更可信）。
3. **数据**：价格使用 `auto_adjust=True`；动量超过 ±500% 的股票会被过滤。
4. **行业**：来自 Wikipedia GICS 表，不再对 500 只股票逐个调 yfinance API。
5. **配置**：编辑 `config.py` 可调整动量、相关性阈值、权重上限、重平衡频率等。

## 自动化

```bash
python scheduler.py   # 默认月度重平衡；config.DEFAULT_REBALANCE_FREQUENCY='quarterly' 可改季度
```

`config.ENABLE_PAPER_TRADING = True` 时，`auto_rebalance.py` 结束后会用模拟券商调仓。

## 可选 v2 功能

`config.USE_RISK_PARITY_WEIGHTS = True` 启用 `algorithm_v2_improvements.py` 中的风险平价权重。
