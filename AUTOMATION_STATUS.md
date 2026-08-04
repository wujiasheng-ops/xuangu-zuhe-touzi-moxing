# 自动化模块状态

## 已实现

- `auto_rebalance.py`：完整四阶段 + 写入 `results/` + `current_portfolio.json`
- `scheduler.py`：月度/季度重平衡（`config.DEFAULT_REBALANCE_FREQUENCY`）+ 每周相关性检查
- 相关性预警写入 `results/correlation_alerts.json`
- 进程重启后从 `current_portfolio.json` 恢复持仓上下文
- 可选纸面交易：`config.ENABLE_PAPER_TRADING = True`

## 定时任务

| 任务 | 默认 |
|------|------|
| 重平衡 | 每月最后一个工作日 15:00 |
| 相关性检查 | 每周一 09:00 |
| 紧急重平衡 | 相关性 > `CORRELATION_ALERT_THRESHOLD` 时触发 |

## 与主 Pipeline 的关系

自动化模块与 `main_pipeline.py` 共用同一套 stage 脚本，输出格式一致，`final_report.py` 可统一生成报告。

**注意**：选股仍基于当前时点全样本动量（非逐期 walk-forward 选股）；权重样本外验证见 `portfolio_metrics.json` 中的 `walk_forward_metrics`。
