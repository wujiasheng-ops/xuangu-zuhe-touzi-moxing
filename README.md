# 选股组合投资模型

从 **S&P 500** 自动选出约 **10 只** 低相关、行业分散的动量组合，优化权重，并对照 **SPY** 做滚动回测与报告生成。

> 仅供研究学习；历史回测不代表未来收益，不构成投资建议。

---

## 快速开始

```bash
python -m pip install -r requirements.txt

# 实验（写入 results/test，不影响正式状态）
python quick_test.py

# 正式跑（写入 results/live）
python main_pipeline.py

# 生成报告
python final_report.py
```

首次完整运行约 30–45 分钟（下载并缓存周线）；之后有 `cache/` 会快很多。

---

## 算法流水线

```
yfinance 周线 + Wikipedia GICS
  → 流动性过滤（周均成交额）
  → Stage 1  动量漏斗（约 500 → 100，现仓忠诚加成）
  → Stage 2  相关性剪枝（→ 10，|ρ|≤0.6，现仓替换门槛）
  → Stage 3  权重优化（Sharpe；换手收缩 + 权重上下限）
  → Stage 4  样本外滚动回测（含交易成本假设）
  → 决策日志 / SPY 对照
  → results/live 报告
```

状态依赖（可关）：现仓 Stage1 加成、Stage2 替换边际、Stage3 换手收缩、执行层 Deadband。

---

## 环境隔离

| 环境 | 何时 | 输出目录 |
|------|------|----------|
| **PROD**（默认） | `main_pipeline.py` / 正式重平衡 | `results/live/` |
| **TEST** | `quick_test.py` 等实验 | `results/test/` |

正式状态只认 **`results/live`**。可用 `TRADING_ENV=TEST` / `TRADING_ENV=PROD` 切换。

---

## 主要输出（`results/live/`）

| 文件 | 内容 |
|------|------|
| `selection_results.json` | 选股、权重、动量 |
| `final_recommendation.json` / `FINAL_REPORT.md` | 最终持仓与摘要 |
| `optimal_weights.csv` | 权重表 |
| `correlation_matrix.csv` | 相关性矩阵 |
| `portfolio_metrics.json` | 回测与基准相关指标 |
| `benchmark_latest.json` / `equity_curve.csv` | SPY 对照与权益曲线 |
| `decision_log_latest.json` / `logs/` | 结构化决策日志 |
| `rebalance_history.json` / `current_portfolio.json` | 正式持仓状态 |

---

## 项目结构

| 路径 | 说明 |
|------|------|
| `config.py` | 全局参数（环境、流动性、状态依赖、Deadband 等） |
| `data_fetcher.py` | 成分股、周线、行业、成交额、缓存 |
| `stage1_funnel.py` ~ `stage4_backtest.py` | 四阶段核心 |
| `portfolio_state.py` | 读取上一期正式持仓 |
| `benchmark.py` / `decision_log.py` | SPY 对照与决策日志 |
| `main_pipeline.py` | **正式入口** |
| `quick_test.py` | 样本股实验（TEST） |
| `auto_rebalance.py` / `scheduler.py` | 手动 / 定时重平衡 |
| `final_report.py` | 报告生成 |
| `trading_interface.py` | 本地模拟对照（可选扩展） |
| `results/live/` · `results/test/` | 正式 / 实验产物 |
| `cache/` | 本地缓存（不入库） |

---

## 关键配置（`config.py`）

| 参数 | 默认含义 |
|------|----------|
| `TARGET_SIZE` | 最终约 10 只 |
| `CORRELATION_THRESHOLD` | 相关性 \|ρ\| ≤ 0.6 |
| `MAX_WEIGHT` | 单票上限 15% |
| `MIN_AVG_WEEKLY_DOLLAR_VOLUME` | 周均成交额下限（流动性） |
| `DEADBAND_THRESHOLD` | 2.5%，执行层静默阈值 |
| `ENABLE_STATE_DEPENDENCY` | 状态依赖开关 |
| `DEFAULT_REBALANCE_FREQUENCY` | `monthly` / `quarterly` |

可选：`USE_RISK_PARITY_WEIGHTS = True` 启用风险平价（v2）。

---

## 技术栈

| 类别 | 依赖 |
|------|------|
| 数据 | yfinance、Wikipedia GICS |
| 计算 | pandas、numpy、scipy、scikit-learn |
| 优化 | PyPortfolioOpt |
| 调度 | APScheduler |

完整列表见 [`requirements.txt`](requirements.txt)。

---

## 安全

- `config.py` 只有算法参数，可公开。
- `credentials.json`、`.env` 已忽略，勿提交。
- `results/` 为公开股票权重与回测产物，不含账户资金信息。

---

## 更多文档

- [`RUN_GUIDE.md`](RUN_GUIDE.md) — 运行细节
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — 设计说明
- [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) — 环境参考

---

## License

MIT（如需更改请自行添加 LICENSE 文件）
