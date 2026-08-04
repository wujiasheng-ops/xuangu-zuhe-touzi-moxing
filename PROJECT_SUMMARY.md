# 自动选股与配置算法 - 项目说明

## 目标

从 **S&P 500** 中自动选出 **低相关、行业分散** 的动量组合，优化权重，并支持定期重平衡与报告生成。

## 四阶段 Pipeline

```
数据 (yfinance + Wikipedia GICS)
  → Stage 1 动量漏斗 (500→100)
  → Stage 2 相关性剪枝 (100→10, |ρ|≤0.6)
  → Stage 3 权重优化 (Sharpe 或风险平价)
  → Stage 4 样本外滚动回测
```

## 文件结构

| 文件 | 说明 |
|------|------|
| `config.py` | 全局参数 |
| `data_fetcher.py` | 成分股、周线、行业、缓存 |
| `stage1_funnel.py` ~ `stage4_backtest.py` | 四阶段 |
| `main_pipeline.py` | 生产入口 |
| `quick_test.py` | 样本股快速测试 |
| `auto_rebalance.py` | 自动化重平衡 |
| `scheduler.py` | 定时任务 |
| `final_report.py` | 从 `results/` 生成报告 |
| `trading_interface.py` | 模拟 / Alpaca 下单（可选） |
| `algorithm_v2_improvements.py` | 风险平价等 v2 扩展 |

## 如何查看结果

运行 `main_pipeline.py` 或 `quick_test.py` 后：

```bash
python final_report.py
```

或查看 `results/final_recommendation.json`、`FINAL_REPORT.md`（自动生成）。

## 免责声明

仅供研究学习；历史回测不代表未来收益。
