"""
最终选股推荐与风险分析报告
从 main_pipeline / auto_rebalance 写入的 results/ 动态生成，不再使用硬编码快照。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _corr_summary() -> Dict[str, Any]:
    path = os.path.join(config.RESULTS_DIR, 'correlation_matrix.csv')
    if not os.path.isfile(path):
        return {}
    c = pd.read_csv(path, index_col=0)
    v = c.values.astype(float)
    n = v.shape[0]
    if n < 2:
        return {}
    triu_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            triu_vals.append(v[i, j])
    arr = np.array(triu_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    abs_arr = np.abs(arr)
    mx = float(abs_arr.max())
    mn = float(arr.min())
    return {
        'max_abs_correlation': mx,
        'min_correlation': mn,
        'all_below_threshold': mx <= config.CORRELATION_THRESHOLD,
        'status': '✓ 所有股票对 |ρ|≤阈值' if mx <= config.CORRELATION_THRESHOLD else '⚠️ 存在高相关性股票对',
    }


def build_recommendation() -> Dict[str, Any]:
    sel = _load_json('selection_results.json')
    metrics = _load_json('portfolio_metrics.json')
    risk = _load_json('risk_analysis.json')

    if not sel:
        raise FileNotFoundError(
            '未找到 selection_results.json，请先运行 main_pipeline.py 或 auto_rebalance.py'
        )

    tickers = sel.get('selected_tickers', [])
    weights = sel.get('weights', {})
    sectors = sel.get('sectors', {})
    momentum = sel.get('momentum', {})

    stocks = []
    for t in tickers:
        stocks.append({
            'ticker': t,
            'weight': float(weights.get(t, 0)),
            'sector': sectors.get(t, 'Unknown'),
            'momentum': float(momentum.get(t, 0)),
        })

    sector_dist = {}
    for s in stocks:
        sec = s['sector']
        sector_dist[sec] = sector_dist.get(sec, 0) + 1

    in_sample = (metrics or {}).get('in_sample_metrics') or {}
    walk_fwd = (metrics or {}).get('walk_forward_metrics') or {}
    crisis = (metrics or {}).get('metrics_2022') or {}
    tx_costs = (metrics or {}).get('transaction_costs') or {}
    optimization = sel.get('optimization') or {}
    bench = sel.get('benchmark') or _load_json('benchmark_latest.json') or (metrics or {}).get('benchmark')

    return {
        'generation_date': datetime.now().isoformat(),
        'source_run_timestamp': sel.get('timestamp'),
        'trading_env': getattr(config, 'TRADING_ENV', 'PROD'),
        'algorithm_name': '自动选股与配置算法',
        'version': '1.2',
        'data_source': f'Yahoo Finance ({config.LOOKBACK_PERIOD}年周线, auto_adjust)',
        'stage2_tickers': sel.get('stage2_tickers', tickers),
        'selected_stocks': stocks,
        'weight_adjustments': optimization.get('dropped_low_weight'),
        'portfolio_metrics': {
            'in_sample': in_sample,
            'walk_forward': walk_fwd,
            'preferred': 'walk_forward',
            'transaction_costs': tx_costs,
        },
        'benchmark': bench or {},
        'backtest_2022_crisis': crisis,
        'sector_distribution': {
            'count': len(sector_dist),
            'sectors': sector_dist,
        },
        'correlation_matrix_summary': _corr_summary(),
        'risk_analysis': risk or {},
    }


def print_report():
    rec = build_recommendation()
    stocks = rec['selected_stocks']

    print('=' * 70)
    print('自动选股与配置算法 - 最终选股推荐报告')
    print('=' * 70)
    print(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据来源：{rec['data_source']}")
    print(f"对应 pipeline 运行：{rec.get('source_run_timestamp', 'N/A')}")

    print(f"\n【最终持仓 - {len(stocks)} 只股票】")
    print('-' * 70)
    print(f"{'序号':<4} {'代码':<8} {'权重':<10} {'行业':<28} {'Momentum':<10}")
    print('-' * 70)
    total_w = 0.0
    for i, s in enumerate(stocks, 1):
        total_w += s['weight']
        print(
            f"{i:<4} {s['ticker']:<8} {s['weight']*100:>6.2f}% "
            f"{s['sector']:<28} {s['momentum']:>7.2f}%"
        )
    print('-' * 70)
    print(f"{'总计':<22} {total_w*100:>6.2f}%")

    wf = rec['portfolio_metrics'].get('walk_forward') or {}
    ins = rec['portfolio_metrics'].get('in_sample') or {}
    if wf:
        method = wf.get('method', '')
        suffix = f' ({method})' if method else ''
        print(f'\n【样本外回测指标（推荐参考）{suffix}】')
        if wf.get('n_windows'):
            print(f"  独立测试窗数量：{wf['n_windows']}")
        print(f"  平均年化收益率：{wf.get('annual_return', 0)*100:.2f}%")
        print(f"  平均单窗总收益：{wf.get('total_return', 0)*100:.2f}%")
        print(f"  平均最大回撤：{wf.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Sharpe（均值）：{wf.get('sharpe_ratio', 0):.3f}")
    if ins:
        print('\n【样本内指标（仅供参考，可能偏乐观）】')
        print(f"  年化收益率：{ins.get('annual_return', 0)*100:.2f}%")
        print(f"  Sharpe：{ins.get('sharpe_ratio', 0):.3f}")

    c = rec.get('backtest_2022_crisis') or {}
    if c:
        print('\n【2022 年子区间】')
        print(f"  收益率：{c.get('total_return', 0)*100:.2f}%")
        print(f"  最大回撤：{c.get('max_drawdown', 0)*100:.2f}%")

    corr = rec.get('correlation_matrix_summary') or {}
    if corr:
        print('\n【相关性】')
        print(f"  最大 |ρ|：{corr.get('max_abs_correlation', 0):.3f}")
        print(f"  状态：{corr.get('status', '')}")

    dropped = rec.get('weight_adjustments') or {}
    if dropped:
        print('\n【权重回填说明】')
        print('  以下 Stage2 标的优化器给出极低权重，已抬升至 MIN_WEIGHT 后归一化：')
        for t, w in dropped.items():
            print(f'    {t}: 原始权重 {w*100:.4f}%')

    tx = (rec.get('portfolio_metrics') or {}).get('transaction_costs') or {}
    if tx:
        print('\n【交易成本假设】')
        print(f"  佣金：{tx.get('commission_bps', config.TRANSACTION_COST_BPS)} bps，"
              f"滑点：{tx.get('slippage_bps', config.SLIPPAGE_BPS)} bps")
        print(f"  样本内初始建仓成本：{tx.get('in_sample_initial_cost_pct', 0):.3f}%")

    bench = rec.get('benchmark') or {}
    if bench:
        import benchmark as bench_mod
        print('\n' + bench_mod.format_period_comparison(bench))

    print('\n【免责声明】')
    print('本报告由程序根据历史数据自动生成，不构成投资建议。')
    print('=' * 70)


def save_report_to_json():
    rec = build_recommendation()
    path = os.path.join(config.RESULTS_DIR, 'final_recommendation.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n✓ 报告已保存到：{path}')
    return path


def save_report_to_markdown():
    rec = build_recommendation()
    path = os.path.join(config.BASE_DIR, 'FINAL_REPORT.md')
    stocks = rec['selected_stocks']
    wf = rec['portfolio_metrics'].get('walk_forward') or {}

    lines = [
        '# 选股组合投资模型 最终选股报告',
        '',
        f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Pipeline 运行**：{rec.get('source_run_timestamp', 'N/A')}  ",
        f"**数据来源**：{rec['data_source']}  ",
        '',
        f'## 最终持仓（{len(stocks)} 只）',
        '',
        '| 代码 | 权重 | 行业 | Momentum |',
        '|------|------|------|----------|',
    ]
    for s in stocks:
        lines.append(
            f"| {s['ticker']} | {s['weight']*100:.2f}% | {s['sector']} | {s['momentum']:.2f}% |"
        )

    lines.extend([
        '',
        '## 样本外回测（推荐参考）',
        '',
        f"- 年化收益：{wf.get('annual_return', 0)*100:.2f}%",
        f"- 最大回撤：{wf.get('max_drawdown', 0)*100:.2f}%",
        f"- Sharpe：{wf.get('sharpe_ratio', 0):.3f}",
        '',
    ])
    bench = rec.get('benchmark') or {}
    if bench:
        sr = bench.get('strategy_return')
        sp = bench.get('spy_return')
        al = bench.get('alpha')
        lines.extend([
            '## 本期 vs SPY',
            '',
            f"- 策略收益：{(sr or 0)*100:+.2f}%" if sr is not None else '- 策略收益：N/A',
            f"- SPY 收益：{(sp or 0)*100:+.2f}%" if sp is not None else '- SPY 收益：N/A',
            f"- Alpha：{(al or 0)*100:+.2f}%" if al is not None else '- Alpha：N/A',
            '',
        ])
    lines.append(f'> 由 `final_report.py` 根据 `{config.RESULTS_DIR}` 自动生成（ENV={config.TRADING_ENV}）。')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'✓ Markdown 报告已保存到：{path}')
    return path


if __name__ == '__main__':
    print_report()
    save_report_to_json()
    save_report_to_markdown()
