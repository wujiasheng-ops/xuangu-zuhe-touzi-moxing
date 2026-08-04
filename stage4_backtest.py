import pandas as pd
import numpy as np
import config
import stage1_funnel
import stage2_correlation
import stage3_optimizer


def calculate_portfolio_metrics(returns_series, periods_per_year=52):
    if returns_series is None or len(returns_series) == 0:
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'sharpe_ratio': np.nan,
            'volatility': np.nan,
        }

    total_return = (1 + returns_series).prod() - 1
    n = len(returns_series)
    annual_return = (1 + total_return) ** (periods_per_year / n) - 1 if n > 0 else np.nan

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    std = returns_series.std()
    sharpe_ratio = (
        returns_series.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan
    )

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'volatility': std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        'n_weeks': n,
    }


def _turnover_cost_rate(old_weights, new_weights):
    """按换手率扣除单边佣金 + 滑点（long-only：sum(|Δw|) × bps）。"""
    if not new_weights:
        return 0.0
    all_tickers = set(old_weights or {}) | set(new_weights)
    old = {t: float((old_weights or {}).get(t, 0.0)) for t in all_tickers}
    new = {t: float(new_weights.get(t, 0.0)) for t in all_tickers}
    turnover = sum(abs(new[t] - old[t]) for t in all_tickers)
    bps = config.TRANSACTION_COST_BPS + config.SLIPPAGE_BPS
    return turnover * bps / 10000.0


def _apply_rebalance_cost(returns_series, cost_rate):
    if cost_rate <= 0 or returns_series is None or len(returns_series) == 0:
        return returns_series
    out = returns_series.copy()
    out.iloc[0] = out.iloc[0] - cost_rate
    return out


def _portfolio_returns_from_weights(weekly_subset, weights_dict):
    """仅在 weekly_subset 时间范围内计算组合周收益（不引用区间外价格）。"""
    prices = weekly_subset[list(weights_dict.keys())].copy()
    returns = prices.pct_change().dropna()
    if returns.empty:
        return pd.Series(dtype=float)

    w = pd.Series(weights_dict, dtype=float)
    w = w / w.sum()
    w = w.reindex(returns.columns).fillna(0.0)
    return (returns * w).sum(axis=1)


def _assert_no_train_test_overlap(train_slice, test_slice):
    if train_slice.empty or test_slice.empty:
        return
    train_end = train_slice.index.max()
    test_start = test_slice.index.min()
    if train_end >= test_start:
        raise ValueError(
            f'训练/测试日期重叠: train_end={train_end}, test_start={test_start}'
        )


def _stitch_returns(returns_list):
    if not returns_list:
        return pd.Series(dtype=float)
    stitched = pd.concat(returns_list)
    stitched = stitched[~stitched.index.duplicated(keep='first')]
    return stitched.sort_index()


def _average_metrics(metrics_list):
    if not metrics_list:
        return None
    keys = ['total_return', 'annual_return', 'max_drawdown', 'calmar_ratio', 'sharpe_ratio', 'volatility']
    out = {'n_windows': len(metrics_list), 'method': 'mean_of_disjoint_oos_windows'}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and np.isfinite(m[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def _stitch_metrics(returns_list):
    stitched = _stitch_returns(returns_list)
    if stitched.empty:
        return None
    m = calculate_portfolio_metrics(stitched)
    m['method'] = 'stitched_compound_oos_returns'
    return m


def stage4_backtest(
    weekly_data,
    sector_dict,
    outer_selected_tickers=None,
    outer_weights_dict=None,
    weekly_dollar_volume=None,
):
    """
    Stage 4 回测：
    - 每个训练窗内重新执行 Stage 1/Stage 2 选股，实现严格时间隔离
    - 每个训练窗内再执行 Stage 3 权重优化
    - 测试窗结果拼接为真实复利曲线，并报告复合年化收益等指标
    - 可选传入外部最终持仓，计算静态全样本指标供参考
    """
    print('\n=== Stage 4: 滚动回测 ===')
    print(f'可用股票池：{len(weekly_data.columns)} 只')

    subset = weekly_data.copy().sort_index()
    if subset.empty or len(subset) < 52 * 5:
        print('警告：数据不足（需要至少 5 年数据）')
        return None

    dollar_subset = None
    if weekly_dollar_volume is not None and not getattr(weekly_dollar_volume, 'empty', True):
        dollar_subset = (
            weekly_dollar_volume
            .reindex(index=subset.index)
            .reindex(columns=subset.columns)
            .fillna(0.0)
        )

    full_metrics = None
    metrics_2022 = None
    if outer_selected_tickers and outer_weights_dict:
        static_subset = subset[outer_selected_tickers].copy()
        static_portfolio_returns = _portfolio_returns_from_weights(static_subset, outer_weights_dict)
        initial_cost = _turnover_cost_rate({}, outer_weights_dict)
        static_portfolio_returns_net = _apply_rebalance_cost(static_portfolio_returns, initial_cost)
        if initial_cost > 0:
            bps = config.TRANSACTION_COST_BPS + config.SLIPPAGE_BPS
            print(f'\n静态持仓初始建仓成本: {initial_cost * 100:.3f}% （单边 {bps} bps × 换手率）')

        mask_2022 = (static_portfolio_returns_net.index >= '2022-01-01') & (static_portfolio_returns_net.index <= '2022-12-31')
        if mask_2022.sum() > 0:
            metrics_2022 = calculate_portfolio_metrics(static_portfolio_returns_net[mask_2022])
            print('\n2022年表现（静态最终持仓，已扣交易成本）：')
            print(f'  收益率: {metrics_2022["total_return"] * 100:.2f}%')
            print(f'  最大回撤: {metrics_2022["max_drawdown"] * 100:.2f}%')

        full_metrics = calculate_portfolio_metrics(static_portfolio_returns_net)
        print(f'\n样本内静态持仓表现（{len(static_portfolio_returns_net)} 周，已扣初始建仓成本）：')
        print(f'  年化收益率: {full_metrics["annual_return"] * 100:.2f}%')
        print(f'  最大回撤: {full_metrics["max_drawdown"] * 100:.2f}%')
        print(f'  Sharpe Ratio: {full_metrics["sharpe_ratio"]:.3f}')
    else:
        print('警告：未传入静态持仓，样本内静态指标将被跳过')

    train_weeks = 52 * config.TRAIN_YEARS
    test_weeks = 52 * config.TEST_YEARS
    step = 52

    backtest_results = []
    oos_window_metrics = []
    oos_window_returns = []
    prev_oos_weights = None

    roll_start = 0
    while roll_start + train_weeks + test_weeks <= len(subset):
        train_end = roll_start + train_weeks
        test_end = roll_start + train_weeks + test_weeks

        train_slice = subset.iloc[roll_start:train_end]
        test_slice = subset.iloc[train_end:test_end]

        _assert_no_train_test_overlap(train_slice, test_slice)

        print(f'\n--- 回测窗口 {train_slice.index.min().date()} ~ {test_slice.index.max().date()} ---')

        prior_holdings = list(prev_oos_weights.keys()) if prev_oos_weights else None
        dollar_train = dollar_subset.iloc[roll_start:train_end] if dollar_subset is not None else None
        candidates = stage1_funnel.stage1_funnel(
            train_slice, sector_dict, dollar_train, current_holdings=prior_holdings
        )
        if candidates.empty:
            print('  跳过：训练窗内未能选出候选股票')
            roll_start += step
            continue

        selected_df, selected_list, _ = stage2_correlation.stage2_correlation_pruning(
            train_slice, candidates, current_holdings=prior_holdings
        )
        if not selected_list:
            print('  跳过：相关性剪枝后无股票可选')
            roll_start += step
            continue

        wdf, opt_info = stage3_optimizer.stage3_optimizer(
            train_slice, selected_list, quiet=True, current_weights=prev_oos_weights
        )
        if wdf is None or wdf.empty:
            print('  跳过：训练窗优化失败')
            roll_start += step
            continue

        w = dict(zip(wdf['Ticker'], wdf['Weight']))
        if prev_oos_weights and config.ENABLE_STATE_DEPENDENCY:
            from portfolio_state import apply_deadband
            w, _db_meta = apply_deadband(w, prev_oos_weights)

        rebalance_cost = _turnover_cost_rate(prev_oos_weights, w)
        test_rets = _portfolio_returns_from_weights(test_slice, w)
        test_rets = _apply_rebalance_cost(test_rets, rebalance_cost)
        if test_rets.empty:
            print('  跳过：测试窗组合收益为空')
            roll_start += step
            continue

        prev_oos_weights = w

        if not test_rets.index.min() >= test_slice.index.min():
            raise ValueError('测试收益索引越界（混入训练期）')
        if not test_rets.index.max() <= test_slice.index.max():
            raise ValueError('测试收益索引越界（超出测试期）')

        test_m = calculate_portfolio_metrics(test_rets)
        oos_window_metrics.append(test_m)
        oos_window_returns.append(test_rets)

        train_port = _portfolio_returns_from_weights(train_slice, w)
        train_m = calculate_portfolio_metrics(train_port)

        backtest_results.append({
            'period_start': test_slice.index.min(),
            'period_end': test_slice.index.max(),
            'train_start': train_slice.index.min(),
            'train_end': train_slice.index.max(),
            'test_start': test_slice.index.min(),
            'test_end': test_slice.index.max(),
            'optimizer_method': opt_info.get('method'),
            'train_annual_return': train_m['annual_return'],
            'test_annual_return': test_m['annual_return'],
            'test_total_return': test_m['total_return'],
            'test_max_drawdown': test_m['max_drawdown'],
            'test_sharpe': test_m['sharpe_ratio'],
            'test_calmar': test_m['calmar_ratio'],
            'test_weeks': test_m.get('n_weeks'),
            'rebalance_cost_pct': rebalance_cost * 100,
        })
        roll_start += step

    results_df = pd.DataFrame(backtest_results)
    print(f'\n滚动样本外回测：{len(results_df)} 个独立测试窗（训练/测试严格隔离）')

    walk_forward_metrics = _average_metrics(oos_window_metrics)
    stitched_metrics = _stitch_metrics(oos_window_returns)

    if walk_forward_metrics:
        print('\n样本外表现（各测试窗指标算术平均，已扣换手成本）：')
        print(f'  平均年化收益率: {walk_forward_metrics["annual_return"] * 100:.2f}%')
        print(f'  平均单窗总收益: {walk_forward_metrics["total_return"] * 100:.2f}%')
        print(f'  平均最大回撤: {walk_forward_metrics["max_drawdown"] * 100:.2f}%')
        print(f'  Sharpe（均值）: {walk_forward_metrics["sharpe_ratio"]:.3f}')

    if stitched_metrics:
        print('\n拼接样本外表现（真实复利CAGR，按测试窗净值连续拼接）：')
        print(f'  复合年化收益率: {stitched_metrics["annual_return"] * 100:.2f}%')
        print(f'  累计总收益: {stitched_metrics["total_return"] * 100:.2f}%')
        print(f'  最大回撤: {stitched_metrics["max_drawdown"] * 100:.2f}%')
        print(f'  Sharpe: {stitched_metrics["sharpe_ratio"]:.3f}')

    if len(results_df) > 0:
        print('\n各测试窗明细：')
        for _, row in results_df.iterrows():
            print(
                f"  {row['test_start'].date()} ~ {row['test_end'].date()}: "
                f"年化 {row['test_annual_return']*100:.1f}%, "
                f"总收益 {row['test_total_return']*100:.1f}%"
            )

    return {
        'full_metrics': full_metrics,
        'metrics_2022': metrics_2022,
        'walk_forward_metrics': walk_forward_metrics,
        'walk_forward_metrics_stitched': stitched_metrics,
        'rolling_backtest': results_df,
        'transaction_costs': {
            'commission_bps': config.TRANSACTION_COST_BPS,
            'slippage_bps': config.SLIPPAGE_BPS,
            'method': 'turnover_based_sum_abs_delta_w',
        },
    }


if __name__ == '__main__':
    import data_fetcher
    import stage1_funnel
    import stage2_correlation

    tickers, weekly_data, sectors, dollar_vol = data_fetcher.prepare_data()
    if weekly_data is None or weekly_data.empty:
        print('数据获取失败')
    else:
        candidates = stage1_funnel.stage1_funnel(weekly_data, sectors, dollar_vol)
        selected_df, selected_list, _ = stage2_correlation.stage2_correlation_pruning(weekly_data, candidates)
        weights_df, _ = stage3_optimizer.stage3_optimizer(weekly_data, selected_list)
        weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
        stage4_backtest(weekly_data, sectors, selected_list, weights_dict, dollar_vol)
