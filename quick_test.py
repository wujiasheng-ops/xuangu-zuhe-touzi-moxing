"""
快速测试版本 - 20 只样本股验证完整四阶段流程
物理隔离：强制 TRADING_ENV=TEST，写入 results/test/，不污染 live。
"""

import os

# 必须在 import config 之前设置，避免污染正式状态目录
os.environ['TRADING_ENV'] = 'TEST'

import json
from datetime import datetime

import pandas as pd
import yfinance as yf

import config
config.apply_trading_env('TEST')

import data_fetcher
import stage1_funnel
import stage2_correlation
import stage3_optimizer
import stage4_backtest
import decision_log
import portfolio_state


SAMPLE_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JNJ', 'UNH', 'PFE', 'ABBV',
    'JPM', 'BAC', 'GS', 'WFC',
    'XOM', 'CVX',
    'WMT', 'KO', 'MCD',
    'TSLA', 'AMZN',
]

TICKER_SECTOR = {
    'AAPL': 'Information Technology', 'MSFT': 'Information Technology',
    'NVDA': 'Information Technology', 'GOOGL': 'Communication Services',
    'META': 'Communication Services', 'JNJ': 'Health Care', 'UNH': 'Health Care',
    'PFE': 'Health Care', 'ABBV': 'Health Care', 'JPM': 'Financials',
    'BAC': 'Financials', 'GS': 'Financials', 'WFC': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'WMT': 'Consumer Staples',
    'KO': 'Consumer Staples', 'MCD': 'Consumer Staples',
    'TSLA': 'Consumer Discretionary', 'AMZN': 'Consumer Discretionary',
}


def fetch_sample_data_fast():
    print('正在获取样本数据（批量下载，复权）...')
    start_date, end_date = config.get_date_range()

    raw = yf.download(
        SAMPLE_TICKERS,
        start=start_date,
        end=end_date,
        progress=False,
        interval='1d',
        auto_adjust=True,
        threads=True,
        group_by='ticker',
    )

    price_data = {}
    dollar_data = {}
    for ticker in SAMPLE_TICKERS:
        try:
            close, volume = data_fetcher._extract_ohlcv_pair(raw, ticker)
            if close is None or close.empty:
                continue
            weekly = data_fetcher._sanitize_weekly_series(close.resample('W').last())
            if len(weekly) > 52:
                price_data[ticker] = weekly
                dollar_data[ticker] = data_fetcher._weekly_dollar_volume(close, volume)
        except Exception:
            pass

    if not price_data:
        return pd.DataFrame(), [], pd.DataFrame()

    prices = pd.DataFrame(price_data).sort_index()
    dollar_vol = pd.DataFrame(dollar_data).reindex(index=prices.index).fillna(0.0)
    dollar_vol = dollar_vol.reindex(columns=prices.columns).fillna(0.0)
    print(f'成功获取 {len(prices.columns)} 只股票，{len(prices)} 周（含成交额）')
    return prices, list(prices.columns), dollar_vol


def main_quick_test():
    print('=' * 60)
    print('快速测试版本 - 自动选股与配置算法')
    print(f'TRADING_ENV={config.TRADING_ENV}  RESULTS_DIR={config.RESULTS_DIR}')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    decision_log.reset(source='quick_test')

    weekly_data, tickers, weekly_dollar_volume = fetch_sample_data_fast()
    sectors = {t: TICKER_SECTOR.get(t, 'Unknown') for t in tickers}
    if weekly_data.empty:
        print('错误：无法获取数据')
        return

    # 仅从 TEST 目录读上一期，绝不读 live
    current_holdings, current_weights = portfolio_state.load_previous_portfolio()
    if current_holdings:
        print(f'检测到 TEST 目录上一期持仓 {len(current_holdings)} 只，启用状态依赖')

    candidates = stage1_funnel.stage1_funnel(
        weekly_data, sectors, weekly_dollar_volume, current_holdings=current_holdings
    )
    if candidates.empty:
        return

    selected_df, selected_list, corr_matrix = stage2_correlation.stage2_correlation_pruning(
        weekly_data, candidates, current_holdings=current_holdings
    )

    weights_df, opt_metrics = stage3_optimizer.stage3_optimizer(
        weekly_data, selected_list, current_weights=current_weights
    )
    selected_list = weights_df['Ticker'].tolist()
    selected_df = selected_df[selected_df['Ticker'].isin(selected_list)]
    selected_df = selected_df.set_index('Ticker').loc[selected_list].reset_index()
    weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
    weights_dict, deadband_meta = portfolio_state.apply_deadband(weights_dict, current_weights)
    if deadband_meta.get('locked'):
        print(
            f"Deadband：锁定 {len(deadband_meta['locked'])} 只，"
            f"换手 {deadband_meta['pre_turnover']:.3f} → {deadband_meta['post_turnover']:.3f}"
        )
        selected_list = [t for t in weights_dict if weights_dict[t] > 0]
        selected_df = selected_df[selected_df['Ticker'].isin(selected_list)]
        selected_df = selected_df.set_index('Ticker').loc[
            [t for t in selected_list if t in set(selected_df['Ticker'])]
        ].reset_index()
        weights_df = pd.DataFrame(
            [{'Ticker': t, 'Weight': round(weights_dict[t], 6)} for t in selected_list]
        )
        opt_metrics = {**(opt_metrics or {}), 'deadband': deadband_meta}

    backtest_result = stage4_backtest.stage4_backtest(
        weekly_data, sectors, selected_list, weights_dict, weekly_dollar_volume
    )

    output_dict = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick_test',
        'selected_tickers': selected_list,
        'weights': weights_dict,
        'sectors': dict(zip(selected_df['Ticker'], selected_df['Sector'])),
        'momentum': dict(zip(selected_df['Ticker'], selected_df['Momentum'])),
        'optimization': opt_metrics,
    }
    with open(os.path.join(config.RESULTS_DIR, 'selection_results.json'), 'w') as f:
        json.dump(output_dict, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, 'quick_test_results.json'), 'w') as f:
        json.dump(output_dict, f, indent=2)

    weights_df.to_csv(os.path.join(config.RESULTS_DIR, 'optimal_weights.csv'), index=False)
    corr_matrix.to_csv(os.path.join(config.RESULTS_DIR, 'correlation_matrix.csv'))

    if backtest_result:
        rb = backtest_result.get('rolling_backtest')
        if rb is not None and len(rb) > 0:
            rb.to_csv(os.path.join(config.RESULTS_DIR, 'backtest_results.csv'), index=False)
        with open(os.path.join(config.RESULTS_DIR, 'portfolio_metrics.json'), 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'in_sample_metrics': backtest_result.get('full_metrics'),
                'walk_forward_metrics': backtest_result.get('walk_forward_metrics'),
                'walk_forward_metrics_stitched': backtest_result.get('walk_forward_metrics_stitched'),
                'metrics_2022': backtest_result.get('metrics_2022'),
            }, f, indent=2, default=str)

    print('\n【最终选股推荐】')
    for i, ticker in enumerate(selected_list, 1):
        print(
            f"  {i}. {ticker:6s}  权重：{weights_dict[ticker]*100:6.2f}%  "
            f"行业：{sectors.get(ticker, 'Unknown')}"
        )

    try:
        decision_log.save(extra_meta={'mode': 'quick_test'})
    except Exception as e:
        print(f'决策日志：{e}')

    try:
        import final_report
        final_report.print_report()
        final_report.save_report_to_markdown()
    except Exception as e:
        print(f'报告生成：{e}')


if __name__ == '__main__':
    try:
        main_quick_test()
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
