"""
基准对照：强制对齐 SPY，维护 equity_curve.csv。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config


def fetch_spy_weekly(start_date=None, end_date=None) -> pd.Series:
    """下载/缓存 SPY 周线收盘价。"""
    import yfinance as yf
    import data_fetcher

    if start_date is None or end_date is None:
        start_date, end_date = config.get_date_range()

    cache_name = (
        f'spy_weekly_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.pkl'
    )
    cached = data_fetcher.load_cached_data(cache_name)
    if cached is not None and not getattr(cached, 'empty', True):
        return cached

    raw = yf.download(
        config.BENCHMARK_TICKER,
        start=start_date,
        end=end_date,
        progress=False,
        interval='1d',
        auto_adjust=True,
        threads=False,
    )
    close = data_fetcher._extract_close_series(raw, config.BENCHMARK_TICKER)
    if close is None or close.empty:
        return pd.Series(dtype=float, name=config.BENCHMARK_TICKER)
    weekly = data_fetcher._sanitize_weekly_series(close.resample('W').last())
    weekly.name = config.BENCHMARK_TICKER
    if not weekly.empty:
        data_fetcher.cache_data(weekly, cache_name)
    return weekly


def portfolio_buy_and_hold_return(
    weekly_data: pd.DataFrame,
    weights: Dict[str, float],
    start_ts,
    end_ts=None,
) -> Optional[float]:
    """用上一期权重计算区间买入持有收益。"""
    if not weights or weekly_data is None or weekly_data.empty:
        return None
    tickers = [t for t in weights if t in weekly_data.columns]
    if not tickers:
        return None
    w = pd.Series({t: float(weights[t]) for t in tickers}, dtype=float)
    w = w / w.sum()
    sub = weekly_data[tickers].sort_index()
    if start_ts is not None:
        sub = sub[sub.index >= pd.Timestamp(start_ts)]
    if end_ts is not None:
        sub = sub[sub.index <= pd.Timestamp(end_ts)]
    sub = sub.dropna(how='all')
    if len(sub) < 2:
        return None
    start_px = sub.iloc[0]
    end_px = sub.iloc[-1]
    rets = (end_px / start_px - 1.0).replace([np.inf, -np.inf], np.nan)
    aligned = rets.reindex(w.index).fillna(0.0)
    return float((aligned * w).sum())


def spy_return_between(spy: pd.Series, start_ts, end_ts=None) -> Optional[float]:
    if spy is None or len(spy) < 2:
        return None
    s = spy.dropna().sort_index()
    if start_ts is not None:
        s = s[s.index >= pd.Timestamp(start_ts)]
    if end_ts is not None:
        s = s[s.index <= pd.Timestamp(end_ts)]
    if len(s) < 2:
        # 回退：用全样本最近两周
        s = spy.dropna().sort_index()
        if len(s) < 2:
            return None
        return float(s.iloc[-1] / s.iloc[-2] - 1.0)
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def equity_curve_path() -> str:
    return os.path.join(config.RESULTS_DIR, 'equity_curve.csv')


def load_equity_curve() -> pd.DataFrame:
    path = equity_curve_path()
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_equity_curve_row(row: Dict[str, Any]) -> str:
    path = equity_curve_path()
    df = load_equity_curve()
    new = pd.DataFrame([row])
    df = pd.concat([df, new], ignore_index=True) if not df.empty else new
    df.to_csv(path, index=False)
    return path


def update_benchmark_snapshot(
    weekly_data: pd.DataFrame,
    previous_weights: Dict[str, float],
    previous_timestamp: Optional[str] = None,
    strategy_equity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    计算本期策略 vs SPY，并追加 equity_curve.csv。
    若无上一期时间戳，则用最近约 4 周作为对照窗。
    """
    spy = fetch_spy_weekly()
    now = datetime.now()
    start_ts = previous_timestamp
    if start_ts is None and not spy.empty:
        # 约一个月
        start_ts = spy.index[-min(5, len(spy))]

    strat_ret = portfolio_buy_and_hold_return(
        weekly_data, previous_weights, start_ts, end_ts=None
    )
    spy_ret = spy_return_between(spy, start_ts, end_ts=None)
    spy_px = float(spy.dropna().iloc[-1]) if spy is not None and len(spy.dropna()) else np.nan

    hist = load_equity_curve()
    if strategy_equity is None:
        if not hist.empty and 'strategy_equity' in hist.columns:
            last_eq = float(hist['strategy_equity'].iloc[-1])
            strategy_equity = last_eq * (1.0 + (strat_ret or 0.0))
        else:
            strategy_equity = float(config.PAPER_INITIAL_CASH) * (1.0 + (strat_ret or 0.0))

    alpha = None
    if strat_ret is not None and spy_ret is not None:
        alpha = float(strat_ret - spy_ret)

    period_start = str(start_ts) if start_ts is not None else None
    period_end = now.isoformat()

    snap = {
        'timestamp': now.isoformat(),
        'trading_env': config.TRADING_ENV,
        'period_start': period_start,
        'period_end': period_end,
        'strategy_return': strat_ret,
        'spy_return': spy_ret,
        'alpha': alpha,
        'strategy_equity': strategy_equity,
        'spy_price': spy_px,
        'benchmark': config.BENCHMARK_TICKER,
        'beat_spy': bool(alpha is not None and alpha > 0),
    }

    append_equity_curve_row({
        'timestamp': snap['timestamp'],
        'period_start': period_start,
        'period_end': period_end,
        'strategy_equity': strategy_equity,
        'strategy_return': strat_ret,
        'spy_price': spy_px,
        'spy_return': spy_ret,
        'alpha': alpha,
        'trading_env': config.TRADING_ENV,
    })

    # 便于报告直接读取
    out_path = os.path.join(config.RESULTS_DIR, 'benchmark_latest.json')
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snap, f, indent=2, default=str)

    return snap


def format_period_comparison(snap: Dict[str, Any]) -> str:
    if not snap:
        return ''
    sr = snap.get('strategy_return')
    sp = snap.get('spy_return')
    al = snap.get('alpha')
    eq = snap.get('strategy_equity')
    init = config.PAPER_INITIAL_CASH
    lines = [
        f"【当前调仓周期表现】",
        f"周期：{snap.get('period_start', 'N/A')} → {snap.get('period_end', 'N/A')}",
        f"名义权益（累计）：${eq:,.2f}" if eq is not None else '名义权益：N/A',
        f"策略本期收益率：{sr * 100:+.2f}%" if sr is not None else '策略本期收益率：N/A',
        f"同期 {config.BENCHMARK_TICKER} 收益率：{sp * 100:+.2f}%" if sp is not None else f'同期 {config.BENCHMARK_TICKER}：N/A',
        f"Alpha（超额收益）：{al * 100:+.2f}%  "
        + ('🏆 (跑赢)' if snap.get('beat_spy') else '（跑输/持平）')
        if al is not None else 'Alpha：N/A',
        f"（初始名义本金参考 ${init:,.0f}）",
    ]
    return '\n'.join(lines)
