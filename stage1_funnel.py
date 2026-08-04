import pandas as pd
import numpy as np
import config


def _momentum_weeks():
    """12-1 月动量 → 52 周前到 4 周前（相对序列末尾）。"""
    lookback = config.MOMENTUM_WINDOW * config.WEEKS_PER_MONTH
    exclude = config.MOMENTUM_EXCLUDE * config.WEEKS_PER_MONTH
    return lookback, exclude


def calculate_momentum(price_series):
    """
    12-1 Momentum：(P_{T-4w} - P_{T-52w}) / P_{T-52w}

    在去 NaN、按时间排序的周线收盘价上，用**固定行偏移**取价：
    - iloc[-52]：约 12 个月前
    - iloc[-4] ：约 1 个月前（排除最近 1 个月）

    并校验日历跨度与窗口内单周跳变，过滤拆股/坏点。
    """
    s = price_series.dropna().sort_index()
    lookback_w, exclude_w = _momentum_weeks()

    if exclude_w < 1 or lookback_w <= exclude_w:
        return np.nan
    if len(s) < lookback_w + 1:
        return np.nan

    p_start = float(s.iloc[-lookback_w])
    p_end = float(s.iloc[-exclude_w])

    if p_start <= 0 or p_end <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
        return np.nan

    t_start = s.index[-lookback_w]
    t_end = s.index[-exclude_w]
    span_days = (t_end - t_start).days
    # 约 11 个月（52w 与 4w 之间）
    if span_days < 240 or span_days > 400:
        return np.nan

    # 动量窗口内的价格（不含最近 exclude 周）
    window = s.iloc[-lookback_w: -exclude_w] if exclude_w > 0 else s.iloc[-lookback_w:]
    if len(window) < 8:
        return np.nan

    weekly_rets = window.pct_change().dropna()
    if len(weekly_rets) > 0 and weekly_rets.abs().max() > 0.50:
        return np.nan

    momentum = (p_end - p_start) / p_start * 100.0

    if not np.isfinite(momentum) or abs(momentum) > config.MOMENTUM_MAX_PCT:
        return np.nan
    return momentum


def filter_by_liquidity(weekly_data, weekly_dollar_volume):
    """
    宽松流动性安全网：
    - 过去 LIQUIDITY_LOOKBACK_WEEKS 周均成交额 >= MIN_AVG_WEEKLY_DOLLAR_VOLUME
    - 最近 2 周成交额不全为 0（过滤停牌）
    无成交额数据时原样返回（向后兼容）。
    """
    if weekly_dollar_volume is None or getattr(weekly_dollar_volume, 'empty', True):
        return weekly_data, []

    lookback = config.LIQUIDITY_LOOKBACK_WEEKS
    min_adv = config.MIN_AVG_WEEKLY_DOLLAR_VOLUME
    common = [t for t in weekly_data.columns if t in weekly_dollar_volume.columns]
    if not common:
        return weekly_data, []

    dollar = weekly_dollar_volume[common].reindex(index=weekly_data.index).fillna(0.0)
    recent = dollar.tail(lookback)
    if recent.empty:
        return weekly_data, []

    avg_dv = recent.mean()
    last2 = dollar.tail(2)
    halted = (last2.sum() <= 0) if len(last2) > 0 else pd.Series(False, index=common)

    keep = []
    dropped = []
    for t in weekly_data.columns:
        if t not in avg_dv.index:
            dropped.append((t, 'no_volume_data'))
            continue
        if bool(halted.get(t, False)):
            dropped.append((t, 'halted_or_zero_volume'))
            continue
        if float(avg_dv[t]) < min_adv:
            dropped.append((t, f'avg_weekly_dv={float(avg_dv[t]):.0f}'))
            continue
        keep.append(t)

    return weekly_data[keep], dropped


def _ensure_funnel_sector_coverage(momentum_df, sector_dict):
    sort_col = 'RankScore' if 'RankScore' in momentum_df.columns else 'Momentum'
    candidates = momentum_df.sort_values(sort_col, ascending=False).head(config.FUNNEL_SIZE).copy()
    candidates['Sector'] = candidates['Ticker'].map(sector_dict).fillna('Unknown')

    present = set(candidates['Sector']) - {'Unknown'}
    if len(present) >= config.MIN_SECTORS:
        return candidates

    selected_tickers = set(candidates['Ticker'])
    remainder = momentum_df[~momentum_df['Ticker'].isin(selected_tickers)].copy()
    remainder['Sector'] = remainder['Ticker'].map(sector_dict).fillna('Unknown')

    all_sectors = set(sector_dict.values()) - {'Unknown'}
    missing = all_sectors - present

    extras = []
    for sector in missing:
        pool = remainder[remainder['Sector'] == sector]
        if not pool.empty:
            extras.append(pool.iloc[0])

    if not extras:
        return candidates

    extra_df = pd.DataFrame(extras)
    combined = pd.concat([candidates, extra_df], ignore_index=True)
    combined = combined.drop_duplicates(subset='Ticker', keep='first')
    combined = combined.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return combined.head(config.FUNNEL_SIZE)


def stage1_funnel(weekly_data, sector_dict, weekly_dollar_volume=None, current_holdings=None):
    print('\n=== Stage 1: 漏斗式筛选 ===')
    print(f'输入：{len(weekly_data.columns)} 只股票')

    filtered, dropped = filter_by_liquidity(weekly_data, weekly_dollar_volume)
    if dropped:
        print(
            f'流动性过滤：剔除 {len(dropped)} 只'
            f'（门槛：近 {config.LIQUIDITY_LOOKBACK_WEEKS} 周均成交额 ≥ '
            f'${config.MIN_AVG_WEEKLY_DOLLAR_VOLUME:,.0f}）'
        )
        if len(dropped) <= 15:
            for t, reason in dropped:
                print(f'  - {t}: {reason}')
        else:
            for t, reason in dropped[:10]:
                print(f'  - {t}: {reason}')
            print(f'  ... 另有 {len(dropped) - 10} 只')
    elif weekly_dollar_volume is not None and not getattr(weekly_dollar_volume, 'empty', True):
        print(
            f'流动性过滤：全部通过'
            f'（近 {config.LIQUIDITY_LOOKBACK_WEEKS} 周均成交额 ≥ '
            f'${config.MIN_AVG_WEEKLY_DOLLAR_VOLUME:,.0f}）'
        )
    print(f'流动性过滤后剩余：{len(filtered.columns)} 只')

    momentum_scores = {}
    for ticker in filtered.columns:
        m = calculate_momentum(filtered[ticker])
        if not np.isnan(m):
            momentum_scores[ticker] = m

    print(f'有效 Momentum 数据的股票数：{len(momentum_scores)}')
    if not momentum_scores:
        print('警告：没有有效的 Momentum 数据')
        return pd.DataFrame()

    holdings_all = set(current_holdings or [])
    holdings = {t for t in holdings_all if t in momentum_scores}
    use_loyalty = (
        config.ENABLE_STATE_DEPENDENCY
        and holdings
        and config.LOYALTY_MULTIPLIER > 1.0
    )
    mult = config.LOYALTY_MULTIPLIER if use_loyalty else 1.0

    rows = []
    boosted = []
    for t, m in momentum_scores.items():
        score = m * mult if t in holdings else m
        if t in holdings and mult != 1.0:
            boosted.append(t)
        rows.append({'Ticker': t, 'Momentum': m, 'RankScore': score})

    momentum_df = (
        pd.DataFrame(rows)
        .sort_values('RankScore', ascending=False)
        .reset_index(drop=True)
    )

    if config.ENABLE_STATE_DEPENDENCY and holdings_all:
        outside = len(holdings_all - holdings)
        print(
            f'动量缓冲带：对 {len(boosted)} 只现有持仓施加 ×{mult:.2f} 排序加成'
            f'（持仓输入 {len(holdings_all)} 只'
            + (f'，其中 {outside} 只不在当前股票池' if outside else '')
            + '）'
        )

    candidates = _ensure_funnel_sector_coverage(momentum_df, sector_dict)
    candidates['Sector'] = candidates['Ticker'].map(sector_dict).fillna('Unknown')

    print(f'选定前 {len(candidates)} 只候选股票')
    print('\nTop 10 Momentum:')
    print(candidates.head(10)[['Ticker', 'Momentum', 'Sector']])

    sector_counts = candidates['Sector'].value_counts()
    print(f'\n行业覆盖 ({len(sector_counts)} 个不同行业):')
    print(sector_counts)

    return candidates


if __name__ == '__main__':
    import data_fetcher

    tickers, weekly_data, sectors, dollar_vol = data_fetcher.prepare_data()
    if weekly_data is None or weekly_data.empty:
        print('数据获取失败')
    else:
        candidates = stage1_funnel(weekly_data, sectors, dollar_vol)
        if 'INTC' in weekly_data.columns:
            m = calculate_momentum(weekly_data['INTC'])
            print(f'\nINTC 动量校验: {m}')
        print(f'\n✓ Stage 1 完成，选定 {len(candidates)} 只候选股票')
