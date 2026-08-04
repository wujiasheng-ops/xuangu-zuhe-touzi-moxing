import pandas as pd
import numpy as np
import config


def calculate_correlation_matrix(weekly_data, tickers):
    subset = weekly_data[tickers].copy()
    returns = subset.pct_change().dropna()
    if len(returns) == 0:
        print('警告：无法计算收益率')
        return pd.DataFrame()
    return returns.corr()


def _sector_counts(selected, sector_map):
    counts = {}
    for t in selected:
        s = sector_map.get(t, 'Unknown')
        counts[s] = counts.get(s, 0) + 1
    return counts


def _remove_high_corr(corr_matrix, anchor, remaining, protected=None):
    """剔除与锚点高相关的股票；protected 内标的不被剔除（除非就是锚点自身）。"""
    protected = set(protected or [])
    to_remove = {anchor}
    if anchor in corr_matrix.index:
        corrs = corr_matrix[anchor]
        high = corrs[corrs.abs() > config.CORRELATION_THRESHOLD].index.tolist()
        for t in high:
            if t in protected and t != anchor:
                continue
            to_remove.add(t)
    return remaining - to_remove, to_remove


def _eligible_by_sector(remaining, selected, sector_map):
    """行业硬性上限：单行业最多 MAX_STOCKS_PER_SECTOR 只。"""
    counts = _sector_counts(selected, sector_map)
    cap = config.MAX_STOCKS_PER_SECTOR
    return [
        t for t in remaining
        if counts.get(sector_map.get(t, 'Unknown'), 0) < cap
    ]


def _pick_sector_fill(selected, remaining, sector_map, corr_matrix):
    counts = _sector_counts(selected, sector_map)
    cap = config.MAX_STOCKS_PER_SECTOR
    selected_sectors = {sector_map.get(t) for t in selected} - {'Unknown'}
    uncovered = {s for s in set(sector_map.values()) if s != 'Unknown'} - selected_sectors
    if not uncovered:
        return None

    best = None
    best_score = np.inf
    for ticker in remaining:
        sector = sector_map.get(ticker, 'Unknown')
        if sector not in uncovered:
            continue
        if counts.get(sector, 0) >= cap:
            continue
        if ticker not in corr_matrix.index:
            continue
        score = corr_matrix.loc[ticker, selected].abs().sum()
        if score < best_score:
            best_score = score
            best = ticker
    return best


def _seed_protected_holdings(
    candidates_df,
    corr_matrix,
    sector_map,
    momentum_map,
    current_holdings,
):
    """
    将现有持仓预置入组合（持仓优先权）：
    - 仅考虑仍在候选池中的旧仓
    - 遵守行业上限与相关性阈值
    - 若与已预置旧仓高相关，保留动量更高者
    """
    holdings = [t for t in (current_holdings or []) if t in momentum_map]
    if not holdings or not config.ENABLE_STATE_DEPENDENCY:
        return [], set()

    holdings = sorted(holdings, key=lambda t: momentum_map.get(t, -np.inf), reverse=True)
    selected = []
    protected = set()

    for t in holdings:
        if len(selected) >= config.TARGET_SIZE:
            break
        sec = sector_map.get(t, 'Unknown')
        if _sector_counts(selected, sector_map).get(sec, 0) >= config.MAX_STOCKS_PER_SECTOR:
            print(f'  持仓保护跳过 {t}：行业 {sec} 已满')
            try:
                import decision_log
                decision_log.add(
                    ticker=t,
                    action='SKIPPED_SEED_SECTOR_CAP',
                    reason=f'Sector {sec} already at MAX_STOCKS_PER_SECTOR.',
                )
            except Exception:
                pass
            continue

        conflict = None
        if t in corr_matrix.index and selected:
            for s in selected:
                if s not in corr_matrix.columns:
                    continue
                if abs(float(corr_matrix.loc[t, s])) > config.CORRELATION_THRESHOLD:
                    conflict = s
                    break

        if conflict is not None:
            # 两只旧仓互斥：保留动量更高者
            if momentum_map.get(t, -np.inf) > momentum_map.get(conflict, -np.inf):
                print(f'  持仓保护替换：{conflict} → {t}（旧仓内动量更高）')
                selected.remove(conflict)
                protected.discard(conflict)
                selected.append(t)
                protected.add(t)
                try:
                    import decision_log
                    decision_log.add(
                        ticker=conflict,
                        action='REPLACED_WITHIN_HOLDINGS',
                        reason=f'Replaced by fellow holding {t} with higher momentum.',
                        competitor_ticker=t,
                        momentum_diff=float(momentum_map[t] - momentum_map[conflict]),
                    )
                except Exception:
                    pass
            else:
                print(f'  持仓保护跳过 {t}：与已保护持仓 {conflict} 高相关')
                try:
                    import decision_log
                    decision_log.add(
                        ticker=t,
                        action='SKIPPED_SEED_HIGH_CORR',
                        reason=f'High corr with protected holding {conflict}.',
                        competitor_ticker=conflict,
                    )
                except Exception:
                    pass
            continue

        selected.append(t)
        protected.add(t)
        print(
            f'  持仓保护预置：{t} '
            f'(Momentum: {momentum_map[t]:.2f}%, Sector: {sec})'
        )
        try:
            import decision_log
            decision_log.add(
                ticker=t,
                action='SEEDED_PROTECTED_HOLDING',
                reason='Existing holding pre-seeded under holdings priority.',
            )
        except Exception:
            pass

    return selected, protected


def _can_replace_protected(new_ticker, protected_ticker, momentum_map):
    """新票动量需显著高于被保护旧仓，才允许挤占。"""
    margin = config.HOLDINGS_REPLACE_MARGIN
    new_m = momentum_map.get(new_ticker, -np.inf)
    old_m = momentum_map.get(protected_ticker, -np.inf)
    return (new_m - old_m) >= margin


def stage2_correlation_pruning(weekly_data, candidates_df, current_holdings=None):
    print('\n=== Stage 2: 相关性剪枝 ===')
    print(f'输入：{len(candidates_df)} 只候选股票')
    print(f'行业硬性上限：每行业最多 {config.MAX_STOCKS_PER_SECTOR} 只')

    candidate_tickers = candidates_df['Ticker'].tolist()
    corr_matrix = calculate_correlation_matrix(weekly_data, candidate_tickers)

    if corr_matrix.empty:
        print('无法计算相关性矩阵，退回动量 Top N（仍遵守行业上限）')
        fallback = []
        counts = {}
        for _, row in candidates_df.iterrows():
            t, sec = row['Ticker'], row['Sector']
            if counts.get(sec, 0) >= config.MAX_STOCKS_PER_SECTOR:
                continue
            fallback.append(t)
            counts[sec] = counts.get(sec, 0) + 1
            if len(fallback) >= config.TARGET_SIZE:
                break
        fb_df = candidates_df[candidates_df['Ticker'].isin(fallback)].copy()
        fl = fallback
        sub = pd.DataFrame(np.eye(len(fl)), index=fl, columns=fl)
        return fb_df.set_index('Ticker').loc[fl].reset_index(), fl, sub

    selected = []
    remaining = set(candidate_tickers)
    sector_map = dict(zip(candidates_df['Ticker'], candidates_df['Sector']))
    momentum_map = dict(zip(candidates_df['Ticker'], candidates_df['Momentum']))
    protected = set()

    if current_holdings and config.ENABLE_STATE_DEPENDENCY:
        print(
            f'持仓优先权：输入 {len(current_holdings)} 只现有持仓，'
            f'替换门槛 ΔMomentum ≥ {config.HOLDINGS_REPLACE_MARGIN:.1f}pp'
        )
        selected, protected = _seed_protected_holdings(
            candidates_df, corr_matrix, sector_map, momentum_map, current_holdings
        )
        remaining -= set(selected)
        # 预置后，清除与保护持仓高相关的非保护候选（保护持仓彼此已处理）
        for anchor in list(selected):
            remaining, _ = _remove_high_corr(corr_matrix, anchor, remaining, protected=protected)

    while len(selected) < config.TARGET_SIZE and remaining:
        eligible = _eligible_by_sector(remaining, selected, sector_map)
        if not eligible:
            print('  警告：剩余候选均违反行业上限，停止选股')
            break

        best = max(eligible, key=lambda t: momentum_map.get(t, -np.inf))

        # 若新票与某保护持仓高相关，需动量优势足够才允许替换
        blocking = None
        if protected and best in corr_matrix.index:
            for p in list(protected):
                if p not in corr_matrix.columns or p not in selected:
                    continue
                if abs(float(corr_matrix.loc[best, p])) > config.CORRELATION_THRESHOLD:
                    diff = float(momentum_map[best] - momentum_map[p])
                    if _can_replace_protected(best, p, momentum_map):
                        print(
                            f'  动量黑马替换保护持仓：{p} → {best} '
                            f'(Δ={diff:.1f}pp ≥ '
                            f'{config.HOLDINGS_REPLACE_MARGIN:.1f})'
                        )
                        try:
                            import decision_log
                            decision_log.add(
                                ticker=p,
                                action='REPLACED_BY_BLACK_HORSE',
                                reason=(
                                    f'{best} advantage ({diff:.1f}pp) >= '
                                    f'HOLDINGS_REPLACE_MARGIN ({config.HOLDINGS_REPLACE_MARGIN:.1f}).'
                                ),
                                competitor_ticker=best,
                                momentum_diff=diff,
                            )
                        except Exception:
                            pass
                        selected.remove(p)
                        protected.discard(p)
                        remaining.add(p)
                    else:
                        blocking = p
                        try:
                            import decision_log
                            decision_log.add(
                                ticker=p,
                                action='REJECTED_REPLACEMENT',
                                reason=(
                                    f'{best} advantage ({diff:.1f}pp) < '
                                    f'HOLDINGS_REPLACE_MARGIN ({config.HOLDINGS_REPLACE_MARGIN:.1f}). '
                                    f'{p} protected.'
                                ),
                                competitor_ticker=best,
                                momentum_diff=diff,
                                target_weight_raw=0.0,
                            )
                        except Exception:
                            pass
                    break

        if blocking is not None:
            print(
                f'  持仓保护拦截：跳过 {best}（与 {blocking} 高相关且动量优势不足）'
            )
            remaining.discard(best)
            continue

        selected.append(best)
        sec = sector_map.get(best, 'Unknown')
        print(
            f'第 {len(selected)} 选：{best} '
            f'(Momentum: {momentum_map[best]:.2f}%, Sector: {sec}, '
            f'行业内已选: {_sector_counts(selected, sector_map).get(sec, 0)}/{config.MAX_STOCKS_PER_SECTOR})'
        )

        remaining, removed = _remove_high_corr(corr_matrix, best, remaining, protected=protected)
        n_removed = len(removed - {best})
        print(f'  移除 {n_removed} 只高相关性股票，剩余 {len(remaining)} 只')

        if len(selected) < config.TARGET_SIZE:
            sectors_now = {sector_map.get(t) for t in selected} - {'Unknown'}
            print(f'  已覆盖行业数：{len(sectors_now)}')

            if len(selected) >= config.TARGET_SIZE - 2 and len(sectors_now) < config.MIN_SECTORS:
                fill = _pick_sector_fill(selected, remaining, sector_map, corr_matrix)
                if fill and len(selected) < config.TARGET_SIZE:
                    if _sector_counts(selected, sector_map).get(sector_map.get(fill), 0) < config.MAX_STOCKS_PER_SECTOR:
                        print(f'  从未覆盖行业选择：{fill}')
                        selected.append(fill)
                        remaining, _ = _remove_high_corr(
                            corr_matrix, fill, remaining, protected=protected
                        )

    selected = selected[:config.TARGET_SIZE]

    sc = _sector_counts(selected, sector_map)
    print(f'\n最终行业分布：{sc}')
    over = {k: v for k, v in sc.items() if v > config.MAX_STOCKS_PER_SECTOR}
    if over:
        print(f'⚠️ 行业上限违规：{over}')

    print(f'\n✓ Stage 2 完成，选定 {len(selected)} 只股票')
    print(f'最终选股：{selected}')

    selected_df = candidates_df[candidates_df['Ticker'].isin(selected)].copy()
    selected_df = selected_df.set_index('Ticker').loc[selected].reset_index()
    sub_corr = corr_matrix.loc[selected, selected]

    return selected_df, selected, sub_corr


if __name__ == '__main__':
    import data_fetcher
    import stage1_funnel

    tickers, weekly_data, sectors, dollar_vol = data_fetcher.prepare_data()
    if weekly_data is None or weekly_data.empty:
        print('数据获取失败')
    else:
        candidates = stage1_funnel.stage1_funnel(weekly_data, sectors, dollar_vol)
        selected_df, selected_list, corr = stage2_correlation_pruning(weekly_data, candidates)
        print('\n相关性矩阵：')
        print(corr)
