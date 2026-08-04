import pandas as pd
import numpy as np
import config


def _normalize_weights(weights_df):
    s = weights_df['Weight'].sum()
    if s > 1e-12:
        weights_df = weights_df.copy()
        weights_df['Weight'] = weights_df['Weight'] / s
    return weights_df


def _sanitize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """截断极端周收益并清理 inf/nan，避免协方差收缩时溢出警告。"""
    r = returns.replace([np.inf, -np.inf], np.nan)
    cap = config.MAX_WEEKLY_RETURN
    r = r.clip(lower=-cap, upper=cap)
    return r.dropna(how='all')


def finalize_portfolio_weights(cleaned_weights: dict, stage2_tickers):
    """
    保留 Stage2 全部标的。优化器给出的极低权重记录到 dropped_low_weight，
    并将这些标的抬升至 MIN_WEIGHT 后重新归一化，避免最终名单从 10 只变成 9 只。
    """
    stage2_tickers = list(stage2_tickers)
    w = {t: float(cleaned_weights.get(t, 0.0)) for t in stage2_tickers}
    dropped_low_weight = {t: v for t, v in w.items() if v < config.MIN_WEIGHT}

    for t in dropped_low_weight:
        w[t] = config.MIN_WEIGHT

    total = sum(w.values())
    if total > 1e-12:
        w = {t: v / total for t, v in w.items()}

    # 归一化后极小权重可能再次低于 MIN_WEIGHT；对 Stage2 全部标的保留，不再二次剔除
    weights_df = pd.DataFrame(
        [{'Ticker': t, 'Weight': round(w[t], 6)} for t in stage2_tickers]
    )
    weights_df = weights_df[weights_df['Weight'] > 0]
    weights_df = weights_df.sort_values('Weight', ascending=False).reset_index(drop=True)

    meta = {
        'dropped_low_weight': {t: round(v, 8) for t, v in dropped_low_weight.items()},
        'n_stage2': len(stage2_tickers),
        'n_final': len(weights_df),
        'weight_floor_applied': len(dropped_low_weight) > 0,
    }
    return weights_df, meta


def stage3_optimizer_simple(selected_tickers, reason='equal_weight_fallback'):
    print(f'使用等权重分配（原因：{reason}）...')
    n = len(selected_tickers)
    if n == 0:
        return pd.DataFrame(columns=['Ticker', 'Weight']), {'method': reason}
    w = 1.0 / n
    weights_df = pd.DataFrame({'Ticker': selected_tickers, 'Weight': [w] * n})
    return weights_df, {'method': reason, 'ret': np.nan, 'vol': np.nan, 'sharpe': np.nan}


def _build_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf 收缩协方差；失败时对角加载。"""
    clean = _sanitize_returns(returns).dropna(axis=1, how='all')
    if clean.shape[1] < 2:
        clean = _sanitize_returns(returns)
    try:
        from pypfopt.risk_models import CovarianceShrinkage
        with np.errstate(over='ignore', invalid='ignore'):
            cov = CovarianceShrinkage(clean).ledoit_wolf()
    except Exception:
        with np.errstate(over='ignore', invalid='ignore'):
            cov = clean.cov()
    cov = cov * 52
    arr = np.nan_to_num(cov.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr + np.eye(len(arr)) * 1e-5
    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)


def _weights_to_df(weights: dict, selected_tickers) -> pd.DataFrame:
    weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
    weights_df['Weight'] = weights_df['Weight'].round(6)
    weights_df = weights_df[weights_df['Weight'] >= config.MIN_WEIGHT]
    if weights_df.empty:
        return weights_df
    weights_df = _normalize_weights(weights_df)
    order = [t for t in selected_tickers if t in set(weights_df['Ticker'])]
    extra = [t for t in weights_df['Ticker'] if t not in order]
    weights_df = weights_df.set_index('Ticker').loc[order + extra].reset_index()
    return weights_df.sort_values('Weight', ascending=False).reset_index(drop=True)


def _apply_turnover_shrinkage(
    ideal_weights: dict,
    selected_tickers,
    mu: pd.Series,
    cov: pd.DataFrame,
    current_weights: dict = None,
    max_weight: float = None,
    quiet: bool = False,
):
    """
    二次收缩：在理想权重附近，用 L1 项把权重拉向旧仓，降低无谓换手。

    minimize ||w - w_ideal||² + γ Σ|w - w_old|
    s.t. Σw=1, 0≤w≤max_w

    若优化后换手反而上升，则拒绝并保留理想权重。
    """
    from portfolio_state import align_current_weights

    if not config.ENABLE_STATE_DEPENDENCY or not current_weights:
        return ideal_weights, {'turnover_penalty_applied': False}

    tickers = list(selected_tickers)
    w_old = align_current_weights(tickers, current_weights)
    if sum(w_old.values()) <= 1e-12:
        return ideal_weights, {'turnover_penalty_applied': False}

    gamma = float(getattr(config, 'TURNOVER_SHRINK_GAMMA', 0.05))
    upper = float(max_weight if max_weight is not None else config.MAX_WEIGHT)

    w_ideal = np.array([float(ideal_weights.get(t, 0.0)) for t in tickers], dtype=float)
    w_ideal = np.clip(w_ideal, 0.0, upper)
    if w_ideal.sum() <= 1e-12:
        w_ideal = np.ones(len(tickers)) / len(tickers)
    else:
        w_ideal = w_ideal / w_ideal.sum()
    w_old_vec = np.array([w_old[t] for t in tickers], dtype=float)

    pre_to = float(np.sum(np.abs(w_ideal - w_old_vec)))

    def objective(w):
        tracking = float(np.sum((w - w_ideal) ** 2))
        turnover = float(np.sum(np.abs(w - w_old_vec)))
        return tracking + gamma * turnover

    try:
        from scipy.optimize import minimize

        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, upper)] * len(tickers)
        res = minimize(
            objective,
            w_ideal.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False},
        )
        if not res.success:
            raise RuntimeError(res.message)

        w_new = np.clip(res.x, 0.0, upper)
        s = w_new.sum()
        if s <= 1e-12:
            raise RuntimeError('degenerate weights')
        w_new = w_new / s
        post_to = float(np.sum(np.abs(w_new - w_old_vec)))

        if post_to > pre_to + 1e-6:
            if not quiet:
                print(
                    f'  换手收缩未改善（{pre_to:.3f} → {post_to:.3f}），保留理想权重'
                )
            return ideal_weights, {
                'turnover_penalty_applied': False,
                'turnover_before': round(pre_to, 6),
                'turnover_after': round(pre_to, 6),
                'rejected_increase': True,
            }

        out = {t: float(w_new[i]) for i, t in enumerate(tickers)}
        meta = {
            'turnover_penalty_applied': True,
            'turnover_lambda': float(config.TURNOVER_LAMBDA),
            'turnover_shrink_gamma': gamma,
            'turnover_before': round(pre_to, 6),
            'turnover_after': round(post_to, 6),
        }
        if not quiet:
            print(
                f'  换手收缩：turnover {pre_to:.3f} → {post_to:.3f} '
                f'(γ={gamma})'
            )
        try:
            import decision_log
            decision_log.add(
                ticker='PORTFOLIO',
                action='TURNOVER_SHRINK_APPLIED',
                reason=f'Turnover {pre_to:.3f} → {post_to:.3f} with γ={gamma}.',
                turnover_before=pre_to,
                turnover_after=post_to,
            )
        except Exception:
            pass
        return out, meta
    except Exception as e:
        if not quiet:
            print(f'  换手收缩失败，保留理想权重：{e}')
        return ideal_weights, {'turnover_penalty_applied': False, 'error': str(e)}


def stage3_optimizer(weekly_data, selected_tickers, quiet=False, current_weights=None):
    """
    权重优化链：Max Sharpe（15% 上限）→ 放宽上限 → 无上限 → Min Volatility → 等权。
    成功后若提供 current_weights，再做换手惩罚二次收缩。
    """

    def log(msg):
        if not quiet:
            print(msg)

    if not quiet:
        print('\n=== Stage 3: 权重优化 ===')
        print(f'输入：{len(selected_tickers)} 只选定股票')

    if not selected_tickers:
        return stage3_optimizer_simple([])

    subset = weekly_data[selected_tickers].copy()
    returns = _sanitize_returns(subset.pct_change().dropna(how='all'))

    if len(returns) < 10:
        log('警告：收益样本不足，使用等权重')
        return stage3_optimizer_simple(selected_tickers, 'insufficient_returns')

    rf = config.RISK_FREE_RATE
    n = len(selected_tickers)
    max_w = config.MAX_WEIGHT
    relaxed_w = min(0.25, max(1.0 / n + 0.02, max_w))

    if config.USE_RISK_PARITY_WEIGHTS:
        try:
            from algorithm_v2_improvements import risk_parity_weights
            wdict = risk_parity_weights(returns, selected_tickers)
            shrunk, to_meta = _apply_turnover_shrinkage(
                wdict, selected_tickers, returns.mean() * 52,
                _build_covariance(returns), current_weights, max_w, quiet,
            )
            weights_df, finalize_meta = finalize_portfolio_weights(shrunk, selected_tickers)
            if not weights_df.empty:
                return weights_df, {'method': 'risk_parity', **finalize_meta, **to_meta}
        except Exception as e:
            log(f'风险平价失败 ({e})，改用 Sharpe / MinVol')

    mu = returns.mean() * 52
    cov = _build_covariance(returns)

    if not quiet:
        log('预期年化收益率 (%):')
        for ticker in selected_tickers:
            log(f'  {ticker}: {mu[ticker] * 100:.2f}%')

    try:
        from pypfopt.efficient_frontier import EfficientFrontier
    except ImportError as e:
        log(
            f'PyPortfolioOpt 导入失败: {e}；请运行 pip install PyPortfolioOpt ' \
            '或检查当前 Python 环境是否已激活。'
        )
        return stage3_optimizer_simple(selected_tickers, 'import_error')

    attempts = [
        ('max_sharpe', max_w, 'max_sharpe'),
        ('max_sharpe', relaxed_w, 'max_sharpe_relaxed'),
        ('max_sharpe', 1.0, 'max_sharpe_uncapped'),
        ('min_volatility', max_w, 'min_volatility'),
        ('min_volatility', 1.0, 'min_volatility_uncapped'),
    ]

    last_error = None
    for method, upper_bound, label in attempts:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, upper_bound))
            if method == 'max_sharpe':
                raw = ef.max_sharpe(risk_free_rate=rf)
            else:
                raw = ef.min_volatility()

            cleaned = ef.clean_weights()
            shrunk, to_meta = _apply_turnover_shrinkage(
                cleaned, selected_tickers, mu, cov, current_weights, upper_bound, quiet,
            )
            weights_df, finalize_meta = finalize_portfolio_weights(shrunk, selected_tickers)
            if weights_df.empty:
                raise ValueError('优化结果权重全为零')

            ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rf)
            log(f'\n✓ 优化成功：{label}（单票上限 {upper_bound * 100:.0f}%）')
            if finalize_meta.get('dropped_low_weight') and not quiet:
                log(f'  极低权重回填: {finalize_meta["dropped_low_weight"]}')
            if not quiet:
                log(f'  预期收益率: {float(ret) * 100:.2f}%')
                log(f'  波动率: {float(vol) * 100:.2f}%')
                log(f'  Sharpe: {float(sharpe):.3f}')
                log(str(weights_df))

            return weights_df, {
                'method': label,
                'ret': ret,
                'vol': vol,
                'sharpe': sharpe,
                'max_weight_bound': upper_bound,
                **finalize_meta,
                **to_meta,
            }
        except Exception as e:
            last_error = e
            log(f'  ✗ {label} (上限={upper_bound:.2f}) 失败: {type(e).__name__}: {e}')
            if not quiet:
                import traceback
                traceback.print_exc()

    log(f'\n所有优化策略均失败，最后错误: {last_error}')
    return stage3_optimizer_simple(selected_tickers, 'all_optimizers_failed')


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
        weights_df, metrics = stage3_optimizer(weekly_data, selected_list)
        print('\n✓ Stage 3 完成', metrics)
