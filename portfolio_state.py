"""
状态依赖工具：读取上一期持仓、执行层 Deadband、权重对齐。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import config


def _normalize_weight_dict(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {str(t): float(w) for t, w in (weights or {}).items() if float(w) > 0}
    total = sum(clean.values())
    if total <= 1e-12:
        return {}
    return {t: w / total for t, w in clean.items()}


def _candidate_state_dirs():
    """当前 RESULTS_DIR 优先；PROD 时额外兼容旧版扁平 results/。"""
    dirs = [config.RESULTS_DIR]
    root = getattr(config, 'RESULTS_ROOT', None)
    if root and root not in dirs and getattr(config, 'TRADING_ENV', 'PROD') == 'PROD':
        dirs.append(root)
    return dirs


def load_previous_portfolio(
    history: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    按优先级读取上一期**实盘/正式**持仓：
    1) rebalance_history 传入对象（auto_rebalance）
    2) RESULTS_DIR（及 PROD 下旧版 results/）的 rebalance_history.json
    3) current_portfolio.json
    4) selection_results.json（跳过 mode=quick_test）
    """
    def _from_dict(data: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
        stocks = list(data.get('selected_stocks') or data.get('selected_tickers') or [])
        weights = _normalize_weight_dict(data.get('weights') or {})
        if not stocks and weights:
            stocks = list(weights.keys())
        return stocks, weights

    if history and history.get('rebalances'):
        last = history['rebalances'][-1].get('portfolio') or {}
        stocks, weights = _from_dict(last)
        if stocks or weights:
            return stocks, weights

    for base in _candidate_state_dirs():
        hist_path = os.path.join(base, 'rebalance_history.json')
        if os.path.isfile(hist_path):
            try:
                with open(hist_path, 'r', encoding='utf-8') as f:
                    disk_hist = json.load(f)
                if disk_hist.get('rebalances'):
                    last = disk_hist['rebalances'][-1].get('portfolio') or {}
                    stocks, weights = _from_dict(last)
                    if stocks or weights:
                        return stocks, weights
            except Exception:
                pass

        cur_path = os.path.join(base, 'current_portfolio.json')
        if os.path.isfile(cur_path):
            try:
                with open(cur_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stocks, weights = _from_dict(data)
                if stocks or weights:
                    return stocks, weights
            except Exception:
                pass

        sel_path = os.path.join(base, 'selection_results.json')
        if os.path.isfile(sel_path):
            try:
                with open(sel_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('mode') == 'quick_test':
                    continue
                stocks, weights = _from_dict(data)
                if stocks or weights:
                    return stocks, weights
            except Exception:
                pass

    return [], {}


def align_current_weights(
    selected_tickers: List[str],
    current_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """将旧权重投影到当前候选：不在名单内视为 0，缺失视为 0。"""
    cur = current_weights or {}
    return {t: float(cur.get(t, 0.0)) for t in selected_tickers}


def apply_deadband(
    target_weights: Dict[str, float],
    current_weights: Optional[Dict[str, float]],
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    执行层静默阈值：
    - 对仍持有且 |Δw| < threshold 的标的，锁定为当前权重（不交易）
    - 其余可交易标的在剩余权重预算内按目标比例归一化
    - 无当前持仓时原样返回
    """
    thr = config.DEADBAND_THRESHOLD if threshold is None else threshold
    target = _normalize_weight_dict(target_weights)
    current = _normalize_weight_dict(current_weights or {})

    meta = {
        'deadband_threshold': thr,
        'locked': {},
        'adjusted': False,
        'pre_turnover': 0.0,
        'post_turnover': 0.0,
    }

    if not target:
        return {}, meta
    if not current or not config.ENABLE_STATE_DEPENDENCY:
        meta['pre_turnover'] = sum(abs(target.get(t, 0) - current.get(t, 0)) for t in set(target) | set(current))
        meta['post_turnover'] = meta['pre_turnover']
        return target, meta

    all_tickers = set(target) | set(current)
    meta['pre_turnover'] = sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in all_tickers)

    locked = {}
    free_target = {}
    for t, tw in target.items():
        cw = current.get(t, 0.0)
        if cw > 0 and abs(tw - cw) < thr:
            locked[t] = cw
            meta['locked'][t] = {
                'current': round(cw, 6),
                'target': round(tw, 6),
                'delta': round(tw - cw, 6),
            }
        else:
            free_target[t] = tw

    locked_sum = sum(locked.values())
    if locked_sum >= 1.0 - 1e-9:
        # 几乎全锁：保持当前持仓归一化
        out = _normalize_weight_dict(current)
        meta['adjusted'] = True
        meta['post_turnover'] = sum(abs(out.get(t, 0) - current.get(t, 0)) for t in set(out) | set(current))
        try:
            import decision_log
            decision_log.extend_from_deadband(meta, current)
        except Exception:
            pass
        return out, meta

    remaining = max(0.0, 1.0 - locked_sum)
    free_sum = sum(free_target.values())

    out = dict(locked)
    if free_sum > 1e-12 and remaining > 1e-12:
        for t, tw in free_target.items():
            out[t] = remaining * (tw / free_sum)
    elif remaining > 1e-12 and not free_target:
        # 无自由票时把剩余按当前未锁定持仓分配
        unlocked_cur = {t: w for t, w in current.items() if t not in locked}
        u_sum = sum(unlocked_cur.values())
        if u_sum > 1e-12:
            for t, w in unlocked_cur.items():
                out[t] = remaining * (w / u_sum)
        else:
            # 极端情况：均分给目标票
            n = max(len(target), 1)
            for t in target:
                if t not in out:
                    out[t] = remaining / n

    out = _normalize_weight_dict(out)
    meta['adjusted'] = bool(meta['locked'])
    meta['post_turnover'] = sum(abs(out.get(t, 0) - current.get(t, 0)) for t in set(out) | set(current))
    try:
        import decision_log
        decision_log.extend_from_deadband(meta, current)
    except Exception:
        pass
    return out, meta
