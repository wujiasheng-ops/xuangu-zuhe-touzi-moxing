"""
结构化决策日志：记录为何换仓 / 为何不换，便于虚拟盘复盘。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import config

_entries: List[Dict[str, Any]] = []
_meta: Dict[str, Any] = {}


def reset(run_id: Optional[str] = None, **meta):
    global _entries, _meta
    _entries = []
    _meta = {
        'run_id': run_id or datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trading_env': getattr(config, 'TRADING_ENV', 'PROD'),
        'timestamp': datetime.now().isoformat(),
        **meta,
    }


def add(
    ticker: str,
    action: str,
    reason: str,
    current_weight: float = None,
    target_weight_raw: float = None,
    competitor_ticker: str = None,
    momentum_diff: float = None,
    **extra,
):
    entry = {
        'ticker': ticker,
        'action': action,
        'reason': reason,
    }
    if current_weight is not None:
        entry['current_weight'] = round(float(current_weight), 6)
        entry['current_weight_pct'] = f'{float(current_weight) * 100:.2f}%'
    if target_weight_raw is not None:
        entry['target_weight_raw'] = round(float(target_weight_raw), 6)
        entry['target_weight_raw_pct'] = f'{float(target_weight_raw) * 100:.2f}%'
    if competitor_ticker is not None:
        entry['competitor_ticker'] = competitor_ticker
    if momentum_diff is not None:
        entry['momentum_diff_pp'] = round(float(momentum_diff), 2)
        entry['momentum_diff'] = f'{float(momentum_diff):+.1f}pp'
    entry.update(extra)
    _entries.append(entry)


def extend_from_deadband(deadband_meta: Dict[str, Any], current_weights: Dict[str, float] = None):
    locked = (deadband_meta or {}).get('locked') or {}
    thr = (deadband_meta or {}).get('deadband_threshold', config.DEADBAND_THRESHOLD)
    for t, info in locked.items():
        cw = float(info.get('current', (current_weights or {}).get(t, 0)))
        tw = float(info.get('target', 0))
        delta = abs(float(info.get('delta', tw - cw)))
        add(
            ticker=t,
            action='SKIPPED_REBALANCE',
            reason=(
                f'Weight diff ({delta * 100:.2f}%) < DEADBAND_THRESHOLD '
                f'({thr * 100:.1f}%). Locked.'
            ),
            current_weight=cw,
            target_weight_raw=tw,
        )


def entries() -> List[Dict[str, Any]]:
    return list(_entries)


def save(extra_meta: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        **_meta,
        **(extra_meta or {}),
        'n_decisions': len(_entries),
        'decisions': _entries,
    }
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    run_id = payload.get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(config.LOGS_DIR, f'decision_log_{run_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    # 同时写一份 latest 方便报告读取
    latest = os.path.join(config.RESULTS_DIR, 'decision_log_latest.json')
    with open(latest, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f'✓ 决策日志已保存：{path}（{len(_entries)} 条）')
    return path
