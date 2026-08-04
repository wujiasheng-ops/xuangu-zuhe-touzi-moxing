"""
自动化投资组合管理系统 v1.0
功能：
1. 定期重新优化投资组合（月度/季度）
2. 监控相关性变化，触发预警和自动换血
3. 记录变动历史和绩效
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import config
import data_fetcher
import yfinance as yf
import stage1_funnel
import stage2_correlation
import stage3_optimizer
import stage4_backtest
import portfolio_state


def compute_live_max_abs_correlation(tickers: List[str], period: str = None) -> Optional[float]:
    """
    用最近行情重算持仓两两绝对相关系数的最大值（与 Stage2 的 |ρ| 阈值语义一致）。
    失败时返回 None，由调用方回退到历史快照。
    """
    if not tickers or len(tickers) < 2:
        return 0.0

    if period is None:
        period = config.LIVE_CORRELATION_PERIOD

    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval="1wk",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

        prices = prices.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if prices.shape[1] < 2 or len(prices) < 8:
            return None

        rets = prices.pct_change().dropna()
        if rets.shape[0] < 5:
            return None

        c = rets.corr()
        v = np.asarray(c.values, dtype=float)
        triu = np.triu_indices_from(v, k=1)
        off = v[triu]
        off = off[np.isfinite(off)]
        if off.size == 0:
            return None
        return float(np.nanmax(np.abs(off)))
    except Exception:
        return None


class PortfolioAutoManager:
    """自动化投资组合管理器"""

    def __init__(self, rebalance_frequency=None):
        self.frequency = rebalance_frequency or config.DEFAULT_REBALANCE_FREQUENCY
        self.history_file = os.path.join(config.RESULTS_DIR, 'rebalance_history.json')
        self.correlation_alert_file = os.path.join(config.RESULTS_DIR, 'correlation_alerts.json')
        self.current_portfolio = None
        self.correlation_threshold = config.CORRELATION_THRESHOLD
        self.history = self._load_history()
        self._restore_current_portfolio_from_disk()

    def _restore_current_portfolio_from_disk(self):
        """进程重启后从磁盘恢复最近一次组合，供周度相关性检查使用。"""
        path = os.path.join(config.RESULTS_DIR, "current_portfolio.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("selected_stocks"):
                self.current_portfolio = data
        except Exception:
            pass

    def _load_history(self):
        """加载历史重平衡记录"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'rebalances': []}

    def _save_history(self):
        """保存历史记录"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _save_correlation_alert(self, alert: Dict[str, Any]):
        path = self.correlation_alert_file
        alerts = []
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []
        alerts.append(alert)
        with open(path, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)

    def _persist_results_files(
        self,
        selected_df,
        selected_list,
        weights_df,
        weights_dict,
        corr_matrix,
        backtest_result,
        opt_metrics,
    ):
        """写入与 main_pipeline 兼容的 results/ 文件，供 final_report 使用。"""
        output_dict = {
            'timestamp': datetime.now().isoformat(),
            'selected_tickers': selected_list,
            'weights': weights_dict,
            'sectors': dict(zip(selected_df['Ticker'], selected_df['Sector'])),
            'momentum': dict(zip(selected_df['Ticker'], selected_df['Momentum'])),
            'optimization': opt_metrics,
        }
        with open(os.path.join(config.RESULTS_DIR, 'selection_results.json'), 'w') as f:
            json.dump(output_dict, f, indent=2)

        weights_df.to_csv(os.path.join(config.RESULTS_DIR, 'optimal_weights.csv'), index=False)
        corr_matrix.to_csv(os.path.join(config.RESULTS_DIR, 'correlation_matrix.csv'))

        if backtest_result:
            rb = backtest_result.get('rolling_backtest')
            if rb is not None and len(rb) > 0:
                rb.to_csv(os.path.join(config.RESULTS_DIR, 'backtest_results.csv'), index=False)
            payload = {
                'timestamp': datetime.now().isoformat(),
                'in_sample_metrics': backtest_result.get('full_metrics'),
                'metrics_2022': backtest_result.get('metrics_2022'),
                'walk_forward_metrics': backtest_result.get('walk_forward_metrics'),
            }
            with open(os.path.join(config.RESULTS_DIR, 'portfolio_metrics.json'), 'w') as f:
                json.dump(payload, f, indent=2, default=str)

    def run_optimization(self):
        """
        执行完整的优化流程：数据获取 -> 四阶段筛选 -> 权重优化
        """
        print("\n" + "="*70)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] 启动投资组合重新优化")
        print("="*70)

        try:
            # 1. 获取数据
            print("\n[1/5] 获取数据...")
            tickers, weekly_data, sectors, weekly_dollar_volume = data_fetcher.prepare_data()

            if weekly_data is None or weekly_data.empty:
                print("❌ 数据获取失败")
                return None

            import decision_log
            decision_log.reset(source='auto_rebalance')
            print(f'TRADING_ENV={config.TRADING_ENV}  RESULTS_DIR={config.RESULTS_DIR}')

            current_holdings, current_weights = portfolio_state.load_previous_portfolio(self.history)
            if current_holdings:
                print(f"上一期持仓 {len(current_holdings)} 只，启用状态依赖")
            else:
                print("无历史持仓，冷启动")

            # 2. Stage 1: 漏斗筛选
            print("\n[2/5] 执行漏斗筛选...")
            candidates = stage1_funnel.stage1_funnel(
                weekly_data, sectors, weekly_dollar_volume, current_holdings=current_holdings
            )

            if candidates.empty:
                print("❌ Stage 1 失败")
                return None

            # 3. Stage 2: 相关性剪枝
            print("\n[3/5] 执行相关性剪枝...")
            selected_df, selected_list, corr_matrix = stage2_correlation.stage2_correlation_pruning(
                weekly_data, candidates, current_holdings=current_holdings
            )

            if len(selected_list) == 0:
                print("❌ Stage 2 失败")
                return None

            # 4. Stage 3: 权重优化
            print("\n[4/5] 执行权重优化...")
            weights_df, opt_metrics = stage3_optimizer.stage3_optimizer(
                weekly_data, selected_list, current_weights=current_weights
            )

            if weights_df is None or weights_df.empty:
                print('❌ Stage 3 失败')
                return None

            stage2_list = list(selected_list)
            selected_list = weights_df['Ticker'].tolist()
            selected_df = selected_df[selected_df['Ticker'].isin(selected_list)].copy()
            selected_df = selected_df.set_index('Ticker').loc[selected_list].reset_index()
            weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))

            weights_dict, deadband_meta = portfolio_state.apply_deadband(weights_dict, current_weights)
            if deadband_meta.get('locked'):
                print(
                    f"执行层 Deadband：锁定 {len(deadband_meta['locked'])} 只，"
                    f"换手 {deadband_meta['pre_turnover']:.3f} → {deadband_meta['post_turnover']:.3f}"
                )
                selected_list = [t for t in weights_dict if weights_dict[t] > 0]
                selected_df = selected_df[selected_df['Ticker'].isin(selected_list)].copy()
                selected_df = selected_df.set_index('Ticker').loc[
                    [t for t in selected_list if t in set(selected_df['Ticker'])]
                ].reset_index()
                weights_df = pd.DataFrame(
                    [{'Ticker': t, 'Weight': round(weights_dict[t], 6)} for t in selected_list]
                ).sort_values('Weight', ascending=False).reset_index(drop=True)
                opt_metrics = {**(opt_metrics or {}), 'deadband': deadband_meta}

            print('\n[5/6] 执行滚动回测...')
            backtest_result = stage4_backtest.stage4_backtest(
                weekly_data, sectors, selected_list, weights_dict, weekly_dollar_volume
            )

            print('\n[6/6] 记录优化结果...')
            triu_idx = np.triu_indices_from(corr_matrix.values, k=1)
            off_diag = corr_matrix.values[triu_idx]
            # 与 Stage2 一致：用 |ρ| 衡量共变风险（强负相关同样集中风险）
            max_corr = float(np.nanmax(np.abs(off_diag))) if off_diag.size > 0 else 0.0

            raw_w = weights_dict
            weights_norm = raw_w

            self._persist_results_files(
                selected_df, selected_list, weights_df, weights_norm,
                corr_matrix, backtest_result, opt_metrics,
            )

            result = {
                'timestamp': datetime.now().isoformat(),
                'selected_stocks': selected_list,
                'weights': weights_norm,
                'sectors': dict(zip(selected_df['Ticker'], selected_df['Sector'])),
                'momentum': dict(zip(selected_df['Ticker'], selected_df['Momentum'])),
                'correlation_matrix': corr_matrix.to_dict(),
                'max_correlation': max_corr,
                'backtest': backtest_result,
            }

            self.current_portfolio = result
            print("\n✅ 优化完成")
            print(f"选定股票：{selected_list}")

            return result

        except Exception as e:
            print(f"\n❌ 优化过程出错：{e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_with_previous(self, new_portfolio):
        """
        对比新旧投资组合，识别变化
        """
        if not self.history['rebalances'] or len(self.history['rebalances']) == 0:
            print("\n✓ 首次优化，无历史组合对比")
            return {
                'new_stocks': new_portfolio['selected_stocks'],
                'removed_stocks': [],
                'changed_weights': {},
                'correlation_change': 'N/A',
                'old_max_corr': None,
                'new_max_corr': new_portfolio.get('max_correlation', 0),
            }

        last_portfolio = self.history['rebalances'][-1]['portfolio']
        old_stocks = set(last_portfolio['selected_stocks'])
        new_stocks = set(new_portfolio['selected_stocks'])

        changes = {
            'new_stocks': list(new_stocks - old_stocks),
            'removed_stocks': list(old_stocks - new_stocks),
            'changed_weights': {},
            'old_max_corr': last_portfolio.get('max_correlation', 0),
            'new_max_corr': new_portfolio.get('max_correlation', 0),
        }

        # 计算权重变化
        for ticker in old_stocks & new_stocks:
            old_w = last_portfolio['weights'].get(ticker, 0)
            new_w = new_portfolio['weights'].get(ticker, 0)
            change_pct = (new_w - old_w) * 100

            if abs(change_pct) > 0.5:  # 超过0.5%则记录
                changes['changed_weights'][ticker] = {
                    'old': old_w,
                    'new': new_w,
                    'change_pct': change_pct
                }

        return changes

    def check_correlation_alert(
        self,
        portfolio: Dict[str, Any],
        recompute_from_market: bool = True,
    ):
        """
        检查相关性是否超过阈值，触发预警。

        默认用最近周线重算 |ρ|_max（反映当前市场），失败时回退到上次优化写入的快照。
        """
        stored = float(portfolio.get("max_correlation", 0) or 0)
        max_corr = stored
        source = "snapshot"

        if recompute_from_market:
            tickers = portfolio.get("selected_stocks") or []
            live = compute_live_max_abs_correlation(tickers)
            if live is not None:
                max_corr = live
                source = "live_market"
            else:
                max_corr = stored
                source = "snapshot_fallback"

        alert_threshold = config.CORRELATION_ALERT_THRESHOLD

        if max_corr > alert_threshold:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'severity': 'HIGH' if max_corr > 0.75 else 'MEDIUM',
                'max_correlation': max_corr,
                'max_correlation_source': source,
                'max_correlation_snapshot': stored,
                'message': (
                    f'|ρ|_max={max_corr:.3f}（{source}），超过预警阈值 {alert_threshold}'
                ),
                'action': '建议立即执行重新优化'
            }

            print(f"\n⚠️ 相关性预警：{alert['message']}")
            print(f"   严重级别：{alert['severity']}")
            print(f'   建议：{alert["action"]}')
            self._save_correlation_alert(alert)

            return alert
        return None

    def record_rebalance(self, new_portfolio, changes, reason='scheduled'):
        """
        记录一次重平衡事件
        reason: 'scheduled', 'correlation_alert', 'manual'
        """
        rebalance_record = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'portfolio': new_portfolio,
            'changes': changes
        }

        self.history['rebalances'].append(rebalance_record)
        self._save_history()

        print(f"\n✅ 重平衡记录已保存")
        if changes['new_stocks']:
            print(f"   新增股票：{changes['new_stocks']}")
        if changes['removed_stocks']:
            print(f"   移除股票：{changes['removed_stocks']}")
        if changes['changed_weights']:
            print(f"   权重调整：{len(changes['changed_weights'])}只")

    def generate_rebalance_report(self, new_portfolio, changes):
        """生成重平衡报告"""
        report = f"""
{'='*70}
投资组合重平衡报告
{'='*70}

【重平衡时间】
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【当前投资组合】
选定股票数：{len(new_portfolio['selected_stocks'])}
股票列表：{', '.join(new_portfolio['selected_stocks'])}

【权重分配】
"""
        for ticker, weight in sorted(new_portfolio['weights'].items(), key=lambda x: x[1], reverse=True):
            sector = new_portfolio['sectors'].get(ticker, 'Unknown')
            report += f"  {ticker:<8} {weight*100:>6.2f}%  ({sector})\n"

        report += f"""
【变化分析】
新增股票：{changes['new_stocks'] if changes['new_stocks'] else '无'}
移除股票：{changes['removed_stocks'] if changes['removed_stocks'] else '无'}
权重调整数：{len(changes['changed_weights'])}只

"""
        if changes['changed_weights']:
            report += "【权重变化详情】\n"
            for ticker, change in changes['changed_weights'].items():
                old_w = change['old'] * 100
                new_w = change['new'] * 100
                change_pct = change['change_pct']
                direction = "↑" if change_pct > 0 else "↓"
                report += f"  {ticker}: {old_w:6.2f}% → {new_w:6.2f}% ({direction} {abs(change_pct):>5.2f}%)\n"

        new_corr = changes.get('new_max_corr', new_portfolio.get('max_correlation', 0))
        old_corr_disp = changes.get('old_max_corr')
        old_corr_str = f"{old_corr_disp:.3f}" if old_corr_disp is not None else 'N/A'

        report += f"""
【相关性监控】
历史最大相关性：{old_corr_str}
当前最大相关性：{new_corr:.3f}
约束阈值：{self.correlation_threshold}
状态：{'✓ 正常' if new_corr <= 0.6 else '⚠️ 预警'}

{'='*70}
"""
        return report

    def print_portfolio_summary(self):
        """打印当前投资组合摘要"""
        if not self.current_portfolio:
            print("❌ 当前无投资组合")
            return

        portfolio = self.current_portfolio
        print("\n" + "="*70)
        print("【当前投资组合摘要】")
        print("="*70)
        print(f"更新时间：{portfolio['timestamp']}")
        print(f"\n选定股票（{len(portfolio['selected_stocks'])}只）：")
        for i, ticker in enumerate(portfolio['selected_stocks'], 1):
            weight = portfolio['weights'][ticker] * 100
            sector = portfolio['sectors'].get(ticker, 'Unknown')
            print(f"  {i}. {ticker:<8} {weight:>6.2f}% ({sector})")

        print(f"\n相关性：最大值 {portfolio['max_correlation']:.3f} (阈值 {self.correlation_threshold})")

    def save_portfolio_json(self, filename='current_portfolio.json'):
        """保存当前投资组合为JSON"""
        if not self.current_portfolio:
            print("❌ 当前无投资组合")
            return

        filepath = os.path.join(config.RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(self.current_portfolio, f, indent=2, default=str)

        print(f"\n✅ 投资组合已保存到 {filepath}")
        return filepath

def main():
    """主程序：运行自动化重平衡"""
    manager = PortfolioAutoManager(rebalance_frequency='monthly')

    # 运行优化
    new_portfolio = manager.run_optimization()

    if new_portfolio:
        # 对比历史（先取旧仓用于基准，再写入新仓）
        old_weights = {}
        old_ts = None
        if manager.history.get('rebalances'):
            last = manager.history['rebalances'][-1]
            old_ts = last.get('timestamp')
            old_weights = (last.get('portfolio') or {}).get('weights') or {}

        changes = manager.compare_with_previous(new_portfolio)

        # 检查相关性预警
        alert = manager.check_correlation_alert(new_portfolio)

        # 生成报告
        report = manager.generate_rebalance_report(new_portfolio, changes)
        print(report)

        # 记录重平衡
        reason = 'correlation_alert' if alert else 'scheduled'
        manager.record_rebalance(new_portfolio, changes, reason=reason)

        # 保存投资组合
        manager.save_portfolio_json()
        manager.print_portfolio_summary()

        try:
            import decision_log
            import benchmark
            decision_log.save(extra_meta={'selected_stocks': new_portfolio.get('selected_stocks')})
            _, weekly_data, _, _ = data_fetcher.prepare_data()
            if weekly_data is not None:
                bench = benchmark.update_benchmark_snapshot(
                    weekly_data,
                    previous_weights=old_weights,
                    previous_timestamp=old_ts,
                )
                print('\n' + benchmark.format_period_comparison(bench))
        except Exception as e:
            print(f'决策日志/基准更新失败：{e}')

        try:
            import final_report
            final_report.save_report_to_json()
            final_report.save_report_to_markdown()
        except Exception as e:
            print(f'生成报告失败：{e}')

        # 始终走统一入口：ENABLE_PAPER_TRADING=False 时为 Dry Run
        try:
            from trading_interface import execute_paper_rebalance
            print(
                f"\n[纸交易] ENABLE_PAPER_TRADING={config.ENABLE_PAPER_TRADING}，"
                f"broker={config.PAPER_BROKER}"
            )
            execute_paper_rebalance(
                new_portfolio['weights'],
                current_weights=old_weights,
            )
        except Exception as e:
            print(f'纸交易入口失败：{e}')

if __name__ == '__main__':
    main()
