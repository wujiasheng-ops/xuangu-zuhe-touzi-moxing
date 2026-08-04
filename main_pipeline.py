"""
自动选股与配置算法 - 主pipeline
四阶段执行：
1. Funnel (漏斗筛选) - 从500只选100只
2. Correlation Pruning (相关性剪枝) - 从100只选10只
3. Portfolio Optimizer (权重优化) - 最大化Sharpe Ratio
4. Rolling Backtest (稳定性校验) - 验证投资组合稳健性
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

import config
import data_fetcher
import stage1_funnel
import stage2_correlation
import stage3_optimizer
import stage4_backtest
import portfolio_state
import decision_log
import benchmark


def main():
    print("=" * 60)
    print("自动选股与配置算法 - 主Pipeline")
    print(f"TRADING_ENV={config.TRADING_ENV}  RESULTS_DIR={config.RESULTS_DIR}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    decision_log.reset(source='main_pipeline')

    # ========== 数据准备 ==========
    print("\n[准备] 获取数据...")
    tickers, weekly_data, sectors, weekly_dollar_volume = data_fetcher.prepare_data()

    if weekly_data is None or weekly_data.empty:
        print("错误：数据获取失败，无法继续")
        return

    print(f"成功获取 {len(weekly_data.columns)} 只股票的 {len(weekly_data)} 周数据")

    current_holdings, current_weights = portfolio_state.load_previous_portfolio()
    prev_ts = None
    for fname in ('current_portfolio.json', 'selection_results.json'):
        p = os.path.join(config.RESULTS_DIR, fname)
        if not os.path.isfile(p) and hasattr(config, 'RESULTS_ROOT'):
            p = os.path.join(config.RESULTS_ROOT, fname)
        if os.path.isfile(p):
            try:
                with open(p, 'r') as f:
                    prev_ts = json.load(f).get('timestamp')
                if prev_ts:
                    break
            except Exception:
                pass

    if current_holdings:
        print(f"检测到上一期持仓 {len(current_holdings)} 只，启用状态依赖选股/优化")
    else:
        print("无上一期持仓记录，按空仓冷启动")

    # ========== Stage 1: Funnel ==========
    print("\n[Stage 1] 执行漏斗式筛选...")
    candidates = stage1_funnel.stage1_funnel(
        weekly_data, sectors, weekly_dollar_volume, current_holdings=current_holdings
    )

    if candidates.empty:
        print("错误：Stage 1 未能选出任何候选股票")
        return

    # ========== Stage 2: Correlation Pruning ==========
    print("\n[Stage 2] 执行相关性剪枝...")
    selected_df, selected_list, corr_matrix = stage2_correlation.stage2_correlation_pruning(
        weekly_data, candidates, current_holdings=current_holdings
    )

    if len(selected_list) == 0:
        print("错误：Stage 2 未能选出任何股票")
        return

    print(f"✓ 已选出 {len(selected_list)} 只股票：{selected_list}")

    # ========== Stage 3: Portfolio Optimizer ==========
    print("\n[Stage 3] 执行权重优化...")
    weights_df, opt_metrics = stage3_optimizer.stage3_optimizer(
        weekly_data, selected_list, current_weights=current_weights
    )

    if weights_df is None or weights_df.empty:
        print("错误：Stage 3 权重优化失败")
        return

    stage2_list = list(selected_list)
    selected_list = weights_df['Ticker'].tolist()
    selected_df = selected_df[selected_df['Ticker'].isin(selected_list)].copy()
    selected_df = selected_df.set_index('Ticker').loc[selected_list].reset_index()

    weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
    weights_dict, deadband_meta = portfolio_state.apply_deadband(weights_dict, current_weights)
    if deadband_meta.get('locked'):
        print(
            f"执行层 Deadband：锁定 {len(deadband_meta['locked'])} 只"
            f"（阈值 ±{deadband_meta['deadband_threshold']*100:.1f}%），"
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

    # ========== Stage 4: Rolling Backtest ==========
    print("\n[Stage 4] 执行滚动回测...")
    backtest_result = stage4_backtest.stage4_backtest(
        weekly_data, sectors, selected_list, weights_dict, weekly_dollar_volume
    )

    # ========== 生成输出 ==========
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)

    print(f"\n【选定 {len(selected_list)} 只股票】")
    for i, ticker in enumerate(selected_list, 1):
        sector = selected_df[selected_df['Ticker'] == ticker]['Sector'].values[0]
        weight = weights_dict[ticker] * 100
        print(f"  {i}. {ticker:6s} - 权重：{weight:6.2f}% - 行业：{sector}")

    # ========== 保存结果 ==========
    # 1. 选股结果
    output_dict = {
        'timestamp': datetime.now().isoformat(),
        'stage2_tickers': stage2_list,
        'selected_tickers': selected_list,
        'weights': weights_dict,
        'sectors': dict(zip(selected_df['Ticker'], selected_df['Sector'])),
        'momentum': dict(zip(selected_df['Ticker'], selected_df['Momentum'])),
        'optimization': opt_metrics,
    }

    output_file = os.path.join(config.RESULTS_DIR, 'selection_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n选股结果已保存到：{output_file}")

    # 2. 权重CSV
    weights_df.to_csv(os.path.join(config.RESULTS_DIR, 'optimal_weights.csv'), index=False)

    # 3. 相关性矩阵
    corr_matrix.to_csv(os.path.join(config.RESULTS_DIR, 'correlation_matrix.csv'))

    # 4. 回测与组合指标
    if backtest_result:
        if backtest_result.get('rolling_backtest') is not None and len(backtest_result['rolling_backtest']) > 0:
            backtest_result['rolling_backtest'].to_csv(
                os.path.join(config.RESULTS_DIR, 'backtest_results.csv'),
                index=False
            )

        metrics_payload = {
            'timestamp': datetime.now().isoformat(),
            'in_sample_metrics': backtest_result.get('full_metrics'),
            'metrics_2022': backtest_result.get('metrics_2022'),
            'walk_forward_metrics': backtest_result.get('walk_forward_metrics'),
            'walk_forward_metrics_stitched': backtest_result.get('walk_forward_metrics_stitched'),
            'transaction_costs': backtest_result.get('transaction_costs'),
            'note': 'walk_forward_metrics 为各独立 OOS 测试窗指标均值；walk_forward_metrics_stitched 为拼接测试窗净值后的复合年化表现',
        }
        with open(os.path.join(config.RESULTS_DIR, 'portfolio_metrics.json'), 'w') as f:
            json.dump(metrics_payload, f, indent=2, default=str)

    # ========== 风险分析 ==========
    print("\n" + "=" * 60)
    print("风险分析与改进建议")
    print("=" * 60)

    risk_analysis = {
        'correlation_risk': analyze_correlation_risk(corr_matrix),
        'sector_concentration': analyze_sector_concentration(selected_df),
        'momentum_risk': analyze_momentum_risk(selected_df),
        'backtest_performance': analyze_backtest_performance(backtest_result)
    }

    print_risk_analysis(risk_analysis)

    # 保存风险分析
    risk_file = os.path.join(config.RESULTS_DIR, 'risk_analysis.json')
    with open(risk_file, 'w') as f:
        json.dump(risk_analysis, f, indent=2, default=str)
    print(f"\n风险分析已保存到：{risk_file}")

    # ========== P0: 决策日志 / SPY 基准 / 纸交易 Dry Run ==========
    try:
        decision_log.save(extra_meta={
            'selected_tickers': selected_list,
            'weights': weights_dict,
        })
    except Exception as e:
        print(f'决策日志保存失败：{e}')

    try:
        bench = benchmark.update_benchmark_snapshot(
            weekly_data,
            previous_weights=current_weights,
            previous_timestamp=prev_ts,
        )
        output_dict['benchmark'] = bench
        with open(output_file, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)
        print('\n' + benchmark.format_period_comparison(bench))
        if backtest_result:
            metrics_payload = {
                'timestamp': datetime.now().isoformat(),
                'in_sample_metrics': backtest_result.get('full_metrics'),
                'metrics_2022': backtest_result.get('metrics_2022'),
                'walk_forward_metrics': backtest_result.get('walk_forward_metrics'),
                'walk_forward_metrics_stitched': backtest_result.get('walk_forward_metrics_stitched'),
                'transaction_costs': backtest_result.get('transaction_costs'),
                'benchmark': bench,
                'note': 'walk_forward_metrics 为各独立 OOS 测试窗指标均值；walk_forward_metrics_stitched 为拼接测试窗净值后的复合年化表现',
            }
            with open(os.path.join(config.RESULTS_DIR, 'portfolio_metrics.json'), 'w') as f:
                json.dump(metrics_payload, f, indent=2, default=str)
    except Exception as e:
        print(f'基准对照更新失败：{e}')

    try:
        from trading_interface import execute_paper_rebalance
        print('\n[纸交易] ENABLE_PAPER_TRADING='
              f'{config.ENABLE_PAPER_TRADING}，broker={config.PAPER_BROKER}')
        execute_paper_rebalance(weights_dict, current_weights=current_weights)
    except Exception as e:
        print(f'纸交易入口失败：{e}')

    try:
        import final_report
        final_report.print_report()
        final_report.save_report_to_json()
        final_report.save_report_to_markdown()
    except Exception as e:
        print(f"\n生成最终报告时出错：{e}")

    print("\n" + "=" * 60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def analyze_correlation_risk(corr_matrix):
    """分析相关性风险"""
    risks = []

    # 找出高相关性对
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            ticker_i = corr_matrix.index[i]
            ticker_j = corr_matrix.index[j]
            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) > config.CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    'pair': f"{ticker_i}-{ticker_j}",
                    'correlation': corr_value
                })

    if high_corr_pairs:
        print(f"\n⚠️ 相关性风险：存在 {len(high_corr_pairs)} 对高相关性股票对")
        for pair in high_corr_pairs[:5]:
            print(f"    {pair['pair']}: {pair['correlation']:.3f}")
        risks.append({
            'level': 'medium',
            'description': f"{len(high_corr_pairs)} 对股票 |ρ|>{config.CORRELATION_THRESHOLD}",
            'pairs': high_corr_pairs
        })
    else:
        print(f'\n✓ 相关性风险：良好，所有股票对 |ρ|≤{config.CORRELATION_THRESHOLD}')

    return risks

def analyze_sector_concentration(selected_df):
    """分析行业集中度风险"""
    sector_counts = selected_df['Sector'].value_counts()
    print(f"\n行业分布：")
    for sector, count in sector_counts.items():
        pct = count / len(selected_df) * 100
        print(f"    {sector}: {count}只 ({pct:.1f}%)")

    risks = []
    if sector_counts.max() > 4:
        print(f"\n⚠️ 行业集中风险：{sector_counts.index[0]} 占比过高")
        risks.append({
            'level': 'medium',
            'description': f"最大行业占比：{sector_counts.max()/len(selected_df)*100:.1f}%"
        })

    return risks

def analyze_momentum_risk(selected_df):
    """分析Momentum偏差风险"""
    momentum_mean = selected_df['Momentum'].mean()
    momentum_std = selected_df['Momentum'].std()
    momentum_min = selected_df['Momentum'].min()

    print(f"\nMomentum统计：")
    print(f"    平均值: {momentum_mean:.2f}%")
    print(f"    标准差: {momentum_std:.2f}%")
    print(f"    最小值: {momentum_min:.2f}%")

    risks = []
    if momentum_std > 100:
        print(f"\n⚠️ Momentum偏差风险：标准差过大，选股间差异大")
        risks.append({
            'level': 'low',
            'description': f"Momentum标准差：{momentum_std:.2f}%"
        })

    return risks

def analyze_backtest_performance(backtest_result):
    """分析回测表现"""
    risks = []

    if not backtest_result or 'full_metrics' not in backtest_result:
        return risks

    full_metrics = backtest_result['full_metrics']
    print(f"\n回测表现总结：")
    print(f"    年化收益率: {full_metrics['annual_return']*100:.2f}%")
    print(f"    最大回撤: {full_metrics['max_drawdown']*100:.2f}%")
    print(f"    Sharpe Ratio: {full_metrics['sharpe_ratio']:.3f}")

    wf = backtest_result.get('walk_forward_metrics')
    if wf:
        print(f"\n样本外回测（训练窗重优化权重，更可信）：")
        print(f"    年化收益率: {wf['annual_return']*100:.2f}%")
        print(f"    最大回撤: {wf['max_drawdown']*100:.2f}%")
        print(f"    Sharpe Ratio: {wf['sharpe_ratio']:.3f}")

        in_ann = full_metrics['annual_return']
        oos_ann = wf['annual_return']
        if np.isfinite(in_ann) and np.isfinite(oos_ann) and in_ann > oos_ann * 1.5:
            risks.append({
                'level': 'medium',
                'description': (
                    f'样本内年化 {in_ann*100:.1f}% 显著高于样本外 {oos_ann*100:.1f}%，'
                    '存在前视/过拟合风险'
                ),
            })

    if backtest_result.get('metrics_2022'):
        m2022 = backtest_result['metrics_2022']
        print(f"\n2022年加息周期表现（关键风险测试）：")
        print(f"    收益率: {m2022['total_return']*100:.2f}%")
        print(f"    最大回撤: {m2022['max_drawdown']*100:.2f}%")
        print(f"    Calmar Ratio: {m2022['calmar_ratio']:.3f}")

        if m2022['max_drawdown'] < -0.30:
            print(f"\n⚠️ 加息周期风险：2022年回撤超过30%")
            risks.append({
                'level': 'high',
                'description': f"2022年最大回撤：{m2022['max_drawdown']*100:.2f}%"
            })

    return risks

def print_risk_analysis(risk_analysis):
    """打印风险分析总结"""
    print("\n【风险评估】")

    all_risks = []
    for category, risks in risk_analysis.items():
        all_risks.extend(risks)

    if not all_risks:
        print("✓ 总体风险评估：低风险，投资组合分散良好")
    else:
        high_risks = [r for r in all_risks if r.get('level') == 'high']
        medium_risks = [r for r in all_risks if r.get('level') == 'medium']

        if high_risks:
            print(f"\n⚠️ 高风险项 ({len(high_risks)}):")
            for risk in high_risks:
                print(f"    - {risk['description']}")

        if medium_risks:
            print(f"\n⚠️ 中风险项 ({len(medium_risks)}):")
            for risk in medium_risks:
                print(f"    - {risk['description']}")

    print("\n【改进建议】")
    print("1. 考虑定期调整权重（每季度或每半年）")
    print("2. 在加息周期中监控行业轮动，可增加防守性行业权重")
    print("3. 定期重新评估动量，剔除衰退的行业")
    print("4. 监控个股相关性，如果相关性上升，及时调整")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n执行出错: {e}")
        import traceback
        traceback.print_exc()
