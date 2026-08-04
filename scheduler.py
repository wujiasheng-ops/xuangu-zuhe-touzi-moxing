"""
定时执行器 - 使用 APScheduler 实现自动化
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import os
import json

from auto_rebalance import PortfolioAutoManager
import config


class ScheduledRebalancer:
    def __init__(self, rebalance_frequency=None):
        freq = rebalance_frequency or config.DEFAULT_REBALANCE_FREQUENCY
        self.manager = PortfolioAutoManager(rebalance_frequency=freq)
        self.log_file = os.path.join(config.RESULTS_DIR, 'scheduler.log')
        self.scheduler = BackgroundScheduler()

    def _log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f'[{timestamp}] {message}\n'
        print(log_msg.strip())
        with open(self.log_file, 'a') as f:
            f.write(log_msg)

    def _after_rebalance(self, new_portfolio, reason='scheduled'):
        if not new_portfolio:
            return
        changes = self.manager.compare_with_previous(new_portfolio)
        self.manager.record_rebalance(new_portfolio, changes, reason=reason)
        self.manager.save_portfolio_json()
        try:
            import final_report
            final_report.save_report_to_json()
            final_report.save_report_to_markdown()
        except Exception as e:
            self._log(f'报告生成失败：{e}')

    def monthly_rebalance(self):
        self._log('=' * 70)
        self._log('执行重平衡任务')
        try:
            new_portfolio = self.manager.run_optimization()
            if new_portfolio:
                self._after_rebalance(new_portfolio, reason='scheduled')
                self._log('✅ 重平衡完成')
            else:
                self._log('❌ 重平衡失败')
        except Exception as e:
            self._log(f'❌ 重平衡出错：{e}')

    def weekly_correlation_check(self):
        self._log('执行相关性检查...')
        try:
            if not self.manager.current_portfolio:
                self._log('当前无投资组合，跳过检查')
                return
            alert = self.manager.check_correlation_alert(self.manager.current_portfolio)
            if alert:
                self._log(f'⚠️ 相关性预警：{alert["message"]}')
                self._trigger_emergency_rebalance()
            else:
                self._log('✓ 相关性正常')
        except Exception as e:
            self._log(f'❌ 相关性检查出错：{e}')

    def _trigger_emergency_rebalance(self):
        self._log('=' * 70)
        self._log('触发紧急重平衡（相关性预警）')
        try:
            new_portfolio = self.manager.run_optimization()
            if new_portfolio:
                self._after_rebalance(new_portfolio, reason='correlation_alert')
                self._log('✅ 紧急重平衡完成')
            else:
                self._log('❌ 紧急重平衡失败')
        except Exception as e:
            self._log(f'❌ 紧急重平衡出错：{e}')

    def setup_schedule(self):
        freq = self.manager.frequency
        if freq == 'quarterly':
            # 每季度末月最后一个工作日 15:00
            self.scheduler.add_job(
                self.monthly_rebalance,
                CronTrigger(month='3,6,9,12', day='L', hour=15, minute=0, day_of_week='0-4'),
                id='quarterly_rebalance',
                name='Quarterly Portfolio Rebalance',
            )
            self._log('✅ 定时任务：季度重平衡（3/6/9/12 月末工作日 15:00）')
        else:
            self.scheduler.add_job(
                self.monthly_rebalance,
                CronTrigger(day='L', hour=15, minute=0, day_of_week='0-4'),
                id='monthly_rebalance',
                name='Monthly Portfolio Rebalance',
            )
            self._log('✅ 定时任务：月度重平衡（每月末工作日 15:00）')

        self.scheduler.add_job(
            self.weekly_correlation_check,
            CronTrigger(day_of_week=0, hour=9, minute=0),
            id='weekly_correlation_check',
            name='Weekly Correlation Check',
        )
        self._log('✅ 定时任务：相关性检查（每周一 09:00）')

    def start(self):
        self._log('=' * 70)
        self._log('定时执行器启动')
        self._log('=' * 70)
        self.setup_schedule()
        self._log('执行器运行中，等待任务触发...')
        self.scheduler.start()
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            self._log('执行器已停止（用户中止）')
            self.scheduler.shutdown()


if __name__ == '__main__':
    print('启动定时执行器（config.DEFAULT_REBALANCE_FREQUENCY）...')
    ScheduledRebalancer().start()
