import os
from datetime import datetime, timedelta

# --- 目录配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
RESULTS_ROOT = os.path.join(BASE_DIR, 'results')

# PROD = 正式/虚拟盘状态；TEST = quick_test 等开发跑批（物理隔离，防污染）
TRADING_ENV = os.getenv('TRADING_ENV', 'PROD').strip().upper()
if TRADING_ENV not in ('PROD', 'TEST'):
    TRADING_ENV = 'PROD'

RESULTS_DIR = os.path.join(
    RESULTS_ROOT, 'test' if TRADING_ENV == 'TEST' else 'live'
)
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, 'live'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, 'test'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, 'live', 'logs'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, 'test', 'logs'), exist_ok=True)


def _bootstrap_live_from_legacy():
    """若 results/live 为空且旧版扁平 results/ 有正式产物，则复制一份（一次性迁移）。"""
    if TRADING_ENV != 'PROD':
        return
    import shutil
    markers = (
        'selection_results.json',
        'current_portfolio.json',
        'rebalance_history.json',
        'portfolio_metrics.json',
        'optimal_weights.csv',
        'correlation_matrix.csv',
        'final_recommendation.json',
        'backtest_results.csv',
    )
    live_has = any(os.path.isfile(os.path.join(RESULTS_DIR, m)) for m in markers)
    if live_has:
        return
    copied = []
    for name in markers:
        src = os.path.join(RESULTS_ROOT, name)
        dst = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            copied.append(name)
    if copied:
        print(f'已将旧版 results/ 迁移到 results/live/：{", ".join(copied)}')


_bootstrap_live_from_legacy()

# --- 数据配置 ---
LOOKBACK_PERIOD = 6            # 年
MOMENTUM_WINDOW = 12           # 月（动量回看）
MOMENTUM_EXCLUDE = 1           # 月（排除最近）
WEEKS_PER_MONTH = 4            # 月线换算为周线
MOMENTUM_MAX_PCT = 200.0       # 动量绝对值上限，过滤拆股/坏数据
FUNNEL_SIZE = 100              # 候选股数量
TARGET_SIZE = 10               # 最终选定数量
MIN_SECTORS = 4                # 最小行业数
MAX_STOCKS_PER_SECTOR = 3      # 最终组合内单行业最多持股数
CORRELATION_THRESHOLD = 0.6    # 相关性阈值（|ρ|）
CORRELATION_ALERT_THRESHOLD = 0.65
MAX_WEIGHT = 0.15              # 单票最大权重
MIN_WEIGHT = 0.001             # 低于此权重视为未持仓；finalize 时抬升至此以免 Stage2 标的被静默剔除
MAX_WEEKLY_RETURN = 0.5        # 周收益率截断上限（绝对值），防止协方差收缩数值溢出
LIQUIDITY_LOOKBACK_WEEKS = 20  # 流动性过滤回看周数
MIN_AVG_WEEKLY_DOLLAR_VOLUME = 50_000_000  # 周均成交额下限（美元），防御脏数据/停牌
BENCHMARK_TICKER = 'SPY'       # 强制基准
RISK_FREE_RATE = 0.05          # Sharpe 无风险利率（年化）
TRAIN_YEARS = 4                # 回测训练年数
TEST_YEARS = 1                 # 回测测试年数
TRANSACTION_COST_BPS = 10      # 再平衡单边佣金（基点）
SLIPPAGE_BPS = 5               # 再平衡单边滑点（基点）

# --- 状态依赖（缓冲带 / 换手惩罚 / 执行静默）---
ENABLE_STATE_DEPENDENCY = True     # False 时退回无记忆行为
LOYALTY_MULTIPLIER = 1.05          # Stage1：现有持仓动量排序加成
HOLDINGS_REPLACE_MARGIN = 20.0     # Stage2：新票动量需高出旧仓（百分点）才可挤掉
TURNOVER_LAMBDA = 0.0015           # 换手摩擦（与 15bps 对齐，用于记录/参考）
TURNOVER_SHRINK_GAMMA = 0.05       # Stage3 二次收缩：向旧权的 L1 拉力（越大越懒得换仓）
DEADBAND_THRESHOLD = 0.025         # 执行层：权重绝对偏差低于此则不调仓

# --- 下载与缓存 ---
DOWNLOAD_CHUNK_SIZE = 50
LIVE_CORRELATION_PERIOD = '2y'

# --- 权重方案 ---
USE_RISK_PARITY_WEIGHTS = False  # True 时使用 algorithm_v2 风险平价

# --- 自动化 / 纸交易 ---
DEFAULT_REBALANCE_FREQUENCY = 'monthly'  # 'monthly' | 'quarterly'
ENABLE_PAPER_TRADING = False       # True 时 auto_rebalance 结束后执行下单（否则 Dry Run 打印）
PAPER_BROKER = 'simulated'         # 'simulated' | 'alpaca'
ALPACA_PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
PAPER_INITIAL_CASH = 100000.0      # SimulatedBroker / equity_curve 名义本金


def apply_trading_env(env: str = None):
    """运行时切换 PROD/TEST（quick_test 应在 import 其他模块前设置环境变量）。"""
    global TRADING_ENV, RESULTS_DIR, LOGS_DIR
    if env is not None:
        TRADING_ENV = str(env).strip().upper()
    else:
        TRADING_ENV = os.getenv('TRADING_ENV', TRADING_ENV).strip().upper()
    if TRADING_ENV not in ('PROD', 'TEST'):
        TRADING_ENV = 'PROD'
    RESULTS_DIR = os.path.join(RESULTS_ROOT, 'test' if TRADING_ENV == 'TEST' else 'live')
    LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    if TRADING_ENV == 'PROD':
        _bootstrap_live_from_legacy()
    return TRADING_ENV


def get_date_range():
    """每次调用时计算当前回看窗口，避免 import 时日期冻结。"""
    end = datetime.now()
    start = end - timedelta(days=365 * LOOKBACK_PERIOD)
    return start, end


# 兼容旧代码：模块级 START/END 仍可用，但新代码应调用 get_date_range()
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365 * LOOKBACK_PERIOD)

# --- 行业映射（离线备选）---
SECTOR_MAPPING = {
    'AAPL': 'Information Technology', 'MSFT': 'Information Technology', 'NVDA': 'Information Technology',
    'META': 'Information Technology', 'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'JNJ': 'Health Care', 'UNH': 'Health Care', 'PFE': 'Health Care',
    'WMT': 'Consumer Staples', 'KO': 'Consumer Staples',
    'VERIZON': 'Communication Services', 'T': 'Communication Services',
}

STANDARD_SECTORS = [
    'Information Technology',
    'Health Care',
    'Financials',
    'Consumer Discretionary',
    'Communication Services',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Real Estate',
    'Materials',
]
