"""
实盘交易接口 - 自动执行交易
支持多个经纪商：
- Alpaca (美国股票)
- 其他经纪商可扩展

使用前需要：
1. 注册经纪商账户
2. 获取API密钥
3. 配置credentials.json
"""

import importlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import config


def _last_tradeable_price(ticker: str) -> float:
    """用于名义头寸估算：优先现价字段，否则用最近收盘价。"""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            p = info.get(key)
            if p is not None and float(p) > 0:
                return float(p)
        hist = t.history(period="10d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            last = float(hist["Close"].iloc[-1])
            if last > 0:
                return last
    except Exception:
        pass
    return 0.0

def _credentials_path():
    return os.path.join(config.BASE_DIR, 'credentials.json')


def _alpaca_base_url(creds: Dict) -> str:
    """强制纸交易 endpoint，防止误连 live。"""
    url = (creds or {}).get('base_url') or config.ALPACA_PAPER_BASE_URL
    url = str(url).rstrip('/')
    if 'paper-api.alpaca.markets' not in url:
        print(
            f'⚠️ Alpaca base_url={url} 非纸交易地址，已强制切换为 '
            f'{config.ALPACA_PAPER_BASE_URL}'
        )
        url = config.ALPACA_PAPER_BASE_URL
    return url


def _normalize_target_weights(target_weights: Dict[str, float]) -> Dict[str, float]:
    tw = {k: float(v) for k, v in (target_weights or {}).items() if float(v) > 0}
    s = sum(tw.values())
    if s > 1e-12:
        tw = {k: v / s for k, v in tw.items()}
    return tw


def _print_rebalance_diff(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    dry_run: bool,
):
    print('\n【对账 Diff】' + ('（Dry Run — 不下单）' if dry_run else '（将提交订单）'))
    print(f"{'Ticker':<8} {'当前%':>8} {'目标%':>8} {'Δ%':>8} {'动作':<8}")
    all_t = sorted(set(current_weights) | set(target_weights))
    for t in all_t:
        cw = float(current_weights.get(t, 0.0))
        tw = float(target_weights.get(t, 0.0))
        d = tw - cw
        if abs(d) < 1e-4:
            action = 'HOLD'
        elif d > 0:
            action = 'BUY'
        else:
            action = 'SELL'
        print(f'{t:<8} {cw*100:8.2f} {tw*100:8.2f} {d*100:8.2f} {action:<8}')


class BrokerageInterface:
    """经纪商交易接口基类"""

    def __init__(self, broker_name: str):
        self.broker_name = broker_name
        self.credentials = self._load_credentials()

    def _load_credentials(self) -> Dict:
        """加载API凭证"""
        creds_file = _credentials_path()

        if not os.path.exists(creds_file):
            if self.broker_name == 'alpaca':
                print(f"⚠️ 未找到凭证文件 {creds_file}")
                print("   请创建 credentials.json，格式如下：")
                print("""
{
  "alpaca": {
    "api_key": "your_paper_api_key",
    "secret_key": "your_paper_secret_key",
    "base_url": "https://paper-api.alpaca.markets"
  }
}
                """)
            return {}

        with open(creds_file, 'r') as f:
            all_creds = json.load(f)

        return all_creds.get(self.broker_name, {})

    def buy_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        """购买股票"""
        raise NotImplementedError

    def sell_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        """卖出股票"""
        raise NotImplementedError

    def get_portfolio_value(self) -> float:
        """获取投资组合总价值"""
        raise NotImplementedError

    def get_position_weights(self) -> Dict[str, float]:
        """当前持仓权重（市值占比）。"""
        raise NotImplementedError

    def rebalance_portfolio(self, target_weights: Dict[str, float], dry_run: Optional[bool] = None):
        """重新平衡投资组合到目标权重"""
        raise NotImplementedError


class AlpacaBroker(BrokerageInterface):
    """Alpaca 纸交易实现（仅 paper-api）。"""

    def __init__(self):
        assert getattr(config, 'TRADING_ENV', 'PROD') == 'PROD', (
            'Danger: Attempting Alpaca trade while TRADING_ENV != PROD'
        )
        super().__init__('alpaca')
        self._setup_alpaca_client()

    def _setup_alpaca_client(self):
        """初始化Alpaca客户端"""
        try:
            tradeapi = importlib.import_module('alpaca_trade_api')
            base_url = _alpaca_base_url(self.credentials)
            self.api = tradeapi.REST(
                self.credentials.get('api_key'),
                self.credentials.get('secret_key'),
                base_url,
                api_version='v2'
            )
            print(f"✅ Alpaca Paper API 连接成功（{base_url}）")
        except ImportError:
            print("❌ 未安装 alpaca-trade-api，请运行：pip install alpaca-trade-api")
            self.api = None
        except Exception as e:
            print(f"❌ Alpaca API 连接失败：{e}")
            self.api = None

    def get_portfolio_value(self) -> float:
        if not self.api:
            print("❌ API未连接")
            return 0

        try:
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            print(f"✓ 投资组合总价值：${portfolio_value:,.2f}")
            return portfolio_value
        except Exception as e:
            print(f"❌ 获取投资组合价值失败：{e}")
            return 0

    def get_holdings(self) -> Dict[str, float]:
        if not self.api:
            return {}

        try:
            positions = self.api.list_positions()
            holdings = {pos.symbol: float(pos.qty) for pos in positions}
            print(f"✓ 当前持仓：{len(holdings)}只股票")
            return holdings
        except Exception as e:
            print(f"❌ 获取持仓失败：{e}")
            return {}

    def get_position_weights(self) -> Dict[str, float]:
        if not self.api:
            return {}
        try:
            positions = self.api.list_positions()
            mv = {pos.symbol: float(pos.market_value) for pos in positions}
            total = sum(abs(v) for v in mv.values())
            if total <= 1e-12:
                return {}
            return {k: v / total for k, v in mv.items()}
        except Exception as e:
            print(f'❌ 获取持仓权重失败：{e}')
            return {}

    def buy_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        if not self.api:
            print(f"⚠️ 模拟：买入 {quantity} 股 {ticker}")
            return True

        try:
            self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"✅ 买入订单已提交：{ticker} x {quantity}")
            return True
        except Exception as e:
            print(f"❌ 买入订单提交失败：{e}")
            return False

    def sell_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        if not self.api:
            print(f"⚠️ 模拟：卖出 {quantity} 股 {ticker}")
            return True

        try:
            self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"✅ 卖出订单已提交：{ticker} x {quantity}")
            return True
        except Exception as e:
            print(f"❌ 卖出订单提交失败：{e}")
            return False

    def rebalance_portfolio(self, target_weights: Dict[str, float], dry_run: Optional[bool] = None):
        if dry_run is None:
            dry_run = not bool(config.ENABLE_PAPER_TRADING)

        print("\n" + "="*70)
        print("执行投资组合重平衡（Alpaca Paper）")
        print("="*70)

        if not self.api:
            print("⚠️ API未连接，改为 Dry Run")
            dry_run = True

        holdings = self.get_holdings()
        portfolio_value = self.get_portfolio_value()
        current_w = self.get_position_weights()
        tw = _normalize_target_weights(target_weights)
        _print_rebalance_diff(current_w, tw, dry_run)

        if portfolio_value == 0 and not dry_run:
            print("❌ 无法获取投资组合价值")
            return False

        if dry_run:
            print('\n✓ Dry Run 完成：未向 Alpaca 提交任何订单')
            return True

        assert getattr(config, 'TRADING_ENV', 'PROD') == 'PROD', (
            'Danger: Attempting to trade in TEST environment!'
        )

        print("\n[2/3] 计算目标头寸...")
        target_positions: Dict[str, float] = {}
        for ticker, weight in tw.items():
            price = _last_tradeable_price(ticker)
            if price <= 0:
                print(f"  ⚠️ {ticker}: 无法取得有效价格，跳过")
                continue
            notional = portfolio_value * weight
            target_positions[ticker] = round(notional / price, 4)
            print(
                f"  {ticker}: 目标 {target_positions[ticker]} 股 "
                f"(权重 {weight * 100:.2f}%, 参考价 ${price:.2f})"
            )

        print("\n[3/3] 执行交易...")
        working = {k: float(v) for k, v in holdings.items()}

        for ticker in list(working.keys()):
            if ticker not in target_positions and working.get(ticker, 0) > 0:
                q = working[ticker]
                if self.sell_stock(ticker, q):
                    working[ticker] = 0.0

        for ticker, target_qty in target_positions.items():
            cur = working.get(ticker, 0.0)
            if target_qty < cur - 1e-6:
                sell_qty = cur - target_qty
                if self.sell_stock(ticker, sell_qty):
                    working[ticker] = working.get(ticker, 0.0) - sell_qty
            elif target_qty > cur + 1e-6:
                buy_qty = target_qty - cur
                if self.buy_stock(ticker, buy_qty):
                    working[ticker] = working.get(ticker, 0.0) + buy_qty
            else:
                print(f"  {ticker}: 不需要调整")

        print("\n✅ 重平衡完成")
        return True


class SimulatedBroker(BrokerageInterface):
    """
    本地模拟经纪商。

    默认从 results/live（或当前 RESULTS_DIR）的上一期持仓权重初始化，
    避免 Dry Run Diff 把满仓误判成空仓全 BUY。
    这反映的是「本地账本」，不是券商账户本身。
    """

    def __init__(
        self,
        current_weights: Optional[Dict[str, float]] = None,
        equity: Optional[float] = None,
        seed_from_live: bool = True,
    ):
        super().__init__('simulated')
        self.portfolio: Dict[str, float] = {}
        self.cash = float(equity if equity is not None else config.PAPER_INITIAL_CASH)
        self._book_weights: Dict[str, float] = {}
        self._equity = float(equity if equity is not None else config.PAPER_INITIAL_CASH)

        weights = current_weights
        if weights is None and seed_from_live:
            try:
                import portfolio_state
                _, weights = portfolio_state.load_previous_portfolio()
            except Exception:
                weights = {}

        if weights:
            self.seed_from_weights(weights, equity=self._equity)

    def seed_from_weights(self, weights: Dict[str, float], equity: Optional[float] = None):
        """按目标权重把本地账本初始化为满仓（现金≈0）。"""
        tw = _normalize_target_weights(weights)
        eq = float(equity if equity is not None else self._equity)
        self._equity = eq
        self._book_weights = dict(tw)
        self.portfolio = {}
        invested = 0.0
        for t, w in tw.items():
            px = _last_tradeable_price(t)
            if px <= 0:
                px = 100.0
            qty = (eq * w) / px
            self.portfolio[t] = qty
            invested += qty * px
        self.cash = max(0.0, eq - invested)
        print(
            f'✓ [模拟] 已用本地持仓账本初始化：{len(tw)} 只，名义权益 ${eq:,.2f}'
            f'（来源：RESULTS_DIR 上一期权重，非券商 API）'
        )

    def get_holdings(self) -> Dict[str, float]:
        return dict(self.portfolio)

    def get_position_weights(self) -> Dict[str, float]:
        if self._book_weights:
            return dict(self._book_weights)
        total = self.get_portfolio_value()
        if total <= 1e-12:
            return {}
        out = {}
        for t, q in self.portfolio.items():
            px = _last_tradeable_price(t) or 100.0
            out[t] = (q * px) / total
        return out

    def buy_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        px = price if price and price > 0 else _last_tradeable_price(ticker)
        if px <= 0:
            px = 100.0
        cost = quantity * px
        if cost > self.cash:
            print(f'❌ 现金不足，需要 ${cost:,.2f}，现有 ${self.cash:,.2f}')
            return False
        self.portfolio[ticker] = self.portfolio.get(ticker, 0) + quantity
        self.cash -= cost
        self._book_weights = {}  # 失效，改按市值重算
        print(f'✅ [模拟] 买入 {quantity} 股 {ticker} @ ${px:.2f}')
        return True

    def sell_stock(self, ticker: str, quantity: float, price: Optional[float] = None):
        px = price if price and price > 0 else _last_tradeable_price(ticker)
        if px <= 0:
            px = 100.0
        if ticker not in self.portfolio or self.portfolio[ticker] < quantity:
            print(f'❌ 持仓不足，无法卖出 {quantity} 股 {ticker}')
            return False
        proceeds = quantity * px
        self.portfolio[ticker] -= quantity
        if self.portfolio[ticker] == 0:
            del self.portfolio[ticker]
        self.cash += proceeds
        self._book_weights = {}
        print(f'✅ [模拟] 卖出 {quantity} 股 {ticker} @ ${px:.2f}')
        return True

    def get_portfolio_value(self) -> float:
        holdings_value = 0.0
        for t, q in self.portfolio.items():
            px = _last_tradeable_price(t)
            holdings_value += q * (px if px > 0 else 100.0)
        return float(self.cash + holdings_value)

    def rebalance_portfolio(self, target_weights: Dict[str, float], dry_run: Optional[bool] = None):
        if dry_run is None:
            dry_run = not bool(config.ENABLE_PAPER_TRADING)

        print('\n' + '=' * 70)
        print('[模拟] 执行投资组合重平衡（对照本地账本，非券商 API）')
        print('=' * 70)
        current_w = self.get_position_weights()
        tw = _normalize_target_weights(target_weights)
        _print_rebalance_diff(current_w, tw, dry_run)

        if dry_run:
            max_abs = 0.0
            for t in set(current_w) | set(tw):
                max_abs = max(max_abs, abs(tw.get(t, 0) - current_w.get(t, 0)))
            if max_abs < config.DEADBAND_THRESHOLD:
                print(
                    f'\n✓ Dry Run：相对本地账本最大偏离 {max_abs*100:.2f}% '
                    f'< Deadband {config.DEADBAND_THRESHOLD*100:.1f}%，建议无交易'
                )
            else:
                print(
                    f'\n✓ Dry Run：相对本地账本最大偏离 {max_abs*100:.2f}%。'
                    f'若你在券商 App 手动调仓，请对照【最终持仓】做多退少补'
                )
            return True

        holdings = self.get_holdings()
        total = self.get_portfolio_value()
        if total <= 0:
            print('❌ 组合价值为 0')
            return False

        target_positions = {}
        for ticker, weight in tw.items():
            px = _last_tradeable_price(ticker)
            if px <= 0:
                continue
            target_positions[ticker] = round(total * weight / px, 4)

        working = {k: float(v) for k, v in holdings.items()}
        for ticker in list(working.keys()):
            if ticker not in target_positions and working.get(ticker, 0) > 0:
                q = working[ticker]
                px = _last_tradeable_price(ticker) or 100.0
                if self.sell_stock(ticker, q, px):
                    working[ticker] = 0.0

        for ticker, target_qty in target_positions.items():
            cur = working.get(ticker, 0.0)
            px = _last_tradeable_price(ticker) or 100.0
            if target_qty < cur - 1e-6:
                if self.sell_stock(ticker, cur - target_qty, px):
                    working[ticker] = target_qty
            elif target_qty > cur + 1e-6:
                if self.buy_stock(ticker, target_qty - cur, px):
                    working[ticker] = target_qty

        self._book_weights = dict(tw)
        print('\n✅ [模拟] 重平衡完成')
        return True


def get_broker(broker_name: str = 'simulated') -> BrokerageInterface:
    """工厂方法：获取经纪商实例"""
    brokers = {
        'alpaca': AlpacaBroker,
        'simulated': SimulatedBroker,
    }

    if broker_name not in brokers:
        print(f"❌ 不支持的经纪商：{broker_name}")
        return SimulatedBroker()

    return brokers[broker_name]()


def execute_paper_rebalance(
    target_weights: Dict[str, float],
    dry_run: Optional[bool] = None,
    current_weights: Optional[Dict[str, float]] = None,
):
    """
    统一纸交易入口：
    - ENABLE_PAPER_TRADING=False → 强制 Dry Run（只打印 Diff）
    - simulated：用「调仓前」本地持仓初始化，避免空仓假 BUY
    - 优先用调用方传入的 current_weights（流水线开头读到的旧仓），
      避免 selection_results 已写入新权重后被误当成当前仓
    - alpaca 仅允许 TRADING_ENV=PROD + paper-api URL
    """
    if dry_run is None:
        dry_run = not bool(config.ENABLE_PAPER_TRADING)
    if config.PAPER_BROKER == 'alpaca' and not dry_run:
        assert config.TRADING_ENV == 'PROD', (
            'Danger: Attempting to trade in TEST environment!'
        )

    if config.PAPER_BROKER == 'alpaca':
        broker = AlpacaBroker()
    else:
        live_w = current_weights
        if live_w is None:
            try:
                import portfolio_state
                _, live_w = portfolio_state.load_previous_portfolio()
            except Exception:
                live_w = {}
        broker = SimulatedBroker(current_weights=live_w or {}, seed_from_live=False)

    return broker.rebalance_portfolio(target_weights, dry_run=dry_run)

def demo_rebalance():
    """演示重平衡流程"""
    print("="*70)
    print("投资组合重平衡演示（模拟模式）")
    print("="*70)

    # 使用模拟经纪商
    broker = get_broker('simulated')

    # 目标权重（来自选股算法）
    target_weights = {
        'GOOGL': 0.15,
        'NVDA': 0.15,
        'XOM': 0.15,
        'WMT': 0.15,
        'ABBV': 0.15,
        'GS': 0.13,
        'JNJ': 0.10,
        'TSLA': 0.02,
    }

    # 执行重平衡
    broker.rebalance_portfolio(target_weights)

if __name__ == '__main__':
    print("实盘交易接口说明：")
    print("="*70)
    print()
    print("支持的经纪商：")
    print("  1. Alpaca (美国股票，免佣金)")
    print("  2. Simulated (模拟，用于测试)")
    print()
    print("设置步骤：")
    print("  1. 注册 Alpaca 账户：https://alpaca.markets")
    print("  2. 获取 API 密钥")
    print("  3. 创建 credentials.json：")
    print("""
{
  "alpaca": {
    "api_key": "your_api_key",
    "secret_key": "your_secret_key",
    "base_url": "https://api.alpaca.markets"
  }
}
    """)
    print()
    print("  4. 运行重平衡：")
    print("     broker = get_broker('alpaca')")
    print("     broker.rebalance_portfolio(target_weights)")
    print()
    print("="*70)
    print()

    # 运行演示
    demo_rebalance()
