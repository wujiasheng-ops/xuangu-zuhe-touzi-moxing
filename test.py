import yfinance as yf
import pandas as pd
import numpy as np
# 如果这里还有红线，直接点运行，有时候运行能通但编辑器反应慢
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

print("🛠️ 正在初始化... 正在从维基百科实时抓取标普500名单...")

# 1. 抓取名单并清洗
try:
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = df['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
    norm_sym = df['Symbol'].astype(str).str.replace('.', '-', regex=False)
    sector_map = df.assign(_Sym=norm_sym).set_index('_Sym')['GICS Sector'].to_dict()
except Exception:
    print("网络请求受阻，使用备选核心池...")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "PG", "V", "XOM", "LLY"]
    sector_map = {}

# 2. 下载数据 (先用前 30 只测试，确保速度)
print("📈 正在获取 6 年 Weekly 数据...")
data = yf.download(tickers[:30], period="6y", interval="1wk", auto_adjust=True)

# 兼容新版 yfinance 的数据结构
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Close']
else:
    prices = data

# 3. 计算 12-1 动量并选出 10 只
returns = prices.pct_change()
# 简单的动量：过去 12 个月的累积收益
mom = (prices.shift(4) / prices.shift(52)) - 1
top_10 = mom.iloc[-1].dropna().nlargest(10).index.tolist()
final_prices = prices[top_10].dropna()

# 4. 你的核心观点：相关性与权重优化
print(f"⚖️ 正在为这 10 只股票优化权重 (15% 上限): {top_10}")
mu = expected_returns.mean_historical_return(final_prices, frequency=52)
S = risk_models.sample_cov(final_prices, frequency=52)

ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: w <= 0.15) # 单票上限 15%
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\n" + "⭐" * 20 + " 结果输出 " + "⭐" * 20)
for t, w in cleaned_weights.items():
    if w > 0:
        s = sector_map.get(t, "其他")
        print(f"股票: {t:6} | 权重: {w:>7.2%} | 行业: {s}")

print("-" * 50)
ef.portfolio_performance(verbose=True)