import logging
import os
import pickle
import re
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _find_sp500_constituents_table(soup):
    table = soup.find('table', {'id': 'constituents'})
    if table is not None:
        return table
    for tbl in soup.find_all('table'):
        row = tbl.find('tr')
        if row is None:
            continue
        head = row.get_text().lower()
        if 'symbol' in head and 'gics' in head:
            return tbl
    return None


def _normalize_ticker(sym: str) -> str:
    sym = re.sub(r'[^A-Za-z0-9.\-]', '', sym.strip())
    return sym.replace('.', '-') if sym else ''


def _parse_sp500_constituents_from_table(table):
    """解析维基成分表：返回 [(ticker, gics_sector), ...]"""
    rows = []
    for row in table.find_all('tr'):
        cells = row.find_all('td', recursive=False)
        if len(cells) < 3:
            continue
        sym = _normalize_ticker(cells[0].get_text(strip=True).split()[0])
        sector = cells[2].get_text(strip=True)
        if sym and sector:
            rows.append((sym, sector))
    return rows


def _fetch_sp500_table_html():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def get_sp500_constituents():
    """
    从 Wikipedia 获取 S&P 500 成分股及 GICS 行业。
    返回 (tickers, sector_dict)
    """
    print('正在获取 S&P 500 成分股及行业...')
    try:
        html = _fetch_sp500_table_html()
        soup = BeautifulSoup(html, 'html.parser')
        table = _find_sp500_constituents_table(soup)
        if table is None:
            print('警告：未找到成分股表格')
            return [], {}

        pairs = _parse_sp500_constituents_from_table(table)
        if not pairs:
            print('警告：表格中未解析到成分股')
            return [], {}

        tickers = [t for t, _ in pairs]
        sector_dict = dict(pairs)
        print(f'成功获取 {len(tickers)} 个 S&P 500 成分股（含 GICS 行业）')
        return tickers, sector_dict
    except Exception as e:
        print(f'获取 S&P 500 列表出错：{e}')
        return [], {}


def get_sp500_tickers():
    tickers, _ = get_sp500_constituents()
    return tickers


def _extract_close_series(df, ticker=None):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            close = df['Close']
            if ticker and ticker in close.columns:
                return close[ticker]
            if close.shape[1] == 1:
                return close.iloc[:, 0]
        return None
    if 'Close' in df.columns:
        return df['Close']
    if 'Adj Close' in df.columns:
        return df['Adj Close']
    return None


def _extract_ohlcv_pair(raw, ticker):
    """从 yfinance 下载结果中提取单票 Close / Volume。"""
    close = None
    volume = None
    try:
        if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            sub = raw[ticker]
            if 'Close' in sub.columns:
                close = sub['Close']
            if 'Volume' in sub.columns:
                volume = sub['Volume']
        else:
            close = _extract_close_series(raw, ticker)
            if isinstance(raw, pd.DataFrame) and 'Volume' in raw.columns:
                volume = raw['Volume']
    except Exception:
        return None, None
    return close, volume


def _sanitize_weekly_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 10:
        return pd.Series(dtype=float)
    # 单周暴涨暴跌（常见于未复权/公司行动）→ 剔除该点
    ret = s.pct_change().abs()
    if ret.max() > 3.0:
        s = s[ret.fillna(0) <= 3.0]
    return s


def _weekly_dollar_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """日成交额按周求和：sum(Close × Volume)。"""
    if close is None or volume is None:
        return pd.Series(dtype=float)
    c = close.astype(float)
    v = volume.astype(float).fillna(0.0)
    aligned = pd.concat([c, v], axis=1, join='inner')
    if aligned.empty or aligned.shape[1] < 2:
        return pd.Series(dtype=float)
    aligned.columns = ['Close', 'Volume']
    dollar = aligned['Close'] * aligned['Volume'].clip(lower=0)
    weekly = dollar.resample('W').sum()
    weekly = weekly.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return weekly


def fetch_weekly_data(tickers, start_date, end_date, progress=True):
    """批量下载日线并转周线：返回 (周收盘价, 周成交额)。"""
    print(f'正在获取 {len(tickers)} 只股票的周线数据 ({start_date.date()} 到 {end_date.date()})...')
    price_data = {}
    dollar_data = {}
    failed = set()
    chunk = config.DOWNLOAD_CHUNK_SIZE

    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        if progress:
            print(f'进度：{i}/{len(tickers)}')

        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                interval='1d',
                auto_adjust=True,
                threads=True,
                group_by='ticker',
            )
        except Exception:
            raw = None

        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            failed.update(batch)
            continue

        for ticker in batch:
            try:
                close, volume = _extract_ohlcv_pair(raw, ticker)
                if close is None or close.empty:
                    failed.add(ticker)
                    continue
                weekly = _sanitize_weekly_series(close.resample('W').last())
                if len(weekly) < 10:
                    failed.add(ticker)
                    continue
                price_data[ticker] = weekly
                dollar_data[ticker] = _weekly_dollar_volume(close, volume)
            except Exception:
                failed.add(ticker)

    if failed:
        failed_list = sorted(failed)
        logger.warning(
            '获取失败的股票数量：%d，失败名单：%s',
            len(failed_list),
            ', '.join(failed_list)
        )

    if not price_data:
        print(f'错误：没有成功获取任何股票数据（失败 {len(failed)}/{len(tickers)}）')
        return pd.DataFrame(), pd.DataFrame()

    prices = pd.DataFrame(price_data).sort_index()
    dollar_vol = pd.DataFrame(dollar_data).reindex(index=prices.index).fillna(0.0)
    dollar_vol = dollar_vol.reindex(columns=prices.columns).fillna(0.0)
    print(f'成功获取 {len(prices.columns)} 只股票的周线数据，共 {len(prices)} 周（含成交额）')
    return prices, dollar_vol


def get_sector_for_ticker(ticker, sector_dict=None):
    if sector_dict and ticker in sector_dict:
        return sector_dict[ticker]
    return config.SECTOR_MAPPING.get(ticker, 'Unknown')


def cache_data(data, filename):
    filepath = os.path.join(config.CACHE_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f'数据已缓存到 {filepath}')


def load_cached_data(filename):
    filepath = os.path.join(config.CACHE_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                print(f'从缓存加载 {filepath}')
                return pickle.load(f)
        except Exception as e:
            print(f'缓存读取失败，将重新获取数据：{e}')
    return None


def _cache_key(prefix, start_date, end_date):
    return f'{prefix}_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.pkl'


def prepare_data():
    """主数据准备：成分股、周线价格、周成交额、行业（GICS 来自维基）。

    返回 (tickers, weekly_prices, sector_dict, weekly_dollar_volume)
    """
    start_date, end_date = config.get_date_range()

    tickers_file = _cache_key('sp500_tickers', start_date, end_date)
    sectors_file = _cache_key('sp500_sectors', start_date, end_date)
    weekly_cache = _cache_key('weekly_data', start_date, end_date)
    dollar_cache = _cache_key('weekly_dollar_volume', start_date, end_date)

    tickers = load_cached_data(tickers_file)
    sector_dict = load_cached_data(sectors_file)

    if not tickers or not sector_dict:
        tickers, sector_dict = get_sp500_constituents()
        if not tickers:
            return None, None, None, None
        cache_data(tickers, tickers_file)
        cache_data(sector_dict, sectors_file)
    else:
        print(f'使用缓存的 S&P 500 名单与行业，共 {len(sector_dict)} 只')

    weekly_data = load_cached_data(weekly_cache)
    weekly_dollar_volume = load_cached_data(dollar_cache)

    need_refetch = (
        weekly_data is None
        or getattr(weekly_data, 'empty', True)
        or weekly_dollar_volume is None
        or getattr(weekly_dollar_volume, 'empty', True)
    )
    if need_refetch:
        if weekly_data is not None and (weekly_dollar_volume is None or getattr(weekly_dollar_volume, 'empty', True)):
            print('检测到旧价格缓存缺少成交额，重新下载以启用流动性过滤...')
        weekly_data, weekly_dollar_volume = fetch_weekly_data(tickers, start_date, end_date)
        if not weekly_data.empty:
            cache_data(weekly_data, weekly_cache)
            cache_data(weekly_dollar_volume, dollar_cache)

    # 仅保留有行情的股票行业；成交额列对齐价格列
    common = [t for t in weekly_data.columns if t in weekly_dollar_volume.columns]
    weekly_data = weekly_data[common]
    weekly_dollar_volume = weekly_dollar_volume[common].reindex(index=weekly_data.index).fillna(0.0)
    sector_dict = {t: get_sector_for_ticker(t, sector_dict) for t in weekly_data.columns}

    return list(weekly_data.columns), weekly_data, sector_dict, weekly_dollar_volume


if __name__ == '__main__':
    tickers, weekly_data, sectors, dollar_vol = prepare_data()
    if weekly_data is not None:
        print(f'\n成功获取 {len(weekly_data.columns)} 只股票的数据')
        print(f'数据形状：{weekly_data.shape}')
        print(f'时间范围：{weekly_data.index[0]} 到 {weekly_data.index[-1]}')
        if dollar_vol is not None and not dollar_vol.empty:
            avg20 = dollar_vol.tail(20).mean()
            print(f'近20周均成交额中位数：${avg20.median():,.0f}')
