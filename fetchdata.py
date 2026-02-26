from pipeline.alpaca import save_bars, clean_market_data, get_rest, _normalize_bars, _parse_timeframe, _to_rfc3339
import pandas as pd

def fetch_bars_chunked(symbol, timeframe='1Min', days=90):
    api = get_rest()
    tf = _parse_timeframe(timeframe)
    all_dfs = []
    
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    
    current_end = end
    while current_end > start:
        current_start = max(current_end - pd.Timedelta(days=7), start)
        bars = api.get_bars(
            symbol, tf,
            start=_to_rfc3339(current_start),
            end=_to_rfc3339(current_end),
            limit=10000,
            feed='iex'
        ).df
        df = _normalize_bars(bars, symbol)
        if not df.empty:
            all_dfs.append(df)
        current_end = current_start

    if not all_dfs:
        raise ValueError(f"No data returned for {symbol}")
    
    result = pd.concat(all_dfs).drop_duplicates().sort_values('Datetime')
    return result

symbol = 'SPY'
timeframe = '1hour'

raw_df = fetch_bars_chunked(symbol, timeframe=timeframe, days=180)
raw_path = save_bars(raw_df, symbol, timeframe, asset_class='stock')
clean_path = clean_market_data(raw_path)
print("Raw:", raw_path)
print("Clean:", clean_path)