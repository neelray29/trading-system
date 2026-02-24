from pipeline.alpaca import fetch_stock_bars, save_bars, clean_market_data

symbol = 'TSLA'
timeframe = '5Min'
limit = 80000

raw_df = fetch_stock_bars(symbol, timeframe=timeframe, limit=limit)
raw_path = save_bars(raw_df, symbol, timeframe, asset_class='stock')
clean_path = clean_market_data(raw_path)

print("Raw:", raw_path)
print("Clean:", clean_path)