class MyStrategy(Strategy):
    """
    Adaptive RSI + MACD + Trend strategy using a voting system.
    - 2 out of 3 indicators must agree to enter a trade
    - Dynamic RSI thresholds based on volatility
    - 200 EMA trend filter (more reliable on 15-min bars)
    - ATR-based adaptive position sizing
    - Minimum holding period to avoid whipsaws
    - ATR-based stop loss
    """

    def __init__(self, position_size: float = 10.0):
        self.position_size = position_size
        self.rsi_period = 14
        self.base_rsi_buy = 45
        self.base_rsi_sell = 55
        self.min_hold = 5          # minimum bars before switching direction
        self.atr_stop_mult = 2.0   # stop loss = 2x ATR away from entry

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- RSI ---
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # --- MACD ---
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # --- Volatility + Dynamic RSI thresholds ---
        df["returns"] = df["Close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        vol_mean = df["volatility"].rolling(100).mean()
        vol_scalar = (df["volatility"] / vol_mean.replace(0, np.nan)).fillna(1.0)
        df["rsi_buy"] = (self.base_rsi_buy - (vol_scalar * 5)).clip(30, 50)
        df["rsi_sell"] = (self.base_rsi_sell + (vol_scalar * 5)).clip(50, 70)

        # --- 200 EMA Trend Filter (more reliable on 15-min bars) ---
        df["ema_trend"] = df["Close"].ewm(span=200, adjust=False).mean()
        df["trend"] = np.where(df["Close"] > df["ema_trend"], 1, -1)

        # --- ATR ---
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        df["atr"] = (
            pd.concat([high_low, high_close, low_close], axis=1)
            .max(axis=1)
            .rolling(14)
            .mean()
        )

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        # --- Voting system ---
        df["rsi_vote"] = np.where(df["rsi"] < df["rsi_buy"], 1,
                         np.where(df["rsi"] > df["rsi_sell"], -1, 0))
        df["macd_vote"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
        df["trend_vote"] = df["trend"]

        df["score"] = df["rsi_vote"] + df["macd_vote"] + df["trend_vote"]

        df.loc[df["score"] >= 2, "signal"] = 1
        df.loc[df["score"] <= -2, "signal"] = -1

        # --- Cooldown: enforce minimum holding period ---
        signal_out = df["signal"].copy()
        last_signal_idx = -self.min_hold
        for i in range(len(df)):
            if df["signal"].iloc[i] != 0:
                if i - last_signal_idx >= self.min_hold:
                    last_signal_idx = i
                else:
                    signal_out.iloc[i] = 0  # too soon, suppress signal
        df["signal"] = signal_out

        # --- Position (hold between signals) ---
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        # --- ATR stop loss ---
        stop_distance = df["atr"] * self.atr_stop_mult
        df["stop_loss"] = df["Close"] - (stop_distance * df["position"])

        # --- ATR adaptive position sizing ---
        atr_scalar = (df["atr"].mean() / df["atr"].clip(lower=0.01)).fillna(1.0)
        df["target_qty"] = (
            df["position"].abs() * self.position_size * atr_scalar
        ).clip(1, 50)

        # --- Debug ---
        print(f"RSI bullish votes:  {(df['rsi_vote'] == 1).sum()}")
        print(f"RSI bearish votes:  {(df['rsi_vote'] == -1).sum()}")
        print(f"MACD bullish votes: {(df['macd_vote'] == 1).sum()}")
        print(f"MACD bearish votes: {(df['macd_vote'] == -1).sum()}")
        print(f"Trend up:           {(df['trend_vote'] == 1).sum()}")
        print(f"Trend down:         {(df['trend_vote'] == -1).sum()}")
        print(f"Long signals:       {(df['signal'] == 1).sum()}")
        print(f"Short signals:      {(df['signal'] == -1).sum()}")

        return df
    