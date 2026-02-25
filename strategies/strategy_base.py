"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        print("1")
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##




    
class MyStrategy(Strategy):
    """
    Z-Score Mean Reversion Strategy.

    Core logic:
      1. MEAN          - Rolling SMA as the reference price level
      2. DEVIATION     - Z-score (standard deviations from mean) to measure how far price has strayed
      3. ENTRY         - Enter long when Z-score < -entry_z (oversold), short when Z-score > +entry_z (overbought)
      4. EXIT/PROFIT   - Close position when Z-score crosses back through exit_z toward zero (partial reversion)
      5. STOP LOSS     - Hard stop at stop_z standard deviations AND time-based stop after max_hold_periods bars
      6. ASSET FIT     - Best on range-bound, liquid instruments; avoid during strong trend regimes (ADX filter)
      7. LOOKBACK      - Configurable window (default 30); key parameter to tune per instrument
      8. POSITION SIZE - Scales position size UP as Z-score grows (fade into the move gradually)

    Parameters:
        lookback        : int   - Rolling window for mean and std (principle 7). Default 30.
        entry_z         : float - Z-score threshold to enter a trade (principle 3). Default 2.0.
        exit_z          : float - Z-score threshold to exit / take profit (principle 4). Default 0.5.
        stop_z          : float - Z-score hard stop loss threshold (principle 5). Default 3.5.
        max_hold_periods: int   - Time-based stop: exit if not reverted within N bars (principle 5). Default 20.
        base_position   : float - Base number of shares/units at exactly entry_z (principle 8). Default 10.0.
        max_position    : float - Maximum position cap to prevent runaway scaling (principle 8). Default 30.0.
        adx_filter      : bool  - If True, skip entries when trend is strong (ADX > adx_threshold) (principle 6).
        adx_threshold   : float - ADX level above which we consider the market trending. Default 25.0.
        adx_window      : int   - Window for ADX calculation. Default 14.

    Usage:
        python run_backtest.py --csv data/AAPL_15Min_stock_alpaca_clean.csv --strategy ZScoreMeanReversionStrategy --plot
        python run_live.py --symbol AAPL --strategy ZScoreMeanReversionStrategy --live
    """

    def __init__(
        self,
        lookback: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        max_hold_periods: int = 20,
        base_position: float = 10.0,
        max_position: float = 30.0,
        adx_filter: bool = True,
        adx_threshold: float = 25.0,
        adx_window: int = 14,
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if not (0 < exit_z < entry_z < stop_z):
            raise ValueError("Must satisfy: 0 < exit_z < entry_z < stop_z.")
        if base_position <= 0 or max_position < base_position:
            raise ValueError("base_position must be positive and <= max_position.")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_hold_periods = max_hold_periods
        self.base_position = base_position
        self.max_position = max_position
        self.adx_filter = adx_filter
        self.adx_threshold = adx_threshold
        self.adx_window = adx_window

    # ------------------------------------------------------------------
    # Principle 1 & 2: Mean + Deviation via Z-score
    # ------------------------------------------------------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rolling mean (the "reference" level) and standard deviation
        df["mr_mean"] = df["Close"].rolling(self.lookback).mean()
        df["mr_std"] = df["Close"].rolling(self.lookback).std()

        # Z-score: how many standard deviations is price from its mean?
        df["z_score"] = (df["Close"] - df["mr_mean"]) / df["mr_std"].replace(0, np.nan)

        # Principle 6: ADX trend filter — skip entries in trending markets
        if self.adx_filter:
            df = self._add_adx(df)
        else:
            df["adx"] = 0.0  # Always pass filter if disabled

        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX) to detect trending markets."""
        high, low, close = df["High"], df["Low"], df["Close"]
        w = self.adx_window

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        # Directional movements
        dm_plus  = ((high - high.shift(1)).clip(lower=0)
                    .where((high - high.shift(1)) > (low.shift(1) - low), 0.0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where((low.shift(1) - low) > (high - high.shift(1)), 0.0))

        # Smoothed with Wilder's EMA (com = window - 1)
        atr      = tr.ewm(com=w - 1, min_periods=w).mean()
        di_plus  = 100 * dm_plus.ewm(com=w - 1, min_periods=w).mean() / atr.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(com=w - 1, min_periods=w).mean() / atr.replace(0, np.nan)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        df["adx"] = dx.ewm(com=w - 1, min_periods=w).mean().fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Principles 3, 4, 5, 8: Entry, Exit, Stop, and Adaptive Sizing
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"]   = 0
        df["position"] = 0.0
        df["target_qty"] = 0.0

        # Track current position state across rows
        current_position = 0       # +1 long, -1 short, 0 flat
        bars_in_trade    = 0       # time-based stop counter

        positions  = []
        signals    = []
        target_qtys = []

        for i in range(len(df)):
            z       = df["z_score"].iloc[i]
            adx_val = df["adx"].iloc[i]
            signal  = 0

            # --- Manage open position ---
            if current_position != 0:
                bars_in_trade += 1

                # Principle 4: Take profit — price has reverted toward the mean
                reverted = (
                    (current_position ==  1 and z >= -self.exit_z) or
                    (current_position == -1 and z <=  self.exit_z)
                )
                # Principle 5a: Hard stop — Z-score moved further against us
                hard_stop = (
                    (current_position ==  1 and z <= -self.stop_z) or
                    (current_position == -1 and z >=  self.stop_z)
                )
                # Principle 5b: Time-based stop
                time_stop = bars_in_trade >= self.max_hold_periods

                if reverted or hard_stop or time_stop:
                    signal = -current_position   # close the position
                    current_position = 0
                    bars_in_trade = 0

            # --- Look for new entry (only when flat) ---
            if current_position == 0:
                # Principle 6: Skip entry if market is trending strongly
                trending = self.adx_filter and (adx_val > self.adx_threshold)

                if not trending and not np.isnan(z):
                    # Principle 3: Enter when Z-score exceeds entry threshold
                    if z <= -self.entry_z:
                        signal = 1                # price too low → buy
                        current_position = 1
                        bars_in_trade = 0
                    elif z >= self.entry_z:
                        signal = -1               # price too high → sell short
                        current_position = -1
                        bars_in_trade = 0

            # Principle 8: Scale position size with |Z-score| (fade into the move)
            if current_position != 0 and not np.isnan(z):
                scale = abs(z) / self.entry_z          # 1.0 at entry_z, grows beyond
                qty = min(self.base_position * scale, self.max_position)
            else:
                qty = 0.0

            positions.append(current_position)
            signals.append(signal)
            target_qtys.append(qty)

        df["position"]   = positions
        df["signal"]     = signals
        df["target_qty"] = target_qtys

        return df