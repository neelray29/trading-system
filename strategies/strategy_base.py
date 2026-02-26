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
    Regime-Switching Strategy: Mean Reversion + Trend Pullback.

    Automatically detects market regime using ADX and switches behavior:

      RANGING REGIME  (ADX < adx_range_max):
        - Pure mean reversion mode
        - Trades both long and short
        - Enters when Z-score deviates beyond entry_z in either direction
        - Bets that price will snap back to the mean

      TRANSITION ZONE (adx_range_max <= ADX <= adx_trend_min):
        - Sit out — market is ambiguous
        - No new entries

      TRENDING REGIME (ADX > adx_trend_min):
        - Trend pullback mode
        - Only trades IN the direction of the long-term EMA trend
        - Enters on pullbacks toward the mean (Z-score entry timing)
        - Bets that price will resume the trend after a pullback

    Exits (same for both regimes):
      1. Take profit  - Z-score reverts through exit_z
      2. Trend exit   - Price crosses trend EMA (trending mode only)
      3. Hard stop    - Z-score hits stop_z
      4. Time stop    - Open more than max_hold_periods bars
      5. Dollar stop  - Trade down more than max_loss_per_trade dollars

    Recommended parameters for JPM 15-min:
        trend_window        = 50
        lookback            = 30
        entry_z             = 1.8
        exit_z              = 0.4
        stop_z              = 2.8
        max_hold_periods    = 8
        max_loss_per_trade  = 75.0
        base_position       = 10.0
        max_position        = 15.0
        adx_range_max       = 20.0   (below this = ranging, use mean reversion)
        adx_trend_min       = 28.0   (above this = trending, use trend pullback)

    Recommended parameters for JPM 5-min:
        trend_window        = 100
        lookback            = 40
        entry_z             = 2.0
        exit_z              = 0.4
        stop_z              = 2.8
        max_hold_periods    = 10
        max_loss_per_trade  = 50.0
        base_position       = 10.0
        max_position        = 15.0
        adx_range_max       = 20.0
        adx_trend_min       = 28.0
    """

    def __init__(
        self,
        trend_window: int = 50,
        lookback: int = 30,
        entry_z: float = 1.8,
        exit_z: float = 0.4,
        stop_z: float = 2.8,
        max_hold_periods: int = 8,
        max_loss_per_trade: float = 75.0,
        base_position: float = 10.0,
        max_position: float = 15.0,
        adx_range_max: float = 20.0,
        adx_trend_min: float = 28.0,
        adx_window: int = 14,
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if trend_window <= lookback:
            raise ValueError("trend_window must be greater than lookback.")
        if not (0 < exit_z < entry_z < stop_z):
            raise ValueError("Must satisfy: 0 < exit_z < entry_z < stop_z.")
        if base_position <= 0 or max_position < base_position:
            raise ValueError("base_position must be positive and <= max_position.")
        if max_loss_per_trade <= 0:
            raise ValueError("max_loss_per_trade must be positive.")
        if adx_range_max >= adx_trend_min:
            raise ValueError("adx_range_max must be less than adx_trend_min.")

        self.trend_window = trend_window
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_hold_periods = max_hold_periods
        self.max_loss_per_trade = max_loss_per_trade
        self.base_position = base_position
        self.max_position = max_position
        self.adx_range_max = adx_range_max
        self.adx_trend_min = adx_trend_min
        self.adx_window = adx_window

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Long-term EMA for trend direction
        df["trend_ema"] = df["Close"].ewm(span=self.trend_window, adjust=False).mean()
        df["trend_dir"] = np.where(df["Close"] > df["trend_ema"], 1, -1)

        # Short-term Z-score for entry timing
        df["mr_mean"] = df["Close"].rolling(self.lookback).mean()
        df["mr_std"]  = df["Close"].rolling(self.lookback).std()
        df["z_score"] = (df["Close"] - df["mr_mean"]) / df["mr_std"].replace(0, np.nan)

        # ADX for regime detection
        df = self._add_adx(df)

        # Regime label for clarity
        df["regime"] = "transition"
        df.loc[df["adx"] < self.adx_range_max, "regime"]  = "ranging"
        df.loc[df["adx"] > self.adx_trend_min, "regime"]  = "trending"

        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["High"], df["Low"], df["Close"]
        w = self.adx_window

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        dm_plus  = ((high - high.shift(1)).clip(lower=0)
                    .where((high - high.shift(1)) > (low.shift(1) - low), 0.0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where((low.shift(1) - low) > (high - high.shift(1)), 0.0))

        atr      = tr.ewm(com=w - 1, min_periods=w).mean()
        di_plus  = 100 * dm_plus.ewm(com=w - 1, min_periods=w).mean() / atr.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(com=w - 1, min_periods=w).mean() / atr.replace(0, np.nan)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        df["adx"] = dx.ewm(com=w - 1, min_periods=w).mean().fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"]     = 0
        df["position"]   = 0.0
        df["target_qty"] = 0.0

        current_position = 0
        bars_in_trade    = 0
        entry_price      = 0.0

        positions   = []
        signals     = []
        target_qtys = []

        for i in range(len(df)):
            z             = df["z_score"].iloc[i]
            adx_val       = df["adx"].iloc[i]
            trend_dir     = df["trend_dir"].iloc[i]
            current_price = df["Close"].iloc[i]
            trend_ema     = df["trend_ema"].iloc[i]
            regime        = df["regime"].iloc[i]
            signal        = 0

            # Warm-up guard
            warmed_up = i >= self.trend_window

            # --- Manage open position ---
            if current_position != 0:
                bars_in_trade += 1

                # Exit 1: Take profit
                reverted = (
                    (current_position ==  1 and z >= -self.exit_z) or
                    (current_position == -1 and z <=  self.exit_z)
                )

                # Exit 2: Trend exit (only meaningful in trending regime)
                trend_exit = (
                    regime == "trending" and (
                        (current_position ==  1 and current_price < trend_ema) or
                        (current_position == -1 and current_price > trend_ema)
                    )
                )

                # Exit 3: Hard Z-score stop
                hard_stop = (
                    (current_position ==  1 and z <= -self.stop_z) or
                    (current_position == -1 and z >=  self.stop_z)
                )

                # Exit 4: Time stop
                time_stop = bars_in_trade >= self.max_hold_periods

                # Exit 5: Dollar stop
                trade_pnl   = (current_price - entry_price) * current_position * self.base_position
                dollar_stop = trade_pnl <= -self.max_loss_per_trade

                if reverted or trend_exit or hard_stop or time_stop or dollar_stop:
                    signal           = -current_position
                    current_position = 0
                    bars_in_trade    = 0
                    entry_price      = 0.0

            # --- Look for new entry ---
            if current_position == 0 and warmed_up and not np.isnan(z):

                # RANGING REGIME: pure mean reversion, trade both directions
                if regime == "ranging":
                    if z <= -self.entry_z:
                        signal           = 1
                        current_position = 1
                        bars_in_trade    = 0
                        entry_price      = current_price
                    elif z >= self.entry_z:
                        signal           = -1
                        current_position = -1
                        bars_in_trade    = 0
                        entry_price      = current_price

                # TRENDING REGIME: only trade in direction of trend
                elif regime == "trending":
                    if trend_dir == 1 and z <= -self.entry_z:
                        signal           = 1
                        current_position = 1
                        bars_in_trade    = 0
                        entry_price      = current_price
                    elif trend_dir == -1 and z >= self.entry_z:
                        signal           = -1
                        current_position = -1
                        bars_in_trade    = 0
                        entry_price      = current_price

                # TRANSITION ZONE: sit out
                # (no entry logic needed, just skip)

            # Adaptive sizing: scales with Z-score magnitude
            if current_position != 0 and not np.isnan(z):
                scale = abs(z) / self.entry_z
                qty   = min(self.base_position * scale, self.max_position)
            else:
                qty = 0.0

            positions.append(current_position)
            signals.append(signal)
            target_qtys.append(qty)

        df["position"]   = positions
        df["signal"]     = signals
        df["target_qty"] = target_qtys

        return df