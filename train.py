from __future__ import annotations

"""
train.py

LuxAlgo-inspired strategy layer for a Karpathy-style autoresearch loop.

Important design choice:
- This file is the single editable strategy surface.
- It does NOT attempt to fully recreate the entire LuxAlgo Pine indicator.
- It assumes prepare.py or upstream data preparation provides the structural
  objects needed by the strategy on each bar.

Expected per-bar inputs
-----------------------
The strategy expects a pandas DataFrame with, at minimum, these columns:

Required market columns:
- open
- high
- low
- close

Required strategy-context columns produced upstream:
- bullish_internal_ob_high
- bullish_internal_ob_low
- bullish_internal_ob_active
- bearish_internal_ob_high
- bearish_internal_ob_low
- bearish_internal_ob_active
- nearest_weak_high_price
- nearest_weak_high_exists
- nearest_weak_low_price
- nearest_weak_low_exists
- bullish_fvg_nearby
- bearish_fvg_nearby

Optional columns:
- timestamp
- symbol

Strategy summary
----------------
Entries are driven purely by MACD histogram momentum:
- Long: MACD histogram crosses above 0 (bullish momentum)
- Short: MACD histogram crosses below 0 (bearish momentum)

Take profit at ±3.5% with trailing stop activation at 1.3% profit.
Position stops use real order block levels from prepare.py when available,
falling back to percentage-based stops otherwise.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StrategyConfig:
    rsi_length: int = 14
    take_profit_pct: float = 0.035
    short_take_profit_pct: Optional[float] = 0.035
    require_fvg_confirmation: bool = False
    entry_on_close: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True
    min_ob_stop_distance_pct: float = 0.005  # OB must be ≥0.5% away to apply stop
    # Trailing stop params
    trailing_stop_enabled: bool = True
    trailing_trigger_pct: float = (
        0.013  # activate trailing stop once price moves 1.3% in profit
    )
    trailing_distance_pct: float = (
        0.002  # symmetric 0.2% trailing distance for both sides
    )


@dataclass
class Position:
    side: str  # "long" or "short"
    entry_index: int
    entry_price: float
    ob_high: float
    ob_low: float
    entry_reason: str
    peak_price: float  # highest price for longs, lowest for shorts
    trailing_activated: bool = False


@dataclass
class Trade:
    side: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    entry_reason: str


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "bullish_internal_ob_high",
    "bullish_internal_ob_low",
    "bullish_internal_ob_active",
    "bearish_internal_ob_high",
    "bearish_internal_ob_low",
    "bearish_internal_ob_active",
    "nearest_weak_high_exists",
    "nearest_weak_low_exists",
    "bullish_fvg_nearby",
    "bearish_fvg_nearby",
}


def validate_input_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Input DataFrame is missing required columns: " + ", ".join(missing)
        )


# -----------------------------
# Indicator helpers
# -----------------------------


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# -----------------------------
# Core strategy predicates
# -----------------------------


def valid_long_signal(row: pd.Series, config: StrategyConfig, prev_macd_hist: Optional[float]) -> bool:
    """Long signal when MACD histogram crosses above 0 (was negative, now positive)."""
    if not config.allow_longs:
        return False
    if "macd_hist" not in row.index or pd.isna(row.get("macd_hist")):
        return False
    
    current_hist = float(row["macd_hist"])
    # Entry on bullish MACD crossover: previous bar was <= 0, current bar is > 0
    if prev_macd_hist is not None and prev_macd_hist <= 0 and current_hist > 0:
        return True
    # Also allow entry if histogram is already positive (momentum is bullish)
    if current_hist > 0:
        return True
    return False


def valid_short_signal(row: pd.Series, config: StrategyConfig, prev_macd_hist: Optional[float]) -> bool:
    """Short signal when MACD histogram crosses below 0 (was positive, now negative)."""
    if not config.allow_shorts:
        return False
    if "macd_hist" not in row.index or pd.isna(row.get("macd_hist")):
        return False
    
    current_hist = float(row["macd_hist"])
    # Entry on bearish MACD crossover: previous bar was >= 0, current bar is < 0
    if prev_macd_hist is not None and prev_macd_hist >= 0 and current_hist < 0:
        return True
    # Also allow entry if histogram is already negative (momentum is bearish)
    if current_hist < 0:
        return True
    return False


# -----------------------------
# Position management
# -----------------------------


def long_take_profit(entry_price: float, config: StrategyConfig) -> float:
    return entry_price * (1.0 + config.take_profit_pct)


def short_take_profit(entry_price: float, config: StrategyConfig) -> float:
    pct = (
        config.short_take_profit_pct
        if config.short_take_profit_pct is not None
        else config.take_profit_pct
    )
    return entry_price * (1.0 - pct)


def ob_stop_distance_pct(entry_price: float, ob_price: float, side: str) -> float:
    """Return the percentage distance from entry to the OB level."""
    return abs(entry_price - ob_price) / entry_price


def get_ob_levels(row: pd.Series, side: str) -> tuple[float, float]:
    """
    Get order block high/low levels from prepare.py data.
    Returns (ob_high, ob_low) for the position side.
    Falls back to entry-based percentages if no valid OB available.
    """
    if side == "long":
        ob_active = bool(row.get("bullish_internal_ob_active", False))
        ob_low = row.get("bullish_internal_ob_low")
        ob_high = row.get("bullish_internal_ob_high")
        if ob_active and pd.notna(ob_low) and pd.notna(ob_high):
            return float(ob_high), float(ob_low)
    else:  # short
        ob_active = bool(row.get("bearish_internal_ob_active", False))
        ob_low = row.get("bearish_internal_ob_low")
        ob_high = row.get("bearish_internal_ob_high")
        if ob_active and pd.notna(ob_low) and pd.notna(ob_high):
            return float(ob_high), float(ob_low)
    
    # Fallback to entry-based bands
    entry_price = float(row["close"])
    return entry_price * 1.01, entry_price * 0.99


def should_exit_position(
    position: Position, row: pd.Series, config: StrategyConfig
) -> Optional[tuple[float, str]]:
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])

    if position.side == "long":
        tp = long_take_profit(position.entry_price, config)
        if high >= tp:
            return tp, "take_profit"

        # Update peak price for trailing stop
        if high > position.peak_price:
            position.peak_price = high

        # Check trailing stop
        if config.trailing_stop_enabled:
            profit_from_entry = (
                position.peak_price - position.entry_price
            ) / position.entry_price
            if profit_from_entry >= config.trailing_trigger_pct:
                position.trailing_activated = True
                trailing_stop_price = position.peak_price * (
                    1.0 - config.trailing_distance_pct
                )
                if close <= trailing_stop_price:
                    return trailing_stop_price, "trailing_stop"

        # OB stop only if far enough from entry - use real OB levels
        ob_dist_pct = ob_stop_distance_pct(
            position.entry_price, position.ob_low, "long"
        )
        if ob_dist_pct >= config.min_ob_stop_distance_pct and close < position.ob_low:
            return close, "close_below_bullish_ob_low"
        return None

    if position.side == "short":
        tp = short_take_profit(position.entry_price, config)
        if low <= tp:
            return tp, "take_profit"

        # Update peak (lowest) price for trailing stop
        if low < position.peak_price:
            position.peak_price = low

        # Check trailing stop
        if config.trailing_stop_enabled:
            profit_from_entry = (
                position.entry_price - position.peak_price
            ) / position.entry_price
            if profit_from_entry >= config.trailing_trigger_pct:
                position.trailing_activated = True
                trailing_stop_price = position.peak_price * (
                    1.0 + config.trailing_distance_pct
                )
                if close >= trailing_stop_price:
                    return trailing_stop_price, "trailing_stop"

        # OB stop only if far enough from entry - use real OB levels
        ob_dist_pct = ob_stop_distance_pct(
            position.entry_price, position.ob_high, "short"
        )
        if ob_dist_pct >= config.min_ob_stop_distance_pct and close > position.ob_high:
            return close, "close_above_bearish_ob_high"
        return None

    raise ValueError(f"Unsupported position side: {position.side}")


# -----------------------------
# Backtest loop
# -----------------------------


def run_strategy(
    df: pd.DataFrame, config: Optional[StrategyConfig] = None
) -> Dict[str, Any]:
    config = config or StrategyConfig()
    validate_input_frame(df)

    working = df.copy()
    working["rsi"] = compute_rsi(working["close"], length=config.rsi_length)
    # MACD histogram
    ema12 = working["close"].ewm(span=12, adjust=False).mean()
    ema26 = working["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    working["macd_hist"] = macd - signal

    position: Optional[Position] = None
    trades: List[Trade] = []

    long_signals: List[bool] = []
    short_signals: List[bool] = []
    position_side: List[Optional[str]] = []

    prev_macd_hist: Optional[float] = None

    for idx, row in working.iterrows():
        current_macd_hist = float(row["macd_hist"]) if pd.notna(row.get("macd_hist")) else None
        
        long_signal = valid_long_signal(row, config, prev_macd_hist)
        short_signal = valid_short_signal(row, config, prev_macd_hist)

        long_signals.append(long_signal)
        short_signals.append(short_signal)

        # No simultaneous long/short entry on the same bar.
        if long_signal and short_signal:
            long_signal = False
            short_signal = False

        if position is not None:
            exit_decision = should_exit_position(position, row, config)
            if exit_decision is not None:
                exit_price, exit_reason = exit_decision
                pnl_pct = (
                    (exit_price - position.entry_price) / position.entry_price
                    if position.side == "long"
                    else (position.entry_price - exit_price) / position.entry_price
                )
                trades.append(
                    Trade(
                        side=position.side,
                        entry_index=position.entry_index,
                        exit_index=int(idx),
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        entry_reason=position.entry_reason,
                    )
                )
                position = None

        if position is None:
            if long_signal:
                entry_price = float(row["close"])
                ob_high, ob_low = get_ob_levels(row, "long")
                position = Position(
                    side="long",
                    entry_index=int(idx),
                    entry_price=entry_price,
                    ob_high=ob_high,
                    ob_low=ob_low,
                    entry_reason="macd_bullish",
                    peak_price=entry_price,
                    trailing_activated=False,
                )
            elif short_signal:
                entry_price = float(row["close"])
                ob_high, ob_low = get_ob_levels(row, "short")
                position = Position(
                    side="short",
                    entry_index=int(idx),
                    entry_price=entry_price,
                    ob_high=ob_high,
                    ob_low=ob_low,
                    entry_reason="macd_bearish",
                    peak_price=entry_price,
                    trailing_activated=False,
                )

        position_side.append(position.side if position is not None else None)
        prev_macd_hist = current_macd_hist

    # Force-close final open position on last close for accounting.
    if position is not None and not working.empty:
        final_idx = int(working.index[-1])
        final_close = float(working.iloc[-1]["close"])
        pnl_pct = (
            (final_close - position.entry_price) / position.entry_price
            if position.side == "long"
            else (position.entry_price - final_close) / position.entry_price
        )
        trades.append(
            Trade(
                side=position.side,
                entry_index=position.entry_index,
                exit_index=final_idx,
                entry_price=position.entry_price,
                exit_price=final_close,
                pnl_pct=pnl_pct,
                exit_reason="end_of_data",
                entry_reason=position.entry_reason,
            )
        )

    working["rsi"] = working["rsi"].astype(float)
    working["long_signal"] = long_signals
    working["short_signal"] = short_signals
    working["position_side"] = position_side

    trades_df = pd.DataFrame([asdict(trade) for trade in trades])

    metrics = summarize_trades(trades_df)
    return {
        "config": asdict(config),
        "metrics": metrics,
        "trades": trades_df,
        "bars": working,
    }


# -----------------------------
# Evaluation helpers
# -----------------------------


def summarize_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "long_count": 0,
            "short_count": 0,
        }

    win_rate = float((trades_df["pnl_pct"] > 0).mean())
    avg_pnl_pct = float(trades_df["pnl_pct"].mean())
    total_pnl_pct = float(trades_df["pnl_pct"].sum())
    long_count = int((trades_df["side"] == "long").sum())
    short_count = int((trades_df["side"] == "short").sum())

    return {
        "trade_count": int(len(trades_df)),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct,
        "total_pnl_pct": total_pnl_pct,
        "long_count": long_count,
        "short_count": short_count,
    }


# -----------------------------
# Example CLI entrypoint
# -----------------------------


def main() -> None:
    """
    Minimal local smoke-test entrypoint.

    prepare.py in your repo should eventually become the real orchestrator and
    data provider. This main exists only so train.py can still be run directly
    during development.
    """
    sample = pd.DataFrame(
        {
            "open": [100, 99, 98, 99, 100, 101],
            "high": [101, 100, 99, 100, 101, 103],
            "low": [99, 98, 97, 98, 99, 100],
            "close": [100, 98.5, 98.0, 99.5, 100.5, 102.5],
            "bullish_internal_ob_high": [100, 100, 100, 100, 100, 100],
            "bullish_internal_ob_low": [98, 98, 98, 98, 98, 98],
            "bullish_internal_ob_active": [False, True, True, True, False, False],
            "bearish_internal_ob_high": [102, 102, 102, 102, 102, 102],
            "bearish_internal_ob_low": [100, 100, 100, 100, 100, 100],
            "bearish_internal_ob_active": [False, False, False, False, False, False],
            "nearest_weak_high_price": [104, 104, 104, 104, 104, 104],
            "nearest_weak_high_exists": [True, True, True, True, True, True],
            "nearest_weak_low_price": [96, 96, 96, 96, 96, 96],
            "nearest_weak_low_exists": [False, False, False, False, False, False],
            "bullish_fvg_nearby": [False, False, True, True, False, False],
            "bearish_fvg_nearby": [False, False, False, False, False, False],
        }
    )

    result = run_strategy(sample, StrategyConfig())
    print("Metrics:")
    print(result["metrics"])
    print("Trades:")
    print(result["trades"])


if __name__ == "__main__":
    main()
