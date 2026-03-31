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
Long:
- latest valid bullish internal OB exists
- close is inside bullish OB (not merely a wick overlap)
- RSI < 45 (oversold, tuned)
- nearest relevant upside liquidity target is a weak high
- optional bullish FVG can strengthen the thesis but is not required
- take profit at +~1.5553% long (`take_profit_pct`; shorts use `short_take_profit_pct` ~1.9337%)
- stop if close breaks below bullish OB low, BUT only if OB is ≥0.5% below entry
  (prevents getting stopped out by tight OBs that don't offer real risk management)

Short:
- latest valid bearish internal OB exists
- close is inside bearish OB (not merely a wick overlap)
- RSI > 55 (overbought, tuned)
- nearest relevant downside liquidity target is a weak low
- optional bearish FVG can strengthen the thesis but is not required
- take profit at -`short_take_profit_pct` (~1.9337% on sample data)
- stop if close breaks above bearish OB high, BUT only if OB is ≥0.5% above entry
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StrategyConfig:
    rsi_length: int = 14
    long_rsi_threshold: float = 45.0
    short_rsi_threshold: float = 55.0
    take_profit_pct: float = 0.0155532  # long TP; shorts use short_take_profit_pct when not None
    short_take_profit_pct: Optional[float] = 0.019337
    require_fvg_confirmation: bool = False
    entry_on_close: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True
    min_ob_stop_distance_pct: float = 0.005  # OB must be ≥0.5% away to apply stop


@dataclass
class Position:
    side: str  # "long" or "short"
    entry_index: int
    entry_price: float
    ob_high: float
    ob_low: float
    entry_reason: str


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

def valid_long_signal(row: pd.Series, config: StrategyConfig) -> bool:
    if not config.allow_longs:
        return False

    bullish_ob_active = bool(row["bullish_internal_ob_active"])
    if not bullish_ob_active:
        return False

    ob_high = float(row["bullish_internal_ob_high"])
    ob_low = float(row["bullish_internal_ob_low"])
    close = float(row["close"])
    if not (ob_low <= close <= ob_high):
        return False

    if float(row["rsi"]) >= config.long_rsi_threshold:
        return False

    if not bool(row["nearest_weak_high_exists"]):
        return False

    if config.require_fvg_confirmation and not bool(row["bullish_fvg_nearby"]):
        return False

    return True



def valid_short_signal(row: pd.Series, config: StrategyConfig) -> bool:
    if not config.allow_shorts:
        return False

    bearish_ob_active = bool(row["bearish_internal_ob_active"])
    if not bearish_ob_active:
        return False

    ob_high = float(row["bearish_internal_ob_high"])
    ob_low = float(row["bearish_internal_ob_low"])
    close = float(row["close"])
    if not (ob_low <= close <= ob_high):
        return False

    if float(row["rsi"]) <= config.short_rsi_threshold:
        return False

    if not bool(row["nearest_weak_low_exists"]):
        return False

    if config.require_fvg_confirmation and not bool(row["bearish_fvg_nearby"]):
        return False

    return True


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


def should_exit_position(position: Position, row: pd.Series, config: StrategyConfig) -> Optional[tuple[float, str]]:
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])

    if position.side == "long":
        tp = long_take_profit(position.entry_price, config)
        if high >= tp:
            return tp, "take_profit_1p5pct"
        # Only apply OB stop if it's far enough from entry
        ob_dist_pct = ob_stop_distance_pct(position.entry_price, position.ob_low, "long")
        if ob_dist_pct >= config.min_ob_stop_distance_pct and close < position.ob_low:
            return close, "close_below_bullish_ob_low"
        return None

    if position.side == "short":
        tp = short_take_profit(position.entry_price, config)
        if low <= tp:
            return tp, "take_profit_1p5pct"
        # Only apply OB stop if it's far enough from entry
        ob_dist_pct = ob_stop_distance_pct(position.entry_price, position.ob_high, "short")
        if ob_dist_pct >= config.min_ob_stop_distance_pct and close > position.ob_high:
            return close, "close_above_bearish_ob_high"
        return None

    raise ValueError(f"Unsupported position side: {position.side}")


# -----------------------------
# Backtest loop
# -----------------------------

def run_strategy(df: pd.DataFrame, config: Optional[StrategyConfig] = None) -> Dict[str, Any]:
    config = config or StrategyConfig()
    validate_input_frame(df)

    working = df.copy()
    working["rsi"] = compute_rsi(working["close"], length=config.rsi_length)

    position: Optional[Position] = None
    trades: List[Trade] = []

    long_signals: List[bool] = []
    short_signals: List[bool] = []
    position_side: List[Optional[str]] = []

    for idx, row in working.iterrows():
        long_signal = valid_long_signal(row, config)
        short_signal = valid_short_signal(row, config)

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
                position = Position(
                    side="long",
                    entry_index=int(idx),
                    entry_price=entry_price,
                    ob_high=float(row["bullish_internal_ob_high"]),
                    ob_low=float(row["bullish_internal_ob_low"]),
                    entry_reason="bullish_internal_ob_overlap_rsi_weak_high",
                )
            elif short_signal:
                entry_price = float(row["close"])
                position = Position(
                    side="short",
                    entry_index=int(idx),
                    entry_price=entry_price,
                    ob_high=float(row["bearish_internal_ob_high"]),
                    ob_low=float(row["bearish_internal_ob_low"]),
                    entry_reason="bearish_internal_ob_overlap_rsi_weak_low",
                )

        position_side.append(position.side if position is not None else None)

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
