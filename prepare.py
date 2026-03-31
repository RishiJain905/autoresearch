from __future__ import annotations

"""
prepare.py

Fixed preparation and evaluation harness for LuxAlgo-style autoresearch experiments.

Role in this repo
-----------------
- prepare.py is the fixed support layer.
- It loads market data, derives strategy context, and evaluates train.py.
- train.py is the single editable strategy surface for the autoresearch loop.

This file replaces Karpathy's original ML data/tokenizer pipeline with a
market-data preparation pipeline while preserving the same repo philosophy:
fixed prep/eval harness + single editable experiment file.

Expected input CSV columns
--------------------------
Required:
- open
- high
- low
- close

Optional:
- timestamp
- volume
- symbol

Typical usage
-------------
python prepare.py --csv data/sample.csv
python prepare.py --csv data/sample.csv --output prepared_market_data.csv
python prepare.py --csv data/sample.csv --run-train
"""

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

BULLISH = 1
BEARISH = -1
BULLISH_LEG = 1
BEARISH_LEG = 0

DEFAULT_INTERNAL_LENGTH = 5
DEFAULT_SWING_LENGTH = 50
DEFAULT_EQUAL_HL_LENGTH = 3
DEFAULT_EQUAL_HL_THRESHOLD = 0.1
DEFAULT_RSI_LENGTH = 14
DEFAULT_ATR_LENGTH = 200
DEFAULT_OB_FILTER = "atr"       # "atr" or "range"
DEFAULT_OB_MITIGATION = "highlow"  # "close" or "highlow"


@dataclass(frozen=True)
class PrepareConfig:
    internal_length: int = DEFAULT_INTERNAL_LENGTH
    swing_length: int = DEFAULT_SWING_LENGTH
    equal_hl_length: int = DEFAULT_EQUAL_HL_LENGTH
    equal_hl_threshold: float = DEFAULT_EQUAL_HL_THRESHOLD
    rsi_length: int = DEFAULT_RSI_LENGTH
    atr_length: int = DEFAULT_ATR_LENGTH
    ob_filter: str = DEFAULT_OB_FILTER
    ob_mitigation: str = DEFAULT_OB_MITIGATION


@dataclass
class PivotState:
    current_level: Optional[float] = None
    last_level: Optional[float] = None
    crossed: bool = False
    bar_index: Optional[int] = None


@dataclass
class TrendState:
    bias: int = 0


@dataclass
class OrderBlockState:
    high: Optional[float] = None
    low: Optional[float] = None
    start_index: Optional[int] = None
    bias: Optional[int] = None
    active: bool = False


# ---------------------------------------------------------------------------
# Validation and basic indicators
# ---------------------------------------------------------------------------

def validate_market_frame(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = compute_true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def compute_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# ---------------------------------------------------------------------------
# LuxAlgo-style parsing helpers
# ---------------------------------------------------------------------------

def add_volatility_parsing(df: pd.DataFrame, config: PrepareConfig) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = compute_atr(out, config.atr_length)
    tr = compute_true_range(out)

    if config.ob_filter == "atr":
        volatility_measure = out["atr"]
    else:
        # Pine uses ta.cum(ta.tr) / bar_index; this is the closest stable equivalent.
        volatility_measure = tr.expanding(min_periods=1).mean()

    out["volatility_measure"] = volatility_measure.bfill().fillna(0.0)
    out["high_volatility_bar"] = (out["high"] - out["low"]) >= (2.0 * out["volatility_measure"])

    out["parsed_high"] = out["high"]
    out["parsed_low"] = out["low"]

    hv_mask = out["high_volatility_bar"]
    out.loc[hv_mask, "parsed_high"] = out.loc[hv_mask, "low"]
    out.loc[hv_mask, "parsed_low"] = out.loc[hv_mask, "high"]
    return out


def compute_leg_series(high: pd.Series, low: pd.Series, size: int) -> pd.Series:
    """
    Approximate the Pine logic:

        newLegHigh = high[size] > ta.highest(size)
        newLegLow  = low[size]  < ta.lowest(size)

    In a forward Python pass, we interpret this as checking whether the current bar
    breaks above/below the previous `size` bars.

    Vectorized via rolling max/min with shift(1), making the window exclusive of the
    current bar — equivalent to high.iloc[i-size:i].max(). High breakout takes
    precedence over low (mirrors the original elif order). Forward-fill carries the
    last confirmed leg direction between breakout events.
    """
    rolling_max = high.rolling(window=size, min_periods=size).max().shift(1)
    rolling_min = low.rolling(window=size, min_periods=size).min().shift(1)

    new_high = high > rolling_max
    new_low  = low  < rolling_min

    events = pd.Series(
        np.where(new_high, BEARISH_LEG, np.where(new_low, BULLISH_LEG, np.nan)),
        index=high.index,
    )
    return events.ffill().fillna(0).astype("int64")


# ---------------------------------------------------------------------------
# Main structure engine
# ---------------------------------------------------------------------------

def build_strategy_frame(raw_df: pd.DataFrame, config: Optional[PrepareConfig] = None) -> pd.DataFrame:
    config = config or PrepareConfig()
    df = raw_df.copy()

    validate_market_frame(df)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    df["rsi"] = compute_rsi(df["close"], config.rsi_length)
    df = add_volatility_parsing(df, config)

    # Output columns expected by train.py
    output = df.copy()

    output["bullish_internal_ob_high"] = np.nan
    output["bullish_internal_ob_low"] = np.nan
    output["bullish_internal_ob_active"] = False

    output["bearish_internal_ob_high"] = np.nan
    output["bearish_internal_ob_low"] = np.nan
    output["bearish_internal_ob_active"] = False

    output["nearest_weak_high_price"] = np.nan
    output["nearest_weak_high_exists"] = False
    output["nearest_weak_low_price"] = np.nan
    output["nearest_weak_low_exists"] = False

    output["bullish_fvg_nearby"] = False
    output["bearish_fvg_nearby"] = False

    output["in_premium_zone"] = False
    output["in_equilibrium_zone"] = False
    output["in_discount_zone"] = False

    output["swing_trend_bias"] = 0
    output["internal_trend_bias"] = 0

    output["last_swing_high"] = np.nan
    output["last_swing_low"] = np.nan
    output["last_internal_high"] = np.nan
    output["last_internal_low"] = np.nan

    output["equal_high"] = False
    output["equal_low"] = False

    swing_leg = compute_leg_series(output["high"], output["low"], config.swing_length)
    internal_leg = compute_leg_series(output["high"], output["low"], config.internal_length)
    equal_leg = compute_leg_series(output["high"], output["low"], config.equal_hl_length)

    swing_high = PivotState()
    swing_low = PivotState()
    internal_high = PivotState()
    internal_low = PivotState()
    equal_high = PivotState()
    equal_low = PivotState()

    swing_trend = TrendState()
    internal_trend = TrendState()

    bullish_internal_ob = OrderBlockState()
    bearish_internal_ob = OrderBlockState()

    trailing_top: Optional[float] = None
    trailing_bottom: Optional[float] = None
    trailing_top_index: Optional[int] = None
    trailing_bottom_index: Optional[int] = None

    def update_pivot(pivot: PivotState, level: float, bar_index: int) -> None:
        pivot.last_level = pivot.current_level
        pivot.current_level = level
        pivot.crossed = False
        pivot.bar_index = bar_index

    def maybe_store_order_block(
        pivot: PivotState,
        current_index: int,
        bias: int,
        internal: bool,
    ) -> None:
        nonlocal bullish_internal_ob, bearish_internal_ob

        if pivot.bar_index is None:
            return

        scan = output.iloc[pivot.bar_index: current_index + 1]
        if scan.empty:
            return

        if bias == BULLISH:
            parsed_idx = scan["parsed_low"].astype(float).idxmin()
            ob_high = float(output.loc[parsed_idx, "parsed_high"])
            ob_low = float(output.loc[parsed_idx, "parsed_low"])
            if internal:
                bullish_internal_ob = OrderBlockState(
                    high=ob_high,
                    low=ob_low,
                    start_index=int(parsed_idx),
                    bias=BULLISH,
                    active=True,
                )
        else:
            parsed_idx = scan["parsed_high"].astype(float).idxmax()
            ob_high = float(output.loc[parsed_idx, "parsed_high"])
            ob_low = float(output.loc[parsed_idx, "parsed_low"])
            if internal:
                bearish_internal_ob = OrderBlockState(
                    high=ob_high,
                    low=ob_low,
                    start_index=int(parsed_idx),
                    bias=BEARISH,
                    active=True,
                )

    def delete_order_blocks(row: pd.Series) -> None:
        nonlocal bullish_internal_ob, bearish_internal_ob

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        if bullish_internal_ob.active and bullish_internal_ob.low is not None:
            if config.ob_mitigation == "close":
                crossed = close < bullish_internal_ob.low
            else:
                crossed = low < bullish_internal_ob.low
            if crossed:
                bullish_internal_ob = OrderBlockState()

        if bearish_internal_ob.active and bearish_internal_ob.high is not None:
            if config.ob_mitigation == "close":
                crossed = close > bearish_internal_ob.high
            else:
                crossed = high > bearish_internal_ob.high
            if crossed:
                bearish_internal_ob = OrderBlockState()

    def process_structure(
        i: int,
        row: pd.Series,
        pivot_high: PivotState,
        pivot_low: PivotState,
        trend: TrendState,
        leg_series: pd.Series,
        size: int,
        internal: bool,
    ) -> None:
        current_close = float(row["close"])

        # Mirrors the confluence filter logic shape from Pine, but always enabled as data context.
        bullish_bar = row["high"] - max(row["close"], row["open"]) > min(
            row["close"], row["open"]
        ) - row["low"]
        bearish_bar = row["high"] - max(row["close"], row["open"]) < min(
            row["close"], row["open"]
        ) - row["low"]

        # Bullish break of structure / CHoCH
        extra_condition = True
        if internal:
            extra_condition = (
                internal_high.current_level is not None
                and swing_high.current_level is not None
                and internal_high.current_level != swing_high.current_level
                and bullish_bar
            )

        if (
            pivot_high.current_level is not None
            and not pivot_high.crossed
            and extra_condition
            and current_close > pivot_high.current_level
        ):
            pivot_high.crossed = True
            trend.bias = BULLISH
            maybe_store_order_block(pivot_high, i, BULLISH, internal=internal)

        # Bearish break of structure / CHoCH
        extra_condition = True
        if internal:
            extra_condition = (
                internal_low.current_level is not None
                and swing_low.current_level is not None
                and internal_low.current_level != swing_low.current_level
                and bearish_bar
            )

        if (
            pivot_low.current_level is not None
            and not pivot_low.crossed
            and extra_condition
            and current_close < pivot_low.current_level
        ):
            pivot_low.crossed = True
            trend.bias = BEARISH
            maybe_store_order_block(pivot_low, i, BEARISH, internal=internal)

        output.at[i, "internal_trend_bias" if internal else "swing_trend_bias"] = trend.bias

    for i in range(len(output)):
        row = output.iloc[i]
        # -------------------------------------------------------------------
        # Current structure updates
        # -------------------------------------------------------------------
        if i >= config.swing_length:
            current_leg = int(swing_leg.iloc[i])
            previous_leg = int(swing_leg.iloc[i - 1])

            new_pivot = current_leg != previous_leg
            pivot_low_trigger = (current_leg - previous_leg) == 1
            pivot_high_trigger = (current_leg - previous_leg) == -1

            if new_pivot:
                pivot_idx = i - config.swing_length
                if pivot_idx >= 0:
                    if pivot_low_trigger:
                        level = float(output.iloc[pivot_idx]["low"])
                        update_pivot(swing_low, level, pivot_idx)
                        trailing_bottom = level
                        trailing_bottom_index = pivot_idx
                    elif pivot_high_trigger:
                        level = float(output.iloc[pivot_idx]["high"])
                        update_pivot(swing_high, level, pivot_idx)
                        trailing_top = level
                        trailing_top_index = pivot_idx

        if i >= config.internal_length:
            current_leg = int(internal_leg.iloc[i])
            previous_leg = int(internal_leg.iloc[i - 1])

            new_pivot = current_leg != previous_leg
            pivot_low_trigger = (current_leg - previous_leg) == 1
            pivot_high_trigger = (current_leg - previous_leg) == -1

            if new_pivot:
                pivot_idx = i - config.internal_length
                if pivot_idx >= 0:
                    if pivot_low_trigger:
                        level = float(output.iloc[pivot_idx]["low"])
                        update_pivot(internal_low, level, pivot_idx)
                    elif pivot_high_trigger:
                        level = float(output.iloc[pivot_idx]["high"])
                        update_pivot(internal_high, level, pivot_idx)

        if i >= config.equal_hl_length:
            current_leg = int(equal_leg.iloc[i])
            previous_leg = int(equal_leg.iloc[i - 1])

            new_pivot = current_leg != previous_leg
            pivot_low_trigger = (current_leg - previous_leg) == 1
            pivot_high_trigger = (current_leg - previous_leg) == -1

            if new_pivot:
                pivot_idx = i - config.equal_hl_length
                if pivot_idx >= 0:
                    atr_measure = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

                    if pivot_low_trigger:
                        level = float(output.iloc[pivot_idx]["low"])
                        if equal_low.current_level is not None and abs(equal_low.current_level - level) < config.equal_hl_threshold * atr_measure:
                            output.at[i, "equal_low"] = True
                        update_pivot(equal_low, level, pivot_idx)

                    elif pivot_high_trigger:
                        level = float(output.iloc[pivot_idx]["high"])
                        if equal_high.current_level is not None and abs(equal_high.current_level - level) < config.equal_hl_threshold * atr_measure:
                            output.at[i, "equal_high"] = True
                        update_pivot(equal_high, level, pivot_idx)

        # -------------------------------------------------------------------
        # Structure display logic translated into state updates
        # -------------------------------------------------------------------
        process_structure(
            i=i,
            row=row,
            pivot_high=internal_high,
            pivot_low=internal_low,
            trend=internal_trend,
            leg_series=internal_leg,
            size=config.internal_length,
            internal=True,
        )
        process_structure(
            i=i,
            row=row,
            pivot_high=swing_high,
            pivot_low=swing_low,
            trend=swing_trend,
            leg_series=swing_leg,
            size=config.swing_length,
            internal=False,
        )

        # -------------------------------------------------------------------
        # OB mitigation / invalidation
        # -------------------------------------------------------------------
        delete_order_blocks(row)

        # -------------------------------------------------------------------
        # Save active OBs
        # -------------------------------------------------------------------
        output.at[i, "bullish_internal_ob_high"] = bullish_internal_ob.high
        output.at[i, "bullish_internal_ob_low"] = bullish_internal_ob.low
        output.at[i, "bullish_internal_ob_active"] = bullish_internal_ob.active

        output.at[i, "bearish_internal_ob_high"] = bearish_internal_ob.high
        output.at[i, "bearish_internal_ob_low"] = bearish_internal_ob.low
        output.at[i, "bearish_internal_ob_active"] = bearish_internal_ob.active

        # -------------------------------------------------------------------
        # Trailing extremes, weak high/low, premium/discount
        # -------------------------------------------------------------------
        output.at[i, "last_swing_high"] = trailing_top
        output.at[i, "last_swing_low"] = trailing_bottom
        output.at[i, "last_internal_high"] = internal_high.current_level
        output.at[i, "last_internal_low"] = internal_low.current_level

        # LuxAlgo labeling logic:
        # top label text = Strong High if swingTrend.bias == BEARISH else Weak High
        # bottom label text = Strong Low if swingTrend.bias == BULLISH else Weak Low
        weak_high = (trailing_top is not None) and (swing_trend.bias != BEARISH) and (trailing_top > float(row["close"]))
        weak_low = (trailing_bottom is not None) and (swing_trend.bias != BULLISH) and (trailing_bottom < float(row["close"]))

        output.at[i, "nearest_weak_high_price"] = trailing_top
        output.at[i, "nearest_weak_high_exists"] = bool(weak_high)
        output.at[i, "nearest_weak_low_price"] = trailing_bottom
        output.at[i, "nearest_weak_low_exists"] = bool(weak_low)

        # Premium / Equilibrium / Discount zones from the exact LuxAlgo blends.
        if trailing_top is not None and trailing_bottom is not None and trailing_top > trailing_bottom:
            close = float(row["close"])

            premium_top = trailing_top
            premium_bottom = 0.95 * trailing_top + 0.05 * trailing_bottom

            equilibrium_top = 0.525 * trailing_top + 0.475 * trailing_bottom
            equilibrium_bottom = 0.525 * trailing_bottom + 0.475 * trailing_top

            discount_top = 0.95 * trailing_bottom + 0.05 * trailing_top
            discount_bottom = trailing_bottom

            output.at[i, "in_premium_zone"] = premium_bottom <= close <= premium_top
            output.at[i, "in_equilibrium_zone"] = equilibrium_bottom <= close <= equilibrium_top
            output.at[i, "in_discount_zone"] = discount_bottom <= close <= discount_top

        # -------------------------------------------------------------------
        # FVG context
        # -------------------------------------------------------------------
        if i >= 2:
            bullish_fvg = float(row["low"]) > float(output.iloc[i - 2]["high"])
            bearish_fvg = float(row["high"]) < float(output.iloc[i - 2]["low"])
            output.at[i, "bullish_fvg_nearby"] = bullish_fvg
            output.at[i, "bearish_fvg_nearby"] = bearish_fvg

    return output


# ---------------------------------------------------------------------------
# train.py integration
# ---------------------------------------------------------------------------

def load_train_module(train_path: str):
    import sys as _sys
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load train module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    _sys.modules["train_module"] = module
    spec.loader.exec_module(module)
    return module


def run_train_on_prepared_frame(prepared_df: pd.DataFrame, train_path: str = "train.py") -> Dict[str, Any]:
    module = load_train_module(train_path)
    if not hasattr(module, "run_strategy"):
        raise AttributeError("train.py must expose run_strategy(df, config=None)")
    return module.run_strategy(prepared_df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LuxAlgo-style market context and optionally run train.py"
    )
    parser.add_argument("--csv", required=True, help="Path to OHLC CSV input")
    parser.add_argument("--output", default="prepared_market_data.csv", help="Path to save prepared frame")
    parser.add_argument("--run-train", action="store_true", help="Run train.py after preparing the frame")
    parser.add_argument("--train-path", default="train.py", help="Path to train.py")
    parser.add_argument(
        "--metrics-output",
        default="strategy_metrics.json",
        help="Where to save metrics JSON if --run-train is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    config = PrepareConfig()
    prepared_df = build_strategy_frame(raw_df, config=config)
    prepared_df.to_csv(args.output, index=False)

    print(f"Prepared frame saved to: {args.output}")
    print(f"Rows: {len(prepared_df)}")

    if args.run_train:
        result = run_train_on_prepared_frame(prepared_df, train_path=args.train_path)
        metrics = result.get("metrics", {})
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.metrics_output}")
        print("Metrics:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()