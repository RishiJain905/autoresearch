# autoresearch

*Inspired by [@karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an autonomous LLM pretraining research loop — adapted for autonomous trading strategy research on LuxAlgo Smart Money Concepts.*

The idea: give an AI agent a trading strategy backtester and let it experiment autonomously. It modifies the strategy code, runs the backtest, checks if `total_pnl_pct` improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better strategy. The core philosophy is the same as Karpathy's — you're not touching `train.py` like you normally would as a researcher. Instead, you let the agent iterate on it while `prepare.py` provides the fixed evaluation harness.

## Pipeline

```mermaid
flowchart TB
    subgraph trigger["GitHub Actions (every 10 min)"]
        cron["Cron trigger<br/>*/10 * * * *"]
    end

    subgraph step["autoresearch_step.py"]
        direction TB
        run1["Step 1: Run current train.py<br/>on data/sample.csv"]
        measure1["Measure current_metrics<br/>(total_pnl_pct, win_rate, trade_count)"]
        api["Step 2: Call Minimax API<br/>Send: program.md + prepare.py<br/>+ train.py + git history<br/>+ current_metrics"]
        extract["Extract proposed train.py<br/>from LLM response"]
        run2["Step 3: Run proposed train.py<br/>on data/sample.csv"]
        measure2["Measure new_metrics"]
        compare{"Step 4: Improved?<br/>new > current AND<br/>trade_count >= 5"}
        keep["git commit train.py + results.tsv<br/>git push"]
        discard["Revert train.py<br/>Commit results.tsv only"]

        run1 --> measure1 --> api --> extract --> run2 --> measure2 --> compare
        compare -->|Yes| keep
        compare -->|No| discard
    end

    subgraph data["Fixed inputs (never modified)"]
        csv["data/sample.csv<br/>BTC/USDT 15m OHLCV"]
        prepare["prepare.py<br/>LuxAlgo SMC engine"]
        program["program.md<br/>Agent instructions"]
    end

    subgraph editable["Editable by agent"]
        train["train.py<br/>Strategy logic"]
        results["results.tsv<br/>Experiment history"]
    end

    cron --> run1
    csv --> run1
    csv --> run2
    prepare -.->|"builds strategy frame"| run1
    prepare -.->|"builds strategy frame"| run2
    program -.->|"system prompt"| api
    train -->|"current version"| run1
    extract -->|"overwrites"| train
    keep --> results
    discard --> results
```

```mermaid
flowchart LR
    subgraph prepare_py["prepare.py — Fixed Support Layer"]
        direction TB
        csv_in["OHLC CSV input"]
        validate["Validate columns"]
        rsi["Compute RSI"]
        atr["Compute ATR"]
        vol["Volatility parsing<br/>(high-volatility bar detection)"]
        legs["Compute leg series<br/>(swing + internal + equal)"]
        pivots["Detect pivots<br/>(swing highs/lows,<br/>internal highs/lows)"]
        structure["Process structure<br/>(CHoCH / BoS detection)"]
        ob["Order Block creation<br/>+ mitigation"]
        zones["Premium / Equilibrium /<br/>Discount zones"]
        fvg["FVG detection"]
        weak["Weak High / Low<br/>classification"]
        frame["Output: strategy frame<br/>(all features per bar)"]

        csv_in --> validate --> rsi --> atr --> vol --> legs --> pivots --> structure --> ob --> zones --> fvg --> weak --> frame
    end

    subgraph train_py["train.py — Editable Strategy"]
        direction TB
        signals["Signal detection<br/>(long / short)"]
        entry["Entry conditions:<br/>OB overlap + RSI +<br/>weak high/low"]
        exit["Exit conditions:<br/>TP at +/-2% or<br/>OB boundary break"]
        trades["Trade list +<br/>PnL calculation"]
        metrics["Metrics:<br/>total_pnl_pct<br/>win_rate<br/>trade_count"]

        signals --> entry --> exit --> trades --> metrics
    end

    frame -->|"prepared DataFrame"| signals
```

## How it works

The repo has a clean separation of concerns:

- **`prepare.py`** — fixed support layer. Loads OHLC CSV, computes all LuxAlgo Smart Money Concepts context (order blocks, swing/internal pivots, FVGs, premium/discount zones, RSI, ATR, equal highs/lows), and runs `train.py` via `run_train_on_prepared_frame`. Not modified.
- **`train.py`** — the single file the agent edits. Contains entry/exit logic, RSI thresholds, take-profit percentage, position management. Everything is fair game. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions that tell the AI what the metric is, what it can change, and how the experiment loop works. **This file is edited and iterated on by the human.**
- **`autoresearch_step.py`** — orchestrator that runs one experiment iteration: measure current strategy, call Minimax for an improvement, test it, keep or revert. Not modified.

The metric is **`total_pnl_pct`** (sum of trade PnL percentages) — higher is better. Trade count must stay above 5 to prevent overfitting to a handful of lucky trades.

## Quick start

**Requirements:** Python 3.10+, a Minimax API key.

```bash
# 1. Install dependencies
pip install numpy pandas requests

# 2. Run the pipeline manually (one experiment iteration)
export MINIMAX_API_KEY=your_key_here
python autoresearch_step.py

# 3. Or just run the backtest without the AI loop
python prepare.py --csv data/sample.csv --run-train
```

## Running autonomously via GitHub Actions

The workflow (`.github/workflows/autoresearch.yml`) triggers every 10 minutes:

1. Add your `MINIMAX_API_KEY` as a repository secret in GitHub Settings.
2. Push to the branch you want the agent to iterate on.
3. The workflow runs `autoresearch_step.py`, which calls Minimax, tests the proposed change, and commits if improved.

With 10-minute cycles, that's ~6 experiments/hour and ~48+ experiments overnight.

## Project structure

```
prepare.py              — SMC feature engine + evaluation harness (do not modify)
train.py                — strategy logic (agent modifies this)
program.md              — agent instructions (human modifies this)
autoresearch_step.py    — experiment orchestrator (do not modify)
data/sample.csv         — BTC/USDT 15m OHLCV dataset (do not modify)
data/merge.py           — utility to merge Binance daily CSVs
results.tsv             — persistent experiment log (committed by agent)
.github/workflows/      — GitHub Actions workflow (10-min cron)
pyproject.toml          — dependencies (numpy, pandas, requests)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed dataset.** The strategy always runs against the same `data/sample.csv`. This makes experiments directly comparable regardless of what the agent changes.
- **Git as state.** The commit history is the experiment log. Each "keep" commit advances `train.py`, each "discard" commit only updates `results.tsv`. You can `git log --oneline` to see the full research trajectory.
- **Self-contained.** No GPU required, no heavy dependencies. One dataset, one file, one metric.

## Acknowledgements

- [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous LLM pretraining research loop that inspired this project.
- [LuxAlgo Smart Money Concepts](https://www.tradingview.com/script/cR52pOF2/) — the TradingView indicator whose logic `prepare.py` translates into Python.

## License

MIT
