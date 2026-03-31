# autoresearch

This is an experiment to have the LLM do its own research.

This workflow is triggered automatically by GitHub Actions every 10 minutes. The Minimax API key is stored as the repo secret `MINIMAX_API_KEY`. Each run is one experiment iteration — read the current state, make one change to `train.py`, evaluate it, and advance or revert.

## Repo structure

The repo is small. The key files:

- `prepare.py` — fixed support layer. Loads OHLC CSV, computes all LuxAlgo SMC context (order blocks, pivots, FVGs, zones, RSI, ATR), and runs `train.py` via `run_train_on_prepared_frame`. **Do not modify.**
- `train.py` — the file you modify. Strategy entry/exit logic, signal parameters, position management.
- `data/sample.csv` — fixed OHLC dataset (BTC/USDT 15m). **Do not modify.**
- `results.tsv` — persistent experiment log, committed on every run.
- `autoresearch_step.py` — orchestrator that runs the experiment loop (calls prepare.py, compares metrics, keeps/reverts). **Do not modify.**

## Experimentation

Run the strategy on the fixed dataset and evaluate it. You launch it as:

```
python prepare.py --csv data/sample.csv --run-train --metrics-output strategy_metrics.json > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: entry/exit conditions, RSI thresholds, take-profit percentage, FVG confirmation toggle, OB overlap logic, zone filters, position sizing logic, long/short asymmetry, custom signal combinations.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed market data engine, all SMC feature computation, and the evaluation harness `run_train_on_prepared_frame`.
- Modify `data/sample.csv`. The dataset is fixed — you are not allowed to cherry-pick data.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the highest `total_pnl_pct`.** This is the sum of PnL percentages across all trades on the fixed dataset. Higher is better. Everything in `train.py` is fair game: change the signal thresholds, combine conditions differently, rethink the exit logic, go long-only or short-only if warranted.

**Trade count** is a soft constraint. A strategy that finds 2 perfect trades is not better than one that finds 20 solid trades. Be suspicious of very low trade counts — they may indicate overfitting. Conversely, a strategy generating hundreds of noisy trades is also suspect. Use judgment: a meaningful result needs enough trades to be statistically credible.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 `total_pnl_pct` improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the strategy as is.

## Output format

Once the script finishes, `strategy_metrics.json` contains:

```json
{
  "trade_count": 18,
  "win_rate": 0.611111,
  "avg_pnl_pct": 0.008432,
  "total_pnl_pct": 0.151776,
  "long_count": 11,
  "short_count": 7
}
```

You can extract the key metrics from the output file:

```
python -c "import json; m=json.load(open('strategy_metrics.json')); print('total_pnl_pct:', m['total_pnl_pct'], '| win_rate:', m['win_rate'], '| trade_count:', m['trade_count'])"
```

If `strategy_metrics.json` is missing or the above crashes, the run failed. Read `run.log` to diagnose.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	total_pnl_pct	win_rate	trade_count	status	description
```

1. git commit hash (short, 7 chars)
2. `total_pnl_pct` achieved (e.g. 0.151776) — use 0.000000 for crashes
3. `win_rate` (e.g. 0.611111) — use 0.000000 for crashes
4. `trade_count` (integer) — use 0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	total_pnl_pct	win_rate	trade_count	status	description
a1b2c3d	0.151776	0.611111	18	keep	baseline
b2c3d4e	0.198340	0.666667	21	keep	lower long RSI threshold to 35
c3d4e5f	0.091200	0.500000	14	discard	require FVG confirmation on both sides
d4e5f6g	0.000000	0.000000	0	crash	invalid TP pct caused ZeroDivisionError
```

NOTE: `results.tsv` is committed to the repo on every run so the full experiment history persists across workflow invocations.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python prepare.py --csv data/sample.csv --run-train --metrics-output strategy_metrics.json > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `python -c "import json; m=json.load(open('strategy_metrics.json')); print('total_pnl_pct:', m['total_pnl_pct'], '| win_rate:', m['win_rate'], '| trade_count:', m['trade_count'])"`
6. If the above crashes or `strategy_metrics.json` is missing, the run failed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If `total_pnl_pct` improved (higher) **and** `trade_count` is credible (≥5), you "advance" the branch, keeping the git commit
9. If `total_pnl_pct` is equal or worse, or trade count is suspiciously low, git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should complete in well under 5 minutes on any reasonably sized CSV. If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, a bad config value, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import, a divide-by-zero from an extreme parameter), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining signals differently, try more radical changes to entry/exit logic, explore zone-based filtering, long-only vs short-only asymmetry, or stricter OB overlap requirements. The loop runs until the human interrupts you, period.

As an example use case, a user might leave the GitHub Actions workflow running overnight. With a 10-minute trigger, that's up to ~6 experiments/hour and ~48+ experiments over 8 hours. The user then wakes up to a full `results.tsv` of experiments completed autonomously.
