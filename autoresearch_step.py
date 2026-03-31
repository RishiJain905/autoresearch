"""
autoresearch_step.py

One iteration of the autonomous trading strategy research loop.
Called by GitHub Actions every 10 minutes.

Flow (same every run — no special first-run logic needed):
  1. Run the current train.py to measure the baseline for this iteration.
  2. Ask Minimax to propose one improvement.
  3. Apply it, run again, compare.
  4. If better (and trade_count >= MIN_TRADE_COUNT): commit and push.
  5. Otherwise: revert train.py, record as discard.

results.tsv is committed on every run so the full experiment history persists
across GitHub Actions invocations. The git commit history is the real record.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_PATH     = "data/sample.csv"
PREPARE_PY   = "prepare.py"
TRAIN_PY     = "train.py"
PROGRAM_MD   = "program.md"
METRICS_PATH = "strategy_metrics.json"
RESULTS_TSV  = "results.tsv"
RUN_LOG      = "run.log"

RESULTS_HEADER = "commit\ttotal_pnl_pct\twin_rate\ttrade_count\tstatus\tdescription\n"

# Minimum credible trade count — strategies below this are treated as overfitted.
MIN_TRADE_COUNT = 5

# ---------------------------------------------------------------------------
# Helpers: file I/O
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_results() -> str:
    if not os.path.exists(RESULTS_TSV):
        return RESULTS_HEADER
    return read_file(RESULTS_TSV)


def append_result(commit: str, metrics: dict | None, status: str, description: str) -> None:
    existing = read_results()
    if metrics:
        row = (
            f"{commit}\t"
            f"{metrics['total_pnl_pct']:.6f}\t"
            f"{metrics['win_rate']:.6f}\t"
            f"{metrics['trade_count']}\t"
            f"{status}\t"
            f"{description}\n"
        )
    else:
        row = f"{commit}\t0.000000\t0.000000\t0\t{status}\t{description}\n"
    write_file(RESULTS_TSV, existing + row)


# ---------------------------------------------------------------------------
# Helpers: git
# ---------------------------------------------------------------------------

def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def git_short_hash() -> str:
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def git_commit(message: str) -> None:
    """Commit train.py + results.tsv together (used on 'keep')."""
    git("add", TRAIN_PY, RESULTS_TSV)
    git("commit", "-m", message)


def git_commit_results(message: str) -> None:
    """Commit only results.tsv (used on 'discard' and 'crash')."""
    git("add", RESULTS_TSV)
    git("commit", "-m", message)


def git_revert_train() -> None:
    git("checkout", "--", TRAIN_PY)


# ---------------------------------------------------------------------------
# Helpers: experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict | None:
    """Run prepare.py --run-train. Returns parsed metrics dict or None on failure."""
    result = subprocess.run(
        [
            sys.executable, PREPARE_PY,
            "--csv", CSV_PATH,
            "--run-train",
            "--metrics-output", METRICS_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    with open(RUN_LOG, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0 or not os.path.exists(METRICS_PATH):
        return None

    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def metrics_str(m: dict) -> str:
    return (
        f"total_pnl_pct={m['total_pnl_pct']:.6f}  "
        f"win_rate={m['win_rate']:.6f}  "
        f"trade_count={m['trade_count']}"
    )


# ---------------------------------------------------------------------------
# Helpers: Minimax API
# ---------------------------------------------------------------------------

def call_minimax(system_prompt: str, user_message: str) -> str:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY environment variable is not set.")

    # International API: api.minimax.io (keys from platform.minimax.io)
    url = "https://api.minimax.io/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "system", "name": "MiniMax AI", "content": system_prompt},
            {"role": "user",   "name": "user",        "content": user_message},
        ],
        "max_completion_tokens": 8192,
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_python_block(text: str) -> str | None:
    """Extract the first ```python ... ``` code block from the LLM response."""
    text = strip_reasoning(text)
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_description(text: str) -> str:
    """Pull the first substantive sentence from the LLM's explanation for the TSV."""
    text = strip_reasoning(text)
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("```") and not line.startswith("#"):
            return line[:80].replace("\t", " ")
    return "no description"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_git_history() -> str:
    """Return a short summary of recent autoresearch commits for context."""
    result = git("log", "--oneline", "-20")
    return result.stdout.strip() or "(no prior commits)"


def main() -> None:
    program_md  = read_file(PROGRAM_MD)
    prepare_py  = read_file(PREPARE_PY)
    train_py    = read_file(TRAIN_PY)
    git_history = build_git_history()

    # ------------------------------------------------------------------
    # Step 1: measure the current strategy (HEAD = best known so far)
    # ------------------------------------------------------------------
    print("Running current strategy...")
    current_metrics = run_experiment()
    if current_metrics is None:
        print("Current strategy crashed. Check run.log.")
        print(open(RUN_LOG).read()[-2000:])
        sys.exit(1)
    print(f"Current: {metrics_str(current_metrics)}")

    # ------------------------------------------------------------------
    # Step 2: ask Minimax for one improvement
    # ------------------------------------------------------------------
    system_prompt = program_md

    user_message = f"""\
You are running one iteration of the autoresearch experiment loop.

## Current strategy (train.py)

```python
{train_py}
```

## Fixed support layer (prepare.py) — READ ONLY, do not modify

```python
{prepare_py}
```

## Recent git history (last 20 commits)

```
{git_history}
```

## Current metrics (this is what you must beat)

{metrics_str(current_metrics)}

## Your task

Propose ONE specific, targeted improvement to `train.py`.
- Goal: increase `total_pnl_pct` while keeping `trade_count` >= {MIN_TRADE_COUNT}.
- Study the commit history carefully — avoid repeating ideas already tried.
- Prefer simple changes. Removing something that hurts is as good as adding something new.
- Do NOT modify `prepare.py` — it is the fixed evaluation harness.

Respond with:
1. A brief explanation (2-4 sentences) of what you are changing and why.
2. The complete updated `train.py` as a single ```python code block.
"""

    print("Calling Minimax API...")
    try:
        response_text = call_minimax(system_prompt, user_message)
    except Exception as exc:
        print(f"Minimax API error: {exc}")
        sys.exit(1)

    print("Response received. Extracting new train.py...")
    new_train_py = extract_python_block(response_text)
    if new_train_py is None:
        print("Could not extract a Python code block from the response.")
        print("--- Response preview ---")
        print(response_text[:800])
        sys.exit(1)

    description = extract_description(response_text)
    print(f"Proposed change: {description}")

    # ------------------------------------------------------------------
    # Step 3: test the proposed strategy
    # ------------------------------------------------------------------
    write_file(TRAIN_PY, new_train_py)

    print("Running experiment with proposed strategy...")
    new_metrics = run_experiment()
    commit = git_short_hash()

    if new_metrics is None:
        print("Proposed strategy crashed. Reverting.")
        print(open(RUN_LOG).read()[-2000:])
        git_revert_train()
        append_result(commit, None, "crash", description)
        git_commit_results(f"crash: {description}")
        return

    print(f"Proposed: {metrics_str(new_metrics)}")

    # ------------------------------------------------------------------
    # Step 4: keep or revert
    # ------------------------------------------------------------------
    improved = (
        new_metrics["total_pnl_pct"] > current_metrics["total_pnl_pct"]
        and new_metrics["trade_count"] >= MIN_TRADE_COUNT
    )

    if improved:
        append_result(commit, new_metrics, "keep", description)
        git_commit(f"keep: {description}")
        commit = git_short_hash()
        print(f"Improvement accepted. New best: {new_metrics['total_pnl_pct']:.6f}")
    else:
        git_revert_train()
        append_result(commit, new_metrics, "discard", description)
        git_commit_results(f"discard: {description}")
        print(
            f"No improvement (or trade_count < {MIN_TRADE_COUNT}). "
            f"Reverted. Best stays: {current_metrics['total_pnl_pct']:.6f}"
        )


if __name__ == "__main__":
    main()
