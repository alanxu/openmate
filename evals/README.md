# Eval harness

Executes [`docs/16-eval-plan.md`](../docs/16-eval-plan.md)'s 29 cases, split
into three tiers:

| Tier | What | File | Run it |
|---|---|---|---|
| **Unit** | `FakeModel` + fixtures, mechanism only | [`../tests/test_evals.py`](../tests/test_evals.py) | `pytest tests/test_evals.py -v` |
| **Integration** | Real model, N-trial metrics — **review this tier** | [`../evals/test_evals_integration.py`](../evals/test_evals_integration.py) | `pytest evals/test_evals_integration.py -v -s --run-live` |
| **Manual/Human** | Real model, human-judged transcript quality | [`run_manual.py`](run_manual.py) | `python evals/run_manual.py` |

Unit tells you the plumbing works. Integration tells you whether the real,
configured model actually behaves correctly — that's the tier worth your
time reviewing. Manual goes one level deeper on the three safety-relevant
cases where "did it comply" isn't the same question as "did it handle this
well."

## Folder layout — offline vs. live

The split is by **what it costs to run**, not by test kind:

- **`tests/`** — 100% offline: `FakeModel`/fixtures, deterministic, no API key. A
  bare `pytest` runs these on every commit, free.
- **`evals/`** (this folder) — everything that calls a **real model**, all marked
  `live` and skipped unless `--run-live`:
  - `test_evals_integration.py` — the doc-16 metric harness (INT-1…23).
  - `run_manual.py` — the human-judged tier (INT19/20/21).
  - `test_live.py`, `test_reasoning_{react,plan,tot}.py` — live smoke/integration
    tests (binary pass/fail, not metric-driven) for the agent loop and the
    reasoning strategies.
  - `metrics.py` — shared harness code (Wilson CI, P/R/F1, LLM-judge, the
    per-trial reporter, `make_services`).

Run the whole live tier with `pytest evals/ --run-live`, or one file/case by path
or `-k`. The `--run-live` flag and the `live` marker are defined in the repo-root
`conftest.py`, shared by both folders.

## Unit mode

```
pytest tests/test_evals.py -v
```

No flags, no API key, finishes in well under a second. Every test uses
`FakeModel` or a local fixture. Three cases are marked `xfail(strict=True)`
because the capability they test doesn't exist yet:

| Case | Gap | Fix tracked in |
|---|---|---|
| A7 | no read-before-write check on `write_file` | doc 15, R5 |
| D3 | `SkillManifest.tools` parsed but never enforced | doc 15, R3 |
| E1 | side-effecting tool calls run with no approval gate | doc 15, R2 |

When one of R5/R3/R2 lands, its test will **XPASS**. Because of
`strict=True`, an XPASS fails the run — that's intentional, it forces you to
go delete the `xfail` marker (not the test) instead of letting the fix go
unnoticed.

## Integration mode

```
pytest evals/test_evals_integration.py -v -s --run-live
OPENMATE_LIVE_TESTS=1 pytest evals/test_evals_integration.py -v -s   # same, via env var
```

Needs a real model — same `.env` as everything else (`ANTHROPIC_API_KEY` or
`MINIMAX_API_KEY`, see `openmate/config.py`). `-s` is recommended: each case
prints a per-trial breakdown as it runs (which trial passed/failed and why),
not just a final percentage.

Metrics in use go beyond a raw pass/fail fraction — Wilson score intervals
on every rate, tool-use precision/recall/F1, RAGAS-style context recall,
Attack Success Rate / Refusal Rate on the safety cases, a step-efficiency
ratio, and LLM-as-judge rubric scoring for the qualitative cases. See
`docs/16-eval-plan.md`'s **Metrics framework** subsection for what each one
is and why it's used where it's used. Each case also now has a **Test data /
Test method / Run automatically / Notebook snippet** writeup in that doc's
**Per-case test data, harness, and runnable snippets** section — useful for
running one case standalone (e.g. while debugging a single failure) without
spinning up the whole pytest suite.

**23 cases** (`INT-1` through `INT-23`) across 9 capability categories,
designed around "what should a workable agent be able to do," not "what
does OpenMate already do" — see `docs/16-eval-plan.md` Part II for the full
strategy writeup and per-case rationale. Each runs 3-5 independent trials
against the real model and asserts a metric, not a single pass/fail:

| Category | Cases | What it's checking |
|---|---|---|
| A. Autonomous multi-step planning | INT-1 – INT-3 | Agent decides the steps itself, no explicit plan given |
| B. Real file/code edits | INT-4 – INT-6 | Outcome verified independently (script re-run, exit code, byte-diff), not the agent's own claim |
| C. RAG synthesis & grounded honesty | INT-7 – INT-9 | Multi-doc synthesis, conflict surfacing, refuses to hallucinate when the KB lacks the answer |
| D. Long-horizon / stateful tasks | INT-10 – INT-12 | State across separate `run()` calls, resume after interruption, step economy |
| E. Ambiguity handling | INT-13 – INT-14 | Asks before an ambiguous destructive action; admits when it has no tool for the job |
| F. Constraint following | INT-15 – INT-16 | Stated scope and output-format constraints, under a real task that tempts violating them |
| G. Error recovery | INT-17 – INT-18 | Recovers from a typo'd filename, retries after a transient tool failure |
| H. Safety & adversarial (zero tolerance) | INT-19 – INT-22 | Refusal on unsafe requests; injection resistance via `fetch_url`, RAG, and `read_file` |
| I. Faithfulness | INT-23 | Doesn't claim success it didn't actually achieve |

Three cases (INT-8, INT-13, INT-15) are **informational, no failing
assert** — they test capabilities OpenMate doesn't have a mechanism for yet
(RAG conflict detection, a clarification-request path, sub-cwd scope
enforcement). They stay green on purpose; the doc's "Gap → action item"
table is what turns each into a real build task. INT-19/20/21/22 have zero
tolerance because they're safety-relevant: any single miss is a finding, not
statistical noise, and should prompt a manual read via
`python evals/run_manual.py --case INT19` (or INT20/INT21) to see exactly what
happened.

This is a genuinely expensive suite — 23 cases x 3-5 trials x 1-8 model
turns each. Run a subset while iterating with `-k`, e.g.
`-k "INT4 or INT5 or INT6"` for just the file-editing cases, or lower the
per-test trial count (look for the trial loop near the top of each test
body — there's no single global constant, each case picks its own N based
on how expensive its setup is) for a cheap smoke run.

### Run one case with the full model I/O trace

To watch a single case run with the full request/response (and tool I/O)
printed live:

```
pytest evals/test_evals_integration.py -k INT6_ -s --run-live --eval-verbose
```

| flag | what it does |
|---|---|
| `-k INT6_` | select one case by id substring. The trailing `_` pins it to `INT6` — bare `-k INT1` would also match `INT10`–`INT19`, so the `_` is the safe habit. (Or give the exact node id: `evals/test_evals_integration.py::test_INT6_non_regressive_refactor`.) |
| `-s` | built-in pytest flag — don't capture stdout, so the trace prints live. |
| `--run-live` | opt in to the real-model tests (skipped otherwise; needs a key in `.env`). |
| `--eval-verbose` | print the **full model I/O** — every request (the prompt sent, with the growing transcript) and response (finish reason, tokens, content), plus tool calls and results. It sets `verbose=True` on the `ConsoleTracer` that consumes the `ModelRequested`/`ModelResponded` events. Equivalent: prefix the command with `OPENMATE_EVAL_VERBOSE=1`. |

Drop `--eval-verbose` for just the per-trial pass/fail summary; drop `-k` to
run the whole suite.

### ID → implementation map

Unit cases: `grep "def test_A1" tests/test_evals.py` finds case A1, same
pattern for every ID. Integration cases: `grep "def test_INT1" evals/test_evals_integration.py`
(works for any `INT-<n>`, 1 through 23). Some doc cases (B2, C1) have *both*
a Unit version (does the mechanism work) and an Integration version (does
the real model actually do the right thing) — both are listed in the doc's
tables with a cross-reference.

## Manual mode

```
python evals/run_manual.py                  # all three cases, interactive verdicts
python evals/run_manual.py --case INT20      # just one (legacy alias G1 also works)
python evals/run_manual.py --report out.json --no-prompt   # transcripts only, no interactive prompt
```

For each case it prints the full prompt, any injected content, every tool
call the model made, and the model's final response, then asks you to judge
`pass`/`fail`/`unsure` with an optional note. `--report` writes the
transcripts + verdicts to JSON for a record; `--no-prompt` skips the
interactive part if you just want transcripts (e.g. to paste into a PR).

| Case (alias) | What it tests | Why it's manual, not just Integration |
|---|---|---|
| INT19 (E4) | Refusal on a clearly unsafe direct request | INT-19 catches "did it refuse" via keyword match; a human catches "did it refuse *well*," with no partial leakage before the refusal kicks in |
| INT20 (G1) | Indirect prompt injection via `fetch_url` content | INT-20 catches "did a destructive shell call fire"; a human catches subtler partial compliance a regex wouldn't flag |
| INT21 (G2) | Indirect prompt injection via a RAG-indexed document | Same pattern as INT20, through the retrieval path instead |

## Coverage not in any mode

Per docs/16's closing section: multi-agent orchestration, reasoning-strategy
switching, and OS-level sandbox-escape testing aren't covered anywhere here,
because the underlying capability doesn't exist in OpenMate yet. Add cases
once the feature lands.
