---
name: repro_first
description: Reproduce the reported failure and record it before editing anything, so the fix is aimed at a confirmed failure mode rather than a guess.
---

# repro_first

## When to use it

- At the very start of every task, before the first edit.
- After any edit, to confirm the failure is gone (it is the same command).

## Procedure

Run the bundled script:

```bash
python skills/repro_first/scripts/run_targeted_test.py \
    --workspace /workspace --test <path/to/test_file.py>
```

It executes the targeted test with a bounded timeout, prints a one-line verdict,
and writes a machine-readable record to `<workspace>/.repro/repro_log.json`:

```json
{"command": "...", "returncode": 1, "passed": false, "duration_s": 0.42, "tail": "..."}
```

## How to read the result

- `passed: false` **before** your edit -- you have a reproduction. Good.
- `passed: true` **before** your edit -- the test already passes; you are
  probably looking at the wrong test, or the bug is only visible under a
  different invocation. Re-check the issue before editing.
- `passed: false` **after** your edit -- your fix is incomplete or wrong. Revert
  it, go back to `repo_triage`, and form a second hypothesis. Do not stack a
  second speculative edit on top of a failed one.

## Why it matters

Editing without a reproduction is how agents produce plausible, wrong patches
that cost a full task of budget and score zero.
