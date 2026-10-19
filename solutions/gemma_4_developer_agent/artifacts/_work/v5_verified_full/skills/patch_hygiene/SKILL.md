---
name: patch_hygiene
description: Inspect the working-tree diff before submitting and reject it if it contains debug leftovers, edits to test files, path traversal or whitespace damage.
---

# patch_hygiene

## When to use it

Immediately before `submit_patch()`, every time. Also when an edit produced a
surprisingly large diff.

## Procedure

```bash
python skills/patch_hygiene/scripts/check_patch.py --workspace /workspace
```

It stages intent-to-add (`git add -N .`), reads `git diff HEAD`, and writes
`<workspace>/.repro/patch_report.json`:

```json
{"ok": false, "n_files": 1, "n_added_lines": 4,
 "checks": [{"id": "no_debug_leftovers", "passed": false, "detail": "print( at line 12"}]}
```

## The checks, and why each one exists

| check | rejected because |
|---|---|
| `diff_non_empty` | an empty patch scores zero; you edited nothing |
| `diff_parses` | a malformed hunk is rejected by `git apply` on the grader side |
| `no_test_file_edits` | grading runs the issue's own tests; editing them cannot help and looks like gaming |
| `no_debug_leftovers` | `print`, `breakpoint`, `pdb.set_trace`, `TODO XXX` left in the diff are the classic small-model residue |
| `no_path_traversal` | `../` in a diff header escapes the sandbox |
| `no_whitespace_damage` | CRLF or leading tabs turn a 3-line fix into a 300-line diff |
| `no_binary_or_huge` | a vendored blob or minified asset swamps the patch |

## On failure

Fix the reported finding and run it again. Do not submit a diff that failed
hygiene -- the checks are cheap and the failure modes are exactly the ones that
turn a correct fix into a rejected patch.
