---
name: repo_triage
description: Turn an issue report into a ranked list of candidate edit sites using the repository's code graph and embeddings, before any file is opened.
---

# repo_triage

Use this skill at the start of a task, and again whenever a first edit did not
make the targeted test pass.

## When to use it

- The issue text names a behaviour but no file.
- The obvious grep returns more than five plausible sites.
- A first edit failed and you need a second hypothesis rather than a second
  guess.

## Procedure

1. Extract 3-8 keywords from the issue: exception type, function name, option
   name, user-visible string.
2. `search_similar_code(query=<keywords>, k=10)`.
3. For each of the top three hits, `get_code_neighbors(node=<hit>)`.
4. Rank the hits by: (a) does a test file for this module exist, (b) is the
   symbol exported / public, (c) is the hit on the path the issue describes.
5. Return the top site and one fallback.

## Resources

- `resources/navigation_protocol.md` -- the ordering rules and the reasons.
- `resources/issue_taxonomy.md` -- how issue text maps onto code locations.
