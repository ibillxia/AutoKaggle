# Issue taxonomy -> where the fix lives

| the issue says | the fix is usually in |
|---|---|
| a raised exception with a type name | the raise site and its input validation |
| "returns None / wrong value" | the return path, often one early-exit branch |
| "should accept X" | the argument parser / validator, not the business logic |
| "crash on empty input" | a guard clause at the top of the public entry point |
| "regression since ..." | a recently added branch; check the newest code in the hot path |
| "slower than expected" | an accidental nested loop or repeated recomputation |
| "documentation says A, code does B" | the code, not the documentation, unless the test agrees with the documentation |

## Rule of thumb

The smallest diff that makes the *validation test* pass without changing the
test is almost always the right diff.
