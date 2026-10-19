# Why patch hygiene is scored, not cosmetic

The grader applies the patch and runs the issue's validation tests. A patch fails that step for reasons that have nothing to do with whether the logic is right:

- a hunk whose line offsets are wrong does not apply at all;
- a CRLF rewrite makes a three-line fix unreadable and unmergeable;
- editing the test file makes the run validate the wrong contract;
- a stray `print(` in a hot path can change output that tests assert on.

Each of these is cheap to detect and cheap to fix, and each of them converts a working fix into a zero.
