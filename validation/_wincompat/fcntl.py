"""Minimal fcntl shim for Windows.

kev imports `fcntl` only to take an advisory lock while rewriting its suite cache
(kev/suite.py:159, kev/experiment.py:155). Windows has no fcntl; flock() there is a
no-op, which is safe for a single-process evaluation run.
"""
LOCK_EX = 2
LOCK_SH = 1
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation):  # noqa: ARG001 - signature parity with the stdlib
    return None


def lockf(fd, cmd, len=0, start=0, whence=0):  # noqa: A002, ARG001
    return None
