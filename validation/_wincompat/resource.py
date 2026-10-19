"""Minimal `resource` shim for Windows.

kev uses it in exactly two places: disabling core dumps (device.py) and reporting peak RSS
(train.py). Neither is load-bearing on Windows, so the shim no-ops one and reports the best
available RSS figure for the other.
"""
import sys

RLIMIT_CORE = 0
RUSAGE_SELF = 0


def setrlimit(_which, _limits):
    return None


class _Usage:
    ru_maxrss = 0


def getrusage(_who):
    # Windows has no ru_maxrss; fall back to the process working set so the log line stays real.
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        # ru_maxrss is in KB on Linux, bytes on macOS; train.py multiplies by 1024 off darwin,
        # so hand back KB to keep the arithmetic right.
        _Usage.ru_maxrss = c.PeakWorkingSetSize // 1024
    except Exception:
        _Usage.ru_maxrss = 0
    return _Usage()


def getpagesize():
    return 4096


# sys.platform on Windows is "win32", which is not "darwin", so train.py's `* 1024` factor applies.
if sys.platform == "win32":
    pass
