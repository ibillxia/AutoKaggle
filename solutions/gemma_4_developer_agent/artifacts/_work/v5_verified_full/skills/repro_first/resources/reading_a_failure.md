# Reading a failure cheaply

A pytest failure is three useful lines inside two hundred useless ones.

1. The `E   ` line: what was expected, what was received.
2. The last `File "..."` frame in the traceback: the code you will edit.
3. The assertion line in the test: the contract you must satisfy.

Everything else -- the platform banner, the plugin list, the rootdir, the number of deselected tests -- is noise. Discard it before it occupies your context.
