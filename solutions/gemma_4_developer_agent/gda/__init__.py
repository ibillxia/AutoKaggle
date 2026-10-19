"""Gemma 4 Developer Agent solution package.

A competition-specific package that

1. defines several *candidate* agent configurations (submission.zip layouts)
   that differ in prompt strategy, tool policy, sub-agent decomposition and
   skills;
2. builds each candidate into a real ``submission.zip``;
3. validates each zip against the harness constraints that are checkable
   offline (file-extension allowlist, closed tool registry, !include
   resolution, sub-agent wiring, sampling keys, budget arithmetic);
4. scores each candidate with a local **proxy**: a deterministic rubric plus
   real execution of the candidate's own skill scripts against fixture repos.

The proxy does NOT reproduce the official score. It reproduces the *gates* and
the *engineering discipline* that the official harness rewards, which is the
only thing that can be measured without the organiser's Docker sandbox, the
``gemma-4-31b-it-qat-w4a16-ct`` weights and the hidden issues.
"""

from gda.paths import (  # noqa: F401
    ALLOWED_EXTENSIONS,
    COMPETITION,
    REQUIRED_MODEL,
    TOOL_REGISTRY,
)

__version__ = "0.1.0"
