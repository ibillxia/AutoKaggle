"""AutoKaggle **DevAgent mode**: a judged-harness SOP for code-agent competitions.

``multi_agents/hackathon`` adapted AutoKaggle to a competition whose deliverable
is a writeup judged by a rubric. This mode adapts it to a competition whose
deliverable is an **agent configuration** judged by a *harness*:

    the tabular SOP        : train.csv -> ... -> submission.csv
    the hackathon SOP      : data     -> ... -> ≤2000-word writeup + notebook
    the DevAgent SOP (here): brief    -> ... -> submission.zip (agent.yaml + skills)

Two structural consequences, and they are why this is a separate mode rather
than a flag on the hackathon one:

1.  There is **no candidate metric at all**. A submission is scored by an
    external harness that has to be run by the organiser, so nothing in this
    repository can produce a leaderboard number. The SOP therefore optimises and
    gates **format compliance** and **engineering discipline**, and refuses to
    state a task pass rate.
2.  The unit tests check the *artefact contract* (agent.yaml at the zip root,
    closed tool registry, resolvable !includes, budget arithmetic) instead of
    dataframe properties.

Like the hackathon pipeline, the phase work is deterministic and needs no LLM;
the SOP's agent roles describe who owns which decision, not an API call.
"""

__all__ = ["state", "unit_tests", "phases", "sop"]
