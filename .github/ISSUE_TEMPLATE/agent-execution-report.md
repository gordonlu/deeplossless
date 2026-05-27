---
name: Agent Execution Report
about: Report interesting, suspicious, or misleading AI execution behavior.
title: ''
labels: integrity
assignees: ''

---

# Agent Execution Report

Describe interesting, suspicious, misleading, or unexpected AI execution behavior.

Examples:

* claimed tests passed without running tests
* silent fallback behavior
* retry loops
* fake completion
* plan divergence
* hidden tool failure
* incomplete verification

---

## Agent / Environment

* Agent:
* Runtime:
* Model:
* Toolchain:
* deeplossless version:

---

## Task

What was the agent trying to do?

---

## Claimed Outcome

What did the agent claim happened?

Examples:

* "tests passed"
* "issue fixed"
* "build successful"

---

## Observed Execution

What actually happened?

Include:

* commands executed
* missing verification steps
* retries/fallbacks
* execution anomalies
* replay observations

---

## Verification Gap / Anomaly

Describe the mismatch between the claim and observed execution.

---

## Replay / Evidence

If available, include:

* replay screenshots
* trace snippets
* logs
* execution timeline
* diff excerpts

Please redact secrets, credentials, and sensitive traces.

---

## Additional Context

Anything else that may help explain the behavior.
