# DS4-34: Prefix Stability Checker

**Status:** Implemented

## Purpose

A standalone utility that compares successive generated text outputs to
detect when the model has settled on a stable prefix. Used internally by
the streaming path to avoid emitting redundant events and to detect
prefix convergence.

## API

### `PrefixStabilityChecker`

- `new(config: StabilityConfig)` — constructor
- `check(old: &str, new: &str) -> (bool, String)` — returns (is_stable, stable_prefix)
- `diff(old: &str, new: &str) -> Vec<DiffOp>` — token-level diff

### `StabilityConfig`

- `prefix_char_count: usize` — how many characters of prefix to examine
  (default: 100)

### `DiffOp`

- `Equal { text }` — common substring
- `Insert { text }` — text only in new
- `Delete { text }` — text only in old
- `Replace { old, new }` — differing substring

## Implementation

- Manual string scanning, no regex dependency
- Prefix check: byte-level comparison of first N characters
- Diff: common prefix + middle (replace/insert/delete) + common suffix
