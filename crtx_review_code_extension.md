# CRTX Task: Extend review-code for Multi-File External Review

## Context

CRTX has `crtx review-code <file>` for single-file code review. We need it to support reviewing multiple files from external projects (outside the current CRTX directory). This is for using CRTX as a code review step when building other projects.

## What to Check First

1. Open `cli.py` (or wherever the `review-code` command is defined)
2. Check: does `review-code` already accept multiple file paths?
3. Check: does it work with absolute paths or paths outside the current directory?
4. Check: does it pass file contents to the Arbiter/review pipeline correctly?

Report what you find before making changes.

## What to Build (if not already supported)

### Multi-file support:
```bash
# Current (single file):
crtx review-code src/event_bus.py

# Needed (multiple files):
crtx review-code src/event_bus.py src/models.py tests/test_event_bus.py

# Needed (glob pattern):
crtx review-code src/*.py

# Needed (directory):
crtx review-code src/
```

### External path support:
```bash
# Review files from another project:
crtx review-code C:\Users\Adam\documents\clawbucks-platform\core\event_bus.py

# Or with relative paths:
crtx review-code ..\clawbucks-platform\core\event_bus.py

# Or review a whole module:
crtx review-code ..\clawbucks-platform\core\
```

### New flag — review with context:
```bash
# Pass a spec/requirements file so the Arbiter knows what the code SHOULD do:
crtx review-code ..\clawbucks-platform\core\event_bus.py --spec ..\clawbucks-platform\CURRENT_BU.md
```

This tells the Arbiter: "Review this code against this spec. Does it implement what the spec requires?"

### Implementation

The `review-code` command should:

1. **Resolve paths** — accept relative, absolute, globs, and directory paths
2. **Read files** — load content from all specified files
3. **Concatenate with file markers** — pass to review pipeline as:
   ```
   === FILE: core/event_bus.py ===
   [file content]
   
   === FILE: core/models.py ===
   [file content]
   
   === FILE: tests/test_event_bus.py ===
   [file content]
   ```
4. **If --spec provided** — prepend the spec content as review criteria:
   ```
   === SPEC/REQUIREMENTS ===
   [spec content]
   
   === FILES TO REVIEW ===
   [files as above]
   ```
5. **Arbiter review prompt** — adjust the review system prompt when --spec is present:
   ```
   Review the following code files against the provided specification.
   Check:
   1. Does the implementation match what the spec requires?
   2. Are all acceptance criteria from the spec addressed?
   3. Are there bugs, edge cases, or missing error handling?
   4. Are the interfaces clean and consistent?
   5. Are tests comprehensive?
   
   Return structured verdict: APPROVE / FLAG / REJECT with specific issues.
   ```

6. **Output** — structured verdict with per-file feedback:
   ```
   ╭─ CRTX Code Review ─────────────────────────────╮
   │                                                  │
   │  Files reviewed: 3                              │
   │  Spec: CURRENT_BU.md                            │
   │                                                  │
   │  core/event_bus.py ................ ✅ APPROVE   │
   │    Clean implementation. All EventBus methods    │
   │    match spec. WAL mode set correctly.           │
   │                                                  │
   │  core/models.py ................... ✅ APPROVE   │
   │    Models match spec exactly.                    │
   │                                                  │
   │  tests/test_event_bus.py .......... ⚠️ FLAG     │
   │    Missing test for stream with `since` param.   │
   │    Missing performance test (1000 events/5s).    │
   │                                                  │
   │  Overall: FLAG — 2 test gaps to address          │
   │  Confidence: 0.87                                │
   │                                                  │
   ╰─────────────────────────────────────────────────╯
   ```

## What NOT to Change

- Do not modify the core Arbiter engine
- Do not change how single-file review works (backward compatible)
- Do not modify the Loop, Router, or Memory modules
- Do not change the review-code → improve → apply flow (just extend the input)

## Acceptance Criteria

- [ ] `crtx review-code file1.py file2.py` reviews multiple files
- [ ] `crtx review-code ../other-project/src/` reviews a directory
- [ ] `crtx review-code file.py --spec spec.md` reviews against a spec
- [ ] Absolute and relative paths work
- [ ] Glob patterns work (*.py)
- [ ] Output shows per-file verdict
- [ ] Single-file review still works exactly as before
- [ ] Existing tests still pass
