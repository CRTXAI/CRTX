# Refactor Legacy Module

Refactor the following legacy code into clean, testable, well-structured Python. Preserve all existing behavior while improving readability, maintainability, and test coverage.

## Legacy Code

```python
import json, os, time, hashlib
from datetime import datetime

data_store = {}
LOG = []

def proc(inp, mode="default", dbg=False):
    global data_store, LOG
    t = time.time()
    if dbg: print(f"DEBUG: proc called with mode={mode}")

    if not inp or len(inp) == 0:
        return {"error": "empty", "code": 400}

    if type(inp) == str:
        try:
            inp = json.loads(inp)
        except:
            return {"error": "bad json", "code": 400}

    if type(inp) != dict:
        return {"error": "not dict", "code": 400}

    k = inp.get("key", None)
    v = inp.get("value", None)
    if k is None:
        return {"error": "no key", "code": 400}

    h = hashlib.md5(str(k).encode()).hexdigest()[:8]

    if mode == "default" or mode == "upsert":
        old = data_store.get(h, None)
        data_store[h] = {"key": k, "value": v, "updated": datetime.now().isoformat(), "hash": h}
        LOG.append({"action": "upsert", "hash": h, "time": t, "had_old": old is not None})
        if dbg: print(f"DEBUG: stored {h}")
        return {"ok": True, "hash": h, "is_update": old is not None}

    elif mode == "get":
        r = data_store.get(h, None)
        if r is None:
            return {"error": "not found", "code": 404}
        LOG.append({"action": "get", "hash": h, "time": t})
        return {"ok": True, "data": r}

    elif mode == "delete":
        if h not in data_store:
            return {"error": "not found", "code": 404}
        del data_store[h]
        LOG.append({"action": "delete", "hash": h, "time": t})
        return {"ok": True, "deleted": h}

    elif mode == "list":
        items = list(data_store.values())
        LOG.append({"action": "list", "time": t, "count": len(items)})
        return {"ok": True, "items": items, "count": len(items)}

    elif mode == "export":
        fn = inp.get("filename", "export.json")
        with open(fn, "w") as f:
            json.dump(list(data_store.values()), f)
        LOG.append({"action": "export", "time": t, "file": fn})
        return {"ok": True, "file": fn, "count": len(data_store)}

    elif mode == "stats":
        return {"ok": True, "total": len(data_store), "log_entries": len(LOG),
                "oldest": LOG[0]["time"] if LOG else None,
                "newest": LOG[-1]["time"] if LOG else None}
    else:
        return {"error": f"unknown mode: {mode}", "code": 400}
```

## Refactoring Goals
- Replace global mutable state with a proper class
- Replace bare `except` with specific exception handling
- Use `isinstance()` instead of `type() ==`
- Replace string-based mode dispatch with proper methods or enum
- Add type hints to all functions
- Add Pydantic models for input validation and response types
- Add proper error classes instead of error dicts
- Make file operations use pathlib
- Add docstrings
- Write comprehensive tests covering all operations and edge cases

## Constraints
- All original behavior must be preserved (same inputs produce equivalent outputs)
- No external dependencies beyond stdlib + Pydantic
- The refactored code should be easy for a new developer to understand

```bash
triad run --task "$(cat examples/refactor_legacy.md)"
```
