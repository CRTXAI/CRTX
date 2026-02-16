# Task Planner

**Pre-Pipeline Prompt Generator**

*Describe what you want. Get a structured task spec. Feed it to the pipeline.*

**Status:** Post-MVP · Target: v0.2.0

---

## Problem

Today, getting a good result from `triad run` requires writing a detailed task prompt — tech stack, requirements, constraints, file patterns, edge cases. Most developers start by describing what they want to a separate LLM session ("I need a REST API for task management with auth"), iterate on the prompt, then paste the refined version into Triad.

That extra window shouldn't be necessary. The pipeline should help you write the prompt.

## Solution

A single pre-pipeline command that expands a rough idea into a structured task specification:

```
triad plan "REST API for task management with user auth and CRUD"
```

One LLM call. One system prompt (`prompts/planner.md`). The output is a structured task spec that the user reviews, edits if needed, and then runs.

## User Flow

### Quick Mode (default)

```
$ triad plan "REST API for task management with auth"

── Task Planner ─────────────────────────────────────────
Model: claude-sonnet (planner)
Expanding your idea...

── Generated Task Spec ──────────────────────────────────

Build a REST API for a task management system with:

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy 2.0, PostgreSQL, Pydantic v2

**Endpoints:**
- POST /auth/register — Create new user account
- POST /auth/login — Authenticate and return JWT
- POST /auth/refresh — Refresh expired JWT
- GET /tasks — List tasks (filterable by status, priority, assignee; paginated)
- POST /tasks — Create task
- GET /tasks/{id} — Get task by ID
- PUT /tasks/{id} — Update task
- DELETE /tasks/{id} — Soft-delete task
- POST /tasks/{id}/assign — Assign task to user

**Data Models:**
- User: id, email, hashed_password, name, role (admin/member), created_at
- Task: id, title, description, status (todo/in_progress/done), priority (low/medium/high/urgent), assignee_id (FK), creator_id (FK), due_date, created_at, updated_at

**Requirements:**
- JWT auth with access + refresh tokens, bcrypt password hashing
- Role-based access: admins can manage all tasks, members only their own
- Pagination on list endpoints (limit/offset, default 20)
- Input validation on all endpoints via Pydantic
- Proper error responses (400, 401, 403, 404, 422) with consistent error schema
- Soft delete (is_deleted flag, excluded from queries by default)
- Comprehensive pytest suite

──────────────────────────────────────────────────────────
[r] Run pipeline  [e] Edit in $EDITOR  [s] Save to file  [q] Quit
>
```

### Interactive Mode

```
$ triad plan --interactive

── Task Planner (Interactive) ───────────────────────────
What do you want to build?
> A task management API

What tech stack? (or press Enter for auto-detect)
> FastAPI, PostgreSQL

Any specific requirements?
> needs JWT auth, role-based access, soft deletes

Any patterns or conventions to follow?
> follow repository pattern, all async

Generating structured task spec...
```

### Pipe to Run

```
# Generate and run immediately (skips review)
$ triad plan "REST API with auth" | triad run --stdin

# Save spec, then run it later
$ triad plan "REST API with auth" --save tasks/api-spec.md
$ triad run --task-file tasks/api-spec.md
```

## Planner Prompt Design

The planner uses a single system prompt (`prompts/planner.md`) that instructs the LLM to:

1. **Expand** the rough idea into concrete requirements
2. **Infer** the tech stack if not specified (based on keywords and conventions)
3. **Enumerate** endpoints, data models, and behaviors explicitly
4. **Add** standard requirements the user likely wants but didn't mention (error handling, validation, pagination, tests)
5. **Structure** the output in a format the pipeline's Architect prompt can consume directly

The planner does NOT generate code. It generates the *task description* that the pipeline's four stages will implement.

## Implementation

### Files

```
triad/
├── planner.py             # Plan generation logic
├── prompts/
│   └── planner.md         # Planner system prompt (Jinja2)
└── cli.py                 # Add `triad plan` command
```

### Schema Addition

```python
class PlannerResult(BaseModel):
    """Output from the task planner."""
    original_input: str          # What the user typed
    expanded_spec: str           # Full structured task spec
    inferred_tech_stack: str     # Detected/suggested tech stack
    model_used: str              # Which model generated the plan
    token_usage: TokenUsage      # Cost of the planning call
```

### CLI Commands

```
triad plan <description>           # Quick mode — expand and show
triad plan --interactive           # Ask clarifying questions first
triad plan <desc> --save <file>    # Save spec to file
triad plan <desc> --run            # Expand then immediately run pipeline
triad plan <desc> --model <model>  # Use specific model for planning
```

### Model Selection

The planner defaults to the cheapest model that meets a minimum fitness threshold for the "architect" role (since planning is closest to architecture). For most registries, this will be Gemini Flash or Claude Haiku — fast, cheap, and good enough for prompt expansion. The user can override with `--model`.

## Scope Boundaries

**In scope:**
- Single-shot prompt expansion (one LLM call)
- Interactive mode with clarifying questions (2-3 LLM calls)
- Save/edit/run flow
- Task file format that `triad run --task-file` can consume

**Out of scope (future):**
- Multi-turn conversation ("tell me more about the auth requirements")
- Codebase-aware planning (scan existing code to inform the spec)
- Template library (pre-built specs for common patterns)
- Plan versioning and diffing

## Why This Matters

The quality of Triad's output is directly proportional to the quality of the input task spec. A vague prompt produces vague architecture. A detailed spec produces clean, complete code. The planner closes the gap between "I have an idea" and "I have a spec the pipeline can execute on" — without the user needing a separate LLM session to get there.

For the open-source story, this also lowers the barrier to trying Triad. Instead of "write a detailed prompt and hope for the best," the onboarding becomes "describe what you want in one sentence and let the planner do the rest."
