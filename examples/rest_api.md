# REST API with JWT Auth

Build a REST API using Python and FastAPI with the following requirements:

## Features
- JWT-based authentication (register, login, refresh token)
- CRUD operations for a "projects" resource (create, read, update, delete)
- Pagination with cursor-based navigation (limit, after/before cursor)
- Input validation with Pydantic models
- Role-based access control (admin, member, viewer)

## Technical Requirements
- FastAPI with async endpoints
- SQLAlchemy 2.0 with async engine (SQLite for dev, PostgreSQL-ready)
- Alembic migrations
- Password hashing with bcrypt
- JWT tokens with configurable expiry
- Proper HTTP status codes and error responses
- OpenAPI docs auto-generated

## Endpoints
- `POST /auth/register` — Create account
- `POST /auth/login` — Get access + refresh tokens
- `POST /auth/refresh` — Refresh access token
- `GET /projects` — List projects (paginated)
- `POST /projects` — Create project
- `GET /projects/{id}` — Get project details
- `PUT /projects/{id}` — Update project
- `DELETE /projects/{id}` — Delete project (admin only)

## Tests
- Auth flow (register → login → access protected endpoint)
- CRUD operations with valid and invalid data
- Pagination edge cases (empty, first page, last page)
- Authorization checks (viewer can't delete, admin can)
- Token expiry and refresh flow

```bash
triad run --task "$(cat examples/rest_api.md)"
```
