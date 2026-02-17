# WebSocket Chat System

Build a real-time chat system with WebSocket support, rooms, presence tracking, and message history.

## Features
- WebSocket-based real-time messaging
- Chat rooms (create, join, leave, list members)
- User presence (online/offline/typing indicators)
- Message history with pagination (last N messages on room join)
- Message types: text, system (join/leave notifications), typing indicator
- Reconnection handling with message catch-up

## Technical Requirements
- Python with FastAPI and WebSocket support
- In-memory room/presence state (no database required for MVP)
- JSON message protocol with typed envelopes
- Heartbeat/ping-pong for connection health
- Graceful disconnect handling (remove from room, notify others)
- Rate limiting (max 10 messages/second per client)
- Message size limit (4KB max)

## Message Protocol
```json
{"type": "join", "room": "general", "user": "alice"}
{"type": "message", "room": "general", "content": "Hello!", "user": "alice"}
{"type": "typing", "room": "general", "user": "alice", "typing": true}
{"type": "leave", "room": "general", "user": "alice"}
```

## Server Events
```json
{"type": "user_joined", "room": "general", "user": "alice", "members": ["alice", "bob"]}
{"type": "message", "room": "general", "user": "alice", "content": "Hello!", "timestamp": "..."}
{"type": "user_left", "room": "general", "user": "alice", "members": ["bob"]}
{"type": "presence", "room": "general", "user": "alice", "status": "typing"}
{"type": "history", "room": "general", "messages": [...]}
```

## Tests
- Connection lifecycle (connect → join → message → leave → disconnect)
- Multi-room support (user in multiple rooms, messages routed correctly)
- Presence accuracy (typing indicators, online/offline transitions)
- Message history (join room, receive last 50 messages)
- Rate limiting (exceed limit → connection warning)
- Reconnection (disconnect, reconnect, receive missed messages)
- Edge cases (empty room, last user leaves, duplicate join)

```bash
triad run --task "$(cat examples/websocket_chat.md)"
```
