# Concurrency Model - Single Process, Async Event Loop

## Answer: Single Process, Async Event Loop (No Threads)

The servers use **async/await with a single event loop** - not threads or processes.

## Architecture

### Server Type: ASGI (Async Server Gateway Interface)

**Green Agent:**
```python
# Line 512-514
uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
uvicorn_server = uvicorn.Server(uvicorn_config)
await uvicorn_server.serve()  # Single async event loop
```

**Purple Agent:**
```python
# Same pattern - uvicorn with async event loop
```

### How It Works

**Single Process:**
- One Python process per agent (green or purple)
- One event loop (asyncio event loop)
- No threads spawned per request
- No processes spawned per request

**Async Concurrency:**
- All handlers are `async def` functions
- Requests are handled concurrently via async/await
- When one request is waiting (e.g., downloading files, calling LLM), the event loop handles other requests

## Example: 2 Concurrent Tasks

### Scenario: 2 ARVO tasks running simultaneously

**Process Structure:**
```
┌─────────────────────────────────────────┐
│  Green Agent Process (Single Process)  │
│  ┌───────────────────────────────────┐ │
│  │  asyncio Event Loop (Single)      │ │
│  │                                    │ │
│  │  Request 1 (context_id="abc-123")  │ │
│  │  ├─ async def execute()               │ │
│  │  ├─ await download_files()         │ │ ← Waiting for I/O
│  │  └─ ...                             │ │
│  │                                    │ │
│  │  Request 2 (context_id="xyz-789")  │ │
│  │  ├─ async def execute()            │ │ ← Can run while Request 1 waits
│  │  ├─ await download_files()         │ │
│  │  └─ ...                             │ │
│  │                                    │ │
│  │  Request 3 (context_id="abc-123")  │ │
│  │  ├─ async def execute()            │ │ ← Can run while others wait
│  │  ├─ await docker_exec()           │ │
│  │  └─ ...                             │ │
│  └───────────────────────────────────┘ │
│                                         │
│  Shared State:                          │
│  self.task_contexts = {                │
│    "abc-123": TaskContext(...),        │
│    "xyz-789": TaskContext(...),        │
│  }                                      │
└─────────────────────────────────────────┘
```

## How Concurrent Requests Work

### Request 1: Task Initialization (arvo:10055)
```python
# Request arrives
async def execute(context, event_queue):
    context_id = "abc-123"
    # Check state
    task_context = self.task_contexts.get("abc-123")  # None
    
    # Start downloading files (I/O operation)
    await prepare_task_assets(...)  # ← Yields control to event loop
    # While waiting, other requests can be processed!
    
    # Store state
    self.task_contexts["abc-123"] = TaskContext(...)
```

### Request 2: Task Initialization (arvo:10056) - While Request 1 is downloading
```python
# Request arrives (while Request 1 is still downloading)
async def execute(context, event_queue):
    context_id = "xyz-789"
    # Check state
    task_context = self.task_contexts.get("xyz-789")  # None
    
    # Start downloading files (I/O operation)
    await prepare_task_assets(...)  # ← Yields control to event loop
    # Both downloads can happen concurrently!
    
    # Store state
    self.task_contexts["xyz-789"] = TaskContext(...)
```

### Request 3: Command Execution (for Task 1)
```python
# Request arrives (while Requests 1 & 2 are downloading)
async def execute(context, event_queue):
    context_id = "abc-123"
    # Check state
    task_context = self.task_contexts.get("abc-123")  # Found!
    
    # Execute command in Docker (I/O operation)
    await docker_exec(...)  # ← Yields control to event loop
    # Can run while other requests wait!
```

## Key Points

### 1. Single Process, Single Event Loop
- ✅ One Python process per agent
- ✅ One asyncio event loop
- ✅ All requests handled by same process
- ❌ No threads spawned
- ❌ No processes spawned

### 2. Async Concurrency
- When `await` is called, control yields to event loop
- Event loop can process other requests while waiting
- Multiple requests handled concurrently (not in parallel)

### 3. Shared State
- `self.task_contexts` is shared across all requests
- All requests in same process access same dictionary
- No locking needed (single event loop = no race conditions)

### 4. I/O Operations
- File downloads: `await prepare_task_assets()` → yields to event loop
- Docker operations: `await docker_exec()` → yields to event loop
- LLM API calls: `await client.chat.completions.create()` → yields to event loop
- Network requests: `await send_message()` → yields to event loop

## Comparison

### Multi-Threaded (NOT USED)
```
Process
├── Thread 1: Handle Request 1
├── Thread 2: Handle Request 2
└── Thread 3: Handle Request 3
```
- ❌ Thread overhead
- ❌ Need locks for shared state
- ❌ GIL limitations in Python

### Multi-Process (NOT USED)
```
Process 1: Handle Request 1
Process 2: Handle Request 2
Process 3: Handle Request 3
```
- ❌ Process overhead
- ❌ Need IPC for shared state
- ❌ Higher memory usage

### Async Event Loop (ACTUALLY USED)
```
Process
└── Event Loop
    ├── Request 1: await download → yields
    ├── Request 2: await download → yields
    └── Request 3: await docker_exec → yields
```
- ✅ Low overhead
- ✅ Shared state (no locks needed)
- ✅ Efficient I/O concurrency
- ✅ Single-threaded (no GIL issues for I/O)

## Example Timeline

**Time 0ms:**
- Request 1 arrives: Task 10055
- Starts: `await prepare_task_assets()` (downloads files)
- Yields to event loop

**Time 10ms:**
- Request 2 arrives: Task 10056
- Starts: `await prepare_task_assets()` (downloads files)
- Yields to event loop

**Time 20ms:**
- Request 3 arrives: Command for Task 10055
- Starts: `await docker_exec()` (executes in ARVO)
- Yields to event loop

**Time 30ms:**
- Request 1's download completes
- Event loop resumes Request 1
- Stores task context

**Time 40ms:**
- Request 2's download completes
- Event loop resumes Request 2
- Stores task context

**Time 50ms:**
- Request 3's docker exec completes
- Event loop resumes Request 3
- Returns command result

## Benefits

1. **Efficient:** No thread/process overhead
2. **Scalable:** Can handle many concurrent requests
3. **Simple:** Shared state without locks
4. **Fast:** I/O operations don't block other requests

## Limitations

1. **CPU-bound operations block:** If you do heavy CPU work (no `await`), it blocks the event loop
2. **Single CPU core:** Can't use multiple CPU cores for CPU-bound work
3. **Shared state:** All requests share same memory (good for our use case)

## For Our Use Case

**Perfect because:**
- ✅ Most operations are I/O-bound (downloads, Docker, LLM API calls)
- ✅ Need shared state (`task_contexts` dictionary)
- ✅ Want efficient concurrency without complexity
- ✅ Multiple tasks can run simultaneously

## Summary

- **Single process** per agent (green or purple)
- **Single event loop** (asyncio)
- **No threads** spawned per request
- **No processes** spawned per request
- **Concurrent** handling via async/await
- **Shared state** (`self.task_contexts`) accessible by all requests

