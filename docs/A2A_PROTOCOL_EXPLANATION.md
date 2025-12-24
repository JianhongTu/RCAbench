# Understanding the A2A (Agent-to-Agent) Protocol

## What is A2A?

**A2A (Agent-to-Agent)** is an open standard protocol for agent interoperability. It defines how AI agents communicate with each other in a standardized way, regardless of:
- Programming language
- Framework or SDK
- Deployment location
- Implementation details

Think of it like **HTTP for agents** - it provides a common language that all agents can understand.

## Why A2A Exists

### The Problem

Without a standard protocol:
- Each agent would need custom integration code for every other agent
- Agents couldn't easily work together
- No interoperability between different platforms
- Difficult to build reusable agent ecosystems

### The Solution

A2A provides:
- ✅ **Standardized communication** - All agents speak the same language
- ✅ **Interoperability** - Any A2A-compliant agent can talk to any other
- ✅ **Platform independence** - Works across different systems
- ✅ **Task tracking** - Built-in support for long-running operations
- ✅ **Artifact sharing** - Standard way to share results and data

## Core Concepts

### 1. Agents as Servers

Every agent runs as an **HTTP server** that:
- Listens for incoming A2A messages
- Processes requests
- Sends A2A-formatted responses
- Exposes an **agent card** (metadata about capabilities)

```
┌─────────────┐         HTTP Request          ┌─────────────┐
│   Agent A   │ ────────────────────────────> │   Agent B   │
│  (Client)   │                                │  (Server)   │
└─────────────┘ <──────────────────────────── └─────────────┘
                A2A-formatted Response
```

### 2. Agent Card

Every agent exposes an **agent card** at `/.well-known/agent-card.json`:

```json
{
  "name": "debater",
  "description": "Participates in a debate.",
  "url": "http://127.0.0.1:9019/",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "capabilities": {
    "streaming": true
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "preferredTransport": "JSONRPC",
  "skills": []
}
```

**Purpose:**
- Describes what the agent can do
- Specifies supported protocols and formats
- Allows clients to discover agent capabilities
- Similar to API documentation

**Discovery Flow:**
```
1. Client knows agent URL: http://127.0.0.1:9019/
2. Client fetches: http://127.0.0.1:9019/.well-known/agent-card.json
3. Client reads capabilities and protocol version
4. Client creates appropriate client based on agent card
```

### 3. Messages

**Messages** are the basic unit of communication in A2A.

**Structure:**
```python
Message(
    kind="message",
    role=Role.user,  # or Role.assistant
    parts=[Part(TextPart(text="Hello, agent!"))],
    message_id="abc123...",
    context_id="conversation-123"  # For conversation tracking
)
```

**Key Fields:**
- `role`: Who sent it (user or assistant)
- `parts`: Content (text, data, images, etc.)
- `message_id`: Unique identifier
- `context_id`: Links messages in the same conversation

**Example from code:**
```python
# In client.py
def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id
    )
```

### 4. Tasks

**Tasks** represent long-running operations (like assessments).

**Task States:**
- `submitted` - Task created, not started
- `working` - Task in progress
- `completed` - Task finished successfully
- `failed` - Task encountered an error

**Task Structure:**
```python
Task(
    id="task-123",
    context_id="conversation-123",
    status=TaskStatus(
        state=TaskState.working,
        message=Message(...)  # Status update
    ),
    artifacts=[...]  # Results produced
)
```

**Why Tasks?**
- Assessments can take minutes or hours
- Need to track progress
- Need to handle failures gracefully
- Allow status updates during execution

**Example from `green_executor.py`:**
```python
# Create task
task = new_task(msg)
await event_queue.enqueue_event(task)

# Update status
await updater.update_status(
    TaskState.working,
    new_agent_text_message("Starting assessment...")
)

# Complete task
await updater.complete()
```

### 5. Artifacts

**Artifacts** are structured outputs produced by agents (results, reports, data).

**Structure:**
```python
Artifact(
    name="Result",
    parts=[
        Part(TextPart(text="Human-readable reason")),
        Part(DataPart(data={"winner": "pro_debater", ...}))
    ]
)
```

**Purpose:**
- Store assessment results
- Share structured data between agents
- Provide both human-readable and machine-readable formats
- Persist results for later analysis

**Example from `debate_judge.py`:**
```python
await updater.add_artifact(
    parts=[
        Part(root=TextPart(text=debate_eval.reason)),  # Human-readable
        Part(root=TextPart(text=result.model_dump_json()))  # Structured JSON
    ],
    name="Result",
)
```

### 6. Context IDs

**Context IDs** link related messages in a conversation.

**How it works:**
1. First message: No `context_id` (or new one created)
2. Server returns `context_id` in response
3. Subsequent messages: Include same `context_id`
4. Server maintains conversation state per `context_id`

**Example from `tool_provider.py`:**
```python
class ToolProvider:
    def __init__(self):
        self._context_ids = {}  # Track context per agent URL

    async def talk_to_agent(self, message: str, url: str, new_conversation: bool = False):
        # Get context_id for this agent (or None for new conversation)
        context_id = None if new_conversation else self._context_ids.get(url, None)
        
        # Send message
        outputs = await send_message(message=message, base_url=url, context_id=context_id)
        
        # Store context_id for next message
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]
```

## Communication Flow

### Basic Message Exchange

```
┌─────────────┐                                    ┌─────────────┐
│   Client    │                                    │   Server    │
│   Agent     │                                    │   Agent     │
└──────┬──────┘                                    └──────┬──────┘
       │                                                   │
       │ 1. GET /.well-known/agent-card.json              │
       │─────────────────────────────────────────────────>│
       │                                                   │
       │ 2. Agent Card (JSON)                             │
       │<─────────────────────────────────────────────────│
       │                                                   │
       │ 3. POST / (A2A Message)                           │
       │    {role: "user", parts: [{text: "Hello"}]}      │
       │─────────────────────────────────────────────────>│
       │                                                   │
       │ 4. Process message                                │
       │                                                   │
       │ 5. A2A Response (Message or Task)                 │
       │    {role: "assistant", parts: [{text: "Hi!"}]}   │
       │<─────────────────────────────────────────────────│
       │                                                   │
```

### Task-Based Communication

```
┌─────────────┐                                    ┌─────────────┐
│   Client    │                                    │   Server    │
│   Agent     │                                    │   Agent     │
└──────┬──────┘                                    └──────┬──────┘
       │                                                   │
       │ 1. POST / (Assessment Request)                    │
       │─────────────────────────────────────────────────>│
       │                                                   │
       │ 2. Create Task                                    │
       │    Task(id="123", state="submitted")             │
       │<─────────────────────────────────────────────────│
       │                                                   │
       │ 3. Status Update (SSE Stream)                    │
       │    Task(id="123", state="working",               │
       │         message="Starting assessment...")        │
       │<─────────────────────────────────────────────────│
       │                                                   │
       │ 4. Status Update                                  │
       │    Task(id="123", state="working",               │
       │         message="Round 1 complete...")            │
       │<─────────────────────────────────────────────────│
       │                                                   │
       │ 5. Task Complete                                  │
       │    Task(id="123", state="completed",             │
       │         artifacts=[{name: "Result", ...}])       │
       │<─────────────────────────────────────────────────│
       │                                                   │
```

## How A2A Works in This Codebase

### 1. Agent Discovery (`client.py`)

```python
async def send_message(message: str, base_url: str, context_id: str | None = None):
    # Step 1: Discover agent capabilities
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    agent_card = await resolver.get_agent_card()  # GET /.well-known/agent-card.json
    
    # Step 2: Create appropriate client based on agent card
    factory = ClientFactory(config)
    client = factory.create(agent_card)  # Chooses JSONRPC, SSE, etc.
    
    # Step 3: Send message
    outbound_msg = create_message(text=message, context_id=context_id)
    async for event in client.send_message(outbound_msg):
        # Handle response events
        ...
```

**What happens:**
1. Fetch agent card to understand capabilities
2. Create client matching agent's preferred transport
3. Send message using appropriate protocol
4. Receive and process response events

### 2. Sending Messages (`tool_provider.py`)

```python
async def talk_to_agent(self, message: str, url: str, new_conversation: bool = False):
    # Get context_id for conversation continuity
    context_id = None if new_conversation else self._context_ids.get(url, None)
    
    # Send A2A message
    outputs = await send_message(
        message=message, 
        base_url=url, 
        context_id=context_id
    )
    
    # Store context_id for next message
    self._context_ids[url] = outputs.get("context_id", None)
    return outputs["response"]
```

**What happens:**
1. Check if we have an existing conversation (`context_id`)
2. Send message via A2A protocol
3. Store returned `context_id` for next message
4. Return response text

### 3. Receiving Messages (`green_executor.py`)

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    # Extract message from A2A context
    request_text = context.get_user_input()
    
    # Parse as EvalRequest
    req: EvalRequest = EvalRequest.model_validate_json(request_text)
    
    # Create A2A task
    task = new_task(msg)
    await event_queue.enqueue_event(task)
    
    # Create updater for status updates
    updater = TaskUpdater(event_queue, task.id, task.context_id)
    
    # Execute evaluation
    await self.agent.run_eval(req, updater)
    
    # Mark complete
    await updater.complete()
```

**What happens:**
1. Receive A2A message in `RequestContext`
2. Extract and parse request
3. Create A2A task for tracking
4. Execute business logic
5. Send status updates via `TaskUpdater`
6. Mark task as complete

### 4. Status Updates (`TaskUpdater`)

The `TaskUpdater` provides a simple interface for sending A2A status updates:

```python
# Update status
await updater.update_status(
    TaskState.working,
    new_agent_text_message("Round 1 complete")
)

# Add artifact
await updater.add_artifact(
    parts=[...],
    name="Result"
)

# Complete task
await updater.complete()
```

**What happens internally:**
- Creates A2A task update events
- Sends them via `EventQueue`
- Client receives them as Server-Sent Events (SSE) or in response

## Transport Protocols

A2A supports multiple transport protocols:

### 1. JSON-RPC (Most Common)

**How it works:**
- HTTP POST requests
- JSON payloads
- Request/response pattern

**Example:**
```http
POST / HTTP/1.1
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "message",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"text": "Hello"}]
    }
  },
  "id": 1
}
```

### 2. Server-Sent Events (SSE)

**How it works:**
- HTTP GET request
- Server streams events
- Used for task updates and streaming responses

**Example:**
```http
GET /events?task_id=123 HTTP/1.1
Accept: text/event-stream

event: task_update
data: {"task_id": "123", "state": "working", ...}

event: task_update
data: {"task_id": "123", "state": "completed", ...}
```

### 3. WebSocket (Optional)

For real-time bidirectional communication (less common in this codebase).

## Real-World Example: Debate Scenario

Let's trace a complete A2A interaction:

### Step 1: Green Agent Receives Request

```json
POST / HTTP/1.1
{
  "participants": {
    "pro_debater": "http://127.0.0.1:9019/",
    "con_debater": "http://127.0.0.1:9018/"
  },
  "config": {
    "topic": "Should AI be regulated?",
    "num_rounds": 3
  }
}
```

### Step 2: Green Agent Creates Task

```json
{
  "task": {
    "id": "task-abc123",
    "context_id": "ctx-xyz789",
    "status": {
      "state": "submitted"
    }
  }
}
```

### Step 3: Green Agent Sends Message to Purple Agent

```python
# In debate_judge.py, turn() function
response = await self._tool_provider.talk_to_agent(
    "Debate Topic: Should AI be regulated? Present your opening argument.",
    "http://127.0.0.1:9019/",  # pro_debater URL
    new_conversation=False
)
```

**What happens:**
1. `talk_to_agent()` calls `send_message()` in `client.py`
2. `send_message()` fetches agent card from `http://127.0.0.1:9019/.well-known/agent-card.json`
3. Creates A2A message:
   ```json
   {
     "role": "user",
     "parts": [{"text": "Debate Topic: Should AI be regulated?..."}],
     "context_id": "ctx-xyz789"
   }
   ```
4. Sends via JSON-RPC POST to `http://127.0.0.1:9019/`
5. Purple agent processes and responds
6. Response returned: `"Ladies and gentlemen, esteemed judges..."`

### Step 4: Green Agent Updates Task Status

```python
await updater.update_status(
    TaskState.working,
    new_agent_text_message("pro_debater: Ladies and gentlemen...")
)
```

**What happens:**
- Creates task update event
- Sends via SSE stream to client
- Client sees: `[Status: working] pro_debater: Ladies and gentlemen...`

### Step 5: Green Agent Completes Task

```python
await updater.add_artifact(
    parts=[...],
    name="Result"
)
await updater.complete()
```

**What happens:**
- Artifact added to task
- Task marked as `completed`
- Final event sent to client
- Client sees: `[Status: completed]` with results

## Key Benefits of A2A

### 1. **Interoperability**

Any A2A-compliant agent can communicate with any other:
- Python agent ↔ JavaScript agent
- Local agent ↔ Cloud agent
- Different frameworks, same protocol

### 2. **Standardization**

- Consistent message format
- Standard error handling
- Uniform task tracking
- Common artifact format

### 3. **Observability**

- Built-in status updates
- Task tracking
- Artifact storage
- Progress visibility

### 4. **Flexibility**

- Multiple transport protocols
- Streaming support
- Extensible message parts
- Custom artifacts

### 5. **Separation of Concerns**

- Protocol handling (A2A SDK)
- Business logic (your code)
- Clear boundaries

## A2A vs Other Protocols

### A2A vs REST API

| Feature | REST API | A2A Protocol |
|---------|----------|--------------|
| Purpose | General web APIs | Agent-to-agent communication |
| Message Format | Custom per API | Standardized |
| Task Tracking | Manual | Built-in |
| Status Updates | Polling or custom | Streaming (SSE) |
| Artifacts | Custom format | Standard format |
| Agent Discovery | Manual | Agent cards |

### A2A vs gRPC

| Feature | gRPC | A2A Protocol |
|---------|------|--------------|
| Protocol | HTTP/2, Protocol Buffers | HTTP/1.1, JSON |
| Language | Language-specific | Language-agnostic |
| Streaming | Bidirectional | Server-to-client (SSE) |
| Discovery | Service registry | Agent cards |
| Focus | Microservices | AI agents |

## Summary

A2A (Agent-to-Agent) protocol provides:

1. ✅ **Standardized communication** between AI agents
2. ✅ **Agent discovery** via agent cards
3. ✅ **Message-based** communication with context tracking
4. ✅ **Task tracking** for long-running operations
5. ✅ **Status updates** via streaming
6. ✅ **Artifact sharing** for results
7. ✅ **Multiple transports** (JSON-RPC, SSE, WebSocket)
8. ✅ **Language/framework agnostic**

**Key Takeaway**: A2A is like HTTP for agents - it provides a common language that enables any agent to communicate with any other agent, regardless of implementation details.

**In this codebase:**
- Agents run as HTTP servers
- Communication happens via A2A messages
- Tasks track long-running assessments
- Artifacts store results
- Status updates provide visibility

All of this happens transparently - you work with simple interfaces like `talk_to_agent()` and `TaskUpdater`, while the A2A SDK handles all the protocol details.

