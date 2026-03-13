# Project Flow — Invoicing Bug Intake System

This document describes the complete flow of this NanoClaw installation. It covers every path a message can take, every agent involved, and every external system touched. Written for any human or AI picking this up cold.

---

## Overview

This is a personal NanoClaw installation running on a Mac Mini. It connects three messaging channels to a set of AI agents that triage bug reports, investigate issues, and file GitHub PRs for the Invoicing .NET web app.

**Channels:**
- **Slack** (`slack:C0AL0D2K79R`) — where the Invoicing app's users report bugs via chat
- **Telegram @Agent47pi_bot** (`tg:` prefix) — admin/control channel, runs Claude container agent
- **Telegram @Mikeollamabot** (`tgo:` prefix) — Ollama control channel + bug intake notifications
- **Telegram "Bugs" group** (`tgo:-5191721027`) — receives structured bug reports from the .NET app

**Agents:**
- **Claude container agent** — full Claude via Agent SDK, runs in Apple Container (Linux VM)
- **Ollama coding agent** — `qwen2.5-coder:14b`, local, handles bug investigation
- **Ollama control assistant** — `llama3.2:1b`, local, parses commands from the Ollama chat
- **Ollama intake agent** — `llama3.2:1b`, local, drafts GitHub issues from bug reports
- **parseIntent** — `llama3.2:1b`, one-shot, parses natural language approval commands

---

## Registered Groups

| JID | Name | Channel | Agent | Trigger Required |
|-----|------|---------|-------|-----------------|
| `tg:8388828787` | Main (admin) | Telegram Claude bot | Claude container | No |
| `tgo:8388828787` | Ollama | Telegram Ollama bot | Ollama | No |
| `slack:C0AL0D2K79R` | Bug Reports | Slack | Ollama intake | No |
| `tgo:-5191721027` | Bug Intake | Telegram Ollama bot | Ollama intake | No |

---

## Message Lifecycle (All Paths)

### 1. Message Arrives

Every incoming message goes through the same entry point:

```
Channel (Telegram/Slack) → onMessage callback → storeMessage() in SQLite
```

The message loop polls SQLite every `POLL_INTERVAL` ms for new messages in registered groups. When found, it routes to `processGroupMessages(chatJid)`.

---

### 2. Path A — Admin Telegram Chat (`tg:8388828787`)

**Trigger:** Any message to @Agent47pi_bot (no trigger word required — it's the main group)

```
Message → formatMessages() → runAgent() → Apple Container
         └─ Container has:
               - Claude Agent SDK (claude-sonnet-4-6)
               - MCP: nanoclaw (schedule tasks, send messages, register groups)
               - MCP: ollama (call local Ollama models)
               - MCP: brave (web search via Brave API)
               - MCP: mslearn (Microsoft Learn docs)
               - Tools: Bash, Read, Write, WebSearch, WebFetch, etc.
               - Mounted: groups/main/, groups/global/
         └─ Output streams back → sendMessage() to Telegram
```

This is the full-power agent. Can do anything: write code, manage tasks, register groups, search the web, read docs.

---

### 3. Path B — Ollama Chat (`tgo:8388828787`)

**Trigger:** Any message to @Mikeollamabot

The Ollama path has three sub-paths checked in order:

#### 3a. Approval Command (pending issue exists)

```
Message → parseIntent(llama3.2:1b)
         └─ Extracts: { decision: "yes|no|yes but X", ref: "abc123" }
         └─ If matched → handleIntakeApproval()
               ├─ "yes" → fileGithubIssue() → gh issue create → addToQueue() → investigation starts
               ├─ "no"  → remove from pending, notify Slack reporter
               └─ "yes but X" → applyModificationAndFile() → Ollama modifies + files issue
```

#### 3b. Control Commands

```
Message → regex checks (before Ollama agent runs):
         ├─ "queue"          → formatQueueStatus() → reply
         ├─ "manual N"       → acknowledge, leave issue open
         └─ "claude N"       → triggerManualInvestigation(N) → addToQueue()
```

#### 3c. General Ollama Chat

```
Message → runOllamaAgent(llama3.2:1b, 4096 ctx)
         └─ System: control assistant for bug intake system
         └─ Tools: queue_investigation (calls addToQueue)
         └─ Tools: bash, write_file, read_file, brave_search,
                   microsoft_docs_search, microsoft_docs_fetch
         └─ Reply → sendMessage() to Telegram
```

---

### 4. Path C — Slack Bug Reports (`slack:C0AL0D2K79R`)

**Trigger:** Any message in the Slack #bug-reports channel

```
Message → "Thanks @reporter, reviewing your report..." (immediate ack)
        → runSlackIntakeAgent()
              └─ runOllamaAgent(llama3.2:1b, 8192 ctx)
                    └─ System: INTAKE_SYSTEM_PROMPT
                    └─ Tools: read_file only
                    └─ Reads: /code/Invoicing/docs/ai-triage/*.md
              └─ Response parsed for <draft>{...}</draft> tags
        → If drafted:
              ├─ Reply to Slack: "Got it @reporter — I've drafted an issue and sent it for review."
              └─ Send to @Mikeollamabot: formatTelegramNotification(issue)
                    └─ Shows title, labels, body preview
                    └─ Shows exact approval commands: "yes (ref: abc123)", etc.
        → If clarification needed:
              └─ Reply to Slack with questions
```

---

### 5. Path D — Telegram Bug Intake Group (`tgo:-5191721027`)

**Trigger:** Any message posted to the "Bugs" Telegram group

This group receives structured bug reports from the Invoicing .NET app. A form in the app posts a formatted message like:
```
🐛 Bug Report
Type: Something looks wrong
Page: https://invoicing.example.com/SiteAdmin/Health
Description: The services aren't all capitalized properly
```

```
Message → "Bug report received. Processing..." (immediate ack to group)
        → runSlackIntakeAgent()
              └─ same intake pipeline as Path C
              └─ all required fields always present (form enforces them)
        → If drafted:
              ├─ Reply to group: "Draft ready — sent for review."
              └─ Send to @Mikeollamabot: formatTelegramNotification(issue)
        → If error:
              └─ Reply to group + @Mikeollamabot with error message
```

**Note:** Telegram does not deliver bot-to-bot messages. The .NET app cannot post directly as a bot — a human must relay the message, or the message must come from a human/non-bot account in the group.

---

## Bug Investigation Flow

Once a GitHub issue is approved (via Path B 3a) or manually queued, it enters the investigation queue.

### Investigation Queue

Stored at `groups/slack_bug_reports/investigation-queue.json`. One investigation runs at a time.

```
addToQueue(issueNumber, title, issue)
  └─ Status: "pending"

Worker loop (every 10s):
  └─ resetTimedOutItems() — reset any "running" items > 2 hours old
  └─ Skip if any item is "running"
  └─ Pick oldest "pending" item
  └─ Set status: "running"
  └─ onStart() → notify @Mikeollamabot: "🔎 Investigating #N: title"
  └─ runBugInvestigation()
  └─ Set status: "done" or "failed"
  └─ onComplete() → notify @Mikeollamabot with result
```

### runBugInvestigation

```
runOllamaAgent(qwen2.5-coder:14b, 32768 ctx, 500 iterations, 30 min timeout)
  └─ Reads: /code/Invoicing/docs/ai-bug-hunting/*.md
  └─ Finds relevant .cs/.razor files via bash (find, grep)
  └─ Attempts fix via write_file
  └─ Validates: dotnet build /code/Invoicing
  └─ If build passes:
        ├─ git checkout -b fix/issue-N
        ├─ git add && git commit
        ├─ git push -u origin fix/issue-N
        ├─ gh pr create
        ├─ gh pr comment "@copilot please review this fix"
        └─ Returns RESULT:FIXED:<pr-url>
  └─ If cannot find code:
        ├─ gh issue comment (investigation findings)
        ├─ gh issue edit --add-assignee @copilot
        └─ Returns RESULT:ASSIGNED:<summary>
```

**Result handling:**
- `RESULT:FIXED` → notify: "✅ #N fixed — PR: <url>"
- `RESULT:ASSIGNED` → notify with summary + options: "manual N" or "claude N"

---

## Scheduled Tasks

Tasks stored in SQLite (`store/messages.db`, `scheduled_tasks` table). Scheduler polls every minute.

| Field | Description |
|-------|-------------|
| `agent_type` | `"claude"` or `"ollama"` |
| `schedule_type` | `"cron"`, `"interval"`, or `"once"` |
| `schedule_value` | cron string, ms interval, or ISO timestamp |
| `chat_jid` | Where to send the result |

**Claude tasks:** spawn container agent, stream output to chat_jid
**Ollama tasks:** call `runOllamaAgent()` directly, send result to chat_jid

**Current tasks:**
- Morning briefing: cron `0 7 * * *`, Ollama, sends to `tgo:8388828787`

---

## GitHub Token Refresh

GitHub tokens are short-lived (GitHub App installation tokens, ~1 hour). The app generates them on-demand via `getGithubToken()` in `src/github-token.ts` using:
- App ID: `3068016`
- Install ID: `115682233`
- PEM: `~/github-app.pem`

A refresh script at `~/bin/gh-refresh-token` updates all `~/code` repo remote URLs with a fresh token. Run it manually before any git operations if authentication fails.

---

## Tool Availability by Agent

| Tool | Claude Container | Ollama Coder | Ollama Control | Ollama Intake |
|------|:---:|:---:|:---:|:---:|
| bash | ✅ | ✅ | ✅ | ❌ |
| write_file | ✅ | ✅ | ✅ | ❌ |
| read_file | ✅ | ✅ | ✅ | ✅ |
| brave_search | ✅ (MCP) | ✅ | ✅ | ❌ |
| microsoft_docs_search | ✅ (MCP) | ✅ | ✅ | ❌ |
| microsoft_docs_fetch | ✅ (MCP) | ✅ | ✅ | ❌ |
| queue_investigation | ❌ | ❌ | ✅ | ❌ |
| schedule_task | ✅ (MCP) | ❌ | ❌ | ❌ |
| WebSearch/WebFetch | ✅ | ❌ | ❌ | ❌ |

---

## Key External Systems

| System | Purpose | Auth |
|--------|---------|------|
| GitHub (`myers-gh1328/Invoicing`) | Issues, PRs, labels | GitHub App (PEM + App ID) |
| Slack workspace | Bug reports from users | Slack bot token (`.env`) |
| Telegram | Admin chat, Ollama chat, bug intake group | Bot tokens (`.env`) |
| Ollama (`localhost:11434`) | Local model inference | None (local) |
| Brave Search API | Web search for agents | `BRAVE_API_KEY` (`.env`) |
| Microsoft Learn | .NET/ASP.NET docs | None (public) |
| Invoicing app (prod) | Posts bug reports to Telegram group | Human relay (bot-to-bot blocked) |

---

## File Layout

```
nanoclaw/
├── src/
│   ├── index.ts              # Main orchestrator, message loop, all routing logic
│   ├── ollama-agent.ts       # Ollama agent runner, all Ollama tool handlers
│   ├── slack-intake.ts       # Intake agent, GitHub issue filing, bug investigation
│   ├── investigation-queue.ts# Persistent queue, worker loop
│   ├── task-scheduler.ts     # Scheduled task runner
│   ├── channels/
│   │   ├── slack.ts          # Slack channel (Socket Mode)
│   │   └── telegram.ts       # Telegram channel (grammy, polling)
│   └── db.ts                 # SQLite: messages, groups, tasks, sessions
├── container/
│   └── agent-runner/src/
│       └── index.ts          # Claude Agent SDK runner inside container
├── groups/
│   ├── slack_bug_reports/    # Intake history, pending issues, investigation queue
│   ├── main/                 # Admin group memory
│   └── ollama_main/          # Ollama chat history
└── store/
    └── messages.db           # SQLite database
```

---

## Approval Command Format

When the intake agent drafts an issue, a notification is sent to @Mikeollamabot with:

```
To approve: yes (ref: abc123)
To reject:  no (ref: abc123)
To modify:  yes but [describe changes] (ref: abc123)
```

The `parseIntent` helper (llama3.2:1b) parses the response flexibly — handles typos, informal phrasing, and missing parentheses. If only one issue is pending, a bare "yes" or "no" is sufficient.
