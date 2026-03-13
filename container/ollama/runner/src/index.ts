/**
 * NanoClaw Ollama Runner
 * Runs inside the nanoclaw-ollama container.
 * Reads prompt + options from stdin, runs the Ollama agentic loop,
 * writes result to stdout using the standard NANOCLAW_OUTPUT markers.
 *
 * Input (stdin JSON):
 *   { prompt, groupFolder, historySubdir?, chatJid, options? }
 *
 * options:
 *   { systemPrompt?, maxIterations?, maxDurationMs?, maxToolOutputLength?, nudgeMessage? }
 *
 * Output (stdout):
 *   ---NANOCLAW_OUTPUT_START---
 *   {"status":"success","result":"..."}
 *   ---NANOCLAW_OUTPUT_END---
 */

import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OLLAMA_HOST =
  process.env.OLLAMA_HOST || 'http://host.docker.internal:11434';
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL || 'llama3.1:8b';
const WORKSPACE_CODE = '/workspace/code';
const WORKSPACE_GROUP = '/workspace/group';
const MAX_HISTORY = 100;
const MAX_TOOL_ITERATIONS = 20;

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToolCall {
  function: {
    name: string;
    arguments: string | Record<string, unknown>;
  };
}

interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: ToolCall[];
}

interface RunnerInput {
  prompt: string;
  groupFolder: string;
  historySubdir?: string;
  chatJid: string;
  options?: {
    systemPrompt?: string;
    maxIterations?: number;
    maxDurationMs?: number;
    maxToolOutputLength?: number;
    nudgeMessage?: string;
    /** Extra tool definitions passed to Ollama. When called, triggers a custom_tool result. */
    customTools?: object[];
  };
}

interface RunnerOutput {
  status: 'success' | 'error' | 'custom_tool';
  result: string | null;
  error?: string;
  /** Populated when status==='custom_tool' */
  toolName?: string;
  toolArgs?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Tools definition
// ---------------------------------------------------------------------------

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'bash',
      description:
        'Execute a shell command. Use for git operations (clone, pull, checkout, commit, push), ' +
        'gh CLI (issues, repos), running tests/node/npm, and any other tasks. ' +
        'The working directory is /workspace/code unless you specify a path.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'Shell command to run',
          },
          workdir: {
            type: 'string',
            description:
              'Working directory for the command (absolute path). Optional.',
          },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description:
        'Write content to a file. Prefer this over bash for writing JS, HTML, CSS, JSON, or any ' +
        'multi-line code — avoids shell escaping issues. Creates parent directories as needed.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Absolute file path',
          },
          content: {
            type: 'string',
            description: 'Full file content to write',
          },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Absolute file path',
          },
        },
        required: ['path'],
      },
    },
  },
];

// ---------------------------------------------------------------------------
// Default system prompt
// ---------------------------------------------------------------------------

const DEFAULT_SYSTEM_PROMPT = `You are a coding assistant. You have tools to read/write files and run shell commands.

CRITICAL RULES — violation is failure:
1. NEVER list steps you are going to take. Just take them immediately using tools.
2. NEVER say "I will do X". Just do X using a tool right now.
3. NEVER describe code changes. Use write_file to make them.
4. NEVER ask the user to run commands. Use bash to run them yourself.
5. After completing all tool calls, write a SHORT summary of what you did.

Tools:
- bash: run any shell command (git, gh, npm, node, ls, grep, etc.)
- write_file(path, content): write a file — ALWAYS use this to write code, never use bash for file writing
- read_file(path): read a file

Workspace: /workspace/code — repos are cloned in subdirectories.
GitHub is authenticated. Use gh for issues. Use git inside a repo directory to commit and push.

When asked to do something: use tools first, summarize after. No planning. No steps. Just do it.`;

// ---------------------------------------------------------------------------
// Dangerous command filter
// ---------------------------------------------------------------------------

const DANGEROUS_PATTERNS = [
  /rm\s+-rf?\s+~\//, // rm -rf ~/...
  /rm\s+-rf?\s+\/Users\//, // rm -rf /Users/...
  /rm\s+-rf?\s+\/$/, // rm -rf /
  /rm\s+-rf?\s+\*$/, // rm -rf *
  /mkfs/, // format disk
  /dd\s+.*of=\/dev/, // write to device
  />\s*\/dev\/sd/, // overwrite device
];

function isDangerous(command: string): boolean {
  return DANGEROUS_PATTERNS.some((p) => p.test(command));
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

function executeBash(command: string, workdir: string): string {
  try {
    const ghToken = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
    const env: Record<string, string> = {
      ...(process.env as Record<string, string>),
      HOME: os.homedir(),
      PATH: `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH || ''}`,
    };
    if (ghToken) {
      env.GH_TOKEN = ghToken;
      env.GITHUB_TOKEN = ghToken;
      env.GIT_CONFIG_COUNT = '1';
      env.GIT_CONFIG_KEY_0 = `url.https://x-access-token:${ghToken}@github.com/.insteadOf`;
      env.GIT_CONFIG_VALUE_0 = 'https://github.com/';
    }
    const output = execSync(command, {
      cwd: workdir,
      env,
      timeout: 120_000,
      maxBuffer: 2 * 1024 * 1024,
    });
    return output.toString().trim() || '(no output)';
  } catch (err: unknown) {
    const e = err as { stdout?: Buffer; stderr?: Buffer; message?: string };
    const out = e.stdout?.toString().trim();
    const errMsg = e.stderr?.toString().trim() || e.message || String(err);
    return out ? `${out}\nError: ${errMsg}` : `Error: ${errMsg}`;
  }
}

// ---------------------------------------------------------------------------
// Content-embedded tool call parser (for models that don't use tool_calls)
// ---------------------------------------------------------------------------

function parseContentToolCall(
  content: string,
): { name: string; arguments: Record<string, unknown> } | null {
  if (!content) return null;

  const candidates: string[] = [];

  const stripped = content
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();
  candidates.push(stripped);

  for (let start = 0; start < content.length; start++) {
    if (content[start] !== '{') continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = start;
    for (; end < content.length; end++) {
      const ch = content[end];
      if (escaped) { escaped = false; continue; }
      if (ch === '\\' && inString) { escaped = true; continue; }
      if (ch === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (ch === '{') depth++;
      else if (ch === '}') { depth--; if (depth === 0) break; }
    }
    if (depth === 0) candidates.push(content.slice(start, end + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      if (typeof parsed.name === 'string' && parsed.arguments) {
        return { name: parsed.name, arguments: parsed.arguments as Record<string, unknown> };
      }
    } catch {
      // Try next
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function historyPath(historySubdir: string): string {
  const dir = historySubdir
    ? path.join(WORKSPACE_GROUP, historySubdir)
    : WORKSPACE_GROUP;
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, 'ollama-history.json');
}

function loadHistory(historySubdir: string): Message[] {
  try {
    return JSON.parse(fs.readFileSync(historyPath(historySubdir), 'utf8')) as Message[];
  } catch {
    return [];
  }
}

function saveHistory(historySubdir: string, history: Message[]): void {
  try {
    fs.writeFileSync(historyPath(historySubdir), JSON.stringify(history, null, 2));
  } catch {
    // Non-fatal
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

function writeOutput(output: RunnerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(msg: string): void {
  console.error(`[ollama-runner] ${msg}`);
}

// ---------------------------------------------------------------------------
// Ollama agent loop
// ---------------------------------------------------------------------------

async function runOllamaLoop(input: RunnerInput): Promise<string> {
  const historySubdir = input.historySubdir || '';
  const opts = input.options || {};
  const maxIterations = opts.maxIterations ?? MAX_TOOL_ITERATIONS;
  const deadline = opts.maxDurationMs ? Date.now() + opts.maxDurationMs : null;
  const maxOutput = opts.maxToolOutputLength ?? 8000;
  const systemPrompt = opts.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;

  const history = loadHistory(historySubdir);
  history.push({ role: 'user', content: input.prompt });
  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  log(`Starting loop: model=${OLLAMA_MODEL}, maxIter=${maxIterations}, group=${input.groupFolder}`);

  for (let i = 0; i < maxIterations; i++) {
    if (deadline && Date.now() > deadline) {
      return 'Error: investigation time limit reached.';
    }

    const allTools = opts.customTools ? [...TOOLS, ...opts.customTools] : TOOLS;
    const customToolNames = new Set(
      (opts.customTools ?? []).map((t) => (t as { function: { name: string } }).function.name)
    );

    const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          ...history,
        ],
        tools: allTools,
        stream: false,
        options: { num_ctx: 32768 },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as {
      message: { role: string; content: string; tool_calls?: ToolCall[] };
    };

    const msg = data.message;
    log(`Response: content=${msg.content?.slice(0, 100)}, tools=${JSON.stringify(msg.tool_calls?.map(t => t.function.name))}`);

    // Some models embed tool calls in content instead of tool_calls
    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const embedded = parseContentToolCall(msg.content);
      if (embedded) {
        msg.tool_calls = [{ function: embedded }];
      }
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const reply = msg.content.trim();
      history.push({ role: 'assistant', content: reply });

      if (opts.nudgeMessage) {
        log(`Nudging model to continue (iteration ${i})`);
        history.push({ role: 'user', content: opts.nudgeMessage });
        continue;
      }

      saveHistory(historySubdir, history);
      log(`Done. Reply length: ${reply.length}`);
      return reply;
    }

    history.push({
      role: 'assistant',
      content: msg.content || '',
      tool_calls: msg.tool_calls,
    });

    for (const tc of msg.tool_calls) {
      const { name, arguments: args } = tc.function;
      const parsed =
        typeof args === 'string'
          ? (JSON.parse(args) as Record<string, unknown>)
          : args;

      log(`Tool call: ${name} ${String(parsed['command'] || parsed['path'] || '').slice(0, 80)}`);

      let result: string;

      if (name === 'bash') {
        const rawWorkdir = parsed['workdir'] as string | undefined;
        const cwd = rawWorkdir && path.isAbsolute(rawWorkdir)
          ? rawWorkdir
          : WORKSPACE_CODE;
        const cmd = parsed['command'] as string;
        if (isDangerous(cmd)) {
          log(`Blocked dangerous command: ${cmd}`);
          result = `Error: command blocked for safety reasons: ${cmd}`;
        } else {
          result = executeBash(cmd, cwd);
        }
      } else if (name === 'write_file') {
        const filePath = parsed['path'] as string;
        const content = parsed['content'] as string;
        try {
          fs.mkdirSync(path.dirname(filePath), { recursive: true });
          fs.writeFileSync(filePath, content, 'utf8');
          result = `Written: ${filePath}`;
        } catch (err) {
          result = `Error writing file: ${err instanceof Error ? err.message : String(err)}`;
        }
      } else if (name === 'read_file') {
        const filePath = parsed['path'] as string;
        try {
          result = fs.readFileSync(filePath, 'utf8');
        } catch (err) {
          result = `Error reading file: ${err instanceof Error ? err.message : String(err)}`;
        }
      } else if (customToolNames.has(name)) {
        // Custom tool — output a special result and exit so main() doesn't write a second block
        saveHistory(historySubdir, history);
        writeOutput({
          status: 'custom_tool',
          result: msg.content?.trim() || null,
          toolName: name,
          toolArgs: parsed,
        });
        process.exit(0);
      } else {
        result = `Unknown tool: ${name}`;
      }

      if (result.length > maxOutput) {
        result =
          result.slice(0, maxOutput) +
          `\n\n[Output truncated at ${maxOutput} chars — ${result.length - maxOutput} chars omitted.]`;
      }

      log(`Tool result: ${name} → ${result.length} chars`);
      history.push({ role: 'tool', content: result });
    }
  }

  return 'Error: exceeded maximum tool iterations — task may be too complex.';
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let input: RunnerInput;

  try {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk as Buffer);
    }
    input = JSON.parse(Buffer.concat(chunks).toString('utf8')) as RunnerInput;
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  try {
    const result = await runOllamaLoop(input);
    writeOutput({ status: 'success', result });
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: err instanceof Error ? err.message : String(err),
    });
    process.exit(1);
  }
}

main();
