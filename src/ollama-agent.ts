/**
 * Ollama agent with tool calling support.
 * Uses qwen2.5-coder (or any tool-capable model) to write code,
 * run git/gh commands, read/write files, and manage GitHub issues.
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
import fs from 'fs';
import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { getGithubToken } from './github-token.js';
import { logger } from './logger.js';

const envConfig = readEnvFile(['OLLAMA_HOST', 'OLLAMA_MODEL', 'BRAVE_API_KEY']);
const OLLAMA_HOST =
  process.env.OLLAMA_HOST || envConfig.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL || envConfig.OLLAMA_MODEL || 'llama3.1:8b';
const BRAVE_API_KEY = process.env.BRAVE_API_KEY || envConfig.BRAVE_API_KEY;
const MAX_HISTORY = 100;
const MAX_TOOL_ITERATIONS = 20;

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

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'brave_search',
      description:
        'Search the web using Brave Search API. Returns titles, URLs, and descriptions for the top results.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query',
          },
          count: {
            type: 'number',
            description: 'Number of results to return (default: 5, max: 20)',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bash',
      description:
        'Execute a shell command. Use for git operations (clone, pull, checkout, commit, push), ' +
        'gh CLI (issues, repos), running tests/node/npm, and any other tasks. ' +
        'The working directory is the group workspace unless you specify a path.',
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
              'Working directory for the command (absolute path or relative to workspace). Optional.',
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
            description: 'File path (absolute or relative to workspace)',
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
            description: 'File path (absolute or relative to workspace)',
          },
        },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'microsoft_docs_search',
      description:
        'Search Microsoft Learn / official Microsoft documentation. Use for .NET, ASP.NET, C#, Azure, and any other Microsoft technology questions.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'microsoft_docs_fetch',
      description:
        'Fetch a specific Microsoft Learn documentation page as markdown.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The Microsoft Learn page URL to fetch',
          },
        },
        required: ['url'],
      },
    },
  },
];

const SYSTEM_PROMPT = `You are a coding assistant. You have tools to read/write files and run shell commands.

CRITICAL RULES — violation is failure:
1. NEVER list steps you are going to take. Just take them immediately using tools.
2. NEVER say "I will do X". Just do X using a tool right now.
3. NEVER describe code changes. Use write_file to make them.
4. NEVER ask the user to run commands. Use bash to run them yourself.
5. After completing all tool calls, write a SHORT summary of what you did.
6. There is NO "gh" tool and NO "git" tool. Run ALL gh and git commands through the "bash" tool (e.g. bash with command "gh pr create ..." or "git push ...").
7. When pushing a new branch for the first time, ALWAYS use "git push -u origin <branch>" (the -u flag is required).

Tools:
- bash: run any shell command (git, gh, npm, node, ls, etc.) — this is the ONLY way to run git and gh commands
- write_file(path, content): write a file — ALWAYS use this to write code, never use bash for file writing
- read_file(path): read a file

Workspace: /Users/nanobot/code — repos are already cloned in subdirectories:
  /Users/nanobot/code/ocean-games
  /Users/nanobot/code/Invoicing
  /Users/nanobot/code/dissingTerry
  /Users/nanobot/code/nanoclaw
To work on a repo, use its full path. NEVER clone or delete anything in /Users/nanobot/code directly.
GitHub is authenticated. Use gh for issues. Use git inside a repo directory to commit and push.

When asked to do something: use tools first, summarize after. No planning. No steps. Just do it.`;

const histories = new Map<string, Message[]>();

function historyPath(groupFolder: string): string {
  return path.join(process.cwd(), 'groups', groupFolder, 'ollama-history.json');
}

export function loadHistory(groupFolder: string): Message[] {
  try {
    const raw = fs.readFileSync(historyPath(groupFolder), 'utf8');
    return JSON.parse(raw) as Message[];
  } catch {
    return [];
  }
}

export function saveHistory(groupFolder: string, history: Message[]): void {
  try {
    const p = historyPath(groupFolder);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify(history, null, 2));
  } catch (err) {
    logger.warn({ groupFolder, err }, 'Failed to save Ollama history');
  }
}

export function clearOllamaHistory(groupFolder: string): void {
  histories.delete(groupFolder);
  try {
    fs.unlinkSync(historyPath(groupFolder));
  } catch {
    /* already gone */
  }
}

function getWorkspaceDir(_groupFolder: string): string {
  const wsDir = path.join(os.homedir(), 'code');
  fs.mkdirSync(wsDir, { recursive: true });
  return wsDir;
}

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

export async function executeBash(
  command: string,
  workdir: string,
  ghToken: string | null,
): Promise<string> {
  const env: Record<string, string> = {
    ...(process.env as Record<string, string>),
    HOME: os.homedir(),
    PATH: `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH || ''}`,
  };
  if (ghToken) {
    env.GH_TOKEN = ghToken;
    env.GITHUB_TOKEN = ghToken;
    // Rewrite https://github.com/ to embed the token so git push works
    // without needing ~/.gitconfig or gh auth login
    env.GIT_CONFIG_COUNT = '1';
    env.GIT_CONFIG_KEY_0 = `url.https://x-access-token:${ghToken}@github.com/.insteadOf`;
    env.GIT_CONFIG_VALUE_0 = 'https://github.com/';
  }
  try {
    const { stdout } = await execAsync(command, {
      cwd: workdir,
      env,
      timeout: 120_000,
      maxBuffer: 2 * 1024 * 1024,
    });
    return stdout.trim() || '(no output)';
  } catch (err: unknown) {
    const e = err as {
      stdout?: string;
      stderr?: string;
      message?: string;
    };
    const out = e.stdout?.trim();
    const errMsg = e.stderr?.trim() || e.message || String(err);
    return out ? `${out}\nError: ${errMsg}` : `Error: ${errMsg}`;
  }
}

function parseContentToolCall(
  content: string,
): { name: string; arguments: Record<string, unknown> } | null {
  if (!content) return null;

  // Try candidates: full content, content stripped of markdown fences,
  // and any JSON object embedded within text
  const candidates: string[] = [];

  // 1. Strip markdown code fences
  const stripped = content
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();
  candidates.push(stripped);

  // 2. Extract embedded JSON objects from within text using brace counting
  // (regex with non-greedy match fails on nested objects)
  for (let start = 0; start < content.length; start++) {
    if (content[start] !== '{') continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = start;
    for (; end < content.length; end++) {
      const ch = content[end];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === '\\' && inString) {
        escaped = true;
        continue;
      }
      if (ch === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) break;
      }
    }
    if (depth === 0) candidates.push(content.slice(start, end + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      if (typeof parsed.name === 'string' && parsed.arguments) {
        return {
          name: parsed.name,
          arguments: parsed.arguments as Record<string, unknown>,
        };
      }
    } catch {
      // Try next candidate
    }
  }
  return null;
}

const INTENT_MODEL = 'llama3.2:1b';

/**
 * Use a fast local model to parse user intent from a free-form message.
 * Handles typos, misspellings, and informal phrasing.
 *
 * @param schema - JSON schema describing the fields to extract
 * @param context - Additional context to help the model (e.g. list of valid refs)
 * @param text - The raw user message to parse
 * @returns Parsed object matching the schema, or null if intent is unclear
 */
export async function parseIntent<T>(
  schema: string,
  context: string,
  text: string,
): Promise<T | null> {
  const prompt = `You are a strict JSON extractor. Extract structured data from the user message below.
${context}

Return ONLY valid JSON matching this schema — no explanation, no markdown:
${schema}

If you cannot confidently extract the data, return: null

User message: "${text}"`;

  try {
    const res = await fetch(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: INTENT_MODEL,
        prompt,
        stream: false,
        format: 'json',
        options: { temperature: 0, num_predict: 200 },
      }),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { response?: string };
    const raw = (data.response ?? '').trim();
    if (!raw || raw === 'null') return null;
    return JSON.parse(raw) as T;
  } catch (err) {
    logger.warn({ err }, 'parseIntent failed');
    return null;
  }
}

export async function runOllamaAgent(
  text: string,
  groupFolder: string,
  options?: {
    maxIterations?: number;
    maxDurationMs?: number;
    maxToolOutputLength?: number;
    systemPrompt?: string;
    model?: string;
    numCtx?: number;
    extraTools?: object[];
    allowedTools?: string[];
    nudgeMessage?: string;
    toolHandler?: (
      name: string,
      args: Record<string, unknown>,
    ) => Promise<{ result: string; stop?: boolean } | null>;
  },
): Promise<string> {
  if (!histories.has(groupFolder)) {
    // Evict oldest entry if cache is full
    if (histories.size >= 100) {
      histories.delete(histories.keys().next().value!);
    }
    histories.set(groupFolder, loadHistory(groupFolder));
  }
  const history = histories.get(groupFolder)!;
  const workspace = getWorkspaceDir(groupFolder);
  const ghToken = await getGithubToken();
  const maxIterations = options?.maxIterations ?? MAX_TOOL_ITERATIONS;
  const deadline = options?.maxDurationMs
    ? Date.now() + options.maxDurationMs
    : null;

  history.push({ role: 'user', content: text });
  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  const model = options?.model ?? OLLAMA_MODEL;
  logger.info({ groupFolder, model }, 'Ollama agent: generating');

  for (let i = 0; i < maxIterations; i++) {
    if (deadline && Date.now() > deadline) {
      return 'Error: investigation time limit reached.';
    }
    const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: options?.systemPrompt ?? SYSTEM_PROMPT },
          ...history,
        ],
        tools: (() => {
          const all = options?.extraTools
            ? [...TOOLS, ...options.extraTools]
            : TOOLS;
          if (options?.allowedTools) {
            return all.filter((t) =>
              options.allowedTools!.includes((t as any).function.name),
            );
          }
          return all;
        })(),
        stream: false,
        options: { num_ctx: options?.numCtx ?? 16384 },
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`,
      );
    }

    const data = (await response.json()) as {
      message: {
        role: string;
        content: string;
        tool_calls?: ToolCall[];
      };
    };

    const msg = data.message;

    logger.info(
      {
        groupFolder,
        content: msg.content?.slice(0, 200),
        toolCalls: JSON.stringify(
          msg.tool_calls?.map((tc) => tc.function.name),
        ),
      },
      'Ollama raw response',
    );

    // Some models embed tool calls as JSON in content instead of tool_calls field
    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const contentToolCall = parseContentToolCall(msg.content);
      if (contentToolCall) {
        msg.tool_calls = [{ function: contentToolCall }];
      }
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const reply = msg.content.trim();
      history.push({ role: 'assistant', content: reply });
      if (options?.nudgeMessage) {
        logger.info(
          { groupFolder, iteration: i },
          'Ollama agent: model stopped — nudging to continue',
        );
        history.push({ role: 'user', content: options.nudgeMessage });
        continue;
      }
      saveHistory(groupFolder, history);
      logger.info(
        { groupFolder, replyLength: reply.length },
        'Ollama agent: done',
      );
      return reply;
    }

    // Push assistant message with tool_calls so the model sees its own calls
    history.push({
      role: 'assistant',
      content: msg.content || '',
      tool_calls: msg.tool_calls,
    });

    // Execute each tool and feed results back
    for (const tc of msg.tool_calls) {
      const { name, arguments: args } = tc.function;
      const parsed =
        typeof args === 'string'
          ? (JSON.parse(args) as Record<string, unknown>)
          : args;

      logger.info(
        { groupFolder, tool: name, command: parsed['command'] },
        'Ollama tool call',
      );

      let result: string;
      if (name === 'bash') {
        const rawWorkdir = parsed['workdir'] as string | undefined;
        const cwd = rawWorkdir
          ? path.isAbsolute(rawWorkdir)
            ? rawWorkdir
            : path.resolve(workspace, rawWorkdir)
          : workspace;
        const cmd = parsed['command'] as string;
        if (isDangerous(cmd)) {
          logger.warn(
            { groupFolder, command: cmd },
            'Ollama: blocked dangerous command',
          );
          result = `Error: command blocked for safety reasons: ${cmd}`;
        } else {
          result = await executeBash(cmd, cwd, ghToken);
        }
      } else if (name === 'write_file') {
        const filePath = parsed['path'] as string;
        const content = parsed['content'] as string;
        const resolved = path.isAbsolute(filePath)
          ? filePath
          : path.resolve(workspace, filePath);
        try {
          fs.mkdirSync(path.dirname(resolved), { recursive: true });
          fs.writeFileSync(resolved, content, 'utf8');
          result = `Written: ${resolved}`;
        } catch (err) {
          result = `Error writing file: ${err instanceof Error ? err.message : String(err)}`;
        }
      } else if (name === 'read_file') {
        const filePath = parsed['path'] as string;
        const resolved = path.isAbsolute(filePath)
          ? filePath
          : path.resolve(workspace, filePath);
        try {
          result = fs.readFileSync(resolved, 'utf8');
        } catch (err) {
          result = `Error reading file: ${err instanceof Error ? err.message : String(err)}`;
        }
      } else if (name === 'brave_search') {
        if (!BRAVE_API_KEY) {
          result = 'Error: BRAVE_API_KEY is not configured.';
        } else {
          const query = parsed['query'] as string;
          const count = (parsed['count'] as number | undefined) ?? 5;
          try {
            const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${Math.min(count, 20)}`;
            const res = await fetch(url, {
              headers: {
                Accept: 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': BRAVE_API_KEY,
              },
            });
            if (!res.ok) {
              result = `Brave API error (${res.status}): ${await res.text()}`;
            } else {
              const data = (await res.json()) as {
                web?: {
                  results?: Array<{
                    title: string;
                    url: string;
                    description?: string;
                  }>;
                };
              };
              const results = data.web?.results ?? [];
              if (results.length === 0) {
                result = 'No results found.';
              } else {
                result = results
                  .map(
                    (r, i) =>
                      `${i + 1}. ${r.title}\n   ${r.url}${r.description ? `\n   ${r.description}` : ''}`,
                  )
                  .join('\n\n');
              }
            }
          } catch (err) {
            result = `Brave search failed: ${err instanceof Error ? err.message : String(err)}`;
          }
        }
      } else if (
        name === 'microsoft_docs_search' ||
        name === 'microsoft_docs_fetch'
      ) {
        const MCP_URL = 'https://learn.microsoft.com/api/mcp';
        try {
          const toolName =
            name === 'microsoft_docs_search'
              ? 'microsoft_docs_search'
              : 'microsoft_docs_fetch';
          const res = await fetch(MCP_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              id: 1,
              method: 'tools/call',
              params: { name: toolName, arguments: parsed },
            }),
          });
          if (!res.ok) {
            result = `Microsoft Learn MCP error (${res.status}): ${await res.text()}`;
          } else {
            const data = (await res.json()) as {
              result?: { content?: Array<{ text?: string }> };
              error?: { message: string };
            };
            if (data.error) {
              result = `Microsoft Learn error: ${data.error.message}`;
            } else {
              result =
                data.result?.content?.map((c) => c.text ?? '').join('\n') ??
                'No results.';
            }
          }
        } catch (err) {
          result = `Microsoft Learn request failed: ${err instanceof Error ? err.message : String(err)}`;
        }
      } else if (options?.toolHandler) {
        const handled = await options.toolHandler(name, parsed);
        if (handled?.stop) {
          saveHistory(groupFolder, history);
          return handled.result;
        }
        result =
          handled?.result ??
          `Error: unknown tool "${name}". There is no "${name}" tool. Use the "bash" tool instead and run the command via shell (e.g. use bash with command "gh pr create ..." instead of calling a gh tool directly).`;
      } else {
        result = `Error: unknown tool "${name}". There is no "${name}" tool. Use the "bash" tool instead and run the command via shell (e.g. use bash with command "gh pr create ..." instead of calling a gh tool directly).`;
      }

      // Truncate large tool outputs so the model isn't overwhelmed
      const maxOutput = options?.maxToolOutputLength ?? 8000;
      if (result.length > maxOutput) {
        result =
          result.slice(0, maxOutput) +
          `\n\n[Output truncated at ${maxOutput} chars — ${result.length - maxOutput} chars omitted. Use a more targeted command to see specific parts.]`;
      }

      logger.info(
        { groupFolder, tool: name, resultLength: result.length },
        'Ollama tool result',
      );

      history.push({ role: 'tool', content: result });
    }
  }

  return 'Error: exceeded maximum tool iterations — task may be too complex.';
}
