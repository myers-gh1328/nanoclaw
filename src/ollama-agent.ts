/**
 * Ollama agent with tool calling support.
 * Uses qwen2.5-coder (or any tool-capable model) to write code,
 * run git/gh commands, read/write files, and manage GitHub issues.
 */

import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { getGithubToken } from './github-token.js';
import { logger } from './logger.js';

const envConfig = readEnvFile(['OLLAMA_HOST', 'OLLAMA_MODEL']);
const OLLAMA_HOST =
  process.env.OLLAMA_HOST || envConfig.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL || envConfig.OLLAMA_MODEL || 'llama3.1:8b';
const MAX_HISTORY = 20;
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
];

const SYSTEM_PROMPT = `You are a coding assistant with full access to the local filesystem and GitHub.

You have the following tools available:
- bash: run any shell command (git, gh, npm, node, etc.)
- write_file: write content to a file (preferred for writing code)
- read_file: read a file's contents

Your working directory is ~/code. You can clone repos there, create projects, read and edit files, run builds and tests, commit and push changes.

GitHub CLI (gh) is available and authenticated. Use it to read/create issues, interact with repos, etc.

Always use your tools to actually perform tasks — do not say you cannot access the filesystem or run commands.`;

// In-memory conversation history per group folder
const histories = new Map<string, Message[]>();

export function clearOllamaHistory(groupFolder: string): void {
  histories.delete(groupFolder);
}

function getWorkspaceDir(_groupFolder: string): string {
  const wsDir = path.join(os.homedir(), 'code');
  fs.mkdirSync(wsDir, { recursive: true });
  return wsDir;
}

async function executeBash(
  command: string,
  workdir: string,
  ghToken: string | null,
): Promise<string> {
  try {
    const env: Record<string, string> = {
      ...(process.env as Record<string, string>),
      HOME: os.homedir(),
      PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin',
    };
    if (ghToken) {
      env.GH_TOKEN = ghToken;
      env.GITHUB_TOKEN = ghToken;
    }
    const output = execSync(command, {
      cwd: workdir,
      env,
      timeout: 120_000,
      maxBuffer: 2 * 1024 * 1024,
    });
    return output.toString().trim() || '(no output)';
  } catch (err: unknown) {
    const e = err as {
      stdout?: Buffer;
      stderr?: Buffer;
      message?: string;
    };
    const out = e.stdout?.toString().trim();
    const errMsg = e.stderr?.toString().trim() || e.message || String(err);
    return out ? `${out}\nError: ${errMsg}` : `Error: ${errMsg}`;
  }
}

export async function runOllamaAgent(
  text: string,
  groupFolder: string,
): Promise<string> {
  if (!histories.has(groupFolder)) {
    histories.set(groupFolder, []);
  }
  const history = histories.get(groupFolder)!;
  const workspace = getWorkspaceDir(groupFolder);
  const ghToken = await getGithubToken();

  history.push({ role: 'user', content: text });
  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  logger.info({ groupFolder, model: OLLAMA_MODEL }, 'Ollama agent: generating');

  for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
    const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...history],
        tools: TOOLS,
        stream: false,
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

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const reply = msg.content.trim();
      history.push({ role: 'assistant', content: reply });
      logger.info({ groupFolder, replyLength: reply.length }, 'Ollama agent: done');
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
        result = await executeBash(parsed['command'] as string, cwd, ghToken);
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
      } else {
        result = `Unknown tool: ${name}`;
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
