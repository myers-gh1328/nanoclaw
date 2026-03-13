/**
 * Ollama Container Runner
 * Spawns the nanoclaw-ollama container to run Ollama agent tasks in isolation.
 * Simpler than container-runner.ts — no IPC, no credential proxy, just stdin/stdout.
 */

import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { GROUPS_DIR } from './config.js';
import { readEnvFile } from './env.js';
import { getGithubToken } from './github-token.js';
import { logger } from './logger.js';
import { CONTAINER_RUNTIME_BIN, hostGatewayArgs } from './container-runtime.js';

const envConfig = readEnvFile(['OLLAMA_HOST', 'OLLAMA_MODEL']);

const OLLAMA_IMAGE =
  process.env.OLLAMA_CONTAINER_IMAGE || 'nanoclaw-ollama:latest';
const OLLAMA_HOST =
  process.env.OLLAMA_HOST || envConfig.OLLAMA_HOST || 'http://host.docker.internal:11434';
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL || envConfig.OLLAMA_MODEL || 'llama3.1:8b';

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

export interface OllamaContainerOptions {
  systemPrompt?: string;
  maxIterations?: number;
  maxDurationMs?: number;
  maxToolOutputLength?: number;
  nudgeMessage?: string;
  /** Extra tool definitions (no bash/file access). When called, customToolHandler is invoked. */
  customTools?: object[];
  /** Called when the model invokes a custom tool. Return the reply to send to the user. */
  customToolHandler?: (name: string, args: Record<string, unknown>) => Promise<string | null>;
}

export async function runOllamaContainer(
  text: string,
  groupFolder: string,
  options?: OllamaContainerOptions,
): Promise<string> {
  const ghToken = await getGithubToken();

  // Split groupFolder into base (for mount) and historySubdir
  // e.g. "slack_engineering/history/userId" → base="slack_engineering", subdir="history/userId"
  const slashIdx = groupFolder.indexOf('/');
  const baseFolder = slashIdx === -1 ? groupFolder : groupFolder.slice(0, slashIdx);
  const historySubdir = slashIdx === -1 ? '' : groupFolder.slice(slashIdx + 1);

  const groupDir = path.join(GROUPS_DIR, baseFolder);
  fs.mkdirSync(groupDir, { recursive: true });

  const codeDir = path.join(os.homedir(), 'code');
  const hostUid = process.getuid?.() ?? 502;
  const hostGid = process.getgid?.() ?? 20;

  const containerName = `nanoclaw-ollama-${baseFolder}-${Date.now()}`;

  const args = [
    'run', '-i', '--rm',
    '--name', containerName,
    '-e', `TZ=${process.env.TZ || 'UTC'}`,
    '-e', `OLLAMA_HOST=${OLLAMA_HOST}`,
    '-e', `OLLAMA_MODEL=${OLLAMA_MODEL}`,
    ...hostGatewayArgs(),
    '--user', `${hostUid}:${hostGid}`,
    '-e', 'HOME=/home/node',
    '-v', `${groupDir}:/workspace/group`,
  ];

  if (ghToken) {
    args.push('-e', `GH_TOKEN=${ghToken}`);
    args.push('-e', `GITHUB_TOKEN=${ghToken}`);
  }

  if (fs.existsSync(codeDir)) {
    args.push('-v', `${codeDir}:/workspace/code`);
  }

  args.push(OLLAMA_IMAGE);

  const { customToolHandler, ...serializableOptions } = options ?? {};
  const input = JSON.stringify({
    prompt: text,
    groupFolder: baseFolder,
    historySubdir: historySubdir || undefined,
    chatJid: '',
    options: serializableOptions,
  });

  logger.info(
    { group: baseFolder, historySubdir: historySubdir || undefined, model: OLLAMA_MODEL },
    'Spawning Ollama container',
  );

  return new Promise((resolve, reject) => {
    const container = spawn(CONTAINER_RUNTIME_BIN, args, { stdio: ['pipe', 'pipe', 'pipe'] });

    let stdout = '';
    let stderr = '';

    container.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    container.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      stderr += text;
      // Log runner messages in real time
      for (const line of text.split('\n')) {
        if (line.trim()) logger.debug({ group: baseFolder, line }, 'ollama-runner');
      }
    });

    // Send input and close stdin
    container.stdin.write(input);
    container.stdin.end();

    container.on('close', (code) => {
      if (code !== 0) {
        logger.error({ group: baseFolder, code, stderr: stderr.slice(-500) }, 'Ollama container exited with error');
        reject(new Error(`Ollama container exited with code ${code}: ${stderr.slice(-200)}`));
        return;
      }

      // Parse result from stdout markers
      const startIdx = stdout.lastIndexOf(OUTPUT_START_MARKER);
      const endIdx = stdout.lastIndexOf(OUTPUT_END_MARKER);

      if (startIdx === -1 || endIdx === -1 || endIdx < startIdx) {
        logger.error({ group: baseFolder, stdout: stdout.slice(-500) }, 'No output markers in Ollama container stdout');
        reject(new Error('Ollama container produced no output'));
        return;
      }

      try {
        const jsonLine = stdout.slice(startIdx + OUTPUT_START_MARKER.length, endIdx).trim();
        const output = JSON.parse(jsonLine) as {
          status: string;
          result: string | null;
          error?: string;
          toolName?: string;
          toolArgs?: Record<string, unknown>;
        };

        if (output.status === 'error') {
          reject(new Error(output.error || 'Ollama agent error'));
          return;
        }

        if (output.status === 'custom_tool' && output.toolName && customToolHandler) {
          logger.info({ group: baseFolder, tool: output.toolName }, 'Ollama container: custom tool call');
          customToolHandler(output.toolName, output.toolArgs ?? {})
            .then((handlerResult) => resolve(handlerResult ?? output.result ?? ''))
            .catch((err) => reject(err));
          return;
        }

        logger.info({ group: baseFolder, resultLength: output.result?.length ?? 0 }, 'Ollama container completed');
        resolve(output.result ?? '');
      } catch (err) {
        reject(new Error(`Failed to parse Ollama container output: ${err instanceof Error ? err.message : String(err)}`));
      }
    });

    container.on('error', (err) => {
      reject(new Error(`Ollama container spawn error: ${err.message}`));
    });
  });
}
