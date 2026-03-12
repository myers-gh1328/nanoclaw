/**
 * Slack intake agent for bug reports and feature requests.
 * Screens reports, asks clarifying questions, drafts GitHub issues,
 * and saves pending confirmations for Telegram approval.
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { getGithubToken } from './github-token.js';
import { logger } from './logger.js';
import {
  executeBash,
  loadHistory,
  saveHistory,
  runOllamaAgent,
} from './ollama-agent.js';
import { readEnvFile } from './env.js';

const envConfig = readEnvFile(['OLLAMA_HOST', 'OLLAMA_MODEL']);
const OLLAMA_HOST =
  process.env.OLLAMA_HOST || envConfig.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL || envConfig.OLLAMA_MODEL || 'llama3.1:8b';

const INVOICING_REPO = 'myers-gh1328/Invoicing';
const INVOICING_PATH = path.join(os.homedir(), 'code', 'Invoicing');

export interface PendingIssue {
  id: string;
  slackJid: string;
  reporterName: string;
  title: string;
  type: 'bug' | 'enhancement';
  body: string;
  labels: string[];
  createdAt: string;
}

function pendingPath(groupFolder: string): string {
  return path.join(process.cwd(), 'groups', groupFolder, 'pending-issues.json');
}

export function loadPendingIssues(groupFolder: string): PendingIssue[] {
  try {
    const raw = fs.readFileSync(pendingPath(groupFolder), 'utf8');
    return JSON.parse(raw) as PendingIssue[];
  } catch {
    return [];
  }
}

export function savePendingIssues(
  groupFolder: string,
  issues: PendingIssue[],
): void {
  fs.writeFileSync(pendingPath(groupFolder), JSON.stringify(issues, null, 2));
}

export function findPendingIssue(
  issues: PendingIssue[],
  id: string,
): PendingIssue | undefined {
  return issues.find((i) => i.id === id);
}

export const APPROVAL_PATTERN =
  /^(yes|no|yes but .+)\s*\(ref:\s*([a-z0-9]+)\)/i;

export function parseApprovalReply(
  text: string,
): { decision: string; ref: string } | null {
  const match = APPROVAL_PATTERN.exec(text.trim());
  if (!match) return null;
  return { decision: match[1].trim(), ref: match[2].trim() };
}

export function formatTelegramNotification(issue: PendingIssue): string {
  const typeLabel = issue.type === 'bug' ? 'Bug report' : 'Feature request';
  const bodyPreview =
    issue.body.length > 300 ? issue.body.slice(0, 300) + '...' : issue.body;
  return (
    `New ${typeLabel} from @${issue.reporterName}:\n` +
    `Title: ${issue.title}\n` +
    `Labels: ${issue.labels.join(', ')}\n\n` +
    `${bodyPreview}\n\n` +
    `Reply "yes", "no", or "yes but [changes]" (ref: ${issue.id})`
  );
}

const INTAKE_SYSTEM_PROMPT = `You are an intake assistant for an invoicing app called Invoicing. Your job is to screen bug reports and feature requests submitted via Slack and help turn them into well-formed GitHub issues.

When a user submits a report, you must decide ONE of two things:

1. ASK A CLARIFYING QUESTION — if the report is missing critical information:
   - For bugs: missing reproduction steps, unclear what went wrong, no platform/browser/version info when relevant
   - For features: no clear problem being solved, too vague to act on
   Ask ONE specific question. Be brief and friendly. Do not explain your reasoning.

2. DRAFT AN ISSUE — if you have enough information to create a useful GitHub issue.
   Output ONLY a JSON object wrapped in <draft> tags like this:
   <draft>{"title": "...", "type": "bug" or "enhancement", "body": "...", "labels": ["bug"] or ["enhancement"]}</draft>

   The body should be formatted markdown with:
   - For bugs: ## Description, ## Steps to Reproduce, ## Expected Behavior, ## Actual Behavior
   - For features: ## Problem, ## Proposed Solution
   Fill in what you know. Do not make up information the user didn't provide.

RULES:
- Never use tools. Never write code. Never explain your process.
- Output either a clarifying question (plain text) OR a <draft> block. Nothing else.
- If the message is not a bug report or feature request (e.g. "hello", spam, off-topic), reply with a brief polite redirect: "This channel is for bug reports and feature requests for the Invoicing app."`;

interface OllamaMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
}

interface DraftJson {
  title: string;
  type: string;
  body: string;
  labels: string[];
}

function extractDraftJson(content: string): DraftJson | null {
  // Candidates: <draft>...</draft> block first, then any embedded JSON object
  const candidates: string[] = [];

  const tagMatch = /<draft>([\s\S]*?)<\/draft>/i.exec(content);
  if (tagMatch) candidates.push(tagMatch[1].trim());

  // Extract any JSON object from the content using brace counting
  for (let start = 0; start < content.length; start++) {
    if (content[start] !== '{') continue;
    let depth = 0,
      inString = false,
      escaped = false,
      end = start;
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
      if (
        typeof parsed.title === 'string' &&
        typeof parsed.type === 'string' &&
        typeof parsed.body === 'string'
      ) {
        return parsed as unknown as DraftJson;
      }
    } catch {
      // try next
    }
  }
  return null;
}

export async function runSlackIntakeAgent(
  text: string,
  groupFolder: string,
  reporterName: string,
  slackJid: string,
): Promise<
  | { type: 'clarification'; message: string }
  | { type: 'drafted'; issue: PendingIssue }
  | { type: 'redirect'; message: string }
> {
  const history = loadHistory(groupFolder) as OllamaMessage[];
  history.push({ role: 'user', content: text });

  logger.info({ groupFolder, reporterName }, 'Slack intake: evaluating report');

  const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages: [{ role: 'system', content: INTAKE_SYSTEM_PROMPT }, ...history],
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama API error: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as { message: { content: string } };
  const reply = data.message.content.trim();

  logger.info(
    { groupFolder, reply: reply.slice(0, 200) },
    'Slack intake: raw reply',
  );

  history.push({ role: 'assistant', content: reply });
  saveHistory(groupFolder, history);

  // Try to extract a valid draft JSON — either from <draft> tags or raw JSON in the reply
  const draftJson = extractDraftJson(reply);
  if (draftJson) {
    const issue: PendingIssue = {
      id: crypto.randomBytes(3).toString('hex'),
      slackJid,
      reporterName,
      title: draftJson.title,
      type: draftJson.type === 'bug' ? 'bug' : 'enhancement',
      body: draftJson.body,
      labels: Array.isArray(draftJson.labels)
        ? draftJson.labels
        : [draftJson.type === 'bug' ? 'bug' : 'enhancement'],
      createdAt: new Date().toISOString(),
    };
    const issues = loadPendingIssues(groupFolder);
    issues.push(issue);
    savePendingIssues(groupFolder, issues);
    logger.info(
      { groupFolder, issueId: issue.id, title: issue.title },
      'Slack intake: drafted issue',
    );
    return { type: 'drafted', issue };
  }

  return { type: 'clarification', message: reply };
}

export async function fileGithubIssue(
  issue: PendingIssue,
  ghToken: string | null,
): Promise<string> {
  const labelsArg = issue.labels.map((l) => `--label "${l}"`).join(' ');
  // Write body to a temp file to avoid shell escaping issues
  const tmpBody = path.join(os.tmpdir(), `nanoclaw-issue-${issue.id}.md`);
  fs.writeFileSync(tmpBody, issue.body, 'utf8');
  const cmd = `gh issue create --repo ${INVOICING_REPO} --title ${JSON.stringify(issue.title)} --body-file ${tmpBody} ${labelsArg}`;
  try {
    const result = await executeBash(cmd, INVOICING_PATH, ghToken);
    fs.unlinkSync(tmpBody);
    return result;
  } catch (err) {
    fs.unlinkSync(tmpBody);
    throw err;
  }
}

export async function applyModificationAndFile(
  issue: PendingIssue,
  modification: string,
  groupFolder: string,
): Promise<string> {
  const ghToken = await getGithubToken();
  const prompt =
    `Apply this modification to the following GitHub issue draft, then file it in the ${INVOICING_REPO} repo using the gh CLI.\n\n` +
    `Modification: ${modification}\n\n` +
    `Issue draft:\n` +
    `Title: ${issue.title}\n` +
    `Type: ${issue.type}\n` +
    `Labels: ${issue.labels.join(', ')}\n` +
    `Body:\n${issue.body}\n\n` +
    `Use gh issue create --repo ${INVOICING_REPO} with the modified title and body. Report the URL when done.`;
  // Inject token into env for the agent's bash calls
  if (ghToken) {
    process.env.GH_TOKEN = ghToken;
    process.env.GITHUB_TOKEN = ghToken;
  }
  return runOllamaAgent(prompt, groupFolder);
}

export async function runBugInvestigation(
  issue: PendingIssue,
  issueNumber: string,
  groupFolder: string,
): Promise<
  { type: 'fixed'; prUrl: string } | { type: 'assigned'; summary: string }
> {
  const ghToken = await getGithubToken();
  if (ghToken) {
    process.env.GH_TOKEN = ghToken;
    process.env.GITHUB_TOKEN = ghToken;
  }

  const branchName = `fix/issue-${issueNumber}`;
  const prompt =
    `You are a software engineer investigating and fixing a bug in the Invoicing app.\n\n` +
    `Bug issue #${issueNumber}: ${issue.title}\n\n` +
    `${issue.body}\n\n` +
    `Repository: ${INVOICING_PATH}\n` +
    `GitHub repo: ${INVOICING_REPO}\n\n` +
    `INSTRUCTIONS:\n` +
    `1. Search the codebase to find the root cause. Use grep, git log, git blame, and read_file strategically — start with keywords from the bug description, find the relevant files, then read them.\n` +
    `2. IF you find the root cause and can confidently fix it:\n` +
    `   a. git checkout -b ${branchName}\n` +
    `   b. Apply the fix using write_file\n` +
    `   c. git add and git commit -m "Fix: ${issue.title} (closes #${issueNumber})"\n` +
    `   d. git push -u origin ${branchName}\n` +
    `   e. gh pr create --repo ${INVOICING_REPO} --title "Fix: ${issue.title}" --body "Closes #${issueNumber}" --head ${branchName}\n` +
    `   f. Post a review request: gh pr comment <pr-number> --repo ${INVOICING_REPO} --body "@copilot please review this fix"\n` +
    `   g. End your summary with: RESULT:FIXED:<pr-url>\n` +
    `3. IF you cannot confidently locate or fix the bug after thorough investigation:\n` +
    `   a. Post your findings: gh issue comment ${issueNumber} --repo ${INVOICING_REPO} --body "Investigation findings: <what you searched and found>"\n` +
    `   b. Assign to Copilot: gh issue edit ${issueNumber} --repo ${INVOICING_REPO} --add-assignee copilot\n` +
    `   c. End your summary with: RESULT:ASSIGNED:<one line summary of what you found>\n\n` +
    `Be thorough. You have plenty of iterations. Do not give up after a few searches.`;

  logger.info({ issueNumber, title: issue.title }, 'Bug investigation started');

  const reply = await runOllamaAgent(prompt, groupFolder, {
    maxDurationMs: 30 * 60 * 1000,
    maxIterations: 500,
  });

  // Parse structured result from agent summary
  const fixedMatch = /RESULT:FIXED:(https?:\/\/\S+)/i.exec(reply);
  if (fixedMatch) {
    return { type: 'fixed', prUrl: fixedMatch[1] };
  }

  const assignedMatch = /RESULT:ASSIGNED:(.+)/i.exec(reply);
  return {
    type: 'assigned',
    summary: assignedMatch ? assignedMatch[1].trim() : reply.slice(0, 200),
  };
}
