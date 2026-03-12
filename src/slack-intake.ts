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
import { executeBash, runOllamaAgent } from './ollama-agent.js';
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

// ---------------------------------------------------------------------------
// Pending issues file
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-user intake history
// ---------------------------------------------------------------------------

function intakeHistoryPath(groupFolder: string, userId: string): string {
  const safeId = userId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.join(
    process.cwd(),
    'groups',
    groupFolder,
    'history',
    `${safeId}.json`,
  );
}

interface OllamaMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
}

function loadIntakeHistory(
  groupFolder: string,
  userId: string,
): OllamaMessage[] {
  try {
    const raw = fs.readFileSync(intakeHistoryPath(groupFolder, userId), 'utf8');
    return JSON.parse(raw) as OllamaMessage[];
  } catch {
    return [];
  }
}

function saveIntakeHistory(
  groupFolder: string,
  userId: string,
  history: OllamaMessage[],
): void {
  const filePath = intakeHistoryPath(groupFolder, userId);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(history, null, 2));
}

// ---------------------------------------------------------------------------
// Approval / notification helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Draft extraction
// ---------------------------------------------------------------------------

interface DraftJson {
  title: string;
  type: string;
  body: string;
  labels: string[];
}

function extractJsonObjects(content: string): string[] {
  const results: string[] = [];
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
    if (depth === 0) results.push(content.slice(start, end + 1));
  }
  return results;
}

function parseDraftJson(candidate: string): DraftJson | null {
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
    /* ignore */
  }
  return null;
}

function extractAllDraftJsons(content: string): DraftJson[] {
  const results: DraftJson[] = [];

  // First pass: explicit <draft>...</draft> tags (supports multiple)
  const tagPattern = /<draft>([\s\S]*?)<\/draft>/gi;
  let tagMatch: RegExpExecArray | null;
  while ((tagMatch = tagPattern.exec(content)) !== null) {
    const parsed = parseDraftJson(tagMatch[1].trim());
    if (parsed) results.push(parsed);
  }
  if (results.length > 0) return results;

  // Fallback: any embedded JSON objects with required fields
  for (const candidate of extractJsonObjects(content)) {
    const parsed = parseDraftJson(candidate);
    if (parsed) results.push(parsed);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Intake system prompt
// ---------------------------------------------------------------------------

const INTAKE_SYSTEM_PROMPT = `You are an intake assistant for an invoicing app called Invoicing. Your job is to screen bug reports and feature requests submitted via Slack and help turn them into well-formed GitHub issues.

App feature areas (use these to classify and label reports):
- Pricing: pricing flow, pricing wizard, pricing timeline, pricing history, caching strategy, conflict validation
- Class Type Wizard: readability enhancements, wizard UI
- Azure Functions: integration, testing
- AI Assistant: per-tenant toggle, Azure AI Foundry / Foundry Local integration, invoice generation, scheduling insights
- General invoicing: invoice creation, tenant management, scheduling, reporting

When a user submits a report, you must decide ONE of two things:

1. ASK A CLARIFYING QUESTION — if the report is missing critical information:
   - For bugs: missing reproduction steps, unclear what went wrong, no platform/browser/version info when relevant
   - For features: no clear problem being solved, too vague to act on
   Ask ONE specific question. Be brief and friendly. Do not explain your reasoning.

2. DRAFT AN ISSUE — if you have enough information to create a useful GitHub issue.
   Output a JSON object wrapped in <draft> tags for each distinct issue:
   <draft>{"title": "...", "type": "bug" or "enhancement", "body": "...", "labels": ["bug"] or ["enhancement"]}</draft>

   If the user describes multiple distinct issues in one message, output one <draft> block per issue.
   If all issues are related, file them as one. Use judgment.

   The body should be formatted markdown with:
   - For bugs: ## Description, ## Steps to Reproduce, ## Expected Behavior, ## Actual Behavior
   - For features: ## Problem, ## Proposed Solution
   Fill in what you know. Do not make up information the user didn't provide.

RULES:
- Never use tools. Never write code. Never explain your process.
- Output either a clarifying question (plain text) OR one or more <draft> blocks. Nothing else.
- If the message is not a bug report or feature request (e.g. "hello", spam, off-topic), reply with a brief polite redirect: "This channel is for bug reports and feature requests for the Invoicing app."`;

// ---------------------------------------------------------------------------
// Intake agent
// ---------------------------------------------------------------------------

export async function runSlackIntakeAgent(
  text: string,
  groupFolder: string,
  reporterName: string,
  slackJid: string,
  userId: string,
): Promise<
  | { type: 'clarification'; message: string }
  | { type: 'drafted'; issues: PendingIssue[] }
  | { type: 'redirect'; message: string }
> {
  const history = loadIntakeHistory(groupFolder, userId);
  history.push({ role: 'user', content: text });

  logger.info(
    { groupFolder, reporterName, userId },
    'Slack intake: evaluating report',
  );

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
  saveIntakeHistory(groupFolder, userId, history);

  const draftJsons = extractAllDraftJsons(reply);
  if (draftJsons.length > 0) {
    const newIssues: PendingIssue[] = draftJsons.map((draftJson) => ({
      id: crypto.randomBytes(3).toString('hex'),
      slackJid,
      reporterName,
      title: draftJson.title,
      type:
        draftJson.type === 'bug' ? ('bug' as const) : ('enhancement' as const),
      body: draftJson.body,
      labels: Array.isArray(draftJson.labels)
        ? draftJson.labels
        : [draftJson.type === 'bug' ? 'bug' : 'enhancement'],
      createdAt: new Date().toISOString(),
    }));
    const existing = loadPendingIssues(groupFolder);
    existing.push(...newIssues);
    savePendingIssues(groupFolder, existing);
    logger.info(
      {
        groupFolder,
        count: newIssues.length,
        titles: newIssues.map((i) => i.title),
      },
      'Slack intake: drafted issues',
    );
    return { type: 'drafted', issues: newIssues };
  }

  return { type: 'clarification', message: reply };
}

// ---------------------------------------------------------------------------
// GitHub issue filing
// ---------------------------------------------------------------------------

export async function fileGithubIssue(
  issue: PendingIssue,
  ghToken: string | null,
): Promise<string> {
  const labelsArg = issue.labels.map((l) => `--label "${l}"`).join(' ');
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
  if (ghToken) {
    process.env.GH_TOKEN = ghToken;
    process.env.GITHUB_TOKEN = ghToken;
  }
  return runOllamaAgent(prompt, groupFolder);
}

// ---------------------------------------------------------------------------
// Bug investigation
// ---------------------------------------------------------------------------

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

  // Ensure isolated history folder exists for this investigation
  fs.mkdirSync(path.join(process.cwd(), 'groups', groupFolder), {
    recursive: true,
  });

  const branchName = `fix/issue-${issueNumber}`;
  const prompt =
    `You are a software engineer investigating and fixing a bug in the Invoicing app.\n\n` +
    `Bug issue #${issueNumber}: ${issue.title}\n\n` +
    `${issue.body}\n\n` +
    `Repository: ${INVOICING_PATH}\n` +
    `GitHub repo: ${INVOICING_REPO}\n\n` +
    `Documentation: ${INVOICING_PATH}/docs/ — read relevant docs before diving into code. Available docs:\n` +
    `  - pricing-flow-review.md, pricing-timeline-architecture.md, pricing-caching-strategy.md\n` +
    `  - pricing-history-requirements.md, pricing_conflict_validation.md\n` +
    `  - classtype-wizard-readability-enhancements.md\n` +
    `  - AZURE_FUNCTIONS_INTEGRATION_TESTING.md\n` +
    `  - ai-assistant-setup.md\n\n` +
    `INSTRUCTIONS:\n` +
    `1. Read any docs relevant to the bug area first (use read_file). Then search the codebase to find the root cause. Use grep, git log, git blame, and read_file strategically — start with keywords from the bug description, find the relevant files, then read them.\n` +
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

  logger.info(
    { issueNumber, title: issue.title, groupFolder },
    'Bug investigation started',
  );

  const reply = await runOllamaAgent(prompt, groupFolder, {
    maxDurationMs: 30 * 60 * 1000,
    maxIterations: 500,
  });

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
