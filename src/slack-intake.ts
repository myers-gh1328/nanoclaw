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

const INTAKE_SYSTEM_PROMPT = `You are an intake assistant for the Invoicing app. Users submit bug reports and feature requests via Slack. Your job is to gather enough information to file a high-quality GitHub issue.

START: Read ${INVOICING_PATH}/docs/triage-index.md first — it has the system overview and links to all triage docs. Then read whichever of these are relevant to the report:
- customer-language-map.md — translate vague customer phrases to technical meaning
- triage-follow-up-questions.md — the right questions to ask per symptom class
- known-issue-signatures.md — recognize recurring known issues

After reading the relevant docs, respond with EXACTLY ONE of:

A) QUESTIONS — if the report needs more information. Use triage-follow-up-questions.md to pick the 2-4 most useful questions for the symptom. Be conversational and friendly.

B) DRAFT — if you have enough to file a useful issue. Use the language map to fill in implied technical context.
   <draft>{"title": "...", "type": "bug" or "enhancement", "body": "...", "labels": [...]}</draft>
   Body format for bugs: ## Description, ## Steps to Reproduce, ## Expected Behavior, ## Actual Behavior, ## Likely Subsystem
   Body format for features: ## Problem, ## Proposed Solution
   One <draft> per distinct issue. Do not invent facts the user didn't provide.

C) REDIRECT — if the message is off-topic: "This channel is for bug reports and feature requests for the Invoicing app."

Output ONLY the questions, draft(s), or redirect. No explanations, no preamble.`;

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
  // Use per-user subfolder so each reporter gets isolated conversation history
  const safeId = userId.replace(/[^a-zA-Z0-9_-]/g, '_');
  const userFolder = `${groupFolder}/history/${safeId}`;

  logger.info(
    { groupFolder, reporterName, userId },
    'Slack intake: evaluating report',
  );

  const reply = await runOllamaAgent(text, userFolder, {
    systemPrompt: INTAKE_SYSTEM_PROMPT,
  });

  logger.info(
    { groupFolder, reply: reply.slice(0, 200) },
    'Slack intake: raw reply',
  );

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
    `INSTRUCTIONS:\n` +
    `1. Start by reading ${INVOICING_PATH}/docs/triage-index.md — it contains the triage workflow, system architecture overview, and links to all other docs. Follow its recommended workflow: customer-language-map → known-issue-signatures → triage-rules.yaml → where-to-look-for-evidence. Read whichever of those docs are relevant to this bug before searching code.\n` +
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
