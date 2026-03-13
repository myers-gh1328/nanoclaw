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
  clearOllamaHistory,
  executeBash,
  runOllamaAgent,
} from './ollama-agent.js';

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
    `To approve: yes (ref: ${issue.id})\n` +
    `To reject: no (ref: ${issue.id})\n` +
    `To modify: yes but [describe changes] (ref: ${issue.id})`
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

START: Read ${INVOICING_PATH}/docs/ai-triage/triage-index.md first — it has the system overview and links to all triage docs. Then read whichever of these are relevant to the report:
- docs/ai-triage/customer-language-map.md — translate vague customer phrases to technical meaning
- docs/ai-triage/triage-follow-up-questions.md — the right questions to ask per symptom class
- docs/ai-triage/known-issue-signatures.md — recognize recurring known issues

REQUIRED FIELDS — a draft may NOT be filed until ALL of these are known:
For bugs: page or URL where the issue occurs (a full URL like "https://example.com/SiteAdmin/Health" or a path like "admin/locations" both count), and a description of what went wrong. Expected behavior and steps to reproduce are optional — include them in the draft if provided, omit if not.
For features: the problem being solved (page/URL only needed if it relates to an existing page — skip if it's a new page or flow)

Reports arrive in a structured format with labeled fields (Page:, Description:, etc.). All required fields will always be present. Extract them directly and proceed immediately to DRAFT.

After reading the relevant docs, respond with EXACTLY ONE of:

A) QUESTIONS — never use this. All required fields are always provided.

B) DRAFT — if ALL required fields are known. Use the language map to fill in implied technical context.
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
    allowedTools: ['read_file'],
    model: 'llama3.2:1b',
    numCtx: 8192,
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
  // Fetch available labels from the repo and filter to only valid ones
  let validLabels = issue.labels;
  try {
    const labelOutput = await executeBash(
      `gh label list --repo ${INVOICING_REPO} --json name --jq '.[].name'`,
      INVOICING_PATH,
      ghToken,
    );
    const repoLabels = new Set(
      labelOutput
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean),
    );
    const filtered = issue.labels.filter((l) => repoLabels.has(l));
    if (filtered.length !== issue.labels.length) {
      const skipped = issue.labels.filter((l) => !repoLabels.has(l));
      logger.warn({ skipped }, 'Skipping labels not found in repo');
    }
    validLabels = filtered;
  } catch (err) {
    logger.warn({ err }, 'Could not fetch repo labels — filing without labels');
    validLabels = [];
  }

  const labelsArg = validLabels
    .map((l) => `--label ${JSON.stringify(l)}`)
    .join(' ');
  const tmpBody = path.join(os.tmpdir(), `nanoclaw-issue-${issue.id}.md`);
  fs.writeFileSync(tmpBody, issue.body, 'utf8');
  const cmd = `gh issue create --repo ${INVOICING_REPO} --title ${JSON.stringify(issue.title)} --body-file ${tmpBody} ${labelsArg}`;
  try {
    const result = await executeBash(cmd, INVOICING_PATH, ghToken);
    return result;
  } finally {
    try {
      fs.unlinkSync(tmpBody);
    } catch {
      /* already gone */
    }
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
    `1. Start by reading ${INVOICING_PATH}/docs/ai-bug-hunting/bug-hunting-index.md — it lists all investigation docs and when to use each. Then read ${INVOICING_PATH}/docs/ai-bug-hunting/quick-start-checklist.md for the step-by-step investigation workflow. Use code-entrypoints-by-symptom.md to find the exact files and methods for the symptom, and codebase-map.md to navigate the project structure.\n` +
    `2. Finding the file — use this strategy:\n` +
    `   a. If the issue mentions a URL path (e.g. "admin/locations"), translate each segment into a filename search: run \`find ${INVOICING_PATH} -iname "*Location*" -o -iname "*Admin*"\` or similar. Razor pages, components, and views are named after the route segment.\n` +
    `   b. Search for key terms case-INSENSITIVELY: \`grep -ri "keyword" ${INVOICING_PATH} --include="*.razor" --include="*.cs" -l\`. Never search for full button/label text verbatim — use one or two words.\n` +
    `   c. If unsure, list the directory: \`ls ${INVOICING_PATH}/Invoicing.Web/Features/Admin/\` and look for the obvious file.\n` +
    `   d. Do NOT give up after one failed search. Try at least 3 different search terms/strategies before concluding a file can't be found.\n` +
    `3. CRITICAL — once you have found the relevant code:\n` +
    `   - Do NOT stop and say you could not find it. If you found the file and line, you found it.\n` +
    `   - Do NOT say "I am not confident" if you can read the code and understand what it does wrong.\n` +
    `   - Attempt a fix. An imperfect fix that gets reviewed is better than no fix.\n` +
    `4. IF you attempt a fix:\n` +
    `   a. git checkout -b ${branchName}\n` +
    `   b. Apply the fix using write_file\n` +
    `   c. Run \`dotnet build ${INVOICING_PATH}\` to verify the fix compiles. If it fails, read the errors and fix them before continuing. Do NOT push broken code.\n` +
    `   d. git add and git commit -m "Fix: ${issue.title} (closes #${issueNumber})"\n` +
    `   e. git push -u origin ${branchName}\n` +
    `   f. gh pr create --repo ${INVOICING_REPO} --title "Fix: ${issue.title}" --body "Closes #${issueNumber}" --head ${branchName}\n` +
    `   g. gh pr comment <pr-number> --repo ${INVOICING_REPO} --body "@copilot please review this fix"\n` +
    `   h. End your final message with: RESULT:FIXED:<pr-url>\n` +
    `5. ONLY IF you have searched extensively and truly cannot locate any relevant code:\n` +
    `   a. gh issue comment ${issueNumber} --repo ${INVOICING_REPO} --body "Investigation findings: <what you searched and found>"\n` +
    `   b. gh issue edit ${issueNumber} --repo ${INVOICING_REPO} --add-assignee @copilot\n` +
    `   c. End your final message with: RESULT:ASSIGNED:<one line summary>\n\n` +
    `You have up to 30 minutes and 500 iterations. Be persistent. Finding the code is the hard part — once you see it, fix it.`;

  // Clear stale history so previous failed runs don't contaminate this one
  clearOllamaHistory(groupFolder);

  logger.info(
    { issueNumber, title: issue.title, groupFolder },
    'Bug investigation started',
  );

  const reply = await runOllamaAgent(prompt, groupFolder, {
    maxDurationMs: 30 * 60 * 1000,
    maxIterations: 500,
    maxToolOutputLength: 15000,
    numCtx: 32768,
    nudgeMessage:
      'You stopped before completing the task. You must keep searching. ' +
      'Do NOT stop until you have either fixed the bug (RESULT:FIXED) or ' +
      'exhausted all search options and assigned the issue to Copilot (RESULT:ASSIGNED). ' +
      'Continue now — use tools to keep searching.',
  });

  const fixedMatch = /RESULT:FIXED:(https?:\/\/\S+)/i.exec(reply);
  if (fixedMatch) {
    return { type: 'fixed', prUrl: fixedMatch[1] };
  }

  const assignedMatch = /RESULT:ASSIGNED:(.+)/i.exec(reply);
  const summary = assignedMatch ? assignedMatch[1].trim() : reply.slice(0, 300);
  return { type: 'assigned', summary };
}
