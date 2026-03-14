/**
 * Intake agent for bug reports and feature requests from any channel.
 * Screens reports, drafts and files GitHub issues,
 * then asks the reporter whether they want to investigate it (via Ollama or Claude).
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { logger } from './logger.js';
import {
  clearOllamaHistory,
  executeBash,
  parseIntent,
  runOllamaAgent,
} from './ollama-agent.js';

export const INVOICING_REPO = 'myers-gh1328/Invoicing';
const INVOICING_PATH = path.join(os.homedir(), 'code', 'Invoicing');

// ---------------------------------------------------------------------------
// Pending investigation decisions
// ---------------------------------------------------------------------------

export interface PendingIssue {
  id: string;
  sourceJid: string; // JID of the channel the report came from
  reporterName: string;
  title: string;
  type: 'bug' | 'enhancement';
  body: string;
  labels: string[];
  createdAt: string;
  issueNumber?: string; // GitHub issue number (set after filing)
  issueUrl?: string; // GitHub issue URL (set after filing)
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

// ---------------------------------------------------------------------------
// Approval / notification helpers
// ---------------------------------------------------------------------------

// Matches: "yes (ref: abc123)", "no (ref: abc123)", "yes claude (ref: abc123)"
// "yes claude" must come before "yes" so the longer token is tried first
export const APPROVAL_PATTERN =
  /^(yes claude|yes|no)\s*\(ref:\s*([a-z0-9]+)\)/i;

export function parseApprovalReply(
  text: string,
): { decision: string; ref: string } | null {
  const match = APPROVAL_PATTERN.exec(text.trim());
  if (!match) return null;
  return { decision: match[1].trim(), ref: match[2].trim() };
}

export function formatInvestigationQuestion(issue: PendingIssue): string {
  const typeLabel = issue.type === 'bug' ? 'Bug' : 'Feature request';
  const issueRef = issue.issueUrl
    ? `Issue filed: ${issue.issueUrl}`
    : `Issue queued for filing`;
  return (
    `${typeLabel} from @${issue.reporterName}: ${issue.title}\n` +
    `${issueRef}\n\n` +
    `Do you want to work on it?\n` +
    `  yes (ref: ${issue.id}) — Ollama investigates\n` +
    `  yes claude (ref: ${issue.id}) — route to Claude agent\n` +
    `  no (ref: ${issue.id}) — skip`
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
// Intake system prompts
// ---------------------------------------------------------------------------

// Fast path: all required fields already present — draft immediately, no tools.
const FAST_DRAFT_PROMPT = `You are drafting a GitHub issue for the Invoicing app from a complete report.
All required information is already present. Draft immediately — no questions, no preamble.

Output format:
<draft>{"title": "...", "type": "bug" or "enhancement", "body": "...", "labels": [...]}</draft>

Body format for bugs: ## Description, ## Steps to Reproduce (omit if not provided), ## Expected Behavior (omit if not provided), ## Actual Behavior, ## Likely Subsystem
Body format for features: ## Problem, ## Proposed Solution

One <draft> per distinct issue. Do not invent facts not in the report.`;

// Conversational path: missing fields — read triage docs, ask questions, then draft.
const CONVERSATIONAL_INTAKE_PROMPT = `You are an intake assistant for the Invoicing app. The report you received is missing some required information. Your job is to gather it and file a high-quality GitHub issue.

START: Read ${INVOICING_PATH}/docs/ai-triage/triage-index.md first, then whichever are relevant:
- ${INVOICING_PATH}/docs/ai-triage/customer-language-map.md — translate vague phrases to technical meaning
- ${INVOICING_PATH}/docs/ai-triage/triage-follow-up-questions.md — the right questions per symptom class
- ${INVOICING_PATH}/docs/ai-triage/known-issue-signatures.md — recognize recurring known issues

REQUIRED FIELDS — gather ALL before drafting:
1. Type: is this a bug (something broken) or a feature request (new capability)?
2. For bugs: the page or URL where it occurs + what went wrong
3. For features: the problem being solved (page/URL only needed if related to an existing page)

After reading relevant docs, respond with EXACTLY ONE of:

A) QUESTIONS — ask ONLY about what is still missing, in plain language. One message, specific questions only.

B) DRAFT — once ALL required fields are known.
   <draft>{"title": "...", "type": "bug" or "enhancement", "body": "...", "labels": [...]}</draft>
   Body format for bugs: ## Description, ## Steps to Reproduce, ## Expected Behavior, ## Actual Behavior, ## Likely Subsystem
   Body format for features: ## Problem, ## Proposed Solution
   One <draft> per distinct issue. Do not invent facts not provided.

C) REDIRECT — if off-topic: "This channel is for bug reports and feature requests for the Invoicing app."

Output ONLY the questions, draft(s), or redirect. No explanations, no preamble.`;

// ---------------------------------------------------------------------------
// Structure pre-check
// ---------------------------------------------------------------------------

interface StructureCheck {
  hasType: boolean; // clearly a bug or feature request
  isBug: boolean; // true = bug, false = feature (only meaningful if hasType)
  hasDescription: boolean; // describes the problem or request
  hasLocation: boolean; // mentions a page, URL, or location in the app
}

async function checkReportStructure(text: string): Promise<StructureCheck> {
  const result = await parseIntent<StructureCheck>(
    '{ "hasType": boolean, "isBug": boolean, "hasDescription": boolean, "hasLocation": boolean }',
    `Analyze this report and determine:\n` +
      `- hasType: is it clearly a bug (something broken) or feature request (new capability)?\n` +
      `- isBug: true if it's a bug, false if it's a feature request\n` +
      `- hasDescription: does it describe what the problem or request is?\n` +
      `- hasLocation: does it mention a specific page, URL, or location in the app?`,
    text,
  );
  return (
    result ?? {
      hasType: false,
      isBug: false,
      hasDescription: false,
      hasLocation: false,
    }
  );
}

// ---------------------------------------------------------------------------
// Intake agent
// ---------------------------------------------------------------------------

export async function runIntakeAgent(
  text: string,
  groupFolder: string,
  reporterName: string,
  sourceJid: string,
  userId: string,
): Promise<
  | { type: 'clarification'; message: string }
  | { type: 'drafted'; issues: PendingIssue[] }
  | { type: 'redirect'; message: string }
> {
  const safeId = userId.replace(/[^a-zA-Z0-9_-]/g, '_');
  const userFolder = `${groupFolder}/history/${safeId}`;

  logger.info(
    { groupFolder, reporterName, userId },
    'Intake: evaluating report',
  );

  // Phase 1: detect whether the report has all required fields
  const structure = await checkReportStructure(text);
  const isStructured =
    structure.hasType &&
    structure.hasDescription &&
    (!structure.isBug || structure.hasLocation); // bugs need a location; features don't

  logger.info(
    { groupFolder, ...structure, isStructured },
    'Intake: structure check',
  );

  let reply: string;

  if (isStructured) {
    // Fast path: draft directly, no tools, no file reading
    const fastFolder = `${groupFolder}/history/_fast`;
    clearOllamaHistory(fastFolder);
    reply = await runOllamaAgent(text, fastFolder, {
      systemPrompt: FAST_DRAFT_PROMPT,
      allowedTools: [],
      numCtx: 4096,
    });
    clearOllamaHistory(fastFolder);
  } else {
    // Conversational path: full model, reads triage docs, asks for missing fields
    reply = await runOllamaAgent(text, userFolder, {
      systemPrompt: CONVERSATIONAL_INTAKE_PROMPT,
      allowedTools: ['read_file'],
      numCtx: 16384,
    });
  }

  logger.info(
    {
      groupFolder,
      isStructured,
      replyLength: reply.length,
      reply: reply.slice(0, 200),
    },
    'Intake: raw reply',
  );

  if (!reply.trim()) {
    logger.warn(
      { groupFolder, reporterName, isStructured },
      'Intake: model returned empty reply',
    );
    return {
      type: 'clarification',
      message:
        'Sorry, I had trouble processing that. Could you try describing the issue again?',
    };
  }

  const draftJsons = extractAllDraftJsons(reply);
  if (draftJsons.length > 0) {
    const newIssues: PendingIssue[] = draftJsons.map((draftJson) => ({
      id: crypto.randomBytes(3).toString('hex'),
      sourceJid,
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
    // Clear conversation history so future messages don't re-draft the same issue
    clearOllamaHistory(userFolder);
    logger.info(
      {
        groupFolder,
        count: newIssues.length,
        titles: newIssues.map((i) => i.title),
      },
      'Intake: drafted issues',
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
