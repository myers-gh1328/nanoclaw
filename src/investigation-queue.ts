/**
 * Persistent investigation queue.
 * Runs one bug investigation at a time, survives restarts.
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

import { logger } from './logger.js';
import { runBugInvestigation, PendingIssue } from './slack-intake.js';
import { Channel } from './types.js';

export interface QueuedInvestigation {
  id: string;
  issueNumber: string;
  issueTitle: string;
  issue: PendingIssue;
  addedAt: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  startedAt?: string;
  completedAt?: string;
  result?:
    | { type: 'fixed'; prUrl: string }
    | { type: 'assigned'; summary: string };
}

export interface WorkerLoopDeps {
  channels: Channel[];
  telegramJid: string;
}

function queuePath(): string {
  return path.join(
    process.cwd(),
    'groups',
    'slack_bug_reports',
    'investigation-queue.json',
  );
}

export function loadQueue(): QueuedInvestigation[] {
  try {
    const raw = fs.readFileSync(queuePath(), 'utf8');
    return JSON.parse(raw) as QueuedInvestigation[];
  } catch {
    return [];
  }
}

export function saveQueue(queue: QueuedInvestigation[]): void {
  fs.writeFileSync(queuePath(), JSON.stringify(queue, null, 2));
}

/** On startup, reset any items stuck in 'running' back to 'pending'. */
export function resetRunningItems(): void {
  const queue = loadQueue();
  const stuck = queue.filter((i) => i.status === 'running');
  if (stuck.length === 0) return;
  for (const item of stuck) {
    item.status = 'pending';
    item.startedAt = undefined;
  }
  saveQueue(queue);
  logger.info(
    { count: stuck.length },
    'Investigation queue: reset stuck running items',
  );
}

export function addToQueue(
  issueNumber: string,
  issueTitle: string,
  issue: PendingIssue,
): QueuedInvestigation {
  const item: QueuedInvestigation = {
    id: crypto.randomBytes(4).toString('hex'),
    issueNumber,
    issueTitle,
    issue,
    addedAt: new Date().toISOString(),
    status: 'pending',
  };
  const queue = loadQueue();
  queue.push(item);
  saveQueue(queue);
  logger.info({ issueNumber, issueTitle }, 'Investigation queue: added');
  return item;
}

export function formatQueueStatus(): string {
  const queue = loadQueue();
  if (queue.length === 0) return 'No investigations queued.';

  const lines: string[] = ['Investigation queue:'];

  const running = queue.find((i) => i.status === 'running');
  if (running) {
    const mins = running.startedAt
      ? Math.round((Date.now() - Date.parse(running.startedAt)) / 60_000)
      : 0;
    lines.push(
      `▶ Running: #${running.issueNumber} "${running.issueTitle}" (started ${mins}m ago)`,
    );
  }

  const pending = queue.filter((i) => i.status === 'pending');
  for (const item of pending) {
    lines.push(`⏳ Pending: #${item.issueNumber} "${item.issueTitle}"`);
  }

  const done = queue
    .filter((i) => i.status === 'done' || i.status === 'failed')
    .sort((a, b) => (b.completedAt ?? '').localeCompare(a.completedAt ?? ''))
    .slice(0, 3);
  if (done.length > 0) {
    lines.push(
      `✅ Done (last ${done.length}): ${done.map((i) => `#${i.issueNumber}`).join(', ')}`,
    );
  }

  if (!running && pending.length === 0) {
    lines.push('(all done)');
  }

  return lines.join('\n');
}

let workerLoopStarted = false;
let currentlyRunning = false;

export function startWorkerLoop(
  deps: WorkerLoopDeps,
  onComplete: (
    item: QueuedInvestigation,
    result: QueuedInvestigation['result'],
  ) => Promise<void>,
): void {
  if (workerLoopStarted) return;
  workerLoopStarted = true;

  const tick = async () => {
    if (currentlyRunning) return;

    const queue = loadQueue();
    if (queue.some((i) => i.status === 'running')) return;

    const next = queue
      .filter((i) => i.status === 'pending')
      .sort((a, b) => a.addedAt.localeCompare(b.addedAt))[0];
    if (!next) return;

    currentlyRunning = true;
    next.status = 'running';
    next.startedAt = new Date().toISOString();
    saveQueue(queue);

    logger.info(
      { issueNumber: next.issueNumber, title: next.issueTitle },
      'Investigation queue: starting',
    );

    try {
      const result = await runBugInvestigation(
        next.issue,
        next.issueNumber,
        `investigation-${next.issueNumber}`,
      );
      const updated = loadQueue();
      const item = updated.find((i) => i.id === next.id);
      if (item) {
        item.status = 'done';
        item.completedAt = new Date().toISOString();
        item.result = result;
        saveQueue(updated);
      }
      logger.info(
        { issueNumber: next.issueNumber, result: result.type },
        'Investigation queue: completed',
      );
      await onComplete(next, result);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const updated = loadQueue();
      const item = updated.find((i) => i.id === next.id);
      if (item) {
        item.status = 'failed';
        item.completedAt = new Date().toISOString();
        item.result = { type: 'assigned', summary: errMsg };
        saveQueue(updated);
      }
      logger.error(
        { issueNumber: next.issueNumber, err },
        'Investigation queue: failed',
      );
      await onComplete(next, { type: 'assigned', summary: errMsg }).catch(
        () => {},
      );
    } finally {
      currentlyRunning = false;
    }
  };

  setInterval(() => {
    tick().catch((err) =>
      logger.error({ err }, 'Investigation worker tick error'),
    );
  }, 10_000);
  // Fire first tick soon after startup
  setTimeout(() => {
    tick().catch((err) =>
      logger.error({ err }, 'Investigation worker tick error'),
    );
  }, 2_000);

  logger.info('Investigation queue worker started');
}
