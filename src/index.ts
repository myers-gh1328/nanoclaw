import fs from 'fs';
import path from 'path';

import {
  ASSISTANT_NAME,
  CREDENTIAL_PROXY_PORT,
  IDLE_TIMEOUT,
  POLL_INTERVAL,
  TIMEZONE,
  TRIGGER_PATTERN,
} from './config.js';
import { startCredentialProxy } from './credential-proxy.js';
import './channels/index.js';
import {
  getChannelFactory,
  getRegisteredChannelNames,
} from './channels/registry.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  cleanupOrphans,
  ensureContainerRuntimeRunning,
  PROXY_BIND_HOST,
} from './container-runtime.js';
import {
  getAllChats,
  getAllRegisteredGroups,
  getAllSessions,
  getAllTasks,
  getMessagesSince,
  getNewMessages,
  getRegisteredGroup,
  getRouterState,
  initDatabase,
  setRegisteredGroup,
  setRouterState,
  setSession,
  deleteSession,
  storeChatMetadata,
  storeMessage,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { startIpcWatcher } from './ipc.js';
import { findChannel, formatMessages, formatOutbound } from './router.js';
import {
  isSenderAllowed,
  isTriggerAllowed,
  loadSenderAllowlist,
  shouldDropMessage,
} from './sender-allowlist.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { executeBash, runOllamaAgent, parseIntent } from './ollama-agent.js';
import {
  runIntakeAgent,
  loadPendingIssues,
  findPendingIssue,
  savePendingIssues,
  fileGithubIssue,
  formatInvestigationQuestion,
  INVOICING_REPO,
} from './issue-intake.js';
import {
  addToQueue,
  startWorkerLoop,
  resetRunningItems,
  formatQueueStatus,
} from './investigation-queue.js';
import { getGithubToken } from './github-token.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';

const TELEGRAM_MAIN_JID = 'tg:8388828787';
const TELEGRAM_OLLAMA_JID = 'tgo:8388828787';
const TELEGRAM_BUG_INTAKE_JID = 'tgo:-5191721027';
const SLACK_INTAKE_JID = 'slack:C0AL0D2K79R';
const INTAKE_GROUP_FOLDER = 'slack_bug_reports';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

let lastTimestamp = '';
let sessions: Record<string, string> = {};
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
const acknowledgedTimestamp: Record<string, string> = {};
let messageLoopRunning = false;

const channels: Channel[] = [];
const queue = new GroupQueue();

function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch {
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  const ackedTs = getRouterState('acknowledged_timestamp');
  try {
    Object.assign(acknowledgedTimestamp, ackedTs ? JSON.parse(ackedTs) : {});
  } catch {
    /* ignore */
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  logger.info(
    { groupCount: Object.keys(registeredGroups).length },
    'State loaded',
  );
}

function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState('last_agent_timestamp', JSON.stringify(lastAgentTimestamp));
  setRouterState(
    'acknowledged_timestamp',
    JSON.stringify(acknowledgedTimestamp),
  );
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(group.folder);
  } catch (err) {
    logger.warn(
      { jid, folder: group.folder, err },
      'Rejecting group registration with invalid folder',
    );
    return;
  }

  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

/**
 * Get available groups list for the agent.
 * Returns groups ordered by most recent activity.
 */
export function getAvailableGroups(): import('./container-runner.js').AvailableGroup[] {
  const chats = getAllChats();
  const registeredJids = new Set(Object.keys(registeredGroups));

  return chats
    .filter((c) => c.jid !== '__group_sync__' && c.is_group)
    .map((c) => ({
      jid: c.jid,
      name: c.name,
      lastActivity: c.last_message_time,
      isRegistered: registeredJids.has(c.jid),
    }));
}

/** @internal - exported for testing */
export function _setRegisteredGroups(
  groups: Record<string, RegisteredGroup>,
): void {
  registeredGroups = groups;
}

async function triggerManualInvestigation(
  issueNum: string,
  telegramChannel: Channel,
  telegramJid: string,
): Promise<void> {
  const ghToken = await getGithubToken();

  let issueData: { title: string; body: string; labels: { name: string }[] };
  try {
    const issueJson = await executeBash(
      `gh issue view ${issueNum} --repo ${INVOICING_REPO} --json title,body,labels`,
      process.cwd(),
      ghToken,
    );
    if (issueJson.startsWith('Error:')) throw new Error(issueJson);
    issueData = JSON.parse(issueJson) as {
      title: string;
      body: string;
      labels: { name: string }[];
    };
  } catch (err) {
    logger.error({ issueNum, err }, 'Failed to fetch issue for investigation');
    await telegramChannel.sendMessage(
      telegramJid,
      `Could not fetch issue #${issueNum}: ${err instanceof Error ? err.message : String(err)}`,
    );
    return;
  }

  const issue = {
    id: `manual-${issueNum}`,
    sourceJid: telegramJid,
    reporterName: 'manual',
    title: issueData.title,
    type: 'bug' as const,
    body: issueData.body ?? '',
    labels: issueData.labels.map((l) => l.name),
    createdAt: new Date().toISOString(),
  };

  addToQueue(issueNum, issueData.title, issue);
  await telegramChannel.sendMessage(
    telegramJid,
    `Added issue #${issueNum} to the investigation queue.`,
  );
}

async function handleIntakeApproval(
  decision: string,
  ref: string,
  channel: Channel,
  chatJid: string,
): Promise<void> {
  const issues = loadPendingIssues(INTAKE_GROUP_FOLDER);
  const issue = findPendingIssue(issues, ref);
  if (!issue) {
    await channel.sendMessage(chatJid, `No pending issue with ref ${ref}.`);
    return;
  }

  const remaining = issues.filter((i) => i.id !== ref);
  const lower = decision.toLowerCase();

  if (lower === 'no') {
    savePendingIssues(INTAKE_GROUP_FOLDER, remaining);
    await channel.sendMessage(
      chatJid,
      `Ok, skipping investigation for issue ${issue.issueNumber ? `#${issue.issueNumber}` : ref}.`,
    );
  } else if (lower === 'yes') {
    savePendingIssues(INTAKE_GROUP_FOLDER, remaining);
    const issueNum = issue.issueNumber;
    if (issueNum) {
      addToQueue(issueNum, issue.title, issue);
      await channel.sendMessage(
        chatJid,
        `Queued #${issueNum} for Ollama investigation.`,
      );
    } else {
      await channel.sendMessage(
        chatJid,
        `Issue has no number — cannot queue for investigation.`,
      );
    }
  } else if (lower === 'yes claude') {
    savePendingIssues(INTAKE_GROUP_FOLDER, remaining);
    const issueNum = issue.issueNumber;
    if (issueNum) {
      await channel.sendMessage(chatJid, `Routing #${issueNum} to Claude agent...`);
      triggerManualInvestigation(issueNum, channel, chatJid).catch((err) => {
        logger.error({ issueNum, err }, 'Claude investigation error');
        channel.sendMessage(
          chatJid,
          `Investigation error: ${err instanceof Error ? err.message : String(err)}`,
        );
      });
    } else {
      await channel.sendMessage(
        chatJid,
        `Issue has no number — cannot route to Claude agent.`,
      );
    }
  }
}

/**
 * Process all pending messages for a group.
 * Called by the GroupQueue when it's this group's turn.
 */
async function processGroupMessages(chatJid: string): Promise<boolean> {
  const group = registeredGroups[chatJid];
  if (!group) return true;

  const channel = findChannel(channels, chatJid);
  if (!channel) {
    logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
    return true;
  }

  const isMainGroup = group.isMain === true;

  const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
  const missedMessages = getMessagesSince(
    chatJid,
    sinceTimestamp,
    ASSISTANT_NAME,
  );

  if (missedMessages.length === 0) return true;

  // For non-main groups, check if trigger is required and present
  if (!isMainGroup && group.requiresTrigger !== false) {
    const allowlistCfg = loadSenderAllowlist();
    const hasTrigger = missedMessages.some(
      (m) =>
        TRIGGER_PATTERN.test(m.content.trim()) &&
        (m.is_from_me || isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
    );
    if (!hasTrigger) return true;
  }

  const prompt = formatMessages(missedMessages, TIMEZONE);

  // Advance cursor so the piping path in startMessageLoop won't re-fetch
  // these messages. Save the old cursor so we can roll back on error.
  const previousCursor = lastAgentTimestamp[chatJid] || '';
  lastAgentTimestamp[chatJid] =
    missedMessages[missedMessages.length - 1].timestamp;
  saveState();

  logger.info(
    { group: group.name, messageCount: missedMessages.length },
    'Processing messages',
  );

  // Check for intake investigation replies in any channel that has pending issues
  // sourced from this JID. Must run before Ollama routing since tgo:/slack: prefixes
  // are also matched by the Ollama routing block below.
  {
    const allPending = loadPendingIssues(INTAKE_GROUP_FOLDER);
    const pendingForChannel = allPending.filter((i) => i.sourceJid === chatJid);
    const pendingRefs = pendingForChannel.map((i) => i.id);

    let commandHandled = false;
    for (const msg of missedMessages) {
      if (msg.is_from_me) continue;
      const text = msg.content.trim();

      const parsed =
        pendingRefs.length > 0
          ? await parseIntent<{ decision: string; ref: string }>(
              `{ "decision": "yes" | "yes claude" | "no", "ref": "<6-char hex id>" }`,
              `The user is deciding whether to investigate a filed GitHub issue.\n` +
                `Valid ref IDs: ${pendingRefs.join(', ')}.\n` +
                `"decision" must be exactly "yes" (Ollama), "yes claude" (Claude agent), or "no" (skip).\n` +
                `If the user says yes/investigate/look into/fix/sure/ok (or misspellings), that is "yes".\n` +
                `If the user says no/skip/ignore/cancel/drop (or misspellings), that is "no".\n` +
                `If the user says "yes claude" or "claude" or "use claude", that is "yes claude".\n` +
                `If there is only one valid ref (${pendingRefs.length === 1 ? pendingRefs[0] : 'N/A'}) and no ref is mentioned, use that one.\n` +
                `If no investigation intent is found, return null.`,
              text,
            )
          : null;

      if (parsed?.decision && parsed?.ref) {
        await channel.sendMessage(chatJid, 'On it...');
        await handleIntakeApproval(
          parsed.decision,
          parsed.ref,
          channel,
          chatJid,
        );
        commandHandled = true;
        continue;
      }

      // Queue status and manual investigation commands — only in the developer's Ollama bot
      if (chatJid === TELEGRAM_OLLAMA_JID) {
        // Queue status command
        if (/^queue$/i.test(text)) {
          await channel.sendMessage(chatJid, formatQueueStatus());
          commandHandled = true;
          continue;
        }

        // Investigation decision: "manual 227" or "claude 227"
        const decisionMatch = /^(manual|claude)\s+(\d+)$/i.exec(text);
        if (decisionMatch) {
          const action = decisionMatch[1].toLowerCase();
          const issueNum = decisionMatch[2];
          if (action === 'claude') {
            await channel.sendMessage(
              chatJid,
              `Queuing #${issueNum} for Claude agent...`,
            );
            triggerManualInvestigation(issueNum, channel, chatJid).catch(
              (err) => {
                logger.error({ issueNum, err }, 'Claude investigation error');
                channel.sendMessage(
                  chatJid,
                  `Investigation error: ${err instanceof Error ? err.message : String(err)}`,
                );
              },
            );
          } else {
            await channel.sendMessage(
              chatJid,
              `Ok, #${issueNum} is on you. I'll leave it open.`,
            );
          }
          commandHandled = true;
          continue;
        }
      }
    }
    if (commandHandled) return true;
  }

  // Route to Ollama if:
  // - chat is on the dedicated Ollama bot (tgo: prefix), OR
  // - last user message starts with @ollama
  const lastUserMsg = [...missedMessages].reverse().find((m) => !m.is_from_me);
  const ollamaPrefix = /^@ollama\s+/i;
  const isOllamaChat =
    chatJid.startsWith('tgo:') || chatJid.startsWith('slack:');
  const isOllamaPrefix =
    !isOllamaChat &&
    lastUserMsg &&
    ollamaPrefix.test(lastUserMsg.content.trim());

  if (isOllamaChat || isOllamaPrefix) {
    const text = isOllamaChat
      ? (lastUserMsg?.content.trim() ?? prompt)
      : lastUserMsg!.content.trim().replace(ollamaPrefix, '');
    await channel.setTyping?.(chatJid, true);
    if (chatJid === TELEGRAM_OLLAMA_JID) {
      await channel.sendMessage(chatJid, 'Routing message to agent...');
    }
    try {
      if (chatJid === SLACK_INTAKE_JID) {
        const userMessages = missedMessages.filter((m) => !m.is_from_me);
        for (const msg of userMessages) {
          const reporterName = msg.sender_name ?? 'unknown';
          const userId = msg.sender;
          const hasImage = msg.content.includes(
            '(Note: the user also attached an image which could not be processed.)',
          );
          if (hasImage) {
            const telegramChannel = findChannel(channels, TELEGRAM_OLLAMA_JID);
            telegramChannel?.sendMessage(
              TELEGRAM_OLLAMA_JID,
              `📎 @${reporterName} attached an image in Slack that couldn't be processed.`,
            );
          }
          await channel.sendMessage(
            chatJid,
            `Thanks @${reporterName}, reviewing your report...`,
          );
          try {
            const result = await runIntakeAgent(
              msg.content,
              group.folder,
              reporterName,
              chatJid,
              userId,
            );
            if (result.type === 'drafted') {
              const ghToken = await getGithubToken();
              for (const issue of result.issues) {
                try {
                  const url = await fileGithubIssue(issue, ghToken);
                  const issueNum = url.trim().match(/\/issues\/(\d+)/)?.[1];
                  issue.issueNumber = issueNum;
                  issue.issueUrl = url.trim();
                  // Update stored pending issue with number/url
                  const pending = loadPendingIssues(INTAKE_GROUP_FOLDER);
                  const stored = pending.find((p) => p.id === issue.id);
                  if (stored) {
                    stored.issueNumber = issue.issueNumber;
                    stored.issueUrl = issue.issueUrl;
                    savePendingIssues(INTAKE_GROUP_FOLDER, pending);
                  }
                  await channel.sendMessage(
                    chatJid,
                    formatInvestigationQuestion(issue),
                  );
                } catch (fileErr) {
                  logger.error({ err: fileErr }, 'Failed to file GitHub issue');
                  await channel.sendMessage(
                    chatJid,
                    `@${reporterName} — issue drafted but filing failed. Please retry or contact support.`,
                  );
                }
              }
            } else {
              await channel.sendMessage(chatJid, result.message);
            }
          } catch (err) {
            logger.error({ reporterName, userId, err }, 'Intake error');
            await channel.sendMessage(
              chatJid,
              `Sorry @${reporterName}, something went wrong. Please try again.`,
            );
          }
        }
      } else if (chatJid === TELEGRAM_OLLAMA_JID) {
        const reply = await runOllamaAgent(text, group.folder, {
          model: 'llama3.2:1b',
          numCtx: 4096,
          systemPrompt:
            `You are a control assistant for the Invoicing bug intake system. ` +
            `You can answer questions about the queue and help the user manage investigations.\n\n` +
            `If the user wants to investigate a GitHub issue — in any phrasing — call the queue_investigation tool with the issue number.\n` +
            `If the user asks about queue status, reply with the current queue state.\n` +
            `Do not attempt to investigate issues yourself using bash or any other tool.`,
          extraTools: [
            {
              type: 'function',
              function: {
                name: 'queue_investigation',
                description:
                  'Queue a GitHub issue for automated investigation. Use this whenever the user asks to investigate, look into, fix, or check any issue.',
                parameters: {
                  type: 'object',
                  properties: {
                    issue_number: {
                      type: 'string',
                      description: 'The GitHub issue number to investigate',
                    },
                  },
                  required: ['issue_number'],
                },
              },
            },
          ],
          toolHandler: async (name, args) => {
            if (name === 'queue_investigation') {
              const issueNum = String(args['issue_number']).replace(/\D/g, '');
              if (!issueNum)
                return { result: 'Error: no issue number provided.' };
              triggerManualInvestigation(issueNum, channel, chatJid).catch(
                (err) => {
                  logger.error({ issueNum, err }, 'Manual investigation error');
                  channel.sendMessage(
                    chatJid,
                    `Investigation error: ${err instanceof Error ? err.message : String(err)}`,
                  );
                },
              );
              return { result: `Issue #${issueNum} queued.`, stop: true };
            }
            return null;
          },
        });
        if (reply) await channel.sendMessage(chatJid, reply);
      } else if (chatJid === TELEGRAM_BUG_INTAKE_JID) {
        const senderName =
          missedMessages.filter((m) => !m.is_from_me).pop()?.sender_name ??
          'App User';
        const userId = `app:${senderName.replace(/\W+/g, '_').toLowerCase()}`;
        await channel.sendMessage(
          chatJid,
          'Bug report received. Processing...',
        );
        try {
          const result = await runIntakeAgent(
            text,
            INTAKE_GROUP_FOLDER,
            senderName,
            chatJid,
            userId,
          );
          if (result.type === 'drafted') {
            const ghToken = await getGithubToken();
            for (const issue of result.issues) {
              try {
                const url = await fileGithubIssue(issue, ghToken);
                const issueNum = url.trim().match(/\/issues\/(\d+)/)?.[1];
                issue.issueNumber = issueNum;
                issue.issueUrl = url.trim();
                // Update stored pending issue with number/url
                const pending = loadPendingIssues(INTAKE_GROUP_FOLDER);
                const stored = pending.find((p) => p.id === issue.id);
                if (stored) {
                  stored.issueNumber = issue.issueNumber;
                  stored.issueUrl = issue.issueUrl;
                  savePendingIssues(INTAKE_GROUP_FOLDER, pending);
                }
                await channel.sendMessage(
                  chatJid,
                  formatInvestigationQuestion(issue),
                );
              } catch (fileErr) {
                logger.error({ err: fileErr }, 'Failed to file GitHub issue');
                await channel.sendMessage(
                  chatJid,
                  `Issue drafted but filing failed. Please retry.`,
                );
              }
            }
          } else if (result.type === 'clarification') {
            await channel.sendMessage(
              chatJid,
              `Could not draft issue: ${result.message}`,
            );
          }
        } catch (err) {
          logger.error({ err }, 'Intake error (Telegram bug intake)');
          await channel.sendMessage(
            chatJid,
            `Intake error: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      } else {
        const reply = await runOllamaAgent(text, group.folder);
        if (reply) await channel.sendMessage(chatJid, reply);
      }
    } catch (err) {
      logger.error({ group: group.name, err }, 'Ollama agent error');
      await channel.sendMessage(chatJid, 'Ollama error — is it running?');
    } finally {
      await channel.setTyping?.(chatJid, false);
    }
    return true;
  }

  // Track idle timer for closing stdin when agent is idle
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      logger.debug(
        { group: group.name },
        'Idle timeout, closing container stdin',
      );
      queue.closeStdin(chatJid);
    }, IDLE_TIMEOUT);
  };

  const latestTimestamp = missedMessages[missedMessages.length - 1].timestamp;
  if (acknowledgedTimestamp[chatJid] !== latestTimestamp) {
    acknowledgedTimestamp[chatJid] = latestTimestamp;
    await channel.sendMessage(chatJid, 'Routing to agent...');
  }
  await channel.setTyping?.(chatJid, true);
  let hadError = false;
  let outputSentToUser = false;

  const output = await runAgent(group, prompt, chatJid, async (result) => {
    // Streaming output callback — called for each agent result
    if (result.result) {
      const raw =
        typeof result.result === 'string'
          ? result.result
          : JSON.stringify(result.result);
      // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
      const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
      logger.info({ group: group.name }, `Agent output: ${raw.slice(0, 200)}`);
      if (text) {
        await channel.sendMessage(chatJid, text);
        outputSentToUser = true;
      }
      // Only reset idle timer on actual results, not session-update markers (result: null)
      resetIdleTimer();
    }

    if (result.status === 'success') {
      queue.notifyIdle(chatJid);
    }

    if (result.status === 'error') {
      hadError = true;
    }
  });

  await channel.setTyping?.(chatJid, false);
  if (idleTimer) clearTimeout(idleTimer);

  if (output === 'error' || hadError) {
    // If we already sent output to the user, don't roll back the cursor —
    // the user got their response and re-processing would send duplicates.
    if (outputSentToUser) {
      logger.warn(
        { group: group.name },
        'Agent error after output was sent, skipping cursor rollback to prevent duplicates',
      );
      return true;
    }
    // Roll back cursor so retries can re-process these messages
    lastAgentTimestamp[chatJid] = previousCursor;
    saveState();
    logger.warn(
      { group: group.name },
      'Agent error, rolled back message cursor for retry',
    );
    return false;
  }

  return true;
}

async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<'success' | 'error'> {
  const isMain = group.isMain === true;
  const sessionId = sessions[group.folder];

  // Update tasks snapshot for container to read (filtered by group)
  const tasks = getAllTasks();
  writeTasksSnapshot(
    group.folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      agent_type: t.agent_type,
      status: t.status,
      next_run: t.next_run,
    })),
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
  );

  // Wrap onOutput to track session ID from streamed results
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId) {
          sessions[group.folder] = output.newSessionId;
          setSession(group.folder, output.newSessionId);
        }
        await onOutput(output);
      }
    : undefined;

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
        assistantName: ASSISTANT_NAME,
      },
      (proc, containerName) =>
        queue.registerProcess(chatJid, proc, containerName, group.folder),
      wrappedOnOutput,
    );

    if (output.status === 'error') {
      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      // Clear a broken session so the next retry starts fresh
      if (
        output.error?.includes('error_during_execution') ||
        output.error?.includes('exited with code 1')
      ) {
        logger.warn({ group: group.name }, 'Clearing broken session');
        delete sessions[group.folder];
        deleteSession(group.folder);
      }
      return 'error';
    }

    if (output.newSessionId) {
      sessions[group.folder] = output.newSessionId;
      setSession(group.folder, output.newSessionId);
    }

    return 'success';
  } catch (err) {
    logger.error({ group: group.name, err }, 'Agent error');
    return 'error';
  }
}

async function startMessageLoop(): Promise<void> {
  if (messageLoopRunning) {
    logger.debug('Message loop already running, skipping duplicate start');
    return;
  }
  messageLoopRunning = true;

  logger.info(`NanoClaw running (trigger: @${ASSISTANT_NAME})`);

  while (true) {
    try {
      const jids = Object.keys(registeredGroups);
      const { messages, newTimestamp } = getNewMessages(
        jids,
        lastTimestamp,
        ASSISTANT_NAME,
      );

      if (messages.length > 0) {
        logger.info({ count: messages.length }, 'New messages');

        // Advance the "seen" cursor for all messages immediately
        lastTimestamp = newTimestamp;
        saveState();

        // Deduplicate by group
        const messagesByGroup = new Map<string, NewMessage[]>();
        for (const msg of messages) {
          const existing = messagesByGroup.get(msg.chat_jid);
          if (existing) {
            existing.push(msg);
          } else {
            messagesByGroup.set(msg.chat_jid, [msg]);
          }
        }

        for (const [chatJid, groupMessages] of messagesByGroup) {
          const group = registeredGroups[chatJid];
          if (!group) continue;

          const channel = findChannel(channels, chatJid);
          if (!channel) {
            logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
            continue;
          }

          const isMainGroup = group.isMain === true;
          const needsTrigger = !isMainGroup && group.requiresTrigger !== false;

          // For non-main groups, only act on trigger messages.
          // Non-trigger messages accumulate in DB and get pulled as
          // context when a trigger eventually arrives.
          if (needsTrigger) {
            const allowlistCfg = loadSenderAllowlist();
            const hasTrigger = groupMessages.some(
              (m) =>
                TRIGGER_PATTERN.test(m.content.trim()) &&
                (m.is_from_me ||
                  isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
            );
            if (!hasTrigger) continue;
          }

          // Pull all messages since lastAgentTimestamp so non-trigger
          // context that accumulated between triggers is included.
          const allPending = getMessagesSince(
            chatJid,
            lastAgentTimestamp[chatJid] || '',
            ASSISTANT_NAME,
          );
          const messagesToSend =
            allPending.length > 0 ? allPending : groupMessages;
          const formatted = formatMessages(messagesToSend, TIMEZONE);

          if (queue.sendMessage(chatJid, formatted)) {
            logger.debug(
              { chatJid, count: messagesToSend.length },
              'Piped messages to active container',
            );
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            // Show typing indicator while the container processes the piped message
            channel
              .setTyping?.(chatJid, true)
              ?.catch((err) =>
                logger.warn({ chatJid, err }, 'Failed to set typing indicator'),
              );
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
          }
        }
      }
    } catch (err) {
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
    const pending = getMessagesSince(chatJid, sinceTimestamp, ASSISTANT_NAME);
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  ensureContainerRuntimeRunning();
  cleanupOrphans();
}

async function main(): Promise<void> {
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();
  resetRunningItems();

  // Start credential proxy (containers route API calls through this)
  const proxyServer = await startCredentialProxy(
    CREDENTIAL_PROXY_PORT,
    PROXY_BIND_HOST,
  );

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    proxyServer.close();
    await queue.shutdown(10000);
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Channel callbacks (shared by all channels)
  const channelOpts = {
    onMessage: (chatJid: string, msg: NewMessage) => {
      // Sender allowlist drop mode: discard messages from denied senders before storing
      if (!msg.is_from_me && !msg.is_bot_message && registeredGroups[chatJid]) {
        const cfg = loadSenderAllowlist();
        if (
          shouldDropMessage(chatJid, cfg) &&
          !isSenderAllowed(chatJid, msg.sender, cfg)
        ) {
          if (cfg.logDenied) {
            logger.debug(
              { chatJid, sender: msg.sender },
              'sender-allowlist: dropping message (drop mode)',
            );
          }
          return;
        }
      }
      storeMessage(msg);
    },
    onChatMetadata: (
      chatJid: string,
      timestamp: string,
      name?: string,
      channel?: string,
      isGroup?: boolean,
    ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
    registeredGroups: () => registeredGroups,
  };

  // Create and connect all registered channels.
  // Each channel self-registers via the barrel import above.
  // Factories return null when credentials are missing, so unconfigured channels are skipped.
  for (const channelName of getRegisteredChannelNames()) {
    const factory = getChannelFactory(channelName)!;
    const channel = factory(channelOpts);
    if (!channel) {
      logger.warn(
        { channel: channelName },
        'Channel installed but credentials missing — skipping. Check .env or re-run the channel skill.',
      );
      continue;
    }
    channels.push(channel);
    await channel.connect();
  }
  if (channels.length === 0) {
    logger.fatal('No channels connected');
    process.exit(1);
  }

  // Start subsystems (independently of connection handler)
  startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    getSessions: () => sessions,
    queue,
    onProcess: (groupJid, proc, containerName, groupFolder) =>
      queue.registerProcess(groupJid, proc, containerName, groupFolder),
    sendMessage: async (jid, rawText) => {
      const channel = findChannel(channels, jid);
      if (!channel) {
        logger.warn({ jid }, 'No channel owns JID, cannot send message');
        return;
      }
      const text = formatOutbound(rawText);
      if (text) await channel.sendMessage(jid, text);
    },
  });
  startIpcWatcher({
    sendMessage: (jid, text) => {
      const channel = findChannel(channels, jid);
      if (!channel) throw new Error(`No channel for JID: ${jid}`);
      return channel.sendMessage(jid, text);
    },
    registeredGroups: () => registeredGroups,
    registerGroup,
    syncGroups: async (force: boolean) => {
      await Promise.all(
        channels
          .filter((ch) => ch.syncGroups)
          .map((ch) => ch.syncGroups!(force)),
      );
    },
    getAvailableGroups,
    writeGroupsSnapshot: (gf, im, ag, rj) =>
      writeGroupsSnapshot(gf, im, ag, rj),
  });
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();
  startWorkerLoop(
    { channels, telegramJid: TELEGRAM_OLLAMA_JID },
    async (item, result) => {
      const telegramChannel = findChannel(channels, TELEGRAM_OLLAMA_JID);
      if (!telegramChannel) return;
      let msg: string;
      if (result?.type === 'fixed') {
        msg = `✅ #${item.issueNumber} fixed — PR: ${result.prUrl}`;
      } else {
        const summary =
          result?.type === 'assigned'
            ? result.summary
            : 'No details available.';
        msg =
          `🔍 #${item.issueNumber}: ${item.issueTitle}\n\n` +
          `Couldn't fix automatically.\n\n${summary}\n\n` +
          `Reply "manual ${item.issueNumber}" to handle it yourself, or "claude ${item.issueNumber}" to pass it to the Claude agent.`;
      }
      await telegramChannel.sendMessage(TELEGRAM_OLLAMA_JID, msg);
    },
    async (item) => {
      const telegramChannel = findChannel(channels, TELEGRAM_OLLAMA_JID);
      if (!telegramChannel) return;
      await telegramChannel.sendMessage(
        TELEGRAM_OLLAMA_JID,
        `🔎 Investigating #${item.issueNumber}: ${item.issueTitle}`,
      );
    },
  );
  startMessageLoop().catch((err) => {
    logger.fatal({ err }, 'Message loop crashed unexpectedly');
    process.exit(1);
  });
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start NanoClaw');
    process.exit(1);
  });
}
