/**
 * Container runtime abstraction for NanoClaw.
 * All runtime-specific logic lives here so swapping runtimes means changing one file.
 */
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';

import { logger } from './logger.js';

/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN = 'container';

/**
 * Hostname/IP containers use to reach the host machine.
 * Apple Container (macOS): uses the bridge101 gateway IP (192.168.65.1).
 * Detected at startup from host network interfaces; falls back to 192.168.65.1.
 */
export const CONTAINER_HOST_GATEWAY = detectHostGateway();

/**
 * Address the credential proxy binds to.
 * Apple Container (macOS): bind to the bridge IP so containers can reach it.
 * Linux: bind to 0.0.0.0 as fallback.
 */
export const PROXY_BIND_HOST =
  process.env.CREDENTIAL_PROXY_HOST || detectProxyBindHost();

function detectHostGateway(): string {
  // Apple Container bridge interfaces act as gateway (.1 address) for container VMs.
  // Sort bridges descending by number so bridge101 beats bridge100 (Apple Container
  // uses higher-numbered bridges; lower-numbered ones may be VMware/Parallels).
  const ifaces = os.networkInterfaces();
  const bridges = Object.entries(ifaces)
    .filter(([name]) => name.startsWith('bridge'))
    .sort(([a], [b]) => {
      const numA = parseInt(a.replace('bridge', ''), 10) || 0;
      const numB = parseInt(b.replace('bridge', ''), 10) || 0;
      return numB - numA;
    });
  for (const [, addrs] of bridges) {
    if (!addrs) continue;
    const ipv4 = addrs.find(
      (a) =>
        a.family === 'IPv4' &&
        a.address.startsWith('192.168.') &&
        a.address.endsWith('.1'),
    );
    if (ipv4) return ipv4.address;
  }
  return '192.168.65.1';
}

function detectProxyBindHost(): string {
  // Bind to all interfaces so Apple Container VMs can reach the proxy.
  // On a personal Mac this is safe; the credential proxy requires a valid token.
  if (os.platform() === 'darwin') return '0.0.0.0';
  return '0.0.0.0';
}

/** CLI args needed for the container to resolve the host gateway. */
export function hostGatewayArgs(): string[] {
  return [];
}

/** Returns CLI args for a readonly bind mount. */
export function readonlyMountArgs(
  hostPath: string,
  containerPath: string,
): string[] {
  return [
    '--mount',
    `type=bind,source=${hostPath},target=${containerPath},readonly`,
  ];
}

/** Returns the shell command to stop a container by name. */
export function stopContainer(name: string): string {
  return `${CONTAINER_RUNTIME_BIN} stop ${name}`;
}

/** Ensure the container runtime is running, starting it if needed. */
export function ensureContainerRuntimeRunning(): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} system status`, { stdio: 'pipe' });
    logger.debug('Container runtime already running');
    return;
  } catch {
    // Not running — try to start it
  }

  logger.info('Starting container runtime...');
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} system start`, {
      stdio: 'pipe',
      timeout: 30000,
    });
    logger.info('Container runtime started');
  } catch (err) {
    // Container runtime unavailable (e.g. XPC session not ready at boot).
    // Log a warning but don't crash — Ollama agent works without containers,
    // and Claude agent requests will fail gracefully if containers are needed.
    logger.warn(
      { err },
      'Container runtime unavailable at startup — Claude agent will be unavailable until it is running',
    );
  }
}

/** Kill orphaned NanoClaw containers from previous runs. */
export function cleanupOrphans(): void {
  try {
    const output = execSync(`${CONTAINER_RUNTIME_BIN} ls --format json`, {
      stdio: ['pipe', 'pipe', 'pipe'],
      encoding: 'utf-8',
    });
    const containers: { status: string; configuration: { id: string } }[] =
      JSON.parse(output || '[]');
    const orphans = containers
      .filter(
        (c) =>
          c.status === 'running' && c.configuration.id.startsWith('nanoclaw-'),
      )
      .map((c) => c.configuration.id);
    for (const name of orphans) {
      try {
        execSync(stopContainer(name), { stdio: 'pipe' });
      } catch {
        /* already stopped */
      }
    }
    if (orphans.length > 0) {
      logger.info(
        { count: orphans.length, names: orphans },
        'Stopped orphaned containers',
      );
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to clean up orphaned containers');
  }
}
