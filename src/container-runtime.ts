/**
 * Container runtime abstraction for NanoClaw.
 * All runtime-specific logic lives here so swapping runtimes means changing one file.
 */
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';

import { logger } from './logger.js';

/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN =
  process.env.CONTAINER_RUNTIME_BIN ||
  '/Applications/Docker.app/Contents/Resources/bin/docker';

/**
 * Hostname containers use to reach the host machine.
 * Docker Desktop (macOS): host.docker.internal resolves to the host automatically.
 */
export const CONTAINER_HOST_GATEWAY =
  process.env.CONTAINER_HOST_GATEWAY || 'host.docker.internal';

/**
 * Address the credential proxy binds to.
 */
export const PROXY_BIND_HOST = process.env.CREDENTIAL_PROXY_HOST || '0.0.0.0';

/** CLI args needed for the container to resolve the host gateway. */
export function hostGatewayArgs(): string[] {
  // Docker Desktop on macOS provides host.docker.internal automatically,
  // but --add-host ensures it works on Linux Docker too.
  return ['--add-host', 'host.docker.internal:host-gateway'];
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
    execSync(`${CONTAINER_RUNTIME_BIN} info`, { stdio: 'pipe' });
    logger.debug('Container runtime already running');
  } catch (err) {
    logger.warn(
      { err },
      'Docker daemon unavailable — Claude agent will be unavailable until Docker Desktop is running',
    );
  }
}

/** Kill orphaned NanoClaw containers from previous runs. */
export function cleanupOrphans(): void {
  try {
    const output = execSync(
      `${CONTAINER_RUNTIME_BIN} ps --filter name=nanoclaw- --format '{{.Names}}'`,
      { stdio: ['pipe', 'pipe', 'pipe'], encoding: 'utf-8' },
    );
    const orphans = output
      .split('\n')
      .map((n) => n.trim())
      .filter((n) => n.startsWith('nanoclaw-'));
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
