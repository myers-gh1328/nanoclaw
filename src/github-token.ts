/**
 * Generates a GitHub App installation access token.
 * Uses Node's built-in crypto — no external dependencies needed.
 */
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { logger } from './logger.js';

const envConfig = readEnvFile([
  'GITHUB_APP_ID',
  'GITHUB_APP_INSTALL_ID',
  'GITHUB_APP_PEM_PATH',
]);
const APP_ID = process.env.GITHUB_APP_ID || envConfig.GITHUB_APP_ID || '';
const INSTALL_ID =
  process.env.GITHUB_APP_INSTALL_ID || envConfig.GITHUB_APP_INSTALL_ID || '';
const PEM_PATH = (
  process.env.GITHUB_APP_PEM_PATH ||
  envConfig.GITHUB_APP_PEM_PATH ||
  '~/github-app.pem'
).replace(/^~/, os.homedir());

function base64url(buf: Buffer): string {
  return buf
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

function makeJwt(appId: string, privateKey: string): string {
  const now = Math.floor(Date.now() / 1000);
  const header = base64url(
    Buffer.from(JSON.stringify({ alg: 'RS256', typ: 'JWT' })),
  );
  const payload = base64url(
    Buffer.from(JSON.stringify({ iat: now - 60, exp: now + 600, iss: appId })),
  );
  const data = `${header}.${payload}`;
  const sig = crypto.createSign('RSA-SHA256').update(data).sign(privateKey);
  return `${data}.${base64url(sig)}`;
}

export async function getGithubToken(): Promise<string | null> {
  if (!APP_ID || !INSTALL_ID) return null;

  let pem: string;
  try {
    pem = fs.readFileSync(PEM_PATH, 'utf8');
  } catch {
    logger.warn(
      { path: PEM_PATH },
      'GitHub App PEM not found, skipping GH_TOKEN',
    );
    return null;
  }

  try {
    const jwt = makeJwt(APP_ID, pem);
    const res = await fetch(
      `https://api.github.com/app/installations/${INSTALL_ID}/access_tokens`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${jwt}`,
          Accept: 'application/vnd.github+json',
        },
      },
    );
    const data = (await res.json()) as { token?: string; message?: string };
    if (!data.token) {
      logger.warn({ message: data.message }, 'GitHub token request failed');
      return null;
    }
    logger.debug('GitHub App installation token generated');
    return data.token;
  } catch (err) {
    logger.warn({ err }, 'Failed to generate GitHub App token');
    return null;
  }
}
