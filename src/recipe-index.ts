/**
 * Recipe vector index.
 *
 * Two-phase indexing per PDF:
 *   Phase 1 — one LLM call to list all recipe names → saved as 'pending' rows immediately
 *   Phase 2 — one LLM call per recipe to extract ingredients + instructions → saved as 'done'
 *
 * Resumable: if indexing is interrupted, pending rows are picked up on the next run
 * without re-running phase 1. If the file changes (new mtime), phase 1 re-runs from scratch.
 *
 * Search: embeds the query with nomic-embed-text and ranks 'done' chunks by cosine similarity.
 */

import fs from 'fs';
import path from 'path';

import Database from 'better-sqlite3';

import { executeBash } from './ollama-agent.js';
import { RECIPES_DIR } from './recipe-agent.js';
import { logger } from './logger.js';

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const EMBED_MODEL = 'nomic-embed-text';
const PARSE_MODEL = process.env.RECIPE_PARSE_MODEL || 'mistral:7b';
const QC_MODEL = 'gemma3:1b';
const LLM_TIMEOUT_MS = 20 * 60 * 1000; // 20 min per LLM call

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

export function initRecipeIndex(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS recipe_chunks (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      filename    TEXT    NOT NULL,
      chunk_index INTEGER NOT NULL,
      recipe_name TEXT    NOT NULL DEFAULT '',
      text        TEXT    NOT NULL DEFAULT '',
      vector      TEXT    NOT NULL DEFAULT '',
      file_mtime  INTEGER NOT NULL,
      status      TEXT    NOT NULL DEFAULT 'done',
      UNIQUE(filename, chunk_index)
    );
    CREATE INDEX IF NOT EXISTS idx_recipe_chunks_filename
      ON recipe_chunks(filename);
  `);

  // Migrations for older schemas
  for (const col of [
    `ALTER TABLE recipe_chunks ADD COLUMN recipe_name TEXT NOT NULL DEFAULT ''`,
    `ALTER TABLE recipe_chunks ADD COLUMN status TEXT NOT NULL DEFAULT 'done'`,
  ]) {
    try { db.exec(col); } catch { /* already exists */ }
  }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

async function embed(text: string): Promise<number[]> {
  const res = await fetch(`${OLLAMA_HOST}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text }),
  });
  if (!res.ok) throw new Error(`Embed API error: ${res.status}`);
  const data = (await res.json()) as { embedding: number[] };
  return data.embedding;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

// ---------------------------------------------------------------------------
// LLM helpers
// ---------------------------------------------------------------------------

async function ollamaChat(systemPrompt: string, userPrompt: string, format?: object, model?: string): Promise<string> {
  const res = await fetch(`${OLLAMA_HOST}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: model ?? PARSE_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      stream: true,
      format: format ?? undefined,
      options: { num_ctx: 32768, temperature: 0 },
    }),
    signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
  });

  if (!res.ok) throw new Error(`LLM API error: ${res.status}`);

  let content = '';
  const rawText = await res.text();
  for (const line of rawText.split('\n')) {
    if (!line.trim()) continue;
    try {
      const msg = JSON.parse(line) as { message?: { content?: string } };
      if (msg.message?.content) content += msg.message.content;
    } catch { /* ignore malformed lines */ }
  }

  return content
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();
}

// Phase 1: return all recipe names found in the PDF text
async function extractRecipeNames(pdfText: string, filename: string): Promise<string[]> {
  logger.info({ filename, model: PARSE_MODEL }, 'Recipe index: extracting recipe names');

  const namesSchema = {
    type: 'object',
    properties: {
      names: { type: 'array', items: { type: 'string' } },
    },
    required: ['names'],
  };

  const raw = await ollamaChat(
    `You are a recipe extraction tool. List every recipe name found in the text.`,
    `List all recipe names in this PDF text:\n\n${pdfText}`,
    namesSchema,
  );

  const parsed = JSON.parse(raw) as { names: string[] };
  return parsed.names.filter((n) => typeof n === 'string' && n.trim().length > 0);
}

function buildRecipeText(name: string, recipe: { ingredients: string; instructions: string }, qcIssues?: string): string {
  const base = `${name}\n\nIngredients:\n${recipe.ingredients}\n\nInstructions:\n${recipe.instructions}`;
  return qcIssues ? `${base}\n\n⚠️ Potentially incomplete: ${qcIssues}` : base;
}

// Phase 2: extract ingredients + instructions for one named recipe
async function extractSingleRecipe(
  pdfText: string,
  recipeName: string,
  filename: string,
  knownIssues?: string,
): Promise<{ ingredients: string; instructions: string }> {
  logger.info({ filename, recipe: recipeName, retry: !!knownIssues }, 'Recipe index: extracting recipe');

  const recipeSchema = {
    type: 'object',
    properties: {
      ingredients: { type: 'string' },
      instructions: { type: 'string' },
    },
    required: ['ingredients', 'instructions'],
  };

  const issuesClause = knownIssues
    ? `\n\nA previous extraction had these quality issues that you must fix:\n${knownIssues}`
    : '';

  const raw = await ollamaChat(
    `You are a recipe extraction tool. Extract the specified recipe from the text.${issuesClause}`,
    `Extract the recipe called "${recipeName}" from this PDF text:\n\n${pdfText}`,
    recipeSchema,
  );

  const parsed = JSON.parse(raw) as { ingredients?: string; instructions?: string };
  return {
    ingredients: parsed.ingredients ?? '',
    instructions: parsed.instructions ?? '',
  };
}

// Phase 2b: quality-check an extracted recipe against the source PDF
async function qualityCheckRecipe(
  recipeName: string,
  recipe: { ingredients: string; instructions: string },
  pdfText: string,
  filename: string,
): Promise<{ passed: boolean; issues: string }> {
  logger.info({ filename, recipe: recipeName }, 'Recipe index: quality checking');

  const qcSchema = {
    type: 'object',
    properties: {
      passed: { type: 'boolean' },
      issues: { type: 'string' },
    },
    required: ['passed', 'issues'],
  };

  const raw = await ollamaChat(
    `You are a recipe quality control checker. Your job is to verify that a recipe has been correctly and completely extracted from a source document.

Check for:
- Ingredients list is present and complete (no items mentioned in instructions but missing from the list)
- Instructions are present and complete (all steps included, not cut off mid-sentence)
- No obviously missing content compared to the source

Return {"passed": true, "issues": ""} if the recipe looks complete and correct.
Return {"passed": false, "issues": "<description of what is missing or wrong>"} if there are problems.`,
    `Recipe name: "${recipeName}"

Extracted ingredients:
${recipe.ingredients}

Extracted instructions:
${recipe.instructions}

Source PDF text (for reference):
${pdfText}`,
    qcSchema,
    QC_MODEL,
  );

  const parsed = JSON.parse(raw) as { passed?: boolean; issues?: string };
  return {
    passed: parsed.passed ?? false,
    issues: parsed.issues ?? 'QC response could not be parsed',
  };
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

export async function indexRecipeFile(
  db: Database.Database,
  filename: string,
  notify?: (msg: string) => void,
): Promise<void> {
  const filePath = path.join(RECIPES_DIR, filename);
  if (!fs.existsSync(filePath)) return;

  const mtime = Math.floor(fs.statSync(filePath).mtimeMs);

  // Check existing rows for this file
  const rows = db
    .prepare('SELECT status, file_mtime FROM recipe_chunks WHERE filename = ? ORDER BY chunk_index')
    .all(filename) as { status: string; file_mtime: number }[];

  const currentMtime = rows.length > 0 && rows[0].file_mtime === mtime;
  const pendingCount = rows.filter((r) => r.status === 'pending').length;

  if (currentMtime && pendingCount === 0 && rows.length > 0) {
    logger.debug({ filename }, 'Recipe index: already up to date');
    return;
  }

  const pdfText = await executeBash(`pdftotext "${filePath}" -`, RECIPES_DIR, null);
  if (pdfText.startsWith('Error:') || pdfText.trim().length === 0) {
    logger.warn({ filename }, 'Recipe index: could not extract text');
    return;
  }

  // Phase 1: get recipe names if this is a fresh index or stale mtime
  if (!currentMtime || rows.length === 0) {
    logger.info({ filename }, 'Recipe index: phase 1 — listing recipes');
    notify?.(`📖 Indexing *${filename}* — identifying recipes...`);

    const names = await extractRecipeNames(pdfText, filename);
    if (names.length === 0) {
      logger.warn({ filename }, 'Recipe index: no recipes found');
      notify?.(`⚠️ No recipes found in *${filename}*`);
      return;
    }

    logger.info({ filename, count: names.length }, 'Recipe index: found recipes, saving as pending');
    notify?.(`📋 Found ${names.length} recipes in *${filename}* — extracting...`);

    // Delete stale rows only after phase 1 succeeds
    db.prepare('DELETE FROM recipe_chunks WHERE filename = ?').run(filename);

    const insert = db.prepare(`
      INSERT INTO recipe_chunks (filename, chunk_index, recipe_name, text, vector, file_mtime, status)
      VALUES (?, ?, ?, '', '', ?, 'pending')
    `);
    names.forEach((name, i) => insert.run(filename, i, name, mtime));
  }

  // Phase 2: extract each pending recipe
  const pending = db
    .prepare(`SELECT id, chunk_index, recipe_name FROM recipe_chunks WHERE filename = ? AND status = 'pending' ORDER BY chunk_index`)
    .all(filename) as { id: number; chunk_index: number; recipe_name: string }[];

  logger.info({ filename, pending: pending.length }, 'Recipe index: phase 2 — extracting recipes');

  const update = db.prepare(`
    UPDATE recipe_chunks SET text = ?, vector = ?, status = ? WHERE id = ?
  `);

  for (const row of pending) {
    try {
      let recipe = await extractSingleRecipe(pdfText, row.recipe_name, filename);
      const qc1 = await qualityCheckRecipe(row.recipe_name, recipe, pdfText, filename);
      if (!qc1.passed) {
        logger.warn({ filename, recipe: row.recipe_name, issues: qc1.issues }, 'Recipe index: QC failed, re-extracting with issues');
        recipe = await extractSingleRecipe(pdfText, row.recipe_name, filename, qc1.issues);
        const qc2 = await qualityCheckRecipe(row.recipe_name, recipe, pdfText, filename);
        if (!qc2.passed) {
          logger.warn({ filename, recipe: row.recipe_name, issues: qc2.issues }, 'Recipe index: QC failed after retry — saving as partial');
          const fullText = buildRecipeText(row.recipe_name, recipe, qc2.issues);
          const vector = await embed(fullText);
          update.run(fullText, JSON.stringify(vector), 'partial', row.id);
          notify?.(`⚠️ *${row.recipe_name}* saved as potentially incomplete: ${qc2.issues}`);
          continue;
        }
      }
      const fullText = buildRecipeText(row.recipe_name, recipe);
      const vector = await embed(fullText);
      update.run(fullText, JSON.stringify(vector), 'done', row.id);
      logger.info({ filename, recipe: row.recipe_name }, 'Recipe index: saved');
    } catch (err) {
      logger.warn({ filename, recipe: row.recipe_name, err }, 'Recipe index: failed to extract recipe, will retry next run');
    }
  }

  const counts = db
    .prepare(`SELECT status, COUNT(*) as n FROM recipe_chunks WHERE filename = ? GROUP BY status`)
    .all(filename) as { status: string; n: number }[];
  const byStatus = Object.fromEntries(counts.map((r) => [r.status, r.n]));
  const doneCount = (byStatus['done'] ?? 0) + (byStatus['partial'] ?? 0);
  const stillPending = byStatus['pending'] ?? 0;

  logger.info({ filename, ...byStatus }, 'Recipe index: file complete');
  if (stillPending > 0) {
    notify?.(`⚠️ *${filename}*: ${doneCount} recipes indexed, ${stillPending} still pending (will retry on next start)`);
  } else {
    notify?.(`✅ Done indexing *${filename}* — ${doneCount} recipes ready`);
  }
}

export async function indexAllRecipes(
  db: Database.Database,
  notify?: (msg: string) => void,
): Promise<void> {
  fs.mkdirSync(RECIPES_DIR, { recursive: true });
  const files = fs
    .readdirSync(RECIPES_DIR)
    .filter((f) => f.toLowerCase().endsWith('.pdf'));
  for (const file of files) {
    try {
      await indexRecipeFile(db, file, notify);
    } catch (err) {
      logger.warn({ file, err }, 'Recipe index: failed to index file');
      notify?.(`❌ Failed to index *${file}*`);
    }
  }
}

// ---------------------------------------------------------------------------
// File watcher
// ---------------------------------------------------------------------------

export function startRecipeWatcher(db: Database.Database, notify?: (msg: string) => void): void {
  fs.mkdirSync(RECIPES_DIR, { recursive: true });

  const timers = new Map<string, ReturnType<typeof setTimeout>>();

  fs.watch(RECIPES_DIR, (event, filename) => {
    if (!filename || !filename.toLowerCase().endsWith('.pdf')) return;

    const existing = timers.get(filename);
    if (existing) clearTimeout(existing);

    timers.set(
      filename,
      setTimeout(() => {
        timers.delete(filename);
        const filePath = path.join(RECIPES_DIR, filename);

        if (!fs.existsSync(filePath)) {
          db.prepare('DELETE FROM recipe_chunks WHERE filename = ?').run(filename);
          logger.info({ filename }, 'Recipe index: removed deleted file');
          return;
        }

        logger.info({ filename, event }, 'Recipe index: change detected, re-indexing');
        notify?.(`📥 New recipe file detected: *${filename}* — starting indexing...`);
        indexRecipeFile(db, filename, notify).catch((err) =>
          logger.warn({ filename, err }, 'Recipe index: watcher re-index failed'),
        );
      }, 1500),
    );
  });

  logger.info({ dir: RECIPES_DIR }, 'Recipe index: watcher started');
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

export interface RecipeChunk {
  filename: string;
  recipe_name: string;
  text: string;
  score: number;
  status: 'done' | 'partial';
}

export async function searchRecipes(
  db: Database.Database,
  query: string,
  topN = 5,
): Promise<RecipeChunk[]> {
  const queryVec = await embed(query);

  const rows = db
    .prepare(`SELECT filename, recipe_name, text, vector, status FROM recipe_chunks WHERE status IN ('done', 'partial')`)
    .all() as { filename: string; recipe_name: string; text: string; vector: string; status: 'done' | 'partial' }[];

  const scored = rows.map((row) => ({
    filename: row.filename,
    recipe_name: row.recipe_name,
    text: row.text,
    score: cosineSimilarity(queryVec, JSON.parse(row.vector) as number[]),
    status: row.status,
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

export function listIndexedRecipes(
  db: Database.Database,
): { filename: string; recipe_name: string; status: 'done' | 'partial' }[] {
  return db
    .prepare(
      `SELECT filename, recipe_name, status FROM recipe_chunks WHERE status IN ('done', 'partial') ORDER BY filename, chunk_index`,
    )
    .all() as { filename: string; recipe_name: string; status: 'done' | 'partial' }[];
}

// ---------------------------------------------------------------------------
// Corrections
// ---------------------------------------------------------------------------

export function approveRecipe(
  db: Database.Database,
  filename: string,
  recipeName: string,
): boolean {
  const result = db
    .prepare(`UPDATE recipe_chunks SET status = 'done' WHERE filename = ? AND recipe_name = ? AND status = 'partial'`)
    .run(filename, recipeName);
  return result.changes > 0;
}

export async function updateRecipeText(
  db: Database.Database,
  filename: string,
  recipeName: string,
  ingredients: string,
  instructions: string,
): Promise<boolean> {
  const fullText = buildRecipeText(recipeName, { ingredients, instructions });
  const vector = await embed(fullText);
  const result = db
    .prepare(`UPDATE recipe_chunks SET text = ?, vector = ?, status = 'done' WHERE filename = ? AND recipe_name = ?`)
    .run(fullText, JSON.stringify(vector), filename, recipeName);
  return result.changes > 0;
}

export async function reExtractRecipeWithHint(
  db: Database.Database,
  filename: string,
  recipeName: string,
  hint: string,
): Promise<{ saved: boolean; status: 'done' | 'partial'; qcIssues?: string }> {
  const filePath = path.join(RECIPES_DIR, filename);
  const pdfText = await executeBash(`pdftotext "${filePath}" -`, RECIPES_DIR, null);
  if (pdfText.startsWith('Error:') || !pdfText.trim()) {
    throw new Error(`Could not read PDF: ${filename}`);
  }

  const recipe = await extractSingleRecipe(pdfText, recipeName, filename, hint);
  const qc = await qualityCheckRecipe(recipeName, recipe, pdfText, filename);
  const status = qc.passed ? 'done' : 'partial';
  const fullText = buildRecipeText(recipeName, recipe, qc.passed ? undefined : qc.issues);
  const vector = await embed(fullText);

  const result = db
    .prepare(`UPDATE recipe_chunks SET text = ?, vector = ?, status = ? WHERE filename = ? AND recipe_name = ?`)
    .run(fullText, JSON.stringify(vector), status, filename, recipeName);

  logger.info({ filename, recipe: recipeName, status, qcPassed: qc.passed }, 'Recipe index: re-extracted with hint');
  return { saved: result.changes > 0, status, qcIssues: qc.passed ? undefined : qc.issues };
}

// ---------------------------------------------------------------------------
// Model preloading
// ---------------------------------------------------------------------------

/**
 * Pin a model in Ollama's memory so it is never unloaded between requests.
 * Uses keep_alive: -1 (infinite). Call at startup for models that should
 * always be ready (e.g. the intent/QC model).
 */
export async function keepModelLoaded(model: string): Promise<void> {
  try {
    await fetch(`${OLLAMA_HOST}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: 'hi' }],
        keep_alive: -1,
        stream: false,
      }),
      signal: AbortSignal.timeout(5 * 60 * 1000),
    });
    logger.info({ model }, 'Ollama: model pinned in memory');
  } catch (err) {
    logger.warn({ model, err }, 'Ollama: failed to pin model');
  }
}
