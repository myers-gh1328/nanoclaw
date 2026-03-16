import Database from 'better-sqlite3';
import { describe, expect, it, beforeEach } from 'vitest';

import {
  cosineSimilarity,
  initRecipeIndex,
  listIndexedRecipes,
} from './recipe-index.js';

function makeDb(): Database.Database {
  const db = new Database(':memory:');
  initRecipeIndex(db);
  return db;
}

function insertDone(
  db: Database.Database,
  filename: string,
  chunkIndex: number,
  recipeName: string,
  text: string,
  vector: number[],
  mtime = 1000,
): void {
  db.prepare(
    `INSERT INTO recipe_chunks (filename, chunk_index, recipe_name, text, vector, file_mtime, status)
     VALUES (?, ?, ?, ?, ?, ?, 'done')`,
  ).run(filename, chunkIndex, recipeName, text, JSON.stringify(vector), mtime);
}

// ---------------------------------------------------------------------------
// cosineSimilarity
// ---------------------------------------------------------------------------

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const v = [1, 2, 3];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it('handles zero vectors without crashing', () => {
    const result = cosineSimilarity([0, 0], [1, 2]);
    expect(Number.isFinite(result)).toBe(true);
  });

  it('is symmetric', () => {
    const a = [0.5, 0.3, 0.8];
    const b = [0.1, 0.9, 0.2];
    expect(cosineSimilarity(a, b)).toBeCloseTo(cosineSimilarity(b, a));
  });
});

// ---------------------------------------------------------------------------
// initRecipeIndex
// ---------------------------------------------------------------------------

describe('initRecipeIndex', () => {
  it('creates recipe_chunks table', () => {
    const db = makeDb();
    const row = db
      .prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='recipe_chunks'`)
      .get();
    expect(row).toBeTruthy();
  });

  it('is idempotent — calling twice does not throw', () => {
    const db = makeDb();
    expect(() => initRecipeIndex(db)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// listIndexedRecipes
// ---------------------------------------------------------------------------

describe('listIndexedRecipes', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = makeDb();
  });

  it('returns empty array when no recipes indexed', () => {
    expect(listIndexedRecipes(db)).toEqual([]);
  });

  it('returns done recipes ordered by filename and chunk_index', () => {
    insertDone(db, 'italian.pdf', 0, 'Carbonara', 'pasta recipe', [1, 0], 1000);
    insertDone(db, 'italian.pdf', 1, 'Tiramisu', 'dessert recipe', [0, 1], 1000);
    insertDone(db, 'asian.pdf', 0, 'Ramen', 'noodle soup', [0.5, 0.5], 1000);

    const results = listIndexedRecipes(db);
    expect(results).toHaveLength(3);
    // asian.pdf comes first alphabetically
    expect(results[0].filename).toBe('asian.pdf');
    expect(results[0].recipe_name).toBe('Ramen');
    expect(results[1].filename).toBe('italian.pdf');
    expect(results[1].recipe_name).toBe('Carbonara');
    expect(results[2].recipe_name).toBe('Tiramisu');
  });

  it('excludes pending rows', () => {
    db.prepare(
      `INSERT INTO recipe_chunks (filename, chunk_index, recipe_name, text, vector, file_mtime, status)
       VALUES ('new.pdf', 0, 'Draft Recipe', '', '', 1000, 'pending')`,
    ).run();
    insertDone(db, 'italian.pdf', 0, 'Carbonara', 'pasta', [1, 0], 1000);

    const results = listIndexedRecipes(db);
    expect(results).toHaveLength(1);
    expect(results[0].recipe_name).toBe('Carbonara');
  });
});
