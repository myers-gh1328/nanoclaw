/**
 * Recipe assistant agent.
 * Uses a local vector index (nomic-embed-text + SQLite) for fast semantic search
 * across PDF recipe collections. Falls back to direct PDF reads for full recipe text.
 */

import fs from 'fs';
import os from 'os';
import path from 'path';

import { executeBash, runOllamaAgent } from './ollama-agent.js';
import { getDb } from './db.js';
import {
  approveRecipe,
  listIndexedRecipes,
  reExtractRecipeWithHint,
  searchRecipes,
  updateRecipeText,
} from './recipe-index.js';

export const RECIPES_DIR = path.join(os.homedir(), 'recipes');

const RECIPE_SYSTEM_PROMPT = `You are a recipe assistant with access to the user's personal recipe collection.

CRITICAL RULES — violation is failure:
1. NEVER answer a recipe question from memory. Always call a tool first.
2. NEVER describe or list recipe content without a tool call confirming it.
3. NEVER say "I will search for..." or "I would use search_recipes". Just call the tool immediately.
4. To find a recipe: call search_recipes. To get full ingredients/instructions: call read_recipe.
5. To list what's available: call list_recipes.
6. After getting tool results, give a short, helpful reply.

Tools:
- list_recipes: list all indexed recipes (⚠️ = potentially incomplete)
- search_recipes: semantic search by name, ingredient, cuisine, mood, etc.
- read_recipe: read full text of a specific PDF (ingredients + instructions)
- reextract_recipe: re-run extraction for a partial recipe with a hint about what's wrong
- update_recipe: directly correct a recipe with the full corrected text
- approve_recipe: mark a partial recipe as complete

When a recipe is marked ⚠️ (partial):
- Warn the user before using it
- Offer to fix it: re-extract, manual correction, or approve as-is

When making a shopping list:
- Call search_recipes or read_recipe to get full ingredient lists
- Group by category (produce, dairy, meat, pantry, etc.)
- Combine and deduplicate across recipes`;

const RECIPE_EXTRA_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'list_recipes',
      description: 'List all indexed recipe PDFs and how many recipe chunks each contains.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_recipes',
      description:
        'Semantic search across all indexed recipes. Use this to find recipes by name, ingredient, cuisine, dietary preference, or mood. Returns the most relevant recipe sections.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'What to search for, e.g. "pasta with cream sauce" or "something light with chicken"',
          },
          results: {
            type: 'number',
            description: 'Number of results to return (default 4, max 10)',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_recipe',
      description:
        'Read the full text of a specific PDF file. Use when you need complete ingredient quantities or full instructions.',
      parameters: {
        type: 'object',
        properties: {
          filename: {
            type: 'string',
            description: 'The PDF filename from list_recipes, e.g. "italian.pdf"',
          },
        },
        required: ['filename'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'reextract_recipe',
      description: 'Re-run extraction for a partial recipe with a hint about what was wrong. Use when the user describes a specific issue.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'The PDF filename, e.g. "italian.pdf"' },
          recipe_name: { type: 'string', description: 'The exact recipe name as shown in list_recipes' },
          hint: { type: 'string', description: 'Description of what is missing or wrong, e.g. "pecans missing from ingredients"' },
        },
        required: ['filename', 'recipe_name', 'hint'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'update_recipe',
      description: 'Directly correct a recipe by providing the full corrected ingredients and instructions. Use when the user provides the complete corrected text.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'The PDF filename' },
          recipe_name: { type: 'string', description: 'The exact recipe name' },
          ingredients: { type: 'string', description: 'Full corrected ingredients list' },
          instructions: { type: 'string', description: 'Full corrected instructions' },
        },
        required: ['filename', 'recipe_name', 'ingredients', 'instructions'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'approve_recipe',
      description: 'Mark a partial recipe as complete when the user says it looks fine as-is.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'The PDF filename' },
          recipe_name: { type: 'string', description: 'The exact recipe name' },
        },
        required: ['filename', 'recipe_name'],
      },
    },
  },
];

export async function runRecipeAgent(
  text: string,
  groupFolder: string,
): Promise<string> {
  fs.mkdirSync(RECIPES_DIR, { recursive: true });
  const db = getDb();
  const historyKey = `recipes_${groupFolder}`;

  const model = 'qwen3.5:9b';
  const reply = await runOllamaAgent(text, historyKey, {
    model,
    systemPrompt: RECIPE_SYSTEM_PROMPT,
    extraTools: RECIPE_EXTRA_TOOLS,
    allowedTools: ['list_recipes', 'search_recipes', 'read_recipe', 'reextract_recipe', 'update_recipe', 'approve_recipe'],
    toolHandler: async (name, args) => {
      if (name === 'list_recipes') {
        const indexed = listIndexedRecipes(db);
        if (indexed.length === 0) {
          return { result: 'No recipes indexed yet. Add PDF files to ~/recipes/ — they will be indexed automatically.' };
        }
        const byFile = new Map<string, string[]>();
        for (const r of indexed) {
          const key = r.filename.replace(/\.pdf$/i, '');
          if (!byFile.has(key)) byFile.set(key, []);
          const label = r.status === 'partial' ? `⚠️ ${r.recipe_name} (potentially incomplete)` : r.recipe_name;
          byFile.get(key)!.push(label);
        }
        const lines: string[] = [];
        for (const [file, names] of byFile) {
          lines.push(`${file}:\n${names.map((n) => `  - ${n}`).join('\n')}`);
        }
        return { result: `Indexed recipes:\n\n${lines.join('\n\n')}` };
      }

      if (name === 'search_recipes') {
        const query = String(args['query'] ?? '');
        const topN = Math.min(Number(args['results'] ?? 4), 10);
        const results = await searchRecipes(db, query, topN);
        if (results.length === 0) {
          return { result: 'No matching recipes found.' };
        }
        const formatted = results
          .map((r, i) => {
            const partialNote = r.status === 'partial' ? ' ⚠️ POTENTIALLY INCOMPLETE' : '';
            return `--- Result ${i + 1} (from ${r.filename.replace(/\.pdf$/i, '')}, score: ${r.score.toFixed(2)}${partialNote}) ---\n${r.text}`;
          })
          .join('\n\n');
        return { result: formatted };
      }

      if (name === 'read_recipe') {
        const raw = String(args['filename'] ?? '');
        const basename = path.basename(raw);
        const filename = basename.toLowerCase().endsWith('.pdf') ? basename : `${basename}.pdf`;

        // Case-insensitive file lookup
        const files = fs.existsSync(RECIPES_DIR) ? fs.readdirSync(RECIPES_DIR) : [];
        const match = files.find((f) => f.toLowerCase() === filename.toLowerCase());
        if (!match) {
          return { result: `File not found: ${filename}. Use list_recipes to see available files.` };
        }

        const filePath = path.join(RECIPES_DIR, match);
        const extracted = await executeBash(`pdftotext "${filePath}" -`, RECIPES_DIR, null);
        if (extracted.startsWith('Error:')) {
          return { result: `Could not read PDF. Is pdftotext installed? Run: brew install poppler` };
        }
        return { result: extracted.slice(0, 8000) };
      }

      if (name === 'reextract_recipe') {
        const filename = String(args['filename'] ?? '');
        const recipeName = String(args['recipe_name'] ?? '');
        const hint = String(args['hint'] ?? '');
        try {
          const { status, qcIssues } = await reExtractRecipeWithHint(db, filename, recipeName, hint);
          if (status === 'done') {
            return { result: `✅ "${recipeName}" re-extracted and QC passed — now marked as complete.` };
          } else {
            return { result: `⚠️ "${recipeName}" re-extracted but still flagged as potentially incomplete: ${qcIssues}` };
          }
        } catch (err) {
          return { result: `Error re-extracting: ${err instanceof Error ? err.message : String(err)}` };
        }
      }

      if (name === 'update_recipe') {
        const filename = String(args['filename'] ?? '');
        const recipeName = String(args['recipe_name'] ?? '');
        const ingredients = String(args['ingredients'] ?? '');
        const instructions = String(args['instructions'] ?? '');
        try {
          const saved = await updateRecipeText(db, filename, recipeName, ingredients, instructions);
          return { result: saved ? `✅ "${recipeName}" updated and marked as complete.` : `Recipe not found: "${recipeName}"` };
        } catch (err) {
          return { result: `Error updating recipe: ${err instanceof Error ? err.message : String(err)}` };
        }
      }

      if (name === 'approve_recipe') {
        const filename = String(args['filename'] ?? '');
        const recipeName = String(args['recipe_name'] ?? '');
        const approved = approveRecipe(db, filename, recipeName);
        return { result: approved ? `✅ "${recipeName}" approved and marked as complete.` : `Recipe not found or not partial: "${recipeName}"` };
      }

      return null;
    },
  });
  return reply;
}
