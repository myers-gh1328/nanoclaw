/**
 * Brave Search MCP Server for NanoClaw
 * Exposes Brave Search API as a tool for the container agent.
 * Reads BRAVE_API_KEY from environment (injected by container-runner).
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const BRAVE_API_KEY = process.env.BRAVE_API_KEY;
const BRAVE_SEARCH_URL = 'https://api.search.brave.com/res/v1/web/search';

function log(msg: string): void {
  console.error(`[BRAVE] ${msg}`);
}

const server = new McpServer({
  name: 'brave',
  version: '1.0.0',
});

server.tool(
  'brave_search',
  'Search the web using Brave Search API. Returns titles, URLs, and descriptions for the top results.',
  {
    query: z.string().describe('The search query'),
    count: z.number().int().min(1).max(20).optional().describe('Number of results to return (default: 5)'),
  },
  async (args) => {
    if (!BRAVE_API_KEY) {
      return {
        content: [{ type: 'text' as const, text: 'BRAVE_API_KEY is not configured.' }],
        isError: true,
      };
    }

    const count = args.count ?? 5;
    const url = `${BRAVE_SEARCH_URL}?q=${encodeURIComponent(args.query)}&count=${count}`;
    log(`Searching: ${args.query} (count=${count})`);

    try {
      const res = await fetch(url, {
        headers: {
          'Accept': 'application/json',
          'Accept-Encoding': 'gzip',
          'X-Subscription-Token': BRAVE_API_KEY,
        },
      });

      if (!res.ok) {
        const text = await res.text();
        return {
          content: [{ type: 'text' as const, text: `Brave API error (${res.status}): ${text}` }],
          isError: true,
        };
      }

      const data = await res.json() as {
        web?: {
          results?: Array<{
            title: string;
            url: string;
            description?: string;
          }>;
        };
      };

      const results = data.web?.results ?? [];
      if (results.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No results found.' }] };
      }

      const formatted = results.map((r, i) =>
        `${i + 1}. **${r.title}**\n   ${r.url}${r.description ? `\n   ${r.description}` : ''}`
      ).join('\n\n');

      log(`Got ${results.length} results`);
      return { content: [{ type: 'text' as const, text: formatted }] };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `Brave search failed: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
