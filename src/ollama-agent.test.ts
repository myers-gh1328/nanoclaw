import { describe, expect, it } from 'vitest';

import {
  isDangerous,
  parseContentToolCall,
  repairJsonStrings,
} from './ollama-agent.js';

// ---------------------------------------------------------------------------
// isDangerous
// ---------------------------------------------------------------------------

describe('isDangerous', () => {
  it('blocks rm -rf on home directory', () => {
    expect(isDangerous('rm -rf ~/projects')).toBe(true);
    expect(isDangerous('rm -rf /Users/alice/docs')).toBe(true);
  });

  it('blocks rm -rf /', () => {
    expect(isDangerous('rm -rf /')).toBe(true);
  });

  it('blocks rm -rf *', () => {
    expect(isDangerous('rm -rf *')).toBe(true);
  });

  it('blocks disk format and device writes', () => {
    expect(isDangerous('mkfs.ext4 /dev/sdb')).toBe(true);
    expect(isDangerous('dd if=/dev/zero of=/dev/sda')).toBe(true);
    expect(isDangerous('echo foo > /dev/sdb')).toBe(true);
  });

  it('allows safe commands', () => {
    expect(isDangerous('git status')).toBe(false);
    expect(isDangerous('npm install')).toBe(false);
    expect(isDangerous('rm -rf node_modules')).toBe(false);
    expect(isDangerous('ls /Users')).toBe(false);
    expect(isDangerous('cat /etc/hosts')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// repairJsonStrings
// ---------------------------------------------------------------------------

describe('repairJsonStrings', () => {
  it('leaves valid JSON unchanged', () => {
    const input = '{"name":"bash","arguments":{"command":"ls"}}';
    expect(repairJsonStrings(input)).toBe(input);
  });

  it('escapes literal newlines inside string values', () => {
    const input = '{"content":"line one\nline two"}';
    const result = repairJsonStrings(input);
    expect(result).toBe('{"content":"line one\\nline two"}');
    expect(() => JSON.parse(result)).not.toThrow();
  });

  it('escapes literal tabs inside string values', () => {
    const input = '{"content":"col1\tcol2"}';
    const result = repairJsonStrings(input);
    expect(result).toBe('{"content":"col1\\tcol2"}');
  });

  it('escapes literal carriage returns inside string values', () => {
    const input = '{"content":"line\r\n"}';
    const result = repairJsonStrings(input);
    expect(() => JSON.parse(result)).not.toThrow();
  });

  it('does not escape control chars outside strings', () => {
    // Newlines between keys are fine in JSON — should not be double-escaped
    const input = '{\n"a": 1\n}';
    const result = repairJsonStrings(input);
    expect(result).toBe('{\n"a": 1\n}');
  });

  it('handles already-escaped sequences correctly', () => {
    const input = '{"msg":"hello\\nworld"}';
    const result = repairJsonStrings(input);
    expect(result).toBe('{"msg":"hello\\nworld"}');
    expect(JSON.parse(result).msg).toBe('hello\nworld');
  });

  it('handles multi-field objects with embedded newlines', () => {
    const input = '{"name":"write_file","arguments":{"path":"a.ts","content":"const x = 1;\nconst y = 2;"}}';
    const result = repairJsonStrings(input);
    const parsed = JSON.parse(result) as { arguments: { content: string } };
    expect(parsed.arguments.content).toBe('const x = 1;\nconst y = 2;');
  });
});

// ---------------------------------------------------------------------------
// parseContentToolCall
// ---------------------------------------------------------------------------

describe('parseContentToolCall', () => {
  it('returns null for empty content', () => {
    expect(parseContentToolCall('')).toBeNull();
  });

  it('parses a clean JSON tool call', () => {
    const content = JSON.stringify({
      name: 'bash',
      arguments: { command: 'git status' },
    });
    const result = parseContentToolCall(content);
    expect(result?.name).toBe('bash');
    expect(result?.arguments).toEqual({ command: 'git status' });
  });

  it('parses a tool call wrapped in markdown code fences', () => {
    const content = '```json\n{"name":"write_file","arguments":{"path":"a.ts","content":"hello"}}\n```';
    const result = parseContentToolCall(content);
    expect(result?.name).toBe('write_file');
    expect((result?.arguments as Record<string, unknown>)['path']).toBe('a.ts');
  });

  it('extracts embedded JSON from surrounding text', () => {
    const content = 'Sure, I will do that.\n{"name":"bash","arguments":{"command":"npm test"}}\nLet me run that.';
    const result = parseContentToolCall(content);
    expect(result?.name).toBe('bash');
  });

  it('repairs and parses JSON with literal newlines in content field', () => {
    const content = '{"name":"write_file","arguments":{"path":"f.ts","content":"line1\nline2"}}';
    const result = parseContentToolCall(content);
    expect(result?.name).toBe('write_file');
    expect((result?.arguments as Record<string, unknown>)['path']).toBe('f.ts');
  });

  it('returns null when no valid tool call structure found', () => {
    expect(parseContentToolCall('Just a regular message with no JSON.')).toBeNull();
    expect(parseContentToolCall('{"foo": "bar"}')).toBeNull(); // missing name/arguments
  });

  it('ignores objects without arguments field', () => {
    expect(parseContentToolCall('{"name":"bash"}')).toBeNull();
  });
});
