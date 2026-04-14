#!/usr/bin/env node
/**
 * Editor MCP: surgical file editing, grep, and glob.
 *
 * Tools:
 *   - edit: Find-and-replace in a file (surgical, not full rewrite)
 *   - grep: Regex search across files with context lines
 *   - glob: Find files matching glob patterns
 *
 * Cross-platform: macOS, Linux, Windows.
 */

import {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {StdioServerTransport} from '@modelcontextprotocol/sdk/server/stdio.js';
import {z} from 'zod';
import {readFile, writeFile, stat} from 'node:fs/promises';
import {resolve, relative, join} from 'node:path';
import {glob as globAsync} from 'glob';

const server = new McpServer({
	name: 'openagent-editor-mcp',
	version: '1.0.0',
});

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_RESULTS = 500;

// ── Edit tool ──

server.registerTool(
	'edit',
	{
		title: 'Edit File',
		description: `Perform a surgical find-and-replace edit in a file. Replaces the first occurrence of old_string with new_string. Use replace_all=true to replace all occurrences. The old_string must match exactly (including whitespace and indentation).`,
		inputSchema: z.object({
			file_path: z.string().describe('Absolute or relative path to the file to edit'),
			old_string: z.string().describe('The exact text to find and replace'),
			new_string: z.string().describe('The replacement text'),
			replace_all: z.boolean().optional().default(false).describe('Replace all occurrences (default: false, replaces first only)'),
		}).strict(),
	},
	async (args) => {
		const {file_path, old_string, new_string, replace_all} = args as {
			file_path: string;
			old_string: string;
			new_string: string;
			replace_all: boolean;
		};

		const absPath = resolve(file_path);

		let content: string;
		try {
			content = await readFile(absPath, 'utf-8');
		} catch (err: any) {
			throw new Error(`Cannot read file: ${err.message}`);
		}

		if (!content.includes(old_string)) {
			throw new Error(`old_string not found in ${file_path}. Make sure it matches exactly (including whitespace).`);
		}

		// Count occurrences
		const occurrences = content.split(old_string).length - 1;

		let newContent: string;
		if (replace_all) {
			newContent = content.replaceAll(old_string, new_string);
		} else {
			// Replace first occurrence only
			const idx = content.indexOf(old_string);
			newContent = content.slice(0, idx) + new_string + content.slice(idx + old_string.length);
		}

		await writeFile(absPath, newContent, 'utf-8');

		const replaced = replace_all ? occurrences : 1;
		return {
			content: [{
				type: 'text',
				text: JSON.stringify({
					ok: true,
					file: file_path,
					replacements: replaced,
					total_occurrences: occurrences,
				}, null, 2),
			}],
		};
	},
);

// ── Grep tool ──

server.registerTool(
	'grep',
	{
		title: 'Search File Contents',
		description: `Search for a regex pattern across files. Returns matching lines with optional context. Searches recursively in the given directory.`,
		inputSchema: z.object({
			pattern: z.string().describe('Regular expression pattern to search for'),
			path: z.string().optional().default('.').describe('Directory or file to search in (default: current directory)'),
			file_pattern: z.string().optional().describe('Glob pattern to filter files (e.g. "*.ts", "**/*.py")'),
			context: z.number().optional().default(0).describe('Number of context lines before and after each match'),
			case_insensitive: z.boolean().optional().default(false).describe('Case-insensitive search'),
			max_results: z.number().optional().default(100).describe('Maximum number of matches to return'),
		}).strict(),
	},
	async (args) => {
		const {pattern, path: searchPath, file_pattern, context, case_insensitive, max_results} = args as {
			pattern: string;
			path: string;
			file_pattern?: string;
			context: number;
			case_insensitive: boolean;
			max_results: number;
		};

		const absPath = resolve(searchPath);
		const flags = case_insensitive ? 'gi' : 'g';
		let regex: RegExp;
		try {
			regex = new RegExp(pattern, flags);
		} catch (err: any) {
			throw new Error(`Invalid regex pattern: ${err.message}`);
		}

		// Find files to search
		let files: string[];
		const fileStat = await stat(absPath).catch(() => null);

		if (fileStat?.isFile()) {
			files = [absPath];
		} else {
			const globPattern = file_pattern || '**/*';
			files = await globAsync(globPattern, {
				cwd: absPath,
				absolute: true,
				nodir: true,
				ignore: ['**/node_modules/**', '**/.git/**', '**/dist/**'],
			});
		}

		const matches: Array<{file: string; line: number; content: string; context_before: string[]; context_after: string[]}> = [];
		let totalMatches = 0;
		const limit = Math.min(max_results, MAX_RESULTS);

		for (const file of files) {
			if (totalMatches >= limit) break;

			try {
				const fStat = await stat(file);
				if (fStat.size > MAX_FILE_SIZE) continue;

				const content = await readFile(file, 'utf-8');
				const lines = content.split('\n');
				const relPath = relative(process.cwd(), file);

				for (let i = 0; i < lines.length; i++) {
					if (totalMatches >= limit) break;
					regex.lastIndex = 0;

					if (regex.test(lines[i]!)) {
						const contextBefore: string[] = [];
						const contextAfter: string[] = [];

						for (let j = Math.max(0, i - context); j < i; j++) {
							contextBefore.push(lines[j]!);
						}
						for (let j = i + 1; j <= Math.min(lines.length - 1, i + context); j++) {
							contextAfter.push(lines[j]!);
						}

						matches.push({
							file: relPath,
							line: i + 1,
							content: lines[i]!,
							context_before: contextBefore,
							context_after: contextAfter,
						});
						totalMatches++;
					}
				}
			} catch {
				// Skip binary/unreadable files
			}
		}

		return {
			content: [{
				type: 'text',
				text: JSON.stringify({
					pattern,
					total_matches: matches.length,
					truncated: totalMatches >= limit,
					matches,
				}, null, 2),
			}],
		};
	},
);

// ── Glob tool ──

server.registerTool(
	'glob',
	{
		title: 'Find Files',
		description: `Find files matching a glob pattern. Returns file paths sorted by modification time (newest first). Supports patterns like "**/*.ts", "src/**/*.py", "*.json".`,
		inputSchema: z.object({
			pattern: z.string().describe('Glob pattern to match files (e.g. "**/*.ts", "src/**/*.py")'),
			path: z.string().optional().default('.').describe('Base directory to search from (default: current directory)'),
			max_results: z.number().optional().default(200).describe('Maximum number of files to return'),
		}).strict(),
	},
	async (args) => {
		const {pattern: globPattern, path: basePath, max_results} = args as {
			pattern: string;
			path: string;
			max_results: number;
		};

		const absPath = resolve(basePath);
		const limit = Math.min(max_results, MAX_RESULTS);

		const files = await globAsync(globPattern, {
			cwd: absPath,
			absolute: true,
			nodir: true,
			ignore: ['**/node_modules/**', '**/.git/**'],
		});

		// Get modification times and sort newest first
		const withStats = await Promise.all(
			files.slice(0, limit * 2).map(async (f) => {
				try {
					const s = await stat(f);
					return {path: relative(process.cwd(), f), mtime: s.mtimeMs, size: s.size};
				} catch {
					return null;
				}
			}),
		);

		const results = withStats
			.filter((f): f is NonNullable<typeof f> => f !== null)
			.sort((a, b) => b.mtime - a.mtime)
			.slice(0, limit);

		return {
			content: [{
				type: 'text',
				text: JSON.stringify({
					pattern: globPattern,
					base_path: basePath,
					total_files: results.length,
					truncated: files.length > limit,
					files: results,
				}, null, 2),
			}],
		};
	},
);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Editor MCP server running on stdio');
