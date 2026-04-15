#!/usr/bin/env node
/**
 * Cross-platform shell execution MCP server.
 *
 * Tools:
 *   - shell_exec: Run a shell command and return stdout/stderr
 *   - shell_which: Check if a command exists on the system
 *
 * Works on macOS (zsh), Linux (bash), and Windows (cmd/powershell).
 */

import {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {StdioServerTransport} from '@modelcontextprotocol/sdk/server/stdio.js';
import {z} from 'zod';
import {exec, execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {platform} from 'node:os';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

const DEFAULT_TIMEOUT = 120_000; // 2 minutes
// Upper bound the agent can opt into. A macOS Electron build (expo export +
// electron-builder with signing + notarization) can easily take 20+ minutes,
// and the old 10-min cap was hitting before the build finished even when the
// agent asked for more. Raise the ceiling to 30 min to match the model-side
// HARD_TIMEOUT; the agent still has to opt in by passing ``timeout`` explicitly.
const MAX_TIMEOUT = 1_800_000; // 30 minutes
const MAX_OUTPUT = 1_000_000; // 1MB output cap

function getDefaultShell(): string {
	switch (platform()) {
		case 'win32':
			return process.env.COMSPEC || 'cmd.exe';
		case 'darwin':
			return process.env.SHELL || '/bin/zsh';
		default:
			return process.env.SHELL || '/bin/bash';
	}
}

function getShellFlag(): string {
	const shell = getDefaultShell();
	// cmd.exe uses /c, everything else uses -c
	if (shell.includes('cmd')) return '/c';
	return '-c';
}

const server = new McpServer({
	name: 'openagent-shell-mcp',
	version: '1.0.0',
});

server.registerTool(
	'shell_exec',
	{
		title: 'Execute Shell Command',
		description: `Execute a shell command and return its output. Uses the system's default shell (bash/zsh on Unix, cmd on Windows). Commands run with the same permissions as the OpenAgent process.`,
		inputSchema: z.object({
			command: z.string().describe('The shell command to execute'),
			cwd: z.string().optional().describe('Working directory for the command (default: current directory)'),
			timeout: z.number().optional().describe(`Timeout in milliseconds (default: ${DEFAULT_TIMEOUT}, max: ${MAX_TIMEOUT})`),
			env: z.record(z.string()).optional().describe('Additional environment variables to set'),
		}).strict(),
	},
	async (args) => {
		const {command, cwd, timeout: userTimeout, env: userEnv} = args as {
			command: string;
			cwd?: string;
			timeout?: number;
			env?: Record<string, string>;
		};

		const timeout = Math.min(userTimeout || DEFAULT_TIMEOUT, MAX_TIMEOUT);
		const shell = getDefaultShell();

		try {
			const result = await execAsync(command, {
				shell,
				cwd: cwd || process.cwd(),
				timeout,
				maxBuffer: MAX_OUTPUT,
				env: userEnv ? {...process.env, ...userEnv} : process.env,
			});

			const stdout = result.stdout || '';
			const stderr = result.stderr || '';

			return {
				content: [{
					type: 'text',
					text: JSON.stringify({
						exit_code: 0,
						stdout: stdout.slice(0, MAX_OUTPUT),
						stderr: stderr.slice(0, MAX_OUTPUT),
					}, null, 2),
				}],
			};
		} catch (error: any) {
			return {
				content: [{
					type: 'text',
					text: JSON.stringify({
						exit_code: error.code ?? 1,
						stdout: (error.stdout || '').slice(0, MAX_OUTPUT),
						stderr: (error.stderr || error.message || '').slice(0, MAX_OUTPUT),
						killed: error.killed || false,
						signal: error.signal || null,
					}, null, 2),
				}],
			};
		}
	},
);

server.registerTool(
	'shell_which',
	{
		title: 'Check Command Availability',
		description: 'Check if a command/program is available on the system PATH.',
		inputSchema: z.object({
			command: z.string().describe('The command name to check (e.g. "git", "python3", "node")'),
		}).strict(),
	},
	async (args) => {
		const {command: cmd} = args as {command: string};
		const which = platform() === 'win32' ? 'where' : 'which';

		try {
			const result = await execFileAsync(which, [cmd], {timeout: 5000});
			return {
				content: [{
					type: 'text',
					text: JSON.stringify({
						available: true,
						path: result.stdout.trim().split('\n')[0],
					}, null, 2),
				}],
			};
		} catch {
			return {
				content: [{
					type: 'text',
					text: JSON.stringify({available: false}, null, 2),
				}],
			};
		}
	},
);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Shell MCP server running on stdio');
