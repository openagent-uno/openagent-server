#!/usr/bin/env node
/**
 * Messaging MCP: proactive send to Telegram, Discord, WhatsApp.
 *
 * Only registers tools for platforms with configured env vars:
 *   TELEGRAM_BOT_TOKEN → telegram_send_message, telegram_send_file
 *   DISCORD_BOT_TOKEN  → discord_send_message, discord_send_file
 *   WHATSAPP_API_ID + WHATSAPP_API_TOKEN → whatsapp_send_message, whatsapp_send_file
 *
 * File tools accept EITHER ``path`` (absolute local path on the agent's
 * filesystem — uploaded to the provider API via multipart) OR ``url``
 * (a public URL the provider fetches on its own). Exactly one of the
 * two must be provided. Supporting local paths is essential because
 * most agent-generated files live in ``/tmp/<session>/`` and never
 * get uploaded anywhere public — forcing them through a URL-only API
 * meant the agent had to first publish the file somewhere, which
 * breaks for private content and for headless setups.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { z } from 'zod';

// ── Shared helpers ────────────────────────────────────────────────────

/**
 * Read a local file into a Blob for FormData upload. Throws a readable
 * error message if the path is missing or unreadable — the agent sees
 * that text back as the tool's error and can recover (re-check the
 * path, fall back to URL, etc.).
 */
async function readLocalFileAsBlob(path: string): Promise<{ blob: Blob; filename: string }> {
	let buf: Buffer;
	try {
		buf = await readFile(path);
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message : String(e);
		throw new Error(`Cannot read local file ${path}: ${msg}`);
	}
	return { blob: new Blob([new Uint8Array(buf)]), filename: basename(path) };
}

/**
 * Require exactly one of ``path`` / ``url`` on file-send tool inputs.
 * Returns a discriminated object so downstream code doesn't have to
 * repeat the validation.
 */
function pickFileSource(args: { path?: string; url?: string }): { kind: 'path'; path: string } | { kind: 'url'; url: string } {
	const hasPath = !!args.path && args.path.trim() !== '';
	const hasUrl = !!args.url && args.url.trim() !== '';
	if (hasPath === hasUrl) {
		throw new Error('Provide exactly one of `path` (local file) or `url` (remote URL).');
	}
	if (hasPath) return { kind: 'path', path: args.path!.trim() };
	return { kind: 'url', url: args.url!.trim() };
}

const server = new McpServer({
	name: 'openagent-messaging-mcp',
	version: '1.0.0',
});

// ── Status (always registered) ──
//
// The MCP SDK only advertises the `tools` capability when at least one tool
// has been registered. If we conditionally register Telegram/Discord/WhatsApp
// tools and none of their env vars are set, the server starts but `list_tools`
// fails ("method not supported"), which clients log as a hard error.
//
// Registering one always-available status tool guarantees the capability is
// advertised, gives the LLM an affordance for "how do I enable messaging?",
// and keeps the dormant-MCP detection in the OpenAgent agent meaningful.

const TG_TOKEN_PRESENT = !!process.env.TELEGRAM_BOT_TOKEN;
const DC_TOKEN_PRESENT = !!process.env.DISCORD_BOT_TOKEN;
const WA_CREDS_PRESENT = !!process.env.GREEN_API_ID && !!process.env.GREEN_API_TOKEN;

server.registerTool(
	'status',
	{
		title: 'Messaging MCP status',
		description:
			'Return which messaging platforms (Telegram, Discord, WhatsApp) are currently ' +
			'enabled in this MCP server, and how to enable the disabled ones via the ' +
			'OpenAgent config.',
		inputSchema: z.object({}).strict(),
	},
	async () => {
		const status = {
			telegram: TG_TOKEN_PRESENT
				? { enabled: true, tools: ['telegram_send_message', 'telegram_send_file'] }
				: {
						enabled: false,
						how_to_enable:
							'Add `channels.telegram.token: <bot-token>` to openagent.yaml ' +
							'(and restart the agent).',
				  },
			discord: DC_TOKEN_PRESENT
				? { enabled: true, tools: ['discord_send_message', 'discord_send_file'] }
				: {
						enabled: false,
						how_to_enable:
							'Add `channels.discord.token: <bot-token>` to openagent.yaml ' +
							'(and restart the agent).',
				  },
			whatsapp: WA_CREDS_PRESENT
				? { enabled: true, tools: ['whatsapp_send_message', 'whatsapp_send_file'] }
				: {
						enabled: false,
						how_to_enable:
							'Add `channels.whatsapp.green_api_id` and ' +
							'`channels.whatsapp.green_api_token` to openagent.yaml ' +
							'(and restart the agent).',
				  },
		};
		return { content: [{ type: 'text', text: JSON.stringify(status, null, 2) }] };
	},
);

// ── Telegram ──

const TG_TOKEN = process.env.TELEGRAM_BOT_TOKEN;

if (TG_TOKEN) {
	const tgApiJson = async (method: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://api.telegram.org/bot${TG_TOKEN}/${method}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		});
		return res.json();
	};

	const tgApiMultipart = async (method: string, form: FormData) => {
		const res = await fetch(`https://api.telegram.org/bot${TG_TOKEN}/${method}`, {
			method: 'POST',
			body: form,
		});
		return res.json();
	};

	server.registerTool(
		'telegram_send_message',
		{
			title: 'Send Telegram Message',
			description: 'Send a text message to a Telegram chat or user.',
			inputSchema: z.object({
				chat_id: z.string().describe('Telegram chat ID or @username'),
				text: z.string().describe('Message text'),
				parse_mode: z.string().optional().describe('Parse mode: Markdown, HTML, or empty'),
			}).strict(),
		},
		async (args) => {
			const { chat_id, text, parse_mode } = args as { chat_id: string; text: string; parse_mode?: string };
			const result = await tgApiJson('sendMessage', { chat_id, text, parse_mode: parse_mode || undefined });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'telegram_send_file',
		{
			title: 'Send Telegram File',
			description:
				'Send a file/photo/voice/video to a Telegram chat. Provide EITHER ``path`` ' +
				'(absolute local path on the agent server, uploaded via multipart) OR ``url`` ' +
				'(a public URL Telegram will fetch itself). Exactly one of the two is required.',
			inputSchema: z.object({
				chat_id: z.string().describe('Telegram chat ID or @username'),
				path: z.string().optional().describe('Absolute local path to the file on the agent server'),
				url: z.string().optional().describe('Public URL of the file to send'),
				caption: z.string().optional().describe('Optional caption'),
				type: z.enum(['photo', 'document', 'voice', 'video']).optional().describe('File type (default: document)'),
			}).strict(),
		},
		async (args) => {
			const { chat_id, path, url, caption, type: fileType } = args as {
				chat_id: string; path?: string; url?: string; caption?: string; type?: string;
			};
			const method = fileType === 'photo' ? 'sendPhoto' : fileType === 'voice' ? 'sendVoice' : fileType === 'video' ? 'sendVideo' : 'sendDocument';
			const fileKey = fileType === 'photo' ? 'photo' : fileType === 'voice' ? 'voice' : fileType === 'video' ? 'video' : 'document';

			const source = pickFileSource({ path, url });
			let result: unknown;
			if (source.kind === 'path') {
				const { blob, filename } = await readLocalFileAsBlob(source.path);
				const form = new FormData();
				form.append('chat_id', chat_id);
				if (caption) form.append('caption', caption);
				form.append(fileKey, blob, filename);
				result = await tgApiMultipart(method, form);
			} else {
				result = await tgApiJson(method, { chat_id, [fileKey]: source.url, caption });
			}
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('Telegram messaging tools registered');
}

// ── Discord ──

const DC_TOKEN = process.env.DISCORD_BOT_TOKEN;

if (DC_TOKEN) {
	const dcApiJson = async (apiPath: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://discord.com/api/v10${apiPath}`, {
			method: 'POST',
			headers: { 'Authorization': `Bot ${DC_TOKEN}`, 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		});
		return res.json();
	};

	const dcApiMultipart = async (apiPath: string, form: FormData) => {
		const res = await fetch(`https://discord.com/api/v10${apiPath}`, {
			method: 'POST',
			headers: { 'Authorization': `Bot ${DC_TOKEN}` },
			body: form,
		});
		return res.json();
	};

	server.registerTool(
		'discord_send_message',
		{
			title: 'Send Discord Message',
			description: 'Send a text message to a Discord channel.',
			inputSchema: z.object({
				channel_id: z.string().describe('Discord channel ID'),
				text: z.string().describe('Message text'),
			}).strict(),
		},
		async (args) => {
			const { channel_id, text } = args as { channel_id: string; text: string };
			const result = await dcApiJson(`/channels/${channel_id}/messages`, { content: text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'discord_send_file',
		{
			title: 'Send Discord File',
			description:
				'Send a file/image/video as an attachment to a Discord channel. Provide EITHER ' +
				'``path`` (absolute local path on the agent server, uploaded via multipart) OR ' +
				'``url`` (Discord auto-embeds the URL as an attachment if the content-type is an ' +
				'image/video/audio — otherwise it renders inline as a link). Exactly one of the ' +
				'two is required. Optional ``text`` accompanies the attachment as the message body.',
			inputSchema: z.object({
				channel_id: z.string().describe('Discord channel ID'),
				path: z.string().optional().describe('Absolute local path to the file on the agent server'),
				url: z.string().optional().describe('Public URL of the file to send'),
				text: z.string().optional().describe('Optional message text to send alongside the attachment'),
			}).strict(),
		},
		async (args) => {
			const { channel_id, path, url, text } = args as {
				channel_id: string; path?: string; url?: string; text?: string;
			};
			const source = pickFileSource({ path, url });
			let result: unknown;
			if (source.kind === 'path') {
				const { blob, filename } = await readLocalFileAsBlob(source.path);
				const form = new FormData();
				form.append('payload_json', JSON.stringify({ content: text || '' }));
				form.append('files[0]', blob, filename);
				result = await dcApiMultipart(`/channels/${channel_id}/messages`, form);
			} else {
				// Plain URL mode — Discord auto-embeds media from URL content-type,
				// and falls back to a clickable link otherwise. That matches user
				// expectations for ``send me this image`` / ``here's a zip``.
				const body = text ? `${text}\n${source.url}` : source.url;
				result = await dcApiJson(`/channels/${channel_id}/messages`, { content: body });
			}
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('Discord messaging tools registered');
}

// ── WhatsApp (Green API) ──

const WA_ID = process.env.GREEN_API_ID;
const WA_TOKEN = process.env.GREEN_API_TOKEN;

if (WA_ID && WA_TOKEN) {
	const waApiJson = async (method: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://api.green-api.com/waInstance${WA_ID}/${method}/${WA_TOKEN}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		});
		return res.json();
	};

	const waApiMultipart = async (method: string, form: FormData) => {
		const res = await fetch(`https://api.green-api.com/waInstance${WA_ID}/${method}/${WA_TOKEN}`, {
			method: 'POST',
			body: form,
		});
		return res.json();
	};

	const normalizeChatId = (phone: string) => phone.includes('@') ? phone : `${phone}@c.us`;

	server.registerTool(
		'whatsapp_send_message',
		{
			title: 'Send WhatsApp Message',
			description: 'Send a text message via WhatsApp.',
			inputSchema: z.object({
				phone: z.string().describe('Phone number with country code (e.g. 393331234567) or chat ID'),
				text: z.string().describe('Message text'),
			}).strict(),
		},
		async (args) => {
			const { phone, text } = args as { phone: string; text: string };
			const result = await waApiJson('sendMessage', { chatId: normalizeChatId(phone), message: text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'whatsapp_send_file',
		{
			title: 'Send WhatsApp File',
			description:
				'Send a file/image/video/document via WhatsApp. Provide EITHER ``path`` ' +
				'(absolute local path on the agent server, uploaded via the Green API ' +
				'``sendFileByUpload`` multipart endpoint) OR ``url`` (public URL Green API ' +
				'fetches via ``sendFileByUrl``). Exactly one of the two is required.',
			inputSchema: z.object({
				phone: z.string().describe('Phone number with country code (e.g. 393331234567) or chat ID'),
				path: z.string().optional().describe('Absolute local path to the file on the agent server'),
				url: z.string().optional().describe('Public URL of the file to send'),
				caption: z.string().optional().describe('Optional caption'),
				filename: z.string().optional().describe('Filename to display to the recipient. Defaults to the path\'s basename or URL\'s last segment.'),
			}).strict(),
		},
		async (args) => {
			const { phone, path, url, caption, filename } = args as {
				phone: string; path?: string; url?: string; caption?: string; filename?: string;
			};
			const source = pickFileSource({ path, url });
			const chatId = normalizeChatId(phone);
			let result: unknown;
			if (source.kind === 'path') {
				const { blob, filename: inferredName } = await readLocalFileAsBlob(source.path);
				const outName = filename || inferredName;
				const form = new FormData();
				form.append('chatId', chatId);
				if (caption) form.append('caption', caption);
				form.append('fileName', outName);
				form.append('file', blob, outName);
				result = await waApiMultipart('sendFileByUpload', form);
			} else {
				const outName = filename || basename(new URL(source.url).pathname) || 'file';
				result = await waApiJson('sendFileByUrl', {
					chatId,
					urlFile: source.url,
					fileName: outName,
					caption,
				});
			}
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('WhatsApp messaging tools registered');
}

// Start
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Messaging MCP server running on stdio');
