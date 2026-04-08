#!/usr/bin/env node
/**
 * Messaging MCP: proactive send to Telegram, Discord, WhatsApp.
 *
 * Only registers tools for platforms with configured env vars:
 *   TELEGRAM_BOT_TOKEN → telegram_send_message, telegram_send_file
 *   DISCORD_BOT_TOKEN  → discord_send_message, discord_send_file
 *   WHATSAPP_API_ID + WHATSAPP_API_TOKEN → whatsapp_send_message, whatsapp_send_file
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const server = new McpServer({
	name: 'openagent-messaging-mcp',
	version: '1.0.0',
});

// ── Telegram ──

const TG_TOKEN = process.env.TELEGRAM_BOT_TOKEN;

if (TG_TOKEN) {
	const tgApi = async (method: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://api.telegram.org/bot${TG_TOKEN}/${method}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
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
			const result = await tgApi('sendMessage', { chat_id, text, parse_mode: parse_mode || undefined });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'telegram_send_file',
		{
			title: 'Send Telegram File',
			description: 'Send a file/photo/voice to a Telegram chat. Provide a URL or file_id.',
			inputSchema: z.object({
				chat_id: z.string().describe('Telegram chat ID or @username'),
				url: z.string().describe('URL of the file to send'),
				caption: z.string().optional().describe('Optional caption'),
				type: z.enum(['photo', 'document', 'voice', 'video']).optional().describe('File type (default: document)'),
			}).strict(),
		},
		async (args) => {
			const { chat_id, url, caption, type: fileType } = args as { chat_id: string; url: string; caption?: string; type?: string };
			const method = fileType === 'photo' ? 'sendPhoto' : fileType === 'voice' ? 'sendVoice' : fileType === 'video' ? 'sendVideo' : 'sendDocument';
			const fileKey = fileType === 'photo' ? 'photo' : fileType === 'voice' ? 'voice' : fileType === 'video' ? 'video' : 'document';
			const result = await tgApi(method, { chat_id, [fileKey]: url, caption });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('Telegram messaging tools registered');
}

// ── Discord ──

const DC_TOKEN = process.env.DISCORD_BOT_TOKEN;

if (DC_TOKEN) {
	const dcApi = async (path: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://discord.com/api/v10${path}`, {
			method: 'POST',
			headers: { 'Authorization': `Bot ${DC_TOKEN}`, 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
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
			const result = await dcApi(`/channels/${channel_id}/messages`, { content: text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('Discord messaging tools registered');
}

// ── WhatsApp (Green API) ──

const WA_ID = process.env.GREEN_API_ID;
const WA_TOKEN = process.env.GREEN_API_TOKEN;

if (WA_ID && WA_TOKEN) {
	const waApi = async (method: string, body: Record<string, unknown>) => {
		const res = await fetch(`https://api.green-api.com/waInstance${WA_ID}/${method}/${WA_TOKEN}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
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
			const result = await waApi('sendMessage', { chatId: normalizeChatId(phone), message: text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	console.error('WhatsApp messaging tools registered');
}

// Start
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Messaging MCP server running on stdio');
