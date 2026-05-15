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
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { z } from 'zod';

import { loadPhoneConfig, isDestinationAllowed } from './phone-config.js';
import { placeCall, sendSms, sendMms, hangupCall } from './twilio.js';
import {
	createCallSession,
	getCallSession,
	setTwilioSid,
	requestHangup,
	snapshot,
	isFinalStatus,
	waitForChange,
	dailyUsedSeconds,
	setStatus,
	setError,
} from './call-session.js';
import { ensurePhoneServer, twimlUrlFor, statusUrlFor } from './phone-server.js';

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
const PHONE_CONFIG = loadPhoneConfig();

server.registerTool(
	'status',
	{
		title: 'Messaging MCP status',
		description:
			'Return which messaging platforms (Telegram, Discord, WhatsApp, Phone/SMS) ' +
			'are currently enabled in this MCP server, and how to enable the disabled ' +
			'ones via the OpenAgent config.',
		inputSchema: z.object({}).strict(),
	},
	async () => {
		const phoneTools = ['sms_send', 'sms_send_file', 'phone_call_place', 'phone_call_status', 'phone_call_hangup'];
		const phoneBlock: Record<string, unknown> = PHONE_CONFIG
			? {
					enabled: true,
					tools: phoneTools,
					from_number: PHONE_CONFIG.twilioFromNumber,
					public_url: PHONE_CONFIG.publicUrl || '(not set — voice calls disabled, SMS still works)',
					allow_prefixes: PHONE_CONFIG.allowPrefixes,
					max_call_duration_seconds: PHONE_CONFIG.maxDurationSeconds,
					daily_seconds_used: dailyUsedSeconds(),
					daily_seconds_cap: PHONE_CONFIG.maxDailySeconds,
					realtime_model: PHONE_CONFIG.realtimeModel,
				}
			: {
					enabled: false,
					how_to_enable:
						'Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, ' +
						'and `OPENAI_API_KEY` to the messaging MCP env in openagent.yaml ' +
						'(and `OPENAGENT_PHONE_PUBLIC_URL` for voice calls). See docs/guide/phone-mcp.md.',
				};
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
			phone: phoneBlock,
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

// ── Twilio (Voice + SMS) ──
//
// Conditional on the four required env vars (TWILIO_ACCOUNT_SID,
// TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, OPENAI_API_KEY). The
// OPENAI_API_KEY is needed because the live phone-call brain runs
// on the OpenAI Realtime API (g711_ulaw end-to-end with Twilio Media
// Streams). SMS works without OPENAI_API_KEY in principle but we keep
// the gate uniform — if you've configured Twilio you almost certainly
// want voice too. The voice webhook server also requires
// OPENAGENT_PHONE_PUBLIC_URL (an ngrok/cloudflared tunnel); SMS does
// not need that, so phone_call_place fails loudly while sms_send works.

if (PHONE_CONFIG) {
	const cfg = PHONE_CONFIG;

	server.registerTool(
		'sms_send',
		{
			title: 'Send SMS',
			description:
				'Send a plain-text SMS via Twilio Programmable Messaging. The from-number ' +
				'is the configured TWILIO_FROM_NUMBER. Returns the Twilio message SID and ' +
				'submission status.',
			inputSchema: z.object({
				to: z.string().describe('Destination phone number in E.164 format, e.g. +393331234567'),
				text: z.string().describe('Message text. SMS segments at 160 chars (ASCII) or 70 chars (UCS-2); long messages auto-split.'),
			}).strict(),
		},
		async (args) => {
			const { to, text } = args as { to: string; text: string };
			if (cfg.allowPrefixes.length > 0 && !isDestinationAllowed(to, cfg.allowPrefixes)) {
				throw new Error(
					`Destination ${to} is not in OPENAGENT_PHONE_ALLOW_PREFIXES (${cfg.allowPrefixes.join(', ')}). ` +
					`Add the prefix to the messaging MCP env or remove the allowlist to permit it.`,
				);
			}
			const result = await sendSms(cfg, { to, text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'sms_send_file',
		{
			title: 'Send MMS (file via SMS)',
			description:
				'Send a file (image / audio / video / pdf / vcard) as MMS via Twilio. ' +
				'Twilio fetches the media from a public URL — local paths are not ' +
				'supported (Twilio constraint). For private files, host them temporarily ' +
				'(e.g. a signed S3 URL) and pass that URL.',
			inputSchema: z.object({
				to: z.string().describe('Destination phone number in E.164 format'),
				url: z.string().describe('Public URL of the file Twilio should fetch and forward'),
				text: z.string().optional().describe('Optional accompanying text body'),
			}).strict(),
		},
		async (args) => {
			const { to, url, text } = args as { to: string; url: string; text?: string };
			if (cfg.allowPrefixes.length > 0 && !isDestinationAllowed(to, cfg.allowPrefixes)) {
				throw new Error(
					`Destination ${to} is not in OPENAGENT_PHONE_ALLOW_PREFIXES (${cfg.allowPrefixes.join(', ')}).`,
				);
			}
			const result = await sendMms(cfg, { to, mediaUrl: url, text });
			return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
		},
	);

	server.registerTool(
		'phone_call_place',
		{
			title: 'Place AI-driven phone call',
			description:
				'Place an outbound phone call where an embedded AI conducts the live ' +
				'conversation end-to-end (Google-Duplex-style). Provide a ``mission`` ' +
				'(free text — what the call is for) and optionally success criteria, ' +
				'caller identity, language. Returns immediately with a ``call_id``; ' +
				'use ``phone_call_status`` to follow progress and retrieve the final ' +
				'transcript / notes / outcome. The AI is required to disclose itself ' +
				'as an AI assistant on connect.\n\n' +
				'Per-call max duration is clamped down (never up) to the server-wide ' +
				'OPENAGENT_PHONE_MAX_DURATION_SECONDS cap.',
			inputSchema: z.object({
				to: z.string().describe('Destination phone number in E.164 format, e.g. +393331234567'),
				mission: z.string().describe('What the call is for, in free text. Example: "Call this restaurant and book a table for 4 at 7pm tomorrow under the name Alessandro."'),
				caller_identity: z.string().optional().describe('Name to disclose on the user\'s behalf (e.g. "Alessandro Gerelli"). If omitted, the AI says "the user".'),
				language: z.string().optional().describe('Language for the call, e.g. "en-US", "it-IT". Default en-US.'),
				success_criteria: z.string().optional().describe('Optional: what counts as success. The AI uses this to decide outcome=success vs partial vs failure.'),
				max_duration_seconds: z.number().int().positive().optional().describe('Per-call cap in seconds. Clamped to the server max.'),
			}).strict(),
		},
		async (args) => {
			const a = args as {
				to: string;
				mission: string;
				caller_identity?: string;
				language?: string;
				success_criteria?: string;
				max_duration_seconds?: number;
			};
			if (!cfg.publicUrl) {
				throw new Error(
					'OPENAGENT_PHONE_PUBLIC_URL is not set. Voice calls require a public ' +
					'tunnel (ngrok / cloudflared) so Twilio can reach the local webhook ' +
					'server. SMS still works without this.',
				);
			}
			if (!isDestinationAllowed(a.to, cfg.allowPrefixes)) {
				throw new Error(
					`Destination ${a.to} is not in OPENAGENT_PHONE_ALLOW_PREFIXES ` +
					`(${cfg.allowPrefixes.join(', ') || '<empty — allowlist denies all by default>'}). ` +
					`Add the country/area-code prefix to the messaging MCP env to permit it.`,
				);
			}
			if (dailyUsedSeconds() >= cfg.maxDailySeconds) {
				throw new Error(
					`Per-day call-duration cap reached (${cfg.maxDailySeconds}s used today). ` +
					`Raise OPENAGENT_PHONE_MAX_DAILY_SECONDS or wait until tomorrow.`,
				);
			}

			const requestedMax = a.max_duration_seconds ?? cfg.maxDurationSeconds;
			const max = Math.min(requestedMax, cfg.maxDurationSeconds);

			const callId = randomUUID();
			const session = createCallSession({
				id: callId,
				to: a.to,
				from: cfg.twilioFromNumber,
				mission: a.mission,
				caller_identity: a.caller_identity ?? null,
				language: a.language ?? 'en-US',
				success_criteria: a.success_criteria ?? null,
				max_duration_seconds: max,
			});

			await ensurePhoneServer(cfg);

			try {
				const { sid } = await placeCall(cfg, {
					to: a.to,
					twimlUrl: twimlUrlFor(cfg, callId),
					statusCallbackUrl: statusUrlFor(cfg, callId),
				});
				setTwilioSid(callId, sid);
			} catch (err) {
				const msg = err instanceof Error ? err.message : String(err);
				setError(callId, `twilio placeCall failed: ${msg}`);
				setStatus(callId, 'failed');
				throw new Error(`Failed to place call via Twilio: ${msg}`);
			}

			return {
				content: [{
					type: 'text',
					text: JSON.stringify({
						call_id: callId,
						status: session.status,
						twilio_sid: session.twilio_sid,
						hint: 'Poll phone_call_status(call_id, wait=true) for live updates and the final transcript.',
					}, null, 2),
				}],
			};
		},
	);

	server.registerTool(
		'phone_call_status',
		{
			title: 'Get phone-call status',
			description:
				'Return the current state, partial transcript, notes, summary, and ' +
				'outcome of a phone call placed via ``phone_call_place``. With ' +
				'``wait=true``, blocks for up to ~30 seconds waiting for the next ' +
				'state change — this is the friendly way to follow a call: the ' +
				'first wait returns when the call is answered, the next when notes ' +
				'are added or the call ends. ``outcome`` and ``summary`` are only ' +
				'populated once the call is in a final state.',
			inputSchema: z.object({
				call_id: z.string().describe('The call_id returned by phone_call_place'),
				wait: z.boolean().optional().describe('If true, long-poll for up to ~30s waiting for a state change. Default false (returns immediately with current state).'),
			}).strict(),
		},
		async (args) => {
			const { call_id, wait } = args as { call_id: string; wait?: boolean };
			const session = getCallSession(call_id);
			if (!session) {
				throw new Error(`Unknown call_id: ${call_id}`);
			}
			if (wait && !isFinalStatus(session.status)) {
				await waitForChange(call_id, 30_000);
			}
			const fresh = getCallSession(call_id)!;
			return { content: [{ type: 'text', text: JSON.stringify(snapshot(fresh), null, 2) }] };
		},
	);

	server.registerTool(
		'phone_call_hangup',
		{
			title: 'Force-hang-up phone call',
			description:
				'Immediately end an in-flight phone call. Useful when the agent ' +
				'decides the mission is impossible or the call has gone wrong. ' +
				'Returns the final session snapshot. No-op if the call is already ' +
				'in a final state.',
			inputSchema: z.object({
				call_id: z.string().describe('The call_id returned by phone_call_place'),
			}).strict(),
		},
		async (args) => {
			const { call_id } = args as { call_id: string };
			const session = getCallSession(call_id);
			if (!session) {
				throw new Error(`Unknown call_id: ${call_id}`);
			}
			if (isFinalStatus(session.status)) {
				return { content: [{ type: 'text', text: JSON.stringify(snapshot(session), null, 2) }] };
			}
			requestHangup(call_id);
			// Best-effort REST hangup as belt-and-suspenders — the bridge poll
			// will hangup too, but the REST call ensures Twilio drops the line
			// even if the bridge has lost its WS.
			if (session.twilio_sid) {
				try { await hangupCall(cfg, session.twilio_sid); } catch (err) {
					console.error('[phone] hangup REST failed', err);
				}
			}
			// Wait briefly for the bridge to finalise the session.
			await waitForChange(call_id, 3_000);
			const fresh = getCallSession(call_id)!;
			return { content: [{ type: 'text', text: JSON.stringify(snapshot(fresh), null, 2) }] };
		},
	);

	console.error('Twilio (SMS + Voice) messaging tools registered');
}

// Start
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Messaging MCP server running on stdio');
