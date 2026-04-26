/**
 * Singleton HTTP + WebSocket server that fronts the phone MCP for
 * Twilio's webhook traffic.
 *
 * Routes:
 *   POST /twiml/:call_id    → returns the TwiML that points Twilio's
 *                             Media Stream at our /media-stream WS
 *   POST /status/:call_id   → Twilio status callback (ringing / answered / …)
 *   WS   /media-stream      → bidirectional 8 kHz mu-law audio + control events
 *
 * The server lazy-starts on the first ``phone_call_place``: many users
 * never enable phone, and we don't want to bind a port at startup.
 *
 * Both POST endpoints are signature-verified against ``X-Twilio-Signature``.
 * The WS endpoint is implicitly trusted: Twilio negotiates it with a
 * one-shot URL embedded in the just-served TwiML, and the URL itself is
 * parameterised with the call_id we minted.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { URL } from 'node:url';
import { WebSocketServer, type WebSocket } from 'ws';

import type { PhoneConfig } from './phone-config.js';
import { validateSignature } from './twilio.js';
import {
	getCallSession,
	setStatus,
	type CallStatus,
} from './call-session.js';
import { runCallSession } from './phone-bridge.js';

let httpServer: Server | null = null;
let wsServer: WebSocketServer | null = null;
let boundPort: number | null = null;
let activeConfig: PhoneConfig | null = null;

const TWIML_PATH = /^\/twiml\/([^/]+)\/?$/;
const STATUS_PATH = /^\/status\/([^/]+)\/?$/;

async function readForm(req: IncomingMessage): Promise<Record<string, string>> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) chunks.push(chunk as Buffer);
	const body = Buffer.concat(chunks).toString('utf8');
	const params = new URLSearchParams(body);
	const out: Record<string, string> = {};
	for (const [k, v] of params) out[k] = v;
	return out;
}

function sendText(res: ServerResponse, status: number, contentType: string, body: string): void {
	res.statusCode = status;
	res.setHeader('Content-Type', contentType);
	res.end(body);
}

/**
 * Map Twilio's ``CallStatus`` form param to our internal CallStatus.
 * Twilio uses dash-cased values; ours use snake_case.
 */
function twilioStatusToInternal(raw: string): CallStatus | null {
	switch (raw) {
		case 'initiated': return 'initiated';
		case 'ringing': return 'ringing';
		case 'answered': return 'in_progress';
		case 'in-progress': return 'in_progress';
		case 'completed': return 'completed';
		case 'busy': return 'busy';
		case 'failed': return 'failed';
		case 'no-answer': return 'no_answer';
		default: return null;
	}
}

async function handleHttp(cfg: PhoneConfig, req: IncomingMessage, res: ServerResponse): Promise<void> {
	const path = req.url || '/';
	const method = req.method || 'GET';

	const twimlMatch = path.match(TWIML_PATH);
	const statusMatch = path.match(STATUS_PATH);

	if (method !== 'POST' || (!twimlMatch && !statusMatch)) {
		sendText(res, 404, 'text/plain', 'not found');
		return;
	}

	const form = await readForm(req);
	const signature = req.headers['x-twilio-signature'] as string | undefined;
	const reqPath = path.split('?')[0];

	if (!validateSignature(cfg, signature, reqPath, form)) {
		sendText(res, 403, 'text/plain', 'signature mismatch');
		return;
	}

	if (twimlMatch) {
		const callId = decodeURIComponent(twimlMatch[1]);
		const session = getCallSession(callId);
		if (!session) {
			sendText(res, 404, 'text/plain', 'unknown call_id');
			return;
		}
		// Twilio's Media Stream URL must be wss:// (Twilio refuses ws://
		// for outbound). The public URL is HTTP(S); swap the scheme.
		const wsScheme = cfg.publicUrl.startsWith('https') ? 'wss' : 'ws';
		const wsHost = cfg.publicUrl.replace(/^https?:\/\//, '');
		const streamUrl = `${wsScheme}://${wsHost}/media-stream`;
		const twiml =
			`<?xml version="1.0" encoding="UTF-8"?>` +
			`<Response>` +
			`<Connect>` +
			`<Stream url="${streamUrl}">` +
			`<Parameter name="call_id" value="${escapeXml(callId)}"/>` +
			`</Stream>` +
			`</Connect>` +
			`</Response>`;
		sendText(res, 200, 'text/xml', twiml);
		return;
	}

	if (statusMatch) {
		const callId = decodeURIComponent(statusMatch[1]);
		const raw = form['CallStatus'] || '';
		const mapped = twilioStatusToInternal(raw);
		if (mapped) setStatus(callId, mapped);
		sendText(res, 200, 'text/plain', 'ok');
		return;
	}
}

function escapeXml(s: string): string {
	return s.replace(/[<>&'"]/g, (c) =>
		c === '<' ? '&lt;'
			: c === '>' ? '&gt;'
				: c === '&' ? '&amp;'
					: c === '\'' ? '&apos;'
						: '&quot;');
}

function attachWsServer(cfg: PhoneConfig, http: Server): WebSocketServer {
	const wss = new WebSocketServer({ noServer: true });
	http.on('upgrade', (req, socket, head) => {
		const path = (req.url || '').split('?')[0];
		if (path !== '/media-stream') {
			socket.destroy();
			return;
		}
		wss.handleUpgrade(req, socket, head, (ws) => {
			handleMediaStream(cfg, ws as WebSocket);
		});
	});
	return wss;
}

function handleMediaStream(cfg: PhoneConfig, twilioWs: WebSocket): void {
	// We don't know the call_id until Twilio sends the first ``start``
	// frame on this WS. Defer everything to the bridge; it will pull
	// the call_id out of the start event's customParameters.
	runCallSession(cfg, twilioWs).catch((err) => {
		console.error('[phone-bridge] runCallSession failed', err);
		try { twilioWs.close(); } catch { /* ignore */ }
	});
}

/**
 * Start the singleton server if it isn't already up. Returns the bound
 * port — useful for smoke tests, otherwise the public URL is what
 * matters. If ``cfg.publicUrl`` is empty, throws — voice calls require
 * a tunnel so Twilio can reach us.
 */
export async function ensurePhoneServer(cfg: PhoneConfig): Promise<number> {
	if (!cfg.publicUrl) {
		throw new Error(
			'OPENAGENT_PHONE_PUBLIC_URL is not set. Voice calls require a public ' +
			'tunnel (e.g. `ngrok http <port>`) so Twilio can reach the local webhook ' +
			'server. SMS does not need this.',
		);
	}
	if (httpServer && boundPort !== null && activeConfig?.twilioAccountSid === cfg.twilioAccountSid) {
		return boundPort;
	}
	if (httpServer) {
		// Config changed — tear down and rebind.
		await new Promise<void>((resolve) => httpServer!.close(() => resolve()));
		httpServer = null;
		wsServer = null;
		boundPort = null;
	}
	const server = createServer((req, res) => {
		handleHttp(cfg, req, res).catch((err) => {
			console.error('[phone-server] handler error', err);
			if (!res.headersSent) sendText(res, 500, 'text/plain', 'internal error');
			else res.end();
		});
	});
	const wss = attachWsServer(cfg, server);
	const port = await new Promise<number>((resolve, reject) => {
		server.once('error', reject);
		server.listen(0, '127.0.0.1', () => {
			const addr = server.address();
			if (addr && typeof addr === 'object') resolve(addr.port);
			else reject(new Error('failed to bind phone webhook server'));
		});
	});
	httpServer = server;
	wsServer = wss;
	boundPort = port;
	activeConfig = cfg;
	console.error(`[phone-server] listening on 127.0.0.1:${port}, public=${cfg.publicUrl}`);
	return port;
}

/** Build the externally-reachable URLs Twilio should hit for this call. */
export function twimlUrlFor(cfg: PhoneConfig, callId: string): string {
	return `${cfg.publicUrl}/twiml/${encodeURIComponent(callId)}`;
}

export function statusUrlFor(cfg: PhoneConfig, callId: string): string {
	return `${cfg.publicUrl}/status/${encodeURIComponent(callId)}`;
}
