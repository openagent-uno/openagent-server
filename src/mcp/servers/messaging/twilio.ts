/**
 * Thin wrappers around the official Twilio Node SDK. Keeps SDK quirks
 * isolated from the rest of the messaging MCP and lets us swap out the
 * provider later (Vonage / Telnyx) without rippling through the bridge.
 */

import twilio from 'twilio';
import type { PhoneConfig } from './phone-config.js';

let cachedClient: ReturnType<typeof twilio> | null = null;
let cachedClientSid: string | null = null;

function getClient(cfg: PhoneConfig): ReturnType<typeof twilio> {
	// Cache by SID so a config swap (rare, but possible across hot-reloads)
	// invalidates cleanly.
	if (cachedClient && cachedClientSid === cfg.twilioAccountSid) return cachedClient;
	cachedClient = twilio(cfg.twilioAccountSid, cfg.twilioAuthToken);
	cachedClientSid = cfg.twilioAccountSid;
	return cachedClient;
}

export async function placeCall(
	cfg: PhoneConfig,
	opts: { to: string; twimlUrl: string; statusCallbackUrl: string },
): Promise<{ sid: string }> {
	const client = getClient(cfg);
	const call = await client.calls.create({
		to: opts.to,
		from: cfg.twilioFromNumber,
		url: opts.twimlUrl,
		statusCallback: opts.statusCallbackUrl,
		statusCallbackMethod: 'POST',
		statusCallbackEvent: ['initiated', 'ringing', 'answered', 'completed'],
	});
	return { sid: call.sid };
}

export async function hangupCall(cfg: PhoneConfig, callSid: string): Promise<void> {
	const client = getClient(cfg);
	await client.calls(callSid).update({ status: 'completed' });
}

export async function sendSms(
	cfg: PhoneConfig,
	opts: { to: string; text: string },
): Promise<{ sid: string; status: string }> {
	const client = getClient(cfg);
	const msg = await client.messages.create({
		to: opts.to,
		from: cfg.twilioFromNumber,
		body: opts.text,
	});
	return { sid: msg.sid, status: msg.status };
}

export async function sendMms(
	cfg: PhoneConfig,
	opts: { to: string; mediaUrl: string; text?: string },
): Promise<{ sid: string; status: string }> {
	const client = getClient(cfg);
	const msg = await client.messages.create({
		to: opts.to,
		from: cfg.twilioFromNumber,
		body: opts.text || '',
		mediaUrl: [opts.mediaUrl],
	});
	return { sid: msg.sid, status: msg.status };
}

/**
 * Verify ``X-Twilio-Signature`` on an incoming webhook. The canonical
 * URL Twilio signed is the public URL we configured (``OPENAGENT_PHONE_PUBLIC_URL``)
 * + the request path — *not* the local 127.0.0.1 URL the http server
 * actually bound on. This is the single most common signature-failure
 * source; do not "fix" it by reading the request URL from the socket.
 */
export function validateSignature(
	cfg: PhoneConfig,
	signature: string | undefined,
	requestPath: string,
	formParams: Record<string, string>,
): boolean {
	if (!signature) return false;
	const url = `${cfg.publicUrl}${requestPath}`;
	return twilio.validateRequest(cfg.twilioAuthToken, signature, url, formParams);
}
