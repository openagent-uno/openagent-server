/**
 * Phone (Twilio + OpenAI Realtime) configuration loader.
 *
 * Reads env vars at module-import time. The MCP only registers Twilio
 * tools when ``loadPhoneConfig()`` returns a non-null config — same
 * gating pattern as Telegram/Discord/WhatsApp blocks in index.ts.
 *
 * All knobs default to safe values. The destination allowlist defaults
 * to **empty = deny all** so a misconfigured agent can't dial premium-
 * rate numbers; users opt in to country prefixes they actually call.
 */

export type PhoneConfig = {
	twilioAccountSid: string;
	twilioAuthToken: string;
	twilioFromNumber: string;
	openaiApiKey: string;
	publicUrl: string;
	allowPrefixes: string[];
	maxDurationSeconds: number;
	maxDailySeconds: number;
	realtimeModel: string;
	realtimeVoice: string;
};

const DEFAULT_MAX_DURATION = 600;
const DEFAULT_MAX_DAILY = 3600;
const DEFAULT_MODEL = 'gpt-realtime';
const DEFAULT_VOICE = 'alloy';

function parsePrefixes(raw: string | undefined): string[] {
	if (!raw) return [];
	return raw.split(',').map((p) => p.trim()).filter(Boolean);
}

function parseInt10(raw: string | undefined, fallback: number): number {
	if (!raw) return fallback;
	const n = parseInt(raw, 10);
	return Number.isFinite(n) && n > 0 ? n : fallback;
}

/**
 * Returns the live config object if all required env vars are set,
 * otherwise null. ``OPENAGENT_PHONE_PUBLIC_URL`` is *only* required
 * for voice calls; SMS works without it. We split that check at the
 * tool-call site so SMS can register/work without a tunnel.
 */
export function loadPhoneConfig(): PhoneConfig | null {
	const sid = process.env.TWILIO_ACCOUNT_SID;
	const token = process.env.TWILIO_AUTH_TOKEN;
	const from = process.env.TWILIO_FROM_NUMBER;
	const oaiKey = process.env.OPENAI_API_KEY;
	if (!sid || !token || !from || !oaiKey) return null;
	return {
		twilioAccountSid: sid,
		twilioAuthToken: token,
		twilioFromNumber: from,
		openaiApiKey: oaiKey,
		publicUrl: (process.env.OPENAGENT_PHONE_PUBLIC_URL || '').replace(/\/$/, ''),
		allowPrefixes: parsePrefixes(process.env.OPENAGENT_PHONE_ALLOW_PREFIXES),
		maxDurationSeconds: parseInt10(process.env.OPENAGENT_PHONE_MAX_DURATION_SECONDS, DEFAULT_MAX_DURATION),
		maxDailySeconds: parseInt10(process.env.OPENAGENT_PHONE_MAX_DAILY_SECONDS, DEFAULT_MAX_DAILY),
		realtimeModel: process.env.OPENAGENT_PHONE_MODEL || DEFAULT_MODEL,
		realtimeVoice: process.env.OPENAGENT_PHONE_VOICE || DEFAULT_VOICE,
	};
}

/** True if ``to`` (E.164) starts with any allowed prefix. Empty list = deny. */
export function isDestinationAllowed(to: string, allowPrefixes: string[]): boolean {
	if (allowPrefixes.length === 0) return false;
	return allowPrefixes.some((p) => to.startsWith(p));
}
