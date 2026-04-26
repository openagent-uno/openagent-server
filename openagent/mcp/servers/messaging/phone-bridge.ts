/**
 * Bridge between Twilio Media Streams and the OpenAI Realtime API.
 *
 * This is the only place audio touches the wire. Both sides speak the
 * same codec (g711 mu-law, 8 kHz) so we just pass base64 payloads
 * through — no resampling, no PCM detour. Realtime's server VAD
 * detects when the caller starts speaking; on barge-in we cancel the
 * in-flight model response and send Twilio a ``clear`` to flush its
 * audio queue. Without that the bot talks over people and sounds awful.
 *
 * The model has three tools:
 *   - note(text)                       → appended to CallSession.notes
 *   - end_call(summary, outcome)       → finalises the session and
 *                                        triggers a Twilio hangup
 *   - (DTMF outbound is deferred to v2 — Twilio Media Streams don't
 *     support sending DTMF; the only options are ``Calls.update`` with
 *     fresh TwiML which tears down the active stream, or SIP REFER.
 *     Both ruin the live conversation. We accept the limitation.)
 *
 * Inbound DTMF (caller pressed digits — IVR menus) IS handled: Twilio
 * surfaces ``dtmf`` events on the WS and we inject them into Realtime
 * as a system message so the model can react.
 */

import { WebSocket } from 'ws';

import type { PhoneConfig } from './phone-config.js';
import {
	getCallSession,
	setStatus,
	setOutcome,
	setError,
	appendTranscript,
	appendNote,
	type CallSession,
} from './call-session.js';
import { hangupCall } from './twilio.js';

type TwilioStartEvent = {
	event: 'start';
	start: {
		streamSid: string;
		callSid: string;
		customParameters?: Record<string, string>;
	};
};

type TwilioMediaEvent = {
	event: 'media';
	streamSid: string;
	media: { payload: string; track: 'inbound' | 'outbound'; timestamp: string; chunk: string };
};

type TwilioMarkEvent = { event: 'mark'; streamSid: string; mark: { name: string } };
type TwilioStopEvent = { event: 'stop'; streamSid: string };
type TwilioDtmfEvent = { event: 'dtmf'; streamSid: string; dtmf: { digit: string; track: string } };

type TwilioInbound =
	| TwilioStartEvent
	| TwilioMediaEvent
	| TwilioMarkEvent
	| TwilioStopEvent
	| TwilioDtmfEvent
	| { event: 'connected' | string };

const REALTIME_URL_BASE = 'wss://api.openai.com/v1/realtime';

const DISCLOSURE_PREAMBLE =
	'You are an AI voice assistant placing a phone call on behalf of the user. ' +
	'CRITICAL: when the call connects, your FIRST utterance MUST disclose that ' +
	'you are an AI assistant calling on the user\'s behalf — never impersonate ' +
	'a human. If asked directly whether you are an AI or a recording, answer ' +
	'truthfully and immediately. Speak naturally, listen carefully, keep ' +
	'sentences short. If the call goes to voicemail, leave a brief message ' +
	'and call end_call(outcome="voicemail"). If the other party hangs up or the ' +
	'mission becomes impossible, call end_call with the appropriate outcome. ' +
	'Use note(text) to record key facts (names, times, prices, addresses) as ' +
	'you learn them; do not announce that you are taking notes. End the call ' +
	'as soon as the mission is resolved — do not chit-chat.';

function buildSystemPrompt(s: CallSession): string {
	const callerLine = s.caller_identity
		? `You are calling on behalf of: ${s.caller_identity}.`
		: 'You are calling on behalf of the user (identity not provided — say "the user" or "my user").';
	const langLine = `Speak in: ${s.language}.`;
	const missionLine = `Mission: ${s.mission}`;
	const successLine = s.success_criteria
		? `Success criteria: ${s.success_criteria}`
		: '';
	return [
		DISCLOSURE_PREAMBLE,
		callerLine,
		langLine,
		missionLine,
		successLine,
	].filter(Boolean).join('\n\n');
}

const REALTIME_TOOLS = [
	{
		type: 'function',
		name: 'note',
		description:
			'Record a key fact learned during the call (names, times, prices, ' +
			'addresses, decisions). Call this multiple times as the call progresses. ' +
			'Do not announce you are taking notes.',
		parameters: {
			type: 'object',
			properties: { text: { type: 'string' } },
			required: ['text'],
		},
	},
	{
		type: 'function',
		name: 'end_call',
		description:
			'End the call. Call this once the mission is resolved (success/failure), ' +
			'the other party hangs up, or you hit voicemail. After calling this, ' +
			'stop speaking — Twilio will hang up the line.',
		parameters: {
			type: 'object',
			properties: {
				summary: { type: 'string', description: 'One-paragraph summary of what happened on the call.' },
				outcome: {
					type: 'string',
					enum: ['success', 'partial', 'failure', 'no_answer', 'voicemail'],
				},
			},
			required: ['summary', 'outcome'],
		},
	},
];

/**
 * Run one call session: own the Twilio WS until the call ends, drive
 * an OpenAI Realtime session in parallel, finalise the CallSession
 * record on hangup. Throws are caught by the caller (phone-server).
 */
export async function runCallSession(cfg: PhoneConfig, twilioWs: WebSocket): Promise<void> {
	let callId: string | null = null;
	let streamSid: string | null = null;
	let session: CallSession | null = null;
	let openaiWs: WebSocket | null = null;
	let maxDurationTimer: NodeJS.Timeout | null = null;
	let hangupCheckTimer: NodeJS.Timeout | null = null;
	let endCallSent = false;

	// Buffer for the assistant's spoken text per response, so we can
	// commit one transcript line per turn rather than per chunk.
	let assistantTextBuf = '';
	// Same for the caller (Realtime emits transcription deltas).
	let callerTextBuf = '';

	const cleanup = (finalStatus: 'completed' | 'hung_up_by_agent' | 'timed_out' | 'failed') => {
		if (maxDurationTimer) clearTimeout(maxDurationTimer);
		if (hangupCheckTimer) clearInterval(hangupCheckTimer);
		try { openaiWs?.close(); } catch { /* ignore */ }
		try { twilioWs.close(); } catch { /* ignore */ }
		if (callId) setStatus(callId, finalStatus);
		if (callId && session?.twilio_sid && finalStatus !== 'completed') {
			// Tell Twilio to drop the line — completed status comes from Twilio
			// itself, but timed_out / hung_up_by_agent / failed need an explicit hangup.
			hangupCall(cfg, session.twilio_sid).catch((err) => {
				console.error('[phone-bridge] hangup REST failed', err);
			});
		}
	};

	twilioWs.on('error', (err) => {
		console.error('[phone-bridge] twilio ws error', err);
		if (callId) setError(callId, `twilio ws error: ${String(err)}`);
		cleanup('failed');
	});

	twilioWs.on('close', () => {
		// Far end (Twilio) closed — call is over. If we already marked it
		// hung_up_by_agent / timed_out, cleanup() preserves that via the
		// final-state guard in setStatus.
		if (callId && session && !endCallSent) cleanup('completed');
	});

	twilioWs.on('message', (raw) => {
		let evt: TwilioInbound;
		try {
			evt = JSON.parse(raw.toString()) as TwilioInbound;
		} catch {
			return;
		}

		if (evt.event === 'start') {
			const start = (evt as TwilioStartEvent).start;
			streamSid = start.streamSid;
			callId = start.customParameters?.call_id ?? null;
			if (!callId) {
				console.error('[phone-bridge] start event missing call_id custom parameter');
				try { twilioWs.close(); } catch { /* ignore */ }
				return;
			}
			session = getCallSession(callId) ?? null;
			if (!session) {
				console.error(`[phone-bridge] no CallSession for id=${callId}`);
				try { twilioWs.close(); } catch { /* ignore */ }
				return;
			}
			setStatus(callId, 'in_progress');
			openaiWs = openOpenAIRealtime(cfg, session, {
				onAssistantAudio: (b64) => {
					if (!streamSid) return;
					try {
						twilioWs.send(JSON.stringify({
							event: 'media',
							streamSid,
							media: { payload: b64 },
						}));
					} catch { /* ignore */ }
				},
				onAssistantTextDelta: (delta) => { assistantTextBuf += delta; },
				onAssistantTextDone: () => {
					if (callId && assistantTextBuf.trim()) {
						appendTranscript(callId, 'agent', assistantTextBuf.trim());
					}
					assistantTextBuf = '';
				},
				onCallerTextDelta: (delta) => { callerTextBuf += delta; },
				onCallerTextDone: () => {
					if (callId && callerTextBuf.trim()) {
						appendTranscript(callId, 'caller', callerTextBuf.trim());
					}
					callerTextBuf = '';
				},
				onBargeIn: () => {
					if (!streamSid) return;
					try {
						twilioWs.send(JSON.stringify({ event: 'clear', streamSid }));
					} catch { /* ignore */ }
				},
				onNote: (note) => {
					if (callId) appendNote(callId, note);
				},
				onEndCall: (summary, outcome) => {
					if (!callId) return;
					setOutcome(callId, summary, outcome);
					endCallSent = true;
					// Give Realtime a moment to finish the goodbye, then hang up.
					setTimeout(() => cleanup('completed'), 1500);
				},
				onError: (err) => {
					console.error('[phone-bridge] openai error', err);
					if (callId) setError(callId, `openai error: ${err}`);
				},
			});

			// Max-duration guardrail.
			maxDurationTimer = setTimeout(() => {
				console.error(`[phone-bridge] call ${callId} exceeded max duration`);
				cleanup('timed_out');
			}, session.max_duration_seconds * 1000);

			// Poll for agent-requested hangup (set by phone_call_hangup tool).
			hangupCheckTimer = setInterval(() => {
				if (callId && session?.hangupRequested) cleanup('hung_up_by_agent');
			}, 500);

			return;
		}

		if (evt.event === 'media') {
			const m = evt as TwilioMediaEvent;
			if (m.media.track === 'inbound' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
				try {
					openaiWs.send(JSON.stringify({
						type: 'input_audio_buffer.append',
						audio: m.media.payload,
					}));
				} catch { /* ignore */ }
			}
			return;
		}

		if (evt.event === 'dtmf') {
			const d = evt as TwilioDtmfEvent;
			if (callId) appendTranscript(callId, 'system', `Caller pressed DTMF: ${d.dtmf.digit}`);
			if (openaiWs && openaiWs.readyState === WebSocket.OPEN) {
				try {
					openaiWs.send(JSON.stringify({
						type: 'conversation.item.create',
						item: {
							type: 'message',
							role: 'system',
							content: [{ type: 'input_text', text: `[caller pressed DTMF digit: ${d.dtmf.digit}]` }],
						},
					}));
					openaiWs.send(JSON.stringify({ type: 'response.create' }));
				} catch { /* ignore */ }
			}
			return;
		}

		if (evt.event === 'stop') {
			cleanup('completed');
			return;
		}
	});
}

type RealtimeCallbacks = {
	onAssistantAudio: (b64: string) => void;
	onAssistantTextDelta: (delta: string) => void;
	onAssistantTextDone: () => void;
	onCallerTextDelta: (delta: string) => void;
	onCallerTextDone: () => void;
	onBargeIn: () => void;
	onNote: (text: string) => void;
	onEndCall: (summary: string, outcome: 'success' | 'partial' | 'failure' | 'no_answer' | 'voicemail') => void;
	onError: (err: string) => void;
};

function openOpenAIRealtime(cfg: PhoneConfig, session: CallSession, cb: RealtimeCallbacks): WebSocket {
	const url = `${REALTIME_URL_BASE}?model=${encodeURIComponent(cfg.realtimeModel)}`;
	const ws = new WebSocket(url, {
		headers: {
			Authorization: `Bearer ${cfg.openaiApiKey}`,
			'OpenAI-Beta': 'realtime=v1',
		},
	});

	ws.on('open', () => {
		const sessionUpdate = {
			type: 'session.update',
			session: {
				modalities: ['audio', 'text'],
				instructions: buildSystemPrompt(session),
				voice: cfg.realtimeVoice,
				input_audio_format: 'g711_ulaw',
				output_audio_format: 'g711_ulaw',
				input_audio_transcription: { model: 'whisper-1' },
				turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500 },
				tools: REALTIME_TOOLS,
				tool_choice: 'auto',
				temperature: 0.7,
			},
		};
		ws.send(JSON.stringify(sessionUpdate));
		// Kick the model so it speaks first (the disclosure preamble).
		ws.send(JSON.stringify({ type: 'response.create' }));
	});

	ws.on('error', (err) => cb.onError(String(err)));

	ws.on('message', (raw) => {
		let evt: { type: string;[k: string]: unknown };
		try {
			evt = JSON.parse(raw.toString()) as { type: string };
		} catch {
			return;
		}

		switch (evt.type) {
			case 'response.audio.delta': {
				const delta = evt.delta as string | undefined;
				if (delta) cb.onAssistantAudio(delta);
				return;
			}
			case 'response.audio_transcript.delta': {
				const delta = evt.delta as string | undefined;
				if (delta) cb.onAssistantTextDelta(delta);
				return;
			}
			case 'response.audio_transcript.done': {
				cb.onAssistantTextDone();
				return;
			}
			case 'conversation.item.input_audio_transcription.delta': {
				const delta = evt.delta as string | undefined;
				if (delta) cb.onCallerTextDelta(delta);
				return;
			}
			case 'conversation.item.input_audio_transcription.completed': {
				cb.onCallerTextDone();
				return;
			}
			case 'input_audio_buffer.speech_started': {
				// Caller started speaking — flush our queued audio.
				cb.onBargeIn();
				try { ws.send(JSON.stringify({ type: 'response.cancel' })); } catch { /* ignore */ }
				return;
			}
			case 'response.function_call_arguments.done': {
				const name = evt.name as string | undefined;
				const callId = evt.call_id as string | undefined;
				const argsRaw = evt.arguments as string | undefined;
				if (!name || !callId || !argsRaw) return;
				let args: Record<string, unknown> = {};
				try { args = JSON.parse(argsRaw) as Record<string, unknown>; } catch { /* ignore */ }
				if (name === 'note') {
					const text = (args.text as string) || '';
					cb.onNote(text);
					sendFunctionResult(ws, callId, { ok: true });
				} else if (name === 'end_call') {
					const summary = (args.summary as string) || '';
					const outcome = (args.outcome as string) || 'partial';
					const valid = ['success', 'partial', 'failure', 'no_answer', 'voicemail'].includes(outcome);
					const finalOutcome = (valid ? outcome : 'partial') as 'success' | 'partial' | 'failure' | 'no_answer' | 'voicemail';
					cb.onEndCall(summary, finalOutcome);
					sendFunctionResult(ws, callId, { ok: true });
				}
				return;
			}
			case 'error': {
				const err = JSON.stringify(evt.error ?? evt);
				cb.onError(err);
				return;
			}
		}
	});

	return ws;
}

function sendFunctionResult(ws: WebSocket, callId: string, output: unknown): void {
	try {
		ws.send(JSON.stringify({
			type: 'conversation.item.create',
			item: {
				type: 'function_call_output',
				call_id: callId,
				output: JSON.stringify(output),
			},
		}));
		ws.send(JSON.stringify({ type: 'response.create' }));
	} catch { /* ignore */ }
}
