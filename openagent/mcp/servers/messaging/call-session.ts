/**
 * In-memory registry of phone-call sessions. Drives the async-by-default
 * tool model: ``phone_call_place`` creates a session, returns the id,
 * and ``phone_call_status(call_id, wait=true)`` long-polls on the per-
 * session change-notifier.
 *
 * Persistence is intentionally absent in v1 — the agent receives the
 * full result via ``phone_call_status`` and can mirror it into vault
 * memories itself. v2 will add SQLite persistence via the gateway HTTP
 * API so transcripts survive process restart.
 */

import { EventEmitter } from 'node:events';

export type CallStatus =
	| 'initiated' // REST call placed, awaiting Twilio status callback
	| 'ringing'
	| 'in_progress'
	| 'completed'
	| 'busy'
	| 'failed'
	| 'no_answer'
	| 'voicemail'
	| 'hung_up_by_agent'
	| 'timed_out';

export type CallOutcome = 'success' | 'partial' | 'failure' | 'no_answer' | 'voicemail';

export type TranscriptEntry = {
	t: number; // ms epoch
	role: 'agent' | 'caller' | 'system';
	text: string;
};

export type CallSession = {
	id: string;
	to: string;
	from: string;
	mission: string;
	caller_identity: string | null;
	language: string;
	success_criteria: string | null;
	max_duration_seconds: number;
	twilio_sid: string | null;
	status: CallStatus;
	transcript: TranscriptEntry[];
	notes: string[];
	summary: string | null;
	outcome: CallOutcome | null;
	error: string | null;
	started_at: number; // ms epoch
	answered_at: number | null;
	ended_at: number | null;
	// Bridge handle so phone_call_hangup can request closure even
	// while the bridge owns the websockets.
	hangupRequested: boolean;
};

const sessions = new Map<string, CallSession>();
const emitters = new Map<string, EventEmitter>();

// Per-day duration tally — resets at process restart in v1. Twilio's own
// account-level spend caps remain the durable safety net.
let dayKey = currentDayKey();
let dayUsedSeconds = 0;

function currentDayKey(): string {
	const d = new Date();
	return `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`;
}

function rolloverDayIfNeeded(): void {
	const k = currentDayKey();
	if (k !== dayKey) {
		dayKey = k;
		dayUsedSeconds = 0;
	}
}

export function dailyUsedSeconds(): number {
	rolloverDayIfNeeded();
	return dayUsedSeconds;
}

export function recordCallDuration(seconds: number): void {
	rolloverDayIfNeeded();
	dayUsedSeconds += Math.max(0, Math.round(seconds));
}

export function createCallSession(args: {
	id: string;
	to: string;
	from: string;
	mission: string;
	caller_identity: string | null;
	language: string;
	success_criteria: string | null;
	max_duration_seconds: number;
}): CallSession {
	const session: CallSession = {
		id: args.id,
		to: args.to,
		from: args.from,
		mission: args.mission,
		caller_identity: args.caller_identity,
		language: args.language,
		success_criteria: args.success_criteria,
		max_duration_seconds: args.max_duration_seconds,
		twilio_sid: null,
		status: 'initiated',
		transcript: [],
		notes: [],
		summary: null,
		outcome: null,
		error: null,
		started_at: Date.now(),
		answered_at: null,
		ended_at: null,
		hangupRequested: false,
	};
	sessions.set(args.id, session);
	emitters.set(args.id, new EventEmitter());
	return session;
}

export function getCallSession(id: string): CallSession | undefined {
	return sessions.get(id);
}

function emitChange(id: string): void {
	const emitter = emitters.get(id);
	if (emitter) emitter.emit('change');
}

export function setTwilioSid(id: string, sid: string): void {
	const s = sessions.get(id);
	if (!s) return;
	s.twilio_sid = sid;
	emitChange(id);
}

export function setStatus(id: string, status: CallStatus): void {
	const s = sessions.get(id);
	if (!s) return;
	const finalStates: CallStatus[] = [
		'completed', 'busy', 'failed', 'no_answer', 'voicemail',
		'hung_up_by_agent', 'timed_out',
	];
	// Don't regress out of a final state — Twilio sends a "completed" status
	// callback after we've already set "hung_up_by_agent", and we want the
	// agent's reason for the hangup to win.
	if (finalStates.includes(s.status) && !finalStates.includes(status)) return;
	s.status = status;
	if (status === 'in_progress' && s.answered_at === null) s.answered_at = Date.now();
	if (finalStates.includes(status) && s.ended_at === null) {
		s.ended_at = Date.now();
		const durSec = (s.ended_at - (s.answered_at ?? s.started_at)) / 1000;
		recordCallDuration(durSec);
	}
	emitChange(id);
}

export function appendTranscript(id: string, role: TranscriptEntry['role'], text: string): void {
	const s = sessions.get(id);
	if (!s || !text.trim()) return;
	s.transcript.push({ t: Date.now(), role, text });
	emitChange(id);
}

export function appendNote(id: string, note: string): void {
	const s = sessions.get(id);
	if (!s || !note.trim()) return;
	s.notes.push(note);
	emitChange(id);
}

export function setOutcome(id: string, summary: string, outcome: CallOutcome): void {
	const s = sessions.get(id);
	if (!s) return;
	s.summary = summary;
	s.outcome = outcome;
	emitChange(id);
}

export function setError(id: string, error: string): void {
	const s = sessions.get(id);
	if (!s) return;
	s.error = error;
	emitChange(id);
}

export function requestHangup(id: string): void {
	const s = sessions.get(id);
	if (!s) return;
	s.hangupRequested = true;
	emitChange(id);
}

/**
 * Wait for the next state change on a session, or up to ``timeoutMs``,
 * whichever comes first. Returns immediately if the session is already
 * in a final state at call time.
 */
export function waitForChange(id: string, timeoutMs: number): Promise<void> {
	const emitter = emitters.get(id);
	if (!emitter) return Promise.resolve();
	const session = sessions.get(id);
	if (session && isFinalStatus(session.status)) return Promise.resolve();
	return new Promise<void>((resolve) => {
		const timer = setTimeout(() => {
			emitter.off('change', onChange);
			resolve();
		}, timeoutMs);
		const onChange = () => {
			clearTimeout(timer);
			emitter.off('change', onChange);
			resolve();
		};
		emitter.once('change', onChange);
	});
}

export function isFinalStatus(s: CallStatus): boolean {
	return [
		'completed', 'busy', 'failed', 'no_answer', 'voicemail',
		'hung_up_by_agent', 'timed_out',
	].includes(s);
}

/** Snapshot suitable for the ``phone_call_status`` tool's JSON return. */
export function snapshot(s: CallSession): Record<string, unknown> {
	const duration_s = s.ended_at !== null
		? (s.ended_at - (s.answered_at ?? s.started_at)) / 1000
		: null;
	return {
		call_id: s.id,
		to: s.to,
		from: s.from,
		mission: s.mission,
		status: s.status,
		twilio_sid: s.twilio_sid,
		transcript: s.transcript,
		notes: s.notes,
		summary: s.summary,
		outcome: s.outcome,
		error: s.error,
		started_at: s.started_at,
		answered_at: s.answered_at,
		ended_at: s.ended_at,
		duration_s,
	};
}
