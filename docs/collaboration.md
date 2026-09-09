# Shared sessions (additive API)

OpenAgent owns execution; authenticated clients attach to it. This API adds
shared text turns, steering, serialized commands, replay and presence without
requiring a particular host, organization model, identity provider or connector
catalog. It is an optional gateway API, not a replacement for the existing
audio/video or client-machine capability protocol.

## Observe

Open `GET /ws/collaboration` through the same certificate-authenticated transport
as `/ws`. After `{"type":"auth_ok","shared":true}`, send:

```json
{"type":"observe","sessions":["chat:1"],"focus":{"kind":"session","id":"chat:1"}}
```

`sessions` selects up to 16 transcripts. `focus` is optional and can refer to a
`session`, `workflow`, `scheduled_task` or `event`. To observe execution inside
an automation, include its actual child session ID. A separate socket is an
independent presence lease, including two tabs using the same account.

- `shared_state`: `session_id`, monotonic `revision`, and up to eight `turns`.
  Replace the prior snapshot; never append its text as a delta. Turn `runId`
  identifies the submitted `request_id`, not the provider's persisted run ID.
  Messages contain the authenticated author's handle/display and current text.
- `shared_presence`: authorized people with `userId`, `name`, and `target`.
  Same-person/same-target leases are deduplicated. Applications can resolve
  avatars from their own directory using these stable identities.
- `shared_revoked`: discard that session's live cache immediately.
- `resource_event`: refetch the affected resource through its normal API.
- `auth_error` or disconnect: discard live text/presence and reconnect before
  showing another snapshot. Opening/closing this socket never starts/stops work.

Updates coalesce for 40 ms. Producers perform no observer network/authorization
I/O. Replay is bounded to 128 sessions, eight turns/session and 131072 characters
per input/response; `truncated` indicates the cap. Completed tails expire after
60 seconds; durable history remains in the regular session APIs. A reconnect
always receives a fresh authorized snapshot, with no cursor to recover.

## Send, steer, command, stop

Create/claim the durable session through the normal session API first. The
native policy requires ownership or an explicit `admin` ACL grant to write;
public/installation sharing or a `view` grant permits observation only.

```http
POST /api/collaboration/turns
Content-Type: application/json

{"session_id":"chat:1","request_id":"request-uuid","message":"Adjust the answer","delivery":"steer"}
```

`delivery` is `queue` (default) or `steer`. Every message has a distinct author;
messages from different people are never coalesced. Steering cooperatively
interrupts generation, preserves available partial text and serializes the new
turn. A provider that remains detached blocks replacement generation with 409.
Missing terminal frames cannot leave the collector holding the session lock.

Commands are messages (`/compact`, `/model runtime:id`, `/model auto`, `/context`,
`/status`, `/queue`, `/help`, `/usage`, `/stop`). They execute under the same turn lock;
model selection validates native ACLs, enabled models and providers. Command
results are broadcast and recorded as `command/result` in the session journal.
Host administration commands are not part of this session API.

```http
POST /api/collaboration/stop
Content-Type: application/json

{"session_id":"chat:1","request_id":"request-uuid"}
```

Stop targets exactly one active/queued request. Cancelling a queued request
cannot stop another author's active turn. An executing command completes its
atomic operation rather than being cancelled halfway through compaction.

The HTTP result contains `request_id`, `session_id`, `response`, and optional
`model`, `errored`, `interrupted`. Losing/aborting that HTTP connection does not
cancel execution. Retrying the same ID/payload/author returns the same task or
result; changing the payload/author conflicts with 409. Authorization is checked
again before admission, after queue waits, and before returning cached content.
Deduplication is process-local and retained for 60 seconds **after completion**;
it is not an exactly-once guarantee across restarts. Integrations needing that
guarantee must maintain their own durable request ledger. At most 256 requests
are retained; capacity returns 429 rather than allocating an unbounded queue.

An idle legacy session can switch to this transport. A running legacy turn must
finish first (409). While the shared runtime owns the session, legacy stream,
chat, model-pin and delete mutations are rejected rather than running a second
writer. Shared runtimes detach after their request-retention window; observation
does not hold them open. Attachments, audio/video and client-local tools continue
using their existing protocol; this endpoint deliberately accepts text only.

## Host policy boundary

By default the gateway uses its canonical session/resource ACLs and certificate
identity. Device revocation wakes observers and cancels only the revoked
device's active shared turn. Every delivered snapshot is authorized again.

An embedding host may assign an async in-process callable to
`gateway.collaboration_authorizer(request, targets, permission)`. It returns
`{userId, name, allowed}`; `permission` is `view` or `admin`. The result may only
narrow the requested target set. Device authentication/revocation checks still
run before and after the callback. This is not a remotely configurable endpoint
and cannot replace the authenticated tool principal with client-provided text.
Native management operations retain their native authorization checks.

Use `collaboration.service(gateway).hub.resource(kind, action, id)` to invalidate
views after a host-side policy/resource change. OpenAgent's resource broadcasts
already do this. No workspace/project logic, external connector catalog,
host-specific tokens or user-directory synchronization is embedded here.

## Verification

`python -m unittest scripts.tests.test_collaboration -v` exercises two-account
aiohttp sockets, SQLite ACLs, real StreamSession/BatchedChannel execution with a
deterministic model, replay, concurrent steering/commands, revocation, exact
queued stop, missing terminal frames, provider detachment and transport guards.
The model double does not write provider runs; canonical message-part projection
is stubbed in those tests. Native stream, command, cross-device, capability and
session-pin regression suites run separately. No paid provider is needed.
