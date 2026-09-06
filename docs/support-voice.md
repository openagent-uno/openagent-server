# Customer support voice

The controller determines account authority, product facts, evidence, operations,
and the next useful question. These become an internal operational brief. They
are not a customer-facing fallback.

With `OPENAGENT_SUPPORT_HUMAN_VOICE=1` (the production default), every substantive
controller reply is written using the latest message and recent conversation.
The writer acknowledges the person's actual experience, explains useful next steps,
and preserves all relevant facts without pressure, exaggerated empathy, invented
promises or claims of being a human. Cancellation and account deletion receive the
same care without retention friction.

A separate model call checks factual support, complete coverage, relevance,
language and tone. Each required point must have a verbatim supporting quote in
the proposed reply. Numeric and authority checks remain deterministic. Findings
request up to two bounded rewrites. Failure or timeout queues human review and sends no
canned text. A hash binds approval to the exact bytes handed to delivery; a later
guard cannot silently substitute its own wording.

The writer/reviewer can use an enabled registered model through
`OPENAGENT_SUPPORT_VOICE_MODEL`; otherwise the event model is used. No provider is
hardcoded. `OPENAGENT_SUPPORT_VOICE_TIMEOUT_SECONDS` defaults to 45 seconds per
call. A shared 120-second deadline covers the entire voice cycle, including later
guard retries, within the production event's 240-second limit. Tool access remains disabled for writing/review, with the existing model
concurrency gate. Public replies respect actual channel limits; private replies
have space for a complete explanation.

Advertising explanations include the purpose of server funding and the relevant
free options. An optional video started in Settings grants 30 ad-free minutes;
a second adds up to an hour, capped at a two-hour accumulated window. Automatic
ads do not grant the reward, and ad-free time is distinct from Premium. Referral
and eligible Creator rewards are separate routes. Advertising guidance does not
replace diagnosis of an overlapping-audio or playback fault.

Operational unit doubles run with the voice feature disabled to preserve their
isolated planning contracts. The `support_voice` tests explicitly enable it and
cover writer/reviewer, context, bounded repair, failure/hold, authority checks,
quote coverage and delivery integrity. Model replay always enables the feature;
use the same registered voice model as production and inspect the actual replies.

Guidance receives the retrieved product references and operator policy instead of
a generic fallback question. Product steps require supporting source quotations;
ordinary definitions can explain unfamiliar terms without claiming app-specific
capture or navigation behaviour. Reported versions can be acknowledged as context;
they never establish account or payment state. Prose is written directly rather
than embedded in JSON, avoiding broken escaping around quoted UI labels.
