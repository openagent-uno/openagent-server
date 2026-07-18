---
name: support-triage
description: Triage an inbound customer-support message end to end — classify, gather context, resolve or escalate.
category: support
---

# Support triage

A generic first-response playbook for a customer-support agent. Follow the
steps in order; stop as soon as the ticket is resolved.

## 1. Classify

Read the whole message before replying. Bucket it into one of:

- **Question** — the user wants information ("how do I…", "does it support…").
- **Bug** — something worked before / is documented to work and now does not.
- **Billing** — payments, refunds, subscriptions, invoices, entitlements.
- **Account** — login, password, deletion, data export, access.
- **Feedback** — a feature request or opinion, no action expected.

If the message spans several buckets, handle the one that unblocks the user
first (usually access/billing before questions).

## 2. Gather context before answering

- Identify the user (email / account id) and the product/plan they are on.
- Reproduce or confirm the problem in your own words back to them so they
  know they were understood.
- Check for known issues before promising a fix — do not invent a cause.

## 3. Resolve

- **Question** — answer plainly, link the canonical doc, and confirm it
  solved the problem.
- **Bug** — collect exact steps to reproduce, version, and platform. If you
  can work around it, give the workaround now and file the underlying issue.
- **Billing** — verify the charge/entitlement against the record of truth
  before making any promise about money. Never guess a refund is possible.
- **Account** — follow the identity-verification step for the action before
  performing anything destructive (deletion, email change).

## 4. Escalate cleanly when you cannot resolve

Escalate when the fix needs access you do not have, a decision above your
authority, or an engineering change. When you escalate:

- Summarize what the user wants, what you already tried, and what you need.
- Set an expectation for when they will hear back.
- Never leave the user without a next step.

## 5. Close the loop

Confirm the resolution with the user, and record anything reusable (a new
FAQ answer, a recurring bug) so the next person does not start from zero.
