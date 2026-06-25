# Triage rubric (level-3 resource)

A bundled reference the `triage-inbox` skill reads on demand — kept out of the
skill body so it doesn't cost context until it's actually needed.

## Buckets, in priority order

| Bucket | Signal | What to do |
|---|---|---|
| 🔴 **Action** | Addressed to the user with a question, request, deadline, or decision. A real human awaiting a real reply. | Surface first. Extract the single ask. Offer a draft. |
| 🟡 **Waiting** | The user already replied / acted; now blocked on someone else. "I'll get back to you", "pending approval". | Track, don't act. Note who/what is blocking. |
| ⚪ **FYI** | Informational, CC'd, announcements, newsletters worth keeping. No reply expected. | Collapse into a count. List only if asked. |
| 🧾 **Receipts / Automated** | Invoices, order/shipping, CI/build, calendar, system notifications. | Collapse into a count. Flag only money/deadlines. |

## Tie-breakers

- **Deadline beats topic.** Anything with a date/amount (invoice, RSVP, "by
  Friday") ranks above general Action.
- **Named human beats automated**, even on the same thread.
- **Direct `to:` beats `cc:`.** A CC rarely needs the user to act.
- **Thread age matters.** An Action item sitting for days outranks a fresh one.

## Urgency cues (push toward Action / higher rank)

`urgent`, `ASAP`, `EOD`, `today`, `reminder`, `final notice`, `overdue`, a
question mark addressed to the user, an explicit `@`-mention, a due date within 48h.

## De-prioritize

Bulk `list-unsubscribe` headers, `no-reply@` senders, marketing language
("% off", "don't miss"), and anything already labeled by a filter.
