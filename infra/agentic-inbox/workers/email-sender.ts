// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

/**
 * Email sending via Cloudflare Email Service binding.
 *
 * Uses the `send_email` Worker binding (`env.EMAIL.send()`) to send emails.
 *
 * See: https://developers.cloudflare.com/email-service/api/send-emails/workers-api/
 */

// ── Outbound kill-switch + audit log ──────────────────────────────────────────
//
// Kill-switch: write  {"paused":true}  to  config/outbound-paused.json  in R2.
//   - All outbound sends (from any surface) check this flag before calling
//     the Email Service binding. A paused worker returns an error that callers
//     propagate to the user.
//   - To resume: delete the object or set {"paused":false}.
//
// Audit log: every successful or failed outbound send is appended to
//   audit/outbound/<mailboxId>/YYYY-MM.jsonl  in R2 (one entry per line).
//   Each entry is a JSON object with: ts, mailboxId, to, subject, messageId?,
//   error?, surface (http|mcp|agent).
//
// Both features use R2 directly — no DO schema changes needed.

/** R2 key for the outbound kill-switch object. */
export const OUTBOUND_KILL_SWITCH_KEY = "config/outbound-paused.json";

/** Error thrown when the kill-switch is engaged. */
export class OutboundPausedError extends Error {
	constructor() {
		super("Outbound email is currently paused (kill-switch engaged). Contact support.");
		this.name = "OutboundPausedError";
	}
}

/**
 * Check whether the outbound kill-switch is engaged.
 * Returns true if ALL outbound sends should be blocked.
 */
export async function isOutboundPaused(bucket: R2Bucket): Promise<boolean> {
	try {
		const obj = await bucket.get(OUTBOUND_KILL_SWITCH_KEY);
		if (!obj) return false;
		const body = await obj.json<{ paused?: boolean }>();
		return body?.paused === true;
	} catch {
		// If we can't read the flag, fail open (don't block sends on a R2 error).
		return false;
	}
}

interface OutboundAuditEntry {
	ts: string;           // ISO 8601
	mailboxId: string;    // sending mailbox
	to: string;           // recipient(s) as comma-separated string
	subject: string;
	surface: "http" | "mcp" | "agent" | "unknown";
	messageId?: string;   // CF Email Service message ID (on success)
	error?: string;       // error message (on failure)
}

/**
 * Append one line to the per-mailbox monthly outbound audit log in R2.
 *
 * Key pattern: audit/outbound/<mailboxId>/YYYY-MM.jsonl
 *
 * R2 does not support append, so we read → append → put.  This is safe for
 * the low-frequency sends of a support inbox.  For high-volume use a Queue.
 */
async function appendOutboundAuditEntry(
	bucket: R2Bucket,
	entry: OutboundAuditEntry,
): Promise<void> {
	const month = entry.ts.slice(0, 7); // "YYYY-MM"
	const key = `audit/outbound/${entry.mailboxId}/${month}.jsonl`;
	try {
		const existing = await bucket.get(key);
		const prev = existing ? await existing.text() : "";
		const line = JSON.stringify(entry) + "\n";
		await bucket.put(key, prev + line, {
			httpMetadata: { contentType: "application/x-ndjson" },
		});
	} catch (e) {
		// Audit log failure must never block the send or surface to the user.
		console.error("Failed to write outbound audit log:", (e as Error).message);
	}
}

export interface SendEmailParams {
	to: string | string[];
	from: string | { email: string; name: string };
	subject: string;
	html?: string;
	text?: string;
	cc?: string | string[];
	bcc?: string | string[];
	replyTo?: string | { email: string; name: string };
	attachments?: {
		content: string; // base64 encoded
		filename: string;
		type: string;
		disposition: "attachment" | "inline";
		contentId?: string;
	}[];
	headers?: Record<string, string>;
}

/**
 * Send an email using the Cloudflare Email Service binding.
 *
 * @param binding  - The `EMAIL` SendEmail binding from env
 * @param params   - Email parameters (to, from, subject, body, etc.)
 * @returns The send result with messageId
 * @throws On validation or delivery errors (error has `.code` property)
 */
export async function sendEmail(
	binding: SendEmail,
	params: SendEmailParams,
): Promise<{ messageId: string }> {
	const message: Record<string, unknown> = {
		to: params.to,
		from: params.from,
		subject: params.subject,
	};

	if (params.html) message.html = params.html;
	if (params.text) message.text = params.text;
	if (params.cc) message.cc = params.cc;
	if (params.bcc) message.bcc = params.bcc;
	if (params.replyTo) message.replyTo = params.replyTo;

	if (params.headers && Object.keys(params.headers).length > 0) {
		message.headers = params.headers;
	}

	if (params.attachments && params.attachments.length > 0) {
		message.attachments = params.attachments.map((att) => ({
			content: att.content,
			filename: att.filename,
			type: att.type,
			disposition: att.disposition,
			...(att.contentId ? { contentId: att.contentId } : {}),
		}));
	}

	const result = await binding.send(message as any);
	return { messageId: result.messageId };
}

/**
 * Send an email with outbound kill-switch enforcement and audit logging.
 *
 * Use this function instead of `sendEmail` at every outbound surface so that:
 *   1. The kill-switch (`config/outbound-paused.json` in R2) is always checked.
 *   2. Every send (successful or failed) is recorded in the audit log.
 *
 * @param bucket   - R2 bucket binding (for kill-switch + audit log)
 * @param binding  - `EMAIL` SendEmail binding
 * @param mailboxId - The sending mailbox email address (used in audit log key)
 * @param params   - Email parameters
 * @param surface  - Where the send originated ("http" | "mcp" | "agent")
 * @throws OutboundPausedError if the kill-switch is engaged
 * @throws On delivery errors (same as `sendEmail`)
 */
export async function sendEmailAudited(
	bucket: R2Bucket,
	binding: SendEmail,
	mailboxId: string,
	params: SendEmailParams,
	surface: "http" | "mcp" | "agent" = "unknown" as any,
): Promise<{ messageId: string }> {
	// Kill-switch check — fail fast before touching the Email binding.
	if (await isOutboundPaused(bucket)) {
		throw new OutboundPausedError();
	}

	const ts = new Date().toISOString();
	const toStr = Array.isArray(params.to) ? params.to.join(", ") : params.to;
	const entry: OutboundAuditEntry = {
		ts,
		mailboxId,
		to: toStr,
		subject: params.subject,
		surface,
	};

	try {
		const result = await sendEmail(binding, params);
		entry.messageId = result.messageId;
		// Fire-and-forget audit write so it never blocks the caller.
		void appendOutboundAuditEntry(bucket, entry);
		return result;
	} catch (e) {
		entry.error = (e as Error).message;
		void appendOutboundAuditEntry(bucket, entry);
		throw e;
	}
}
