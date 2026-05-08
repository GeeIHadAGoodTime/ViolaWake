/**
 * Cloudflare Email Worker that bridges Cloudflare Email Routing to the
 * ViolaWake backend's /api/email/inbound webhook.
 *
 * Flow:
 *   sender → MX → Cloudflare Email Routing → this Worker → POST → backend
 *   Backend dedupes by sender + 24h, sends an auto-reply via Resend.
 *
 * Also forwards the original message to a human inbox so support emails
 * aren't lost in the auto-reply pipeline.
 */

export interface Env {
  /** Shared secret matching backend settings.email_inbound_webhook_secret. */
  VIOLAWAKE_EMAIL_INBOUND_SECRET: string;
  /** Where to POST the JSON body. */
  VIOLAWAKE_INBOUND_URL: string;
  /** Verified destination address that real humans read. Optional. */
  VIOLAWAKE_FORWARD_TO?: string;
}

interface EmailMessage {
  readonly from: string;
  readonly to: string;
  readonly headers: Headers;
  readonly raw: ReadableStream<Uint8Array>;
  readonly rawSize: number;
  forward(rcptTo: string, headers?: Headers): Promise<void>;
  reply(message: unknown): Promise<void>;
  setReject(reason: string): void;
}

export default {
  async email(message: EmailMessage, env: Env, ctx: ExecutionContext): Promise<void> {
    const subject = message.headers.get("subject") ?? "(no subject)";
    const messageId = message.headers.get("message-id") ?? "";

    const payload = {
      from: message.from,
      to: message.to,
      subject,
      message_id: messageId,
      received_at: new Date().toISOString(),
    };

    // Fire the auto-reply webhook. Use waitUntil so even a slow backend
    // response doesn't delay the actual delivery / forward.
    const webhookPromise = fetch(env.VIOLAWAKE_INBOUND_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-violawake-email-secret": env.VIOLAWAKE_EMAIL_INBOUND_SECRET,
        "user-agent": "violawake-email-worker/1.0",
      },
      body: JSON.stringify(payload),
    }).then(async (resp) => {
      if (!resp.ok) {
        console.warn(
          `inbound webhook returned ${resp.status}: ${(await resp.text()).slice(0, 200)}`,
        );
      }
    }).catch((err) => {
      console.error(`inbound webhook fetch failed: ${err}`);
    });

    ctx.waitUntil(webhookPromise);

    // Forward to a human inbox if configured. Without this the message would
    // be auto-replied but never read.
    if (env.VIOLAWAKE_FORWARD_TO) {
      try {
        await message.forward(env.VIOLAWAKE_FORWARD_TO);
      } catch (err) {
        console.error(`forward to ${env.VIOLAWAKE_FORWARD_TO} failed: ${err}`);
      }
    }
  },
};
