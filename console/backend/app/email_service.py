"""Minimal email service for ViolaWake Console."""

from __future__ import annotations

import asyncio
import logging
from html import escape
from urllib.parse import urlencode, urljoin

import resend

from app.config import settings

logger = logging.getLogger("violawake.email")

FROM_ADDRESS = "ViolaWake <noreply@violawake.com>"


class EmailService:
    """Send transactional emails through Resend."""

    def __init__(
        self,
        api_key: str | None = None,
        console_base_url: str | None = None,
        api_base_url: str | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.resend_api_key).strip()
        base_url = console_base_url if console_base_url is not None else settings.console_base_url
        self._console_base_url = base_url.rstrip("/") + "/"
        api_url = api_base_url if api_base_url is not None else settings.api_base_url
        self._api_base_url = api_url.rstrip("/") + "/"
        self._warned_disabled = False

        if self.enabled:
            resend.api_key = self._api_key
        else:
            self._warn_disabled()

    @property
    def enabled(self) -> bool:
        """Return True when outbound email is configured."""
        return bool(self._api_key)

    async def send_verification_email(self, to: str, token: str, name: str) -> bool:
        """Send a verification link after registration.

        The link points at the backend's own GET ``/api/auth/verify-email``
        endpoint (not the client-rendered SPA route). Verification then happens
        server-side on link click, so it does not depend on the frontend's CDN
        routing serving the SPA page correctly. See the GET handler in
        ``app/routes/auth.py`` and gate ``verification-email-server-side-link``.
        """
        verification_url = self._api_url("/api/auth/verify-email", token=token)
        html = self._render_email(
            heading="Confirm your email",
            intro=f"Hi {escape(name)}, please verify your email to finish setting up ViolaWake Console.",
            button_label="Verify Email",
            button_url=verification_url,
            footer="If you did not create this account, you can ignore this email.",
        )
        return await self._send_email(to, "Verify your ViolaWake email", html)

    async def send_password_reset(self, to: str, token: str, name: str) -> bool:
        """Send a password reset email."""
        reset_url = self._console_url("/reset-password", token=token)
        html = self._render_email(
            heading="Reset your password",
            intro=f"Hi {escape(name)}, use the button below to choose a new password for ViolaWake Console.",
            button_label="Reset Password",
            button_url=reset_url,
            footer="If you did not request a reset, you can ignore this email.",
        )
        return await self._send_email(to, "Reset your ViolaWake password", html)

    async def send_welcome(self, to: str, name: str) -> bool:
        """Send a welcome email after email verification."""
        html = self._render_email(
            heading="Welcome to ViolaWake",
            intro=f"Hi {escape(name)}, your email is verified and your workspace is ready.",
            button_label="Open Console",
            button_url=self._console_url("/dashboard"),
            footer="You can upload recordings, train models, and manage billing from the console.",
        )
        return await self._send_email(to, "Welcome to ViolaWake Console", html)

    async def send_existing_account_notice(self, to: str, name: str) -> bool:
        """Notify a user when someone tries to register an email that already exists."""
        html = self._render_email(
            heading="You already have an account",
            intro=(
                f"Hi {escape(name)}, someone tried to register with your email. "
                "If this was you, try logging in instead."
            ),
            button_label="Log In",
            button_url=self._console_url("/login"),
            footer="If this was not you, no action is required and your account is unchanged.",
        )
        return await self._send_email(to, "Your ViolaWake account already exists", html)

    async def send_training_complete(self, to: str, model_name: str, download_url: str) -> bool:
        """Send a training completion email with a download CTA."""
        html = self._render_email(
            heading="Training complete",
            intro=f"Your model <strong>{escape(model_name)}</strong> is ready to download.",
            button_label="Download Model",
            button_url=self._absolute_url(download_url),
            footer="You can also review metrics and model history in the console.",
        )
        return await self._send_email(to, f"Your ViolaWake model {model_name} is ready", html)

    async def send_training_failed(
        self,
        to: str,
        model_name: str,
        reason: str,
        *,
        charged: bool = True,
    ) -> bool:
        """Notify the customer that a training run failed.

        The console had ``send_training_complete`` but no failure counterpart, so
        a customer whose run failed learned nothing by email -- the missing half of
        the training-outcome channel (#4207). ``charged`` tells the reader whether
        the attempt counted against their monthly quota: our-side infrastructure
        faults are refunded, so the copy must not imply they lost an attempt.
        """
        if charged:
            billing_line = (
                "This run counted as one of your monthly training attempts. "
                "Wake-word training varies run to run, so training again with the "
                "same recordings often succeeds."
            )
        else:
            billing_line = (
                "This failure was on our side, not your recordings, so it did "
                "<strong>not</strong> count against your monthly training attempts."
            )
        html = self._render_email(
            heading="Training didn't finish",
            intro=(
                f"Your training run for <strong>{escape(model_name)}</strong> did not "
                f"complete.<br><br>Reason: {escape(reason)}<br><br>{billing_line}"
            ),
            button_label="Open Console",
            button_url=self._console_url("/dashboard"),
            footer="You can review the run and try again from your dashboard.",
        )
        return await self._send_email(
            to, f"Your ViolaWake training for {model_name} didn't finish", html
        )

    async def send_team_invite(
        self,
        to_email: str,
        team_name: str,
        inviter_name: str,
        accept_url: str,
    ) -> bool:
        """Send a team invitation email with an accept link."""
        html = self._render_email(
            heading="You've been invited to a team",
            intro=(
                f"{escape(inviter_name)} invited you to join "
                f"<strong>{escape(team_name)}</strong> on ViolaWake Console."
            ),
            button_label="Accept Invite",
            button_url=accept_url,
            footer="If you did not expect this invitation, you can ignore this email.",
        )
        return await self._send_email(
            to_email,
            f"You've been invited to join {team_name} on ViolaWake",
            html,
        )

    async def send_quota_warning(self, to: str, used: int, limit: int, tier: str) -> bool:
        """Send a usage warning when the user is near their tier limit."""
        html = self._render_email(
            heading="You are close to your training limit",
            intro=(
                f"You have used <strong>{used}</strong> of <strong>{limit}</strong> "
                f"monthly trainings on the <strong>{escape(tier.title())}</strong> plan."
            ),
            button_label="Review Plans",
            button_url=self._console_url("/pricing"),
            footer="Upgrade before you hit the limit if you need more model training capacity.",
        )
        return await self._send_email(to, "ViolaWake usage warning", html)

    async def send_subscription_activated(self, to: str, name: str, tier: str) -> bool:
        """Send a confirmation email when a paid subscription activates."""
        html = self._render_email(
            heading="Subscription activated",
            intro=(
                f"Hi {escape(name)}, your ViolaWake "
                f"<strong>{escape(tier.title())}</strong> subscription is active."
            ),
            button_label="Open Billing",
            button_url=self._console_url("/billing"),
            footer="You can review usage, invoices, and plan status from the Billing page.",
        )
        return await self._send_email(to, "Your ViolaWake subscription is active", html)

    async def send_account_deleted(
        self,
        to: str,
        name: str,
        scheduled_hard_delete_at: str,
    ) -> bool:
        """Send confirmation after an account deletion request is accepted."""
        html = self._render_email(
            heading="Account deleted",
            intro=(
                f"Hi {escape(name)}, your ViolaWake account has been deleted. "
                "Your account data is no longer accessible and is scheduled "
                f"for permanent deletion on {escape(scheduled_hard_delete_at)}."
            ),
            button_label="ViolaWake",
            button_url=self._console_url("/"),
            footer="If you did not request this, contact privacy@violawake.com immediately.",
        )
        return await self._send_email(to, "Your ViolaWake account has been deleted", html)

    async def send_support_autoreply(self, to: str, ticket_reference: str) -> bool:
        """Send a support inbox auto-reply."""
        html = self._render_email(
            heading="Thanks for contacting ViolaWake",
            intro=(
                "We received your message and aim to respond within 48 hours. "
                f"Your ticket reference is <strong>{escape(ticket_reference)}</strong>."
            ),
            button_label="Open ViolaWake",
            button_url=self._console_url("/"),
            footer="ViolaWake Support",
        )
        return await self._send_email(to, f"ViolaWake support request {ticket_reference}", html)

    def _console_url(self, path: str, **query: str) -> str:
        """Build a console URL from the configured base URL."""
        url = urljoin(self._console_base_url, path.lstrip("/"))
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    def _api_url(self, path: str, **query: str) -> str:
        """Build a backend-API URL from the configured API base URL."""
        url = urljoin(self._api_base_url, path.lstrip("/"))
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    def _absolute_url(self, path_or_url: str) -> str:
        """Normalize relative paths against the console base URL."""
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return self._console_url(path_or_url)

    def _render_email(
        self,
        *,
        heading: str,
        intro: str,
        button_label: str,
        button_url: str,
        footer: str,
    ) -> str:
        """Return a small inline-HTML email body."""
        return f"""
<!DOCTYPE html>
<html lang="en">
  <body style="margin:0;padding:24px;background:#f5f7fb;font-family:Arial,sans-serif;color:#111827;">
    <div style="max-width:560px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:32px;">
      <p style="margin:0 0 12px;font-size:13px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#2563eb;">ViolaWake</p>
      <h1 style="margin:0 0 16px;font-size:28px;line-height:1.2;color:#111827;">{heading}</h1>
      <p style="margin:0 0 24px;font-size:16px;line-height:1.6;color:#374151;">{intro}</p>
      <p style="margin:0 0 24px;">
        <a href="{escape(button_url, quote=True)}" style="display:inline-block;background:#111827;color:#ffffff;text-decoration:none;padding:12px 20px;border-radius:8px;font-size:15px;font-weight:600;">{escape(button_label)}</a>
      </p>
      <p style="margin:0;font-size:14px;line-height:1.6;color:#6b7280;">{footer}</p>
    </div>
  </body>
</html>
""".strip()

    async def _send_email(self, to: str, subject: str, html: str) -> bool:
        """Send an email through Resend, or no-op when disabled."""
        if not self.enabled:
            self._warn_disabled()
            logger.info("Skipping email send to %s for subject %s because Resend is not configured", to, subject)
            return False

        params = {
            "from": FROM_ADDRESS,
            "to": [to],
            "subject": subject,
            "html": html,
        }

        try:
            await asyncio.to_thread(resend.Emails.send, params)
        except Exception:
            logger.exception("Failed to send email to %s for subject %s", to, subject)
            return False

        logger.info("Sent email to %s for subject %s", to, subject)
        return True

    def _warn_disabled(self) -> None:
        """Log once when email delivery is disabled."""
        if self._warned_disabled:
            return
        logger.warning("Resend email delivery is disabled because VIOLAWAKE_RESEND_API_KEY is not set")
        self._warned_disabled = True


_email_service: EmailService | None = None


def get_email_service() -> EmailService:
    """Return the process-wide EmailService singleton."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService(
            api_key=settings.resend_api_key,
            console_base_url=settings.console_base_url,
            api_base_url=settings.api_base_url,
        )
    return _email_service
