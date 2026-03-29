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

    def __init__(self, api_key: str, console_base_url: str) -> None:
        self._api_key = api_key.strip()
        self._console_base_url = console_base_url.rstrip("/") + "/"
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
        """Send a verification link after registration."""
        verification_url = self._console_url("/verify-email", token=token)
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

    async def send_team_invite(self, to_email: str, team_name: str, invite_token: str, invite_url: str) -> bool:
        """Send a team invitation email with a join link."""
        html = self._render_email(
            heading="You've been invited to a team",
            intro=(
                f"You have been invited to join <strong>{escape(team_name)}</strong> "
                f"on ViolaWake Console."
            ),
            button_label="Accept Invite",
            button_url=invite_url,
            footer="If you did not expect this invitation, you can ignore this email.",
        )
        return await self._send_email(to_email, f"Join {team_name} on ViolaWake", html)

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

    def _console_url(self, path: str, **query: str) -> str:
        """Build a console URL from the configured base URL."""
        url = urljoin(self._console_base_url, path.lstrip("/"))
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
        )
    return _email_service
