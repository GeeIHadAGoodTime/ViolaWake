import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { resendVerification } from "../api";

interface LayoutProps {
  children: React.ReactNode;
}

type ResendState = "idle" | "sending" | "sent" | "error";

export default function Layout({ children }: LayoutProps) {
  const { isAuthenticated, logout, user } = useAuth();
  const [resendState, setResendState] = useState<ResendState>("idle");

  // The first verification email is a one-shot send (see backend
  // EmailService._send_email) that can be lost to spam filtering, a
  // delayed/greylisted delivery, or a mistyped address, with no automatic
  // retry -- see GH #520 / GlitchTip glitchtip:22 (a customer had no way to
  // get a second one and had to file a support ticket after a week). This
  // gives every unverified user a visible, self-service retry instead.
  async function handleResendVerification() {
    if (!user?.email || resendState === "sending") {
      return;
    }
    setResendState("sending");
    try {
      await resendVerification(user.email);
      setResendState("sent");
    } catch {
      setResendState("error");
    }
  }

  return (
    <div className="layout">
      <nav className="navbar">
        <Link to="/" className="navbar-brand">
          <span className="brand-icon">W</span>
          <span className="brand-text">ViolaWake</span>
        </Link>

        <div className="navbar-links">
          {isAuthenticated ? (
            <>
              <Link to="/dashboard" className="nav-link">
                Dashboard
              </Link>
              <Link to="/record" className="nav-link">
                Record
              </Link>
              <Link to="/teams" className="nav-link">
                Teams
              </Link>
              <Link to="/billing" className="nav-link">
                Billing
              </Link>
              <Link to="/pricing" className="nav-link">
                Plans
              </Link>
              <Link
                to="/account/password"
                className="nav-link nav-account"
                title={user?.email ?? "Account"}
              >
                <span className="nav-account-label">Account</span>
                {user?.email && (
                  <span className="nav-account-email">{user.email}</span>
                )}
              </Link>
              <button
                onClick={logout}
                className="nav-link nav-button"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/pricing" className="nav-link">
                Pricing
              </Link>
              <Link to="/contact" className="nav-link">
                Contact
              </Link>
              <Link to="/login" className="nav-link">
                Login
              </Link>
              <Link to="/register" className="btn btn-primary btn-nav">
                Get Started
              </Link>
            </>
          )}
        </div>
      </nav>

      {isAuthenticated && user && !user.email_verified && (
        <div className="verification-banner" role="status">
          <span>
            Verify your email to upload recordings, start training, and manage billing.
            Check your inbox for the verification link.
          </span>
          {resendState === "sent" ? (
            <span className="verification-banner-status">
              Verification email sent -- check your inbox (and spam folder).
            </span>
          ) : (
            <button
              type="button"
              className="verification-banner-action"
              onClick={() => void handleResendVerification()}
              disabled={resendState === "sending"}
            >
              {resendState === "sending" ? "Sending..." : "Resend verification email"}
            </button>
          )}
          {resendState === "error" && (
            <span className="verification-banner-status verification-banner-status-error">
              Could not send right now. Try again in a moment.
            </span>
          )}
        </div>
      )}

      <main className="main-content">{children}</main>

      {/* Global footer — visible app-wide so privacy/terms/contact are
          always one click away, including from inside the authed app. */}
      <footer className="app-footer" aria-label="Site footer">
        <div className="app-footer-inner">
          <div className="app-footer-links">
            <Link to="/pricing">Pricing</Link>
            <Link to="/contact">Contact</Link>
            <Link to="/privacy">Privacy</Link>
            <Link to="/terms">Terms</Link>
            <a
              href="https://github.com/GeeIHadAGoodTime/ViolaWake"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </div>
          <div className="app-footer-meta">
            &copy; 2026 ViolaWake. Apache 2.0 licensed SDK.
          </div>
        </div>
      </footer>
    </div>
  );
}
