import { Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { isAuthenticated, logout, user } = useAuth();

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
          Verify your email to upload recordings, start training, and manage billing.
          Check your inbox for the verification link.
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
