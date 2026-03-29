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
          Verify your email to unlock recording, training, and billing features.
          Check your inbox for the verification link.
        </div>
      )}

      <main className="main-content">{children}</main>
    </div>
  );
}
