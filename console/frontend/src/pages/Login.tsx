import { useState, FormEvent } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

interface ValidationErrors {
  email?: string;
  password?: string;
}

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const { login, loading, error, clearError } = useAuth();

  function validate(): ValidationErrors {
    const errors: ValidationErrors = {};
    if (!email.trim()) {
      errors.email = "Email is required";
    }
    if (password.length < 8) {
      errors.password = "Password must be at least 8 characters";
    }
    return errors;
  }

  const currentErrors = validate();
  const isValid = Object.keys(currentErrors).length === 0;

  function handleBlur(field: string) {
    setTouched((prev) => ({ ...prev, [field]: true }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setTouched({ email: true, password: true });
    const errors = validate();
    if (Object.keys(errors).length > 0) return;
    await login(email, password);
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Welcome back</h1>
          <p className="auth-subtitle">
            Sign in to your ViolaWake account
          </p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Sign in">
          {error && (
            <div className="auth-error" onClick={clearError}>
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email" className="form-label">
              Email
            </label>
            <input
              id="email"
              type="email"
              className="form-input"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onBlur={() => handleBlur("email")}
              required
              autoFocus
            />
            {touched.email && currentErrors.email && (
              <span className="form-hint hint-invalid">{currentErrors.email}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">
              Password
            </label>
            <input
              id="password"
              type="password"
              className="form-input"
              placeholder="Your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onBlur={() => handleBlur("password")}
              required
              minLength={8}
              aria-describedby="password-hint"
            />
            <span id="password-hint" className={`form-hint ${touched.password && currentErrors.password ? "hint-invalid" : ""}`}>
              {touched.password && currentErrors.password
                ? currentErrors.password
                : "Minimum 8 characters"}
            </span>
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={loading || !isValid}
          >
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <p className="auth-footer">
          Don&apos;t have an account?{" "}
          <Link to="/register" className="auth-link">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
