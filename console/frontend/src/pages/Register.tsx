import { useState, FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

interface ValidationErrors {
  name?: string;
  email?: string;
  password?: string;
}

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const { register, loading, error, clearError } = useAuth();
  const location = useLocation();
  const loginLink = location.search ? `/login${location.search}` : "/login";

  function validate(): ValidationErrors {
    const errors: ValidationErrors = {};
    if (!name.trim()) {
      errors.name = "Name is required";
    }
    if (!email.includes("@")) {
      errors.email = "Please enter a valid email address";
    }
    if (password.length < 8) {
      errors.password = `${8 - password.length} more characters needed`;
    }
    return errors;
  }

  const currentErrors = validate();
  const isValid = Object.keys(currentErrors).length === 0;
  const passwordValid = password.length >= 8;

  function handleBlur(field: string) {
    setTouched((prev) => ({ ...prev, [field]: true }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setTouched({ name: true, email: true, password: true });
    const errors = validate();
    if (Object.keys(errors).length > 0) return;
    await register(email, password, name);
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Create account</h1>
          <p className="auth-subtitle">
            Train your own custom wake word model
          </p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Create account">
          {error && (
            <div className="auth-error" onClick={clearError}>
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="name" className="form-label">
              Name
            </label>
            <input
              id="name"
              type="text"
              className="form-input"
              placeholder="Your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => handleBlur("name")}
              required
              autoFocus
            />
            {touched.name && currentErrors.name && (
              <span className="form-hint hint-invalid">{currentErrors.name}</span>
            )}
          </div>

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
              placeholder="Min 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onBlur={() => handleBlur("password")}
              required
              minLength={8}
              aria-describedby="register-password-hint"
            />
            <span
              id="register-password-hint"
              className={`form-hint ${password.length > 0 ? (passwordValid ? "hint-valid" : "hint-invalid") : ""}`}
            >
              {password.length === 0
                ? "Minimum 8 characters"
                : passwordValid
                  ? "Password strength: OK"
                  : currentErrors.password}
            </span>
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={loading || !isValid}
          >
            {loading ? "Creating account..." : "Create account"}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account?{" "}
          <Link to={loginLink} className="auth-link">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
