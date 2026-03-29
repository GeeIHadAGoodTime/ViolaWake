import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { resetPassword } from "../api";

interface ValidationErrors {
  password?: string;
  confirmPassword?: string;
}

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const redirectTimerRef = useRef<number | null>(null);

  const validationErrors: ValidationErrors = {};

  if (!password) {
    validationErrors.password = "Password is required.";
  } else if (password.length < 8) {
    validationErrors.password = "Password must be at least 8 characters.";
  }

  if (!confirmPassword) {
    validationErrors.confirmPassword = "Please confirm your password.";
  } else if (confirmPassword !== password) {
    validationErrors.confirmPassword = "Passwords do not match.";
  }

  const isValid = token !== null && Object.keys(validationErrors).length === 0;

  useEffect(() => {
    return () => {
      if (redirectTimerRef.current !== null) {
        window.clearTimeout(redirectTimerRef.current);
      }
    };
  }, []);

  function handleBlur(field: string) {
    setTouched((prev) => ({ ...prev, [field]: true }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched({ password: true, confirmPassword: true });
    setError(null);

    if (!token) {
      setError("This reset link is missing a token.");
      return;
    }

    if (Object.keys(validationErrors).length > 0) {
      return;
    }

    setSubmitting(true);

    try {
      const response = await resetPassword(token, password);
      setSuccess(`${response.message} Redirecting to login...`);
      redirectTimerRef.current = window.setTimeout(() => {
        navigate("/login", { replace: true });
      }, 2000);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "We could not reset your password.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Reset password</h1>
          <p className="auth-subtitle">
            Choose a new password for your ViolaWake account.
          </p>
        </div>

        {!token && (
          <p className="auth-status error">
            This reset link is invalid or missing its token.
          </p>
        )}

        {error && (
          <div className="auth-error" role="alert">
            {error}
          </div>
        )}

        {success && (
          <p className="auth-status success">
            {success}
          </p>
        )}

        <form
          onSubmit={handleSubmit}
          className="auth-form"
          aria-label="Reset password"
        >
          <div className="form-group">
            <label htmlFor="reset-password" className="form-label">
              New password
            </label>
            <input
              id="reset-password"
              type="password"
              className="form-input"
              placeholder="Minimum 8 characters"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              onBlur={() => handleBlur("password")}
              minLength={8}
              required
              autoFocus
              disabled={!token || submitting || success !== null}
            />
            {touched.password && validationErrors.password && (
              <span className="form-hint hint-invalid">
                {validationErrors.password}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="confirm-password" className="form-label">
              Confirm password
            </label>
            <input
              id="confirm-password"
              type="password"
              className="form-input"
              placeholder="Repeat your new password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              onBlur={() => handleBlur("confirmPassword")}
              required
              disabled={!token || submitting || success !== null}
            />
            {touched.confirmPassword && validationErrors.confirmPassword && (
              <span className="form-hint hint-invalid">
                {validationErrors.confirmPassword}
              </span>
            )}
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={!isValid || submitting || success !== null}
          >
            {submitting ? "Resetting password..." : "Reset password"}
          </button>
        </form>

        <p className="auth-footer">
          Remembered your password?{" "}
          <Link to="/login" className="auth-link">
            Back to login
          </Link>
        </p>
      </div>
    </div>
  );
}
