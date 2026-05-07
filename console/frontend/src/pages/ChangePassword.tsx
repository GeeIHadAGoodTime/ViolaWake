import { FormEvent, useState } from "react";
import { changePassword } from "../api";

interface ValidationErrors {
  currentPassword?: string;
  newPassword?: string;
  confirmPassword?: string;
}

export default function ChangePasswordPage() {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const validationErrors: ValidationErrors = {};

  if (!currentPassword) {
    validationErrors.currentPassword = "Current password is required.";
  }

  if (!newPassword) {
    validationErrors.newPassword = "New password is required.";
  } else if (newPassword.length < 8) {
    validationErrors.newPassword = "Password must be at least 8 characters.";
  }

  if (!confirmPassword) {
    validationErrors.confirmPassword = "Please confirm your new password.";
  } else if (confirmPassword !== newPassword) {
    validationErrors.confirmPassword = "Passwords do not match.";
  }

  const isValid = Object.keys(validationErrors).length === 0;

  function handleBlur(field: string) {
    setTouched((prev) => ({ ...prev, [field]: true }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched({
      currentPassword: true,
      newPassword: true,
      confirmPassword: true,
    });
    setError(null);
    setSuccess(null);

    if (!isValid) {
      return;
    }

    setSubmitting(true);

    try {
      const response = await changePassword(currentPassword, newPassword);
      setSuccess(response.message);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setTouched({});
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "We could not change your password.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Change password</h1>
          <p className="auth-subtitle">
            Update the password for your ViolaWake account.
          </p>
        </div>

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
          aria-label="Change password"
        >
          <div className="form-group">
            <label htmlFor="current-password" className="form-label">
              Current password
            </label>
            <input
              id="current-password"
              type="password"
              className="form-input"
              value={currentPassword}
              onChange={(event) => setCurrentPassword(event.target.value)}
              onBlur={() => handleBlur("currentPassword")}
              required
              autoFocus
              disabled={submitting}
            />
            {touched.currentPassword && validationErrors.currentPassword && (
              <span className="form-hint hint-invalid">
                {validationErrors.currentPassword}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="new-password" className="form-label">
              New password
            </label>
            <input
              id="new-password"
              type="password"
              className="form-input"
              placeholder="Minimum 8 characters"
              value={newPassword}
              onChange={(event) => setNewPassword(event.target.value)}
              onBlur={() => handleBlur("newPassword")}
              minLength={8}
              required
              disabled={submitting}
            />
            {touched.newPassword && validationErrors.newPassword && (
              <span className="form-hint hint-invalid">
                {validationErrors.newPassword}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="confirm-password" className="form-label">
              Confirm new password
            </label>
            <input
              id="confirm-password"
              type="password"
              className="form-input"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              onBlur={() => handleBlur("confirmPassword")}
              required
              disabled={submitting}
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
            disabled={!isValid || submitting}
          >
            {submitting ? "Changing password..." : "Change password"}
          </button>
        </form>
      </div>
    </div>
  );
}
