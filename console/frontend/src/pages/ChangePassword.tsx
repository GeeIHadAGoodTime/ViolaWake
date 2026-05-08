import { FormEvent, useState } from "react";
import { changePassword, deleteAccount, exportAccount } from "../api";
import { useAuth } from "../contexts/AuthContext";

interface ValidationErrors {
  currentPassword?: string;
  newPassword?: string;
  confirmPassword?: string;
}

export default function ChangePasswordPage() {
  const { logout, user } = useAuth();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteEmail, setDeleteEmail] = useState("");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [deleting, setDeleting] = useState(false);
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

  async function handleExport() {
    setError(null);
    setSuccess(null);
    setExporting(true);
    try {
      const blob = await exportAccount();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "violawake-account-export.zip";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setSuccess("Account export downloaded.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "We could not export your account data.",
      );
    } finally {
      setExporting(false);
    }
  }

  async function handleDeleteAccount(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccess(null);

    if (!user || deleteEmail.trim().toLowerCase() !== user.email.toLowerCase()) {
      setError("Enter your account email address to confirm deletion.");
      return;
    }

    setDeleting(true);
    try {
      await deleteAccount(deletePassword);
      logout();
      window.location.href = "/";
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "We could not delete your account.",
      );
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Account</h1>
          <p className="auth-subtitle">
            Your ViolaWake account details, password, and data controls.
          </p>
        </div>

        {user && (
          <section className="account-info" aria-label="Account information">
            <div className="account-info-row">
              <span className="account-info-label">Email</span>
              <strong>{user.email}</strong>
            </div>
            {user.name && (
              <div className="account-info-row">
                <span className="account-info-label">Name</span>
                <strong>{user.name}</strong>
              </div>
            )}
            <div className="account-info-row">
              <span className="account-info-label">Email verified</span>
              <strong>{user.email_verified ? "Yes" : "No — check your inbox"}</strong>
            </div>
          </section>
        )}

        <div className="auth-section-header">
          <h2 className="auth-section-title">Change password</h2>
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

        <div className="account-actions">
          <div className="account-action-row">
            <div>
              <h2>Export data</h2>
              <p>Download your account profile, recordings, models, jobs, and audit log entries.</p>
            </div>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleExport}
              disabled={exporting}
            >
              {exporting ? "Exporting..." : "Export data"}
            </button>
          </div>

          <div className="account-action-row account-action-danger">
            <div>
              <h2>Delete account</h2>
              <p>Delete account access now and schedule retained artifacts for permanent deletion in 30 days.</p>
            </div>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => setShowDeleteConfirm(true)}
            >
              Delete account
            </button>
          </div>
        </div>
      </div>

      {showDeleteConfirm && (
        <div className="delete-confirm-overlay" role="dialog" aria-modal="true">
          <form className="delete-confirm" onSubmit={handleDeleteAccount}>
            <p className="delete-confirm-text">
              Delete your ViolaWake account?
            </p>
            <p className="delete-confirm-subtext">
              Enter your account email and current password to confirm.
            </p>
            <div className="form-group">
              <label htmlFor="delete-email" className="form-label">
                Account email
              </label>
              <input
                id="delete-email"
                type="email"
                className="form-input"
                value={deleteEmail}
                onChange={(event) => setDeleteEmail(event.target.value)}
                disabled={deleting}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="delete-password" className="form-label">
                Current password
              </label>
              <input
                id="delete-password"
                type="password"
                className="form-input"
                value={deletePassword}
                onChange={(event) => setDeletePassword(event.target.value)}
                disabled={deleting}
                required
              />
            </div>
            <div className="delete-confirm-actions">
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-danger"
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Delete account"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
