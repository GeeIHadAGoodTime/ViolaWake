import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { verifyEmail } from "../api";

type VerificationState = "verifying" | "success" | "error";

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token");
  const [state, setState] = useState<VerificationState>("verifying");
  const [message, setMessage] = useState("Verifying your email...");
  const hasRequestedRef = useRef(false);
  const redirectTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!token) {
      setState("error");
      setMessage("This verification link is missing a token.");
      return;
    }

    if (hasRequestedRef.current) {
      return;
    }
    hasRequestedRef.current = true;

    let cancelled = false;

    void verifyEmail(token)
      .then((response) => {
        if (cancelled) {
          return;
        }
        setState("success");
        setMessage(`${response.message} Redirecting to your dashboard...`);
        redirectTimerRef.current = window.setTimeout(() => {
          navigate("/dashboard", { replace: true });
        }, 2000);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setState("error");
        setMessage(
          err instanceof Error
            ? err.message
            : "We could not verify your email.",
        );
      });

    return () => {
      cancelled = true;
      if (redirectTimerRef.current !== null) {
        window.clearTimeout(redirectTimerRef.current);
      }
    };
  }, [navigate, token]);

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Verify your email</h1>
          <p className="auth-subtitle">
            Confirming your ViolaWake account.
          </p>
        </div>

        <p className={`auth-status ${state}`}>
          {message}
        </p>

        <div className="auth-status-actions">
          {state === "error" && (
            <Link to="/login" className="btn btn-primary btn-full">
              Back to login
            </Link>
          )}
          {state === "success" && (
            <Link to="/dashboard" className="btn btn-primary btn-full">
              Open dashboard now
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
