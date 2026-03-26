import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { createBillingPortal, getSubscription } from "../api";
import { useAuth } from "../contexts/AuthContext";
import type { SubscriptionResponse } from "../types";
import "./Billing.css";

function formatLabel(value: string): string {
  return value
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

export default function BillingPage() {
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const [subscription, setSubscription] = useState<SubscriptionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [portalLoading, setPortalLoading] = useState(false);
  const [portalError, setPortalError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadSubscription() {
      setLoading(true);
      try {
        const data = await getSubscription();
        if (!cancelled) {
          setSubscription(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to load billing details",
          );
        }
      }
      if (!cancelled) {
        setLoading(false);
      }
    }

    loadSubscription();

    return () => {
      cancelled = true;
    };
  }, []);

  const activated = searchParams.has("session_id");

  const tier = subscription?.tier ?? "free";
  const usage = subscription?.usage;
  const usageLabel = useMemo(() => {
    if (!usage) return "0 / 0";
    if (usage.models_limit === null) {
      return `${usage.models_used} / Unlimited`;
    }
    return `${usage.models_used} / ${usage.models_limit}`;
  }, [usage]);

  const usagePercent = useMemo(() => {
    if (!usage) return 0;
    if (usage.models_limit === null || usage.models_limit <= 0) return 100;
    return Math.min((usage.models_used / usage.models_limit) * 100, 100);
  }, [usage]);

  async function handleManageSubscription() {
    setPortalLoading(true);
    setPortalError(null);
    try {
      const { url } = await createBillingPortal();
      window.location.href = url;
    } catch (err) {
      setPortalError(
        err instanceof Error ? err.message : "Failed to open billing portal",
      );
      setPortalLoading(false);
    }
  }

  return (
    <div className="billing-page">
      <div className="billing-header">
        <div>
          <h1 className="page-title">Billing</h1>
          <p className="page-subtitle">
            Manage your Console plan{user ? ` for ${user.email}` : ""}.
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleManageSubscription}
          disabled={portalLoading || loading || !!error}
        >
          {portalLoading ? "Opening Portal..." : "Manage Subscription"}
        </button>
      </div>

      {activated && (
        <div className="billing-alert billing-alert-success" role="status">
          Subscription activated!
        </div>
      )}

      {portalError && (
        <div className="billing-alert billing-alert-error" role="alert">
          {portalError}
        </div>
      )}

      {loading && (
        <div className="billing-loading">
          <div className="spinner" />
          <p>Loading billing details...</p>
        </div>
      )}

      {error && (
        <div className="billing-error">
          <p>{error}</p>
          <button
            className="btn btn-ghost"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && subscription && (
        <div className="billing-grid">
          <section className="billing-card">
            <div className="billing-card-header">
              <div>
                <h2>Current plan</h2>
                <p>Your active subscription details and renewal date.</p>
              </div>
              <span className="billing-tier-badge">{formatLabel(tier)}</span>
            </div>

            <div className="billing-info-grid">
              <div className="billing-info-item">
                <span className="billing-info-label">Tier</span>
                <strong>{formatLabel(subscription.tier)}</strong>
              </div>
              <div className="billing-info-item">
                <span className="billing-info-label">Status</span>
                <strong>{formatLabel(subscription.status)}</strong>
              </div>
              <div className="billing-info-item">
                <span className="billing-info-label">Current period ends</span>
                <strong>{formatDate(subscription.current_period_end)}</strong>
              </div>
            </div>

            {tier.toLowerCase() === "free" && (
              <div className="billing-upgrade">
                <span>Need more monthly training capacity?</span>
                <Link to="/pricing">Upgrade</Link>
              </div>
            )}
          </section>

          <section className="billing-card">
            <div className="billing-card-header">
              <div>
                <h2>Usage</h2>
                <p>Tracked model training usage for the current billing period.</p>
              </div>
              <span className="billing-usage-value">{usageLabel}</span>
            </div>

            <div
              className="billing-usage-bar"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={usage?.models_limit ?? usage?.models_used ?? 0}
              aria-valuenow={usage?.models_used ?? 0}
              aria-label="Models used this billing period"
            >
              <div
                className="billing-usage-fill"
                style={{ width: `${usagePercent}%` }}
              />
            </div>

            <div className="billing-usage-meta">
              <div className="billing-info-item">
                <span className="billing-info-label">Period start</span>
                <strong>{usage ? formatDate(usage.period_start) : "Unavailable"}</strong>
              </div>
              <div className="billing-info-item">
                <span className="billing-info-label">Period end</span>
                <strong>{usage ? formatDate(usage.period_end) : "Unavailable"}</strong>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
