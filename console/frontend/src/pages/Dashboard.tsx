import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Model, SubscriptionResponse } from "../types";
import { getModels, getSubscription } from "../api";
import ModelCard from "../components/ModelCard";

export default function DashboardPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [subscription, setSubscription] = useState<SubscriptionResponse | null>(
    null,
  );
  const navigate = useNavigate();

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getModels();
      setModels(data);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load models",
      );
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Load subscription so we can show an upgrade CTA on free tier. Failure
  // here is non-fatal — the dashboard still works without the banner.
  useEffect(() => {
    let cancelled = false;
    getSubscription()
      .then((data) => {
        if (!cancelled) setSubscription(data);
      })
      .catch(() => {
        // billing not configured / not available — silently skip the banner
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const tier = subscription?.tier?.toLowerCase() ?? "free";
  const usage = subscription?.usage;
  const showUpgrade = tier === "free";

  const handleModelDeleted = useCallback((modelId: number) => {
    setModels((prev) => prev.filter((m) => m.id !== modelId));
  }, []);

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">Your Wake Word Models</h1>
          <p className="page-subtitle">
            {models.length > 0
              ? `${models.length} model${models.length !== 1 ? "s" : ""} trained`
              : "Train your first custom wake word"}
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => navigate("/record")}
        >
          + Train New Model
        </button>
      </div>

      {showUpgrade && (
        <div className="dashboard-upgrade-banner" role="status">
          <div>
            <strong>You&apos;re on the Free plan.</strong> 3 models per
            month
            {usage
              ? ` — ${usage.models_used} / ${usage.models_limit ?? "∞"} used this period.`
              : "."}{" "}
            Upgrade for 20+ models, priority training, and team features.
          </div>
          <div className="dashboard-upgrade-actions">
            <Link to="/pricing" className="btn btn-primary">
              See plans
            </Link>
            <Link to="/billing" className="btn btn-ghost">
              Manage billing
            </Link>
          </div>
        </div>
      )}

      {loading && (
        <div className="dashboard-loading">
          <div className="spinner" />
          <p>Loading your models...</p>
        </div>
      )}

      {error && (
        <div className="dashboard-error">
          <p>{error}</p>
          <button className="btn btn-ghost" onClick={loadModels}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && models.length === 0 && (
        <div className="dashboard-empty">
          <div className="empty-icon">🎤</div>
          <h2>No models yet</h2>
          <p>
            Record your first wake word and train a custom model.
            It only takes a few minutes.
          </p>
          <button
            className="btn btn-primary"
            onClick={() => navigate("/record")}
          >
            Record Your First Wake Word
          </button>
        </div>
      )}

      {!loading && !error && models.length > 0 && (
        <div className="model-grid" aria-live="polite">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              onDeleted={handleModelDeleted}
            />
          ))}
        </div>
      )}
    </div>
  );
}
