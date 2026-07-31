import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Model, SubscriptionResponse } from "../types";
import { getModels, getSubscription } from "../api";
import ModelCard from "../components/ModelCard";
import TrainingPauseBanner from "../components/TrainingPauseBanner";

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

  // usage.models_used and `models` come from two different sources that can
  // legitimately disagree. usage.models_used counts training jobs SUBMITTED
  // this billing period (backend: routes/billing.py record_usage(), fired
  // from routes/jobs.py submit_training_job() at submit time — every
  // submission burns real training compute regardless of outcome). `models`
  // only lists jobs that finished successfully (routes/models.py
  // list_models() reads the TrainedModel table, populated only on a
  // completed, quality-gate-passing run). So a user whose attempts failed
  // the quality gate (or are still training) can have used their whole
  // period's quota while the model grid is genuinely empty — that's not a
  // data bug, but showing both facts with no connective copy reads as a
  // flat contradiction ("3 of 3 used" next to "No models yet", #3267/#3470).
  // Surface the same two numbers honestly instead of implying they're the
  // same count.
  const attemptsUsedWithNoModels =
    !loading && models.length === 0 && (usage?.models_used ?? 0) > 0;

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
              : attemptsUsedWithNoModels
                ? `${usage!.models_used} training attempt${usage!.models_used !== 1 ? "s" : ""} this period, none completed yet`
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

      <TrainingPauseBanner onResumed={loadModels} />

      {showUpgrade && (
        <div className="dashboard-upgrade-banner" role="status">
          <div>
            <strong>You&apos;re on the Free plan.</strong> 3 training
            attempts per month
            {usage
              ? ` — ${usage.models_used} / ${usage.models_limit ?? "∞"} used this period.`
              : "."}{" "}
            Every submitted training run counts toward this, whether or
            not it finishes successfully. Upgrade for 20+ attempts,
            priority training, and team features.
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
          {attemptsUsedWithNoModels ? (
            <p className="dashboard-empty-note">
              You&apos;ve used {usage!.models_used} of{" "}
              {usage!.models_limit ?? "∞"} training attempts this period,
              but none finished with a working model yet — usually because
              a run is still training or didn&apos;t pass the quality
              check. Open a training run&apos;s progress page to see what
              happened, or try again with a few extra recordings if a
              recent run failed.
            </p>
          ) : (
            <p>
              Record your first wake word and train a custom model.
              It only takes a few minutes.
            </p>
          )}
          <button
            className="btn btn-primary"
            onClick={() => navigate("/record")}
          >
            {attemptsUsedWithNoModels
              ? "Record More Samples"
              : "Record Your First Wake Word"}
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
