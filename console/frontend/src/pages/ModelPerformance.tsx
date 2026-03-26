import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ApiError, getModelPerformance } from "../api";
import type { ModelPerformanceResponse } from "../types";
import "./ModelPerformance.css";

const DISTRIBUTION_BAR_WIDTH = 18;
const DISTRIBUTION_BUCKETS = 8;

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getQualityState(cohenD: number | null): {
  label: string;
  className: string;
} {
  if (cohenD === null) {
    return { label: "Unknown", className: "quality-unknown" };
  }
  if (cohenD < 5) {
    return { label: "Needs Work", className: "quality-low" };
  }
  if (cohenD <= 10) {
    return { label: "Good", className: "quality-medium" };
  }
  return { label: "Excellent", className: "quality-high" };
}

function formatMetric(value: number | null, digits = 2): string {
  if (value === null || Number.isNaN(value)) {
    return "Unavailable";
  }
  return value.toFixed(digits);
}

function buildBar(count: number, maxCount: number, fill: string): string {
  if (count <= 0 || maxCount <= 0) {
    return "";
  }
  const scaled = Math.max(
    1,
    Math.round((count / maxCount) * DISTRIBUTION_BAR_WIDTH),
  );
  return fill.repeat(scaled);
}

function buildDistributionChart(
  positiveScores: number[],
  negativeScores: number[],
): string {
  const allScores = [...positiveScores, ...negativeScores];
  if (allScores.length === 0) {
    return "";
  }

  let minScore = Math.min(...allScores);
  let maxScore = Math.max(...allScores);
  if (minScore === maxScore) {
    minScore -= 0.5;
    maxScore += 0.5;
  }

  const step = (maxScore - minScore) / DISTRIBUTION_BUCKETS;
  const positiveBuckets = Array.from({ length: DISTRIBUTION_BUCKETS }, () => 0);
  const negativeBuckets = Array.from({ length: DISTRIBUTION_BUCKETS }, () => 0);

  function bucketIndex(score: number): number {
    if (step <= 0) return 0;
    const rawIndex = Math.floor((score - minScore) / step);
    return Math.min(DISTRIBUTION_BUCKETS - 1, Math.max(0, rawIndex));
  }

  positiveScores.forEach((score) => {
    positiveBuckets[bucketIndex(score)] += 1;
  });
  negativeScores.forEach((score) => {
    negativeBuckets[bucketIndex(score)] += 1;
  });

  const maxBucketCount = Math.max(
    ...positiveBuckets,
    ...negativeBuckets,
    1,
  );

  const lines = [
    "Range           | Positive              | Negative",
    "------------------------------------------------------",
  ];

  for (let index = 0; index < DISTRIBUTION_BUCKETS; index += 1) {
    const start = minScore + step * index;
    const end = minScore + step * (index + 1);
    const label = `${start.toFixed(2)} to ${end.toFixed(2)}`.padEnd(15, " ");
    const positiveBar = buildBar(
      positiveBuckets[index],
      maxBucketCount,
      "#",
    ).padEnd(DISTRIBUTION_BAR_WIDTH, " ");
    const negativeBar = buildBar(
      negativeBuckets[index],
      maxBucketCount,
      "=",
    ).padEnd(DISTRIBUTION_BAR_WIDTH, " ");

    lines.push(
      `${label} | ${positiveBar} ${String(positiveBuckets[index]).padStart(2, " ")} | ${negativeBar} ${String(negativeBuckets[index]).padStart(2, " ")}`,
    );
  }

  return lines.join("\n");
}

function getRecommendations(
  performance: ModelPerformanceResponse,
): string[] {
  const recommendations: string[] = [];

  if (performance.cohen_d === null) {
    recommendations.push(
      "Performance score unavailable. Train a fresh model run to capture evaluation metrics.",
    );
  } else if (performance.cohen_d < 5) {
    recommendations.push(
      "Consider recording more samples or improving audio quality.",
    );
  } else if (performance.cohen_d <= 10) {
    recommendations.push(
      "Good model. Consider testing with real-world noise.",
    );
  } else {
    recommendations.push(
      "Excellent separation. Ready for production.",
    );
  }

  if (performance.threshold !== null && performance.threshold < 0.35) {
    recommendations.push(
      "Threshold is very low. Raise it slightly if you see false activations.",
    );
  }

  if (performance.threshold !== null && performance.threshold > 0.75) {
    recommendations.push(
      "Threshold is very high. Lower it slightly if the wake word feels hard to trigger.",
    );
  }

  return recommendations;
}

export default function ModelPerformancePage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  const [performance, setPerformance] =
    useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    const parsedModelId = Number(modelId);
    if (!Number.isInteger(parsedModelId) || parsedModelId <= 0) {
      setNotFound(true);
      setLoading(false);
      return;
    }

    let cancelled = false;

    async function loadPerformance() {
      setLoading(true);
      try {
        const data = await getModelPerformance(parsedModelId);
        if (!cancelled) {
          setPerformance(data);
          setError(null);
          setNotFound(false);
        }
      } catch (err) {
        if (cancelled) return;

        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          setError(null);
        } else {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to load model performance",
          );
          setNotFound(false);
        }
      }

      if (!cancelled) {
        setLoading(false);
      }
    }

    loadPerformance();

    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const quality = useMemo(
    () => getQualityState(performance?.cohen_d ?? null),
    [performance?.cohen_d],
  );

  const chart = useMemo(
    () =>
      performance
        ? buildDistributionChart(
            performance.positive_scores,
            performance.negative_scores,
          )
        : "",
    [performance],
  );

  const recommendations = useMemo(
    () => (performance ? getRecommendations(performance) : []),
    [performance],
  );

  return (
    <div className="model-performance-page">
      <div className="model-performance-header">
        <button
          className="btn btn-ghost"
          onClick={() => navigate("/dashboard")}
        >
          Back to Dashboard
        </button>

        <div>
          <h1 className="page-title">
            {performance ? `${performance.model_name} Performance` : "Model Performance"}
          </h1>
          <p className="page-subtitle">
            Review model quality, score separation, and deployment guidance.
          </p>
        </div>
      </div>

      {loading && (
        <div className="model-performance-state">
          <div className="spinner" />
          <p>Loading model performance...</p>
        </div>
      )}

      {!loading && notFound && (
        <div className="model-performance-state model-performance-card">
          <h2>Model not found</h2>
          <p>
            This model no longer exists or you do not have access to it.
          </p>
          <button
            className="btn btn-primary"
            onClick={() => navigate("/dashboard")}
          >
            Return to Dashboard
          </button>
        </div>
      )}

      {!loading && error && !notFound && (
        <div className="model-performance-state model-performance-card">
          <h2>Could not load performance data</h2>
          <p>{error}</p>
          <div className="model-performance-state-actions">
            <button
              className="btn btn-primary"
              onClick={() => window.location.reload()}
            >
              Retry
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => navigate("/dashboard")}
            >
              Back
            </button>
          </div>
        </div>
      )}

      {!loading && !error && !notFound && performance && (
        <div className="model-performance-grid">
          <section className="model-performance-card">
            <div className="model-performance-card-header">
              <div>
                <h2>Model Quality Summary</h2>
                <p>Key metrics from the latest available evaluation data.</p>
              </div>
              <span className={`quality-pill ${quality.className}`}>
                {quality.label}
              </span>
            </div>

            <div className="model-performance-stats">
              <div className="performance-stat">
                <span className="performance-stat-label">Cohen&apos;s d</span>
                <strong className={`performance-stat-value ${quality.className}`}>
                  {formatMetric(performance.cohen_d)}
                </strong>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">Threshold</span>
                <strong className="performance-stat-value">
                  {formatMetric(performance.threshold, 3)}
                </strong>
                <p className="performance-stat-help">
                  Higher thresholds reduce false accepts but can miss softer activations.
                </p>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">File size</span>
                <strong className="performance-stat-value">
                  {formatBytes(performance.file_size)}
                </strong>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">Training date</span>
                <strong className="performance-stat-value">
                  {formatDate(performance.created_at)}
                </strong>
              </div>
            </div>
          </section>

          <section className="model-performance-card">
            <div className="model-performance-card-header">
              <div>
                <h2>Score Distribution</h2>
                <p>ASCII histogram of positive versus negative evaluation scores.</p>
              </div>
            </div>

            {(performance.positive_scores.length > 0 ||
              performance.negative_scores.length > 0) &&
            chart ? (
              <>
                <div className="distribution-legend">
                  <span>Positive `#`</span>
                  <span>Negative `=`</span>
                </div>
                <pre className="distribution-chart">{chart}</pre>
              </>
            ) : (
              <div className="distribution-empty">
                <p>
                  No stored score distributions were found for this model.
                </p>
                <p className="distribution-empty-subtext">
                  Summary metrics are still shown using the saved model record.
                </p>
              </div>
            )}
          </section>

          <section className="model-performance-card">
            <div className="model-performance-card-header">
              <div>
                <h2>Recommendations</h2>
                <p>Actionable guidance based on the current model metrics.</p>
              </div>
            </div>

            <ul className="recommendation-list">
              {recommendations.map((recommendation) => (
                <li key={recommendation}>{recommendation}</li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}
