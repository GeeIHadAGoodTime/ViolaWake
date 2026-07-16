import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ApiError, getModelConfig, getModelPerformance } from "../api";
import type {
  ModelConfig,
  ModelPerformanceResponse,
  QualityGrade,
} from "../types";
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

const GRADE_LABELS: Record<QualityGrade, string> = {
  A: "Excellent",
  B: "Good",
  C: "Acceptable",
  F: "Failed",
};

function normalizeQualityGrade(value: unknown): QualityGrade | null {
  if (typeof value !== "string") {
    return null;
  }

  const grade = value.trim().toUpperCase();
  if (grade === "A" || grade === "B" || grade === "C" || grade === "F") {
    return grade;
  }

  return null;
}

function getConfigQualityGrade(config: ModelConfig | null): QualityGrade | null {
  if (!config) {
    return null;
  }

  return (
    normalizeQualityGrade(config.training_config.quality_grade) ??
    normalizeQualityGrade(config.training_config.quality_gate?.grade)
  );
}

function getGradePill(grade: QualityGrade | null): {
  label: string;
  className: string;
} {
  if (grade === null) {
    return { label: "Grade unavailable", className: "grade-pill-unknown" };
  }

  return {
    label: `Grade ${grade} - ${GRADE_LABELS[grade]}`,
    className: `grade-pill-${grade.toLowerCase()}`,
  };
}

function formatMetric(value: number | null, digits = 2): string {
  if (value === null || Number.isNaN(value)) {
    return "Unavailable";
  }
  return value.toFixed(digits);
}

function formatFarPerHour(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Unavailable";
  }

  const digits = value < 0.01 ? 3 : 2;
  return `~${value.toFixed(digits)}/hr`;
}

function formatRecall(frr: number | null | undefined): string {
  if (frr === null || frr === undefined || Number.isNaN(frr)) {
    return "Unavailable";
  }

  const recall = Math.min(100, Math.max(0, (1 - frr) * 100));
  return `${recall.toFixed(recall === Math.round(recall) ? 0 : 1)}%`;
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
  config: ModelConfig | null,
): string[] {
  const recommendations: string[] = [];
  const grade = getConfigQualityGrade(config);
  const farPerHour = config?.far_per_hour ?? null;
  const hasFar = farPerHour !== null && !Number.isNaN(farPerHour);
  const farText = formatFarPerHour(farPerHour);

  if (grade === null) {
    recommendations.push(
      "SDK quality grade is unavailable. Train a fresh model run to capture the current quality gate.",
    );
  } else if (grade === "A") {
    recommendations.push(
      hasFar
        ? `Grade A passed the quality gate. Validate the ${farText} false-alarm rate in your target environment before broad deployment.`
        : "Grade A passed the quality gate. Run a fresh evaluation before comparing this model to vendor false-alarm KPIs.",
    );
  } else if (grade === "B") {
    recommendations.push(
      hasFar
        ? `Grade B is usable for trials. Add more representative samples if the ${farText} false-alarm rate is too high for the product surface.`
        : "Grade B is usable for trials. Capture false-alarm rate before production deployment.",
    );
  } else if (grade === "C") {
    recommendations.push(
      hasFar
        ? `Grade C is acceptable for testing. Add samples and retrain before production if false alarms remain near ${farText}.`
        : "Grade C is acceptable for testing. Add samples and capture false-alarm rate before production deployment.",
    );
  } else {
    recommendations.push(
      `Grade F: on no-wake audio (silence, speech, or similar-sounding words) the model scored at or above the detection threshold, so it would trigger on the wrong sound and was not saved. This is usually run-to-run training variance, so training again with the same recordings often passes; if it keeps failing, add a few more clear recordings of your wake word.`,
    );
  }

  if (!hasFar) {
    recommendations.push(
      "False-alarm rate is unavailable. Use a fresh evaluation run before comparing this model to wake-word vendor KPIs.",
    );
  } else if (farPerHour > 2) {
    recommendations.push(
      `False alarms are elevated at ${farText}. Add background and confusable negatives or raise the threshold carefully.`,
    );
  } else {
    recommendations.push(
      `False alarms are currently ${farText}. Keep recall checks in the loop when adjusting thresholds.`,
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
  const [config, setConfig] = useState<ModelConfig | null>(null);
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
        const [performanceResult, configResult] = await Promise.allSettled([
          getModelPerformance(parsedModelId),
          getModelConfig(parsedModelId),
        ]);

        if (performanceResult.status === "rejected") {
          throw performanceResult.reason;
        }

        if (!cancelled) {
          setPerformance(performanceResult.value);
          setConfig(
            configResult.status === "fulfilled" ? configResult.value : null,
          );
          setError(null);
          setNotFound(false);
        }
      } catch (err) {
        if (cancelled) return;

        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          setError(null);
          setPerformance(null);
          setConfig(null);
        } else {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to load model performance",
          );
          setNotFound(false);
          setPerformance(null);
          setConfig(null);
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

  const grade = useMemo(
    () => getConfigQualityGrade(config),
    [config],
  );

  const gradePill = useMemo(
    () => getGradePill(grade),
    [grade],
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
    () => (performance ? getRecommendations(performance, config) : []),
    [config, performance],
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
            Review model grade, false-alarm rate, recall, and deployment guidance.
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
          <section className="model-performance-card model-performance-hero-card">
            <div className="model-performance-card-header">
              <div>
                <h2>Deployment Summary</h2>
                <p>Vendor-facing wake-word metrics from the latest evaluation data.</p>
              </div>
              <span className={`grade-pill ${gradePill.className}`}>
                {gradePill.label}
              </span>
            </div>

            <div className="model-performance-stats">
              <div className="performance-stat">
                <span className="performance-stat-label">False alarms/hr</span>
                <strong className="performance-stat-value">
                  {formatFarPerHour(config?.far_per_hour)}
                </strong>
                <p className="performance-stat-help">
                  Expected false accepts per hour.
                </p>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">Recall</span>
                <strong className="performance-stat-value">
                  {formatRecall(config?.frr)}
                </strong>
                <p className="performance-stat-help">
                  Estimated true wake-word acceptance rate.
                </p>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">File size</span>
                <strong className="performance-stat-value">
                  {formatBytes(performance.file_size)}
                </strong>
                <p className="performance-stat-help">
                  Downloadable ONNX wake head.
                </p>
              </div>
              <div className="performance-stat">
                <span className="performance-stat-label">Training date</span>
                <strong className="performance-stat-value">
                  {formatDate(performance.created_at)}
                </strong>
              </div>
            </div>
          </section>

          <details className="model-performance-card advanced-metrics-card">
            <summary className="advanced-metrics-summary">
              <span>
                <strong>Advanced metrics</strong>
                <small>d-prime, threshold, and score distribution</small>
              </span>
              <span aria-hidden="true">+</span>
            </summary>

            <div className="advanced-metrics-content">
              <div className="model-performance-stats advanced-metrics-stats">
                <div className="performance-stat">
                  <span className="performance-stat-label">d&prime; (d-prime)</span>
                  <strong className="performance-stat-value">
                    {formatMetric(performance.d_prime)}
                  </strong>
                  <p className="performance-stat-help">
                    Signal-detection separation, kept for technical debugging.
                  </p>
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
              </div>

              <div className="advanced-distribution">
                <h3>Score Distribution</h3>
                <p>ASCII histogram of positive versus negative evaluation scores.</p>

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
              </div>
            </div>
          </details>

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
