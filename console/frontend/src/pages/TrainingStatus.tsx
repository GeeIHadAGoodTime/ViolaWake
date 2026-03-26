import { useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import TrainingProgress from "../components/TrainingProgress";
import { getModelDownloadUrl } from "../api";

export default function TrainingStatusPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [completedModel, setCompletedModel] = useState<{
    modelId: number;
    dPrime: number;
  } | null>(null);

  const jobIdNum = Number(jobId);

  if (!jobId || isNaN(jobIdNum)) {
    return (
      <div className="training-page">
        <div className="training-error-page">
          <h1>Invalid training job</h1>
          <Link to="/dashboard" className="btn btn-primary">
            Go to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  function handleComplete(modelId: number, dPrime: number) {
    setCompletedModel({ modelId, dPrime });
  }

  function handleDownload() {
    if (!completedModel) return;
    const url = getModelDownloadUrl(completedModel.modelId);
    const a = document.createElement("a");
    a.href = url;
    a.download = "model.onnx";
    a.click();
  }

  return (
    <div className="training-page">
      <h1 className="page-title">Training Progress</h1>
      <p className="page-subtitle">Job #{jobId}</p>

      <div className="training-card">
        <TrainingProgress
          jobId={jobIdNum}
          onComplete={handleComplete}
        />
      </div>

      {completedModel && (
        <div className="training-complete-actions">
          <button
            className="btn btn-primary btn-large"
            onClick={handleDownload}
          >
            Download Model (.onnx)
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => navigate("/dashboard")}
          >
            View on Dashboard
          </button>
        </div>
      )}
    </div>
  );
}
