import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { acceptTeamInvite } from "../api";
import "./TeamDetail.css";

export default function TeamAcceptPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(true);
  const token = searchParams.get("token") ?? "";

  useEffect(() => {
    if (!token) {
      setError("Invite token is missing.");
      setAccepting(false);
      return;
    }

    let cancelled = false;

    async function acceptInvite() {
      try {
        const team = await acceptTeamInvite(token);
        if (!cancelled) {
          navigate(`/teams/${team.id}`, { replace: true });
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to accept invite",
          );
          setAccepting(false);
        }
      }
    }

    acceptInvite();

    return () => {
      cancelled = true;
    };
  }, [navigate, token]);

  return (
    <div className="team-detail-page">
      <section className="team-section team-accept-section">
        <div className="team-section-header">
          <div>
            <h1 className="page-title">Team Invite</h1>
            <p>
              {accepting
                ? "Accepting your invite..."
                : "This invite could not be accepted."}
            </p>
          </div>
        </div>

        {accepting && (
          <div className="team-detail-loading">
            <div className="spinner" />
          </div>
        )}

        {error && (
          <div className="team-alert team-alert-error" role="alert">
            {error}
          </div>
        )}

        {!accepting && (
          <Link to="/teams" className="btn btn-primary">
            Back to Teams
          </Link>
        )}
      </section>
    </div>
  );
}
