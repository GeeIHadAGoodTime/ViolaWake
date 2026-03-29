import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import type { TeamListItem } from "../types";
import { listTeams, createTeam } from "../api";
import "./Teams.css";

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

export default function TeamsPage() {
  const [teams, setTeams] = useState<TeamListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const loadTeams = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listTeams();
      setTeams(data);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load teams",
      );
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadTeams();
  }, [loadTeams]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;

    setCreating(true);
    setCreateError(null);
    try {
      await createTeam(newName.trim());
      setNewName("");
      setShowCreate(false);
      await loadTeams();
    } catch (err) {
      setCreateError(
        err instanceof Error ? err.message : "Failed to create team",
      );
    }
    setCreating(false);
  }

  return (
    <div className="teams-page">
      <div className="teams-header">
        <div>
          <h1 className="page-title">Teams</h1>
          <p className="page-subtitle">
            {teams.length > 0
              ? `${teams.length} team${teams.length !== 1 ? "s" : ""}`
              : "Collaborate on wake word models"}
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => setShowCreate(!showCreate)}
        >
          {showCreate ? "Cancel" : "+ Create Team"}
        </button>
      </div>

      {showCreate && (
        <form className="teams-create-form" onSubmit={handleCreate}>
          <h2>Create a new team</h2>
          <div className="teams-create-fields">
            <div className="form-group">
              <label htmlFor="team-name">Team name</label>
              <input
                id="team-name"
                type="text"
                placeholder="e.g. My Company"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                maxLength={255}
                required
                autoFocus
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={creating || !newName.trim()}
            >
              {creating ? "Creating..." : "Create"}
            </button>
          </div>
          {createError && (
            <p style={{ color: "var(--error)", marginTop: "0.75rem", fontSize: "0.9rem" }}>
              {createError}
            </p>
          )}
        </form>
      )}

      {loading && (
        <div className="teams-loading">
          <div className="spinner" />
          <p>Loading your teams...</p>
        </div>
      )}

      {error && (
        <div className="teams-error">
          <p>{error}</p>
          <button className="btn btn-ghost" onClick={loadTeams}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && teams.length === 0 && (
        <div className="teams-empty">
          <h2>No teams yet</h2>
          <p>
            Create a team to share wake word models and collaborate with others.
          </p>
          <button
            className="btn btn-primary"
            onClick={() => setShowCreate(true)}
          >
            Create Your First Team
          </button>
        </div>
      )}

      {!loading && !error && teams.length > 0 && (
        <div className="teams-grid" aria-live="polite">
          {teams.map((team) => (
            <Link
              key={team.id}
              to={`/teams/${team.id}`}
              className="team-card"
            >
              <h3>{team.name}</h3>
              <div className="team-card-meta">
                <span>{team.member_count} member{team.member_count !== 1 ? "s" : ""}</span>
                <span>Created {formatDate(team.created_at)}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
