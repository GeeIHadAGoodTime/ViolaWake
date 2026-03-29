import { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import type { Team, TeamMemberRole, Model } from "../types";
import {
  getTeam,
  inviteMember,
  removeTeamMember,
  shareModel,
  listTeamModels,
  getModels,
  deleteTeam,
} from "../api";
import { useAuth } from "../contexts/AuthContext";
import "./TeamDetail.css";

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
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

export default function TeamDetailPage() {
  const { teamId } = useParams<{ teamId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();

  const [team, setTeam] = useState<Team | null>(null);
  const [teamModels, setTeamModels] = useState<Model[]>([]);
  const [myModels, setMyModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Invite state
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<TeamMemberRole>("member");
  const [inviting, setInviting] = useState(false);
  const [inviteResult, setInviteResult] = useState<string | null>(null);
  const [inviteError, setInviteError] = useState<string | null>(null);

  // Share model state
  const [shareModelId, setShareModelId] = useState("");
  const [sharing, setSharing] = useState(false);
  const [shareError, setShareError] = useState<string | null>(null);

  // Delete state
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const numericTeamId = Number(teamId);

  const loadTeam = useCallback(async () => {
    if (!teamId || Number.isNaN(numericTeamId)) return;
    setLoading(true);
    try {
      const [teamData, models, userModels] = await Promise.all([
        getTeam(numericTeamId),
        listTeamModels(numericTeamId),
        getModels(),
      ]);
      setTeam(teamData);
      setTeamModels(models);
      setMyModels(userModels);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load team",
      );
    }
    setLoading(false);
  }, [teamId, numericTeamId]);

  useEffect(() => {
    loadTeam();
  }, [loadTeam]);

  const isOwner = team !== null && user !== null && team.owner_id === user.id;
  const currentMember = team?.members.find((m) => m.user_id === user?.id);
  const canManage =
    currentMember?.role === "owner" || currentMember?.role === "admin";

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    if (!inviteEmail.trim()) return;

    setInviting(true);
    setInviteError(null);
    setInviteResult(null);
    try {
      const resp = await inviteMember(numericTeamId, inviteEmail.trim(), inviteRole);
      setInviteResult(resp.message);
      setInviteEmail("");
    } catch (err) {
      setInviteError(
        err instanceof Error ? err.message : "Failed to send invite",
      );
    }
    setInviting(false);
  }

  async function handleRemove(userId: number) {
    if (!window.confirm("Remove this member from the team?")) return;
    try {
      await removeTeamMember(numericTeamId, userId);
      await loadTeam();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to remove member");
    }
  }

  async function handleShare(e: React.FormEvent) {
    e.preventDefault();
    const modelId = Number(shareModelId);
    if (!modelId) return;

    setSharing(true);
    setShareError(null);
    try {
      await shareModel(numericTeamId, modelId);
      setShareModelId("");
      await loadTeam();
    } catch (err) {
      setShareError(
        err instanceof Error ? err.message : "Failed to share model",
      );
    }
    setSharing(false);
  }

  async function handleDelete() {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setDeleting(true);
    try {
      await deleteTeam(numericTeamId);
      navigate("/teams", { replace: true });
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete team");
      setDeleting(false);
      setConfirmDelete(false);
    }
  }

  // Models the user owns that haven't been shared to this team yet
  const unsharableModels = myModels.filter(
    (m) => !teamModels.some((tm) => tm.id === m.id),
  );

  if (loading) {
    return (
      <div className="team-detail-page">
        <div className="team-detail-loading">
          <div className="spinner" />
          <p>Loading team...</p>
        </div>
      </div>
    );
  }

  if (error || !team) {
    return (
      <div className="team-detail-page">
        <div className="team-detail-error">
          <p>{error || "Team not found"}</p>
          <button className="btn btn-ghost" onClick={loadTeam}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="team-detail-page">
      <div className="team-detail-header">
        <div>
          <Link to="/teams" className="team-detail-back">
            &larr; All Teams
          </Link>
          <h1 className="page-title">{team.name}</h1>
          <p className="page-subtitle">
            Created {formatDate(team.created_at)}
          </p>
        </div>
        <div className="team-detail-header-actions">
          {isOwner && (
            <button
              className="btn-danger"
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting
                ? "Deleting..."
                : confirmDelete
                  ? "Confirm Delete"
                  : "Delete Team"}
            </button>
          )}
        </div>
      </div>

      <div className="team-detail-grid">
        {/* ── Members Section ─────────────────────────────────────────── */}
        <section className="team-section">
          <div className="team-section-header">
            <div>
              <h2>Members</h2>
              <p>People with access to this team.</p>
            </div>
            <span className="team-member-count">
              {team.members.length}
            </span>
          </div>

          {canManage && (
            <form className="team-invite-form" onSubmit={handleInvite}>
              <div className="team-invite-fields">
                <div className="form-group" style={{ flex: 2 }}>
                  <label htmlFor="invite-email">Email address</label>
                  <input
                    id="invite-email"
                    type="email"
                    placeholder="colleague@example.com"
                    value={inviteEmail}
                    onChange={(e) => setInviteEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group" style={{ flex: 1 }}>
                  <label htmlFor="invite-role">Role</label>
                  <select
                    id="invite-role"
                    value={inviteRole}
                    onChange={(e) =>
                      setInviteRole(e.target.value as TeamMemberRole)
                    }
                  >
                    <option value="member">Member</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={inviting || !inviteEmail.trim()}
                >
                  {inviting ? "Inviting..." : "Invite"}
                </button>
              </div>

              {inviteResult && (
                <div className="team-invite-token">
                  <strong>Invite sent</strong>
                  {inviteResult}
                </div>
              )}
              {inviteError && (
                <p style={{ color: "var(--error)", marginTop: "0.5rem", fontSize: "0.85rem" }}>
                  {inviteError}
                </p>
              )}
            </form>
          )}

          {team.members.length > 0 ? (
            <ul className="team-member-list">
              {team.members.map((member) => (
                <li key={member.user_id} className="team-member-item">
                  <div className="team-member-info">
                    <div className="team-member-name">{member.name}</div>
                    <div className="team-member-email">{member.email}</div>
                  </div>
                  <div className="team-member-actions">
                    <span
                      className={`team-role-badge role-${member.role}`}
                    >
                      {member.role}
                    </span>
                    {canManage &&
                      member.role !== "owner" &&
                      member.user_id !== user?.id && (
                        <button
                          className="btn-remove"
                          onClick={() => handleRemove(member.user_id)}
                        >
                          Remove
                        </button>
                      )}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="team-empty-list">No members yet.</p>
          )}
        </section>

        {/* ── Shared Models Section ───────────────────────────────────── */}
        <section className="team-section">
          <div className="team-section-header">
            <div>
              <h2>Shared Models</h2>
              <p>Wake word models available to all team members.</p>
            </div>
            <span className="team-member-count">
              {teamModels.length}
            </span>
          </div>

          {canManage && unsharableModels.length > 0 && (
            <form className="team-share-form" onSubmit={handleShare}>
              <div className="team-share-fields">
                <div className="form-group">
                  <label htmlFor="share-model">Share a model</label>
                  <select
                    id="share-model"
                    value={shareModelId}
                    onChange={(e) => setShareModelId(e.target.value)}
                    required
                    style={{
                      width: "100%",
                      padding: "0.6rem 0.875rem",
                      background: "var(--bg-primary)",
                      border: "1px solid var(--border)",
                      borderRadius: "8px",
                      color: "var(--text-primary)",
                      fontSize: "0.9rem",
                    }}
                  >
                    <option value="">Select a model...</option>
                    {unsharableModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.wake_word} (d&prime; {m.d_prime?.toFixed(2) ?? "N/A"})
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={sharing || !shareModelId}
                >
                  {sharing ? "Sharing..." : "Share"}
                </button>
              </div>
              {shareError && (
                <p style={{ color: "var(--error)", marginTop: "0.5rem", fontSize: "0.85rem" }}>
                  {shareError}
                </p>
              )}
            </form>
          )}

          {teamModels.length > 0 ? (
            <ul className="team-model-list">
              {teamModels.map((model) => (
                <li key={model.id} className="team-model-item">
                  <div>
                    <div className="team-model-name">{model.wake_word}</div>
                    <div className="team-model-meta">
                      {formatBytes(model.size_bytes)} &middot; {formatDate(model.created_at)}
                    </div>
                  </div>
                  {model.d_prime != null && (
                    <span className="team-model-dprime">
                      d&prime; {model.d_prime.toFixed(2)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="team-empty-list">
              No models shared yet.
              {canManage && unsharableModels.length > 0
                ? " Use the form above to share one."
                : ""}
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
