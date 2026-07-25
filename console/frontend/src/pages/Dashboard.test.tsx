import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import DashboardPage from "./Dashboard";
import { getModels, getSubscription } from "../api";
import type { SubscriptionResponse } from "../types";
import { ToastProvider } from "../contexts/ToastContext";

// Issues #3267 / #3470: a real customer's dashboard showed "3 of 3 free
// models used" (the upgrade banner, driven by GET /billing/subscription's
// usage.models_used) at the same time as "No models yet" (the empty state,
// driven by GET /models). Root cause: usage.models_used counts training
// jobs SUBMITTED this period (billing.py record_usage(), called from
// jobs.py submit_training_job() at submit time), while GET /models only
// lists jobs that finished successfully (models.py list_models() reads the
// TrainedModel table). A free-tier user whose training attempts all failed
// the quality gate (a real, frequent training-pipeline outcome — see
// docs/knowledge/conclusions/CL-20260714-d2e7.md) can exhaust their whole
// period's quota with zero completed models — that data shape is not a bug,
// but rendering both numbers with no connective copy reads as a flat
// contradiction. This test locks in that the dashboard explains the gap
// instead of asserting two things that look mutually exclusive.
vi.mock("../api", () => ({
  getModels: vi.fn(),
  getSubscription: vi.fn(),
}));

function renderDashboard() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DashboardPage />
      </ToastProvider>
    </MemoryRouter>,
  );
}

const usedUpFreeTierSubscription: SubscriptionResponse = {
  tier: "free",
  status: "active",
  current_period_end: "2026-08-01T00:00:00Z",
  trial_active: false,
  trial_end: null,
  usage: {
    models_used: 3,
    models_limit: 3,
    period_start: "2026-07-01T00:00:00Z",
    period_end: "2026-08-01T00:00:00Z",
  },
};

describe("Dashboard — quota-used-but-no-models contradiction (#3267, #3470)", () => {
  beforeEach(() => {
    vi.mocked(getModels).mockReset();
    vi.mocked(getSubscription).mockReset();
  });

  it("never claims models_used > 0 while the empty state implies zero attempts were ever made", async () => {
    vi.mocked(getModels).mockResolvedValue([]);
    vi.mocked(getSubscription).mockResolvedValue(usedUpFreeTierSubscription);

    renderDashboard();

    // The banner's own numbers must still be present and correct.
    await waitFor(() => {
      expect(screen.getByText(/3 \/ 3 used this period/i)).toBeInTheDocument();
    });

    // The old bug shape: banner says the quota is used, but the empty-state
    // copy ("Record your first wake word") reads as if nothing was ever
    // tried. That specific copy must NOT appear once models_used > 0 with
    // zero completed models — it's what makes the two facts read as
    // contradictory rather than connected.
    expect(
      screen.queryByText(/record your first wake word and train a custom model/i),
    ).not.toBeInTheDocument();

    // The fix: the empty state must explain that attempts were used but
    // none completed, referencing the SAME number the banner shows, so a
    // reader can connect "3 used" to "0 finished" instead of seeing two
    // disconnected claims.
    expect(
      screen.getByText(/used 3 of 3 training attempts this period/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/no models yet/i)).toBeInTheDocument();

    // The banner itself must describe what it actually counts (submitted
    // training runs), not imply the count tracks completed models.
    expect(screen.getByText(/training attempts per month/i)).toBeInTheDocument();
    expect(screen.queryByText(/^3 models per month/i)).not.toBeInTheDocument();
  });

  it("keeps the plain first-time-user empty state when no attempts have been made yet", async () => {
    vi.mocked(getModels).mockResolvedValue([]);
    vi.mocked(getSubscription).mockResolvedValue({
      ...usedUpFreeTierSubscription,
      usage: { ...usedUpFreeTierSubscription.usage, models_used: 0 },
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/no models yet/i)).toBeInTheDocument();
    });

    // A genuine first-time user (0 attempts, 0 models) keeps the original,
    // simple onboarding copy — the explanatory note is only for the
    // attempts-used-but-empty shape.
    expect(
      screen.getByText(/record your first wake word and train a custom model/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/training attempts this period, but none finished/i),
    ).not.toBeInTheDocument();
  });

  it("shows completed models normally when training has actually succeeded", async () => {
    vi.mocked(getModels).mockResolvedValue([
      {
        id: 1,
        wake_word: "hey nova",
        d_prime: 4.2,
        quality_grade: "A",
        created_at: "2026-07-10T00:00:00Z",
        size_bytes: 1024,
      },
    ]);
    vi.mocked(getSubscription).mockResolvedValue(usedUpFreeTierSubscription);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/1 model trained/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/no models yet/i)).not.toBeInTheDocument();
  });
});
