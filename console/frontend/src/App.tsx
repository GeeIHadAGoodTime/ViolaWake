import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import { AuthProvider } from "./contexts/AuthContext";
import { ToastProvider } from "./contexts/ToastContext";
import ToastContainer from "./components/Toast";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import LandingPage from "./pages/Landing";
import LoginPage from "./pages/Login";
import RegisterPage from "./pages/Register";
import DashboardPage from "./pages/Dashboard";
import RecordPage from "./pages/Record";
import AddSamplesPage from "./pages/AddSamples";
import TrainingStatusPage from "./pages/TrainingStatus";
import ModelPerformancePage from "./pages/ModelPerformance";
import PrivacyPage from "./pages/Privacy";
import TermsPage from "./pages/Terms.tsx";
import VerifyEmailPage from "./pages/VerifyEmail";
import ResetPasswordPage from "./pages/ResetPassword";
import ForgotPasswordPage from "./pages/ForgotPassword";
import ChangePasswordPage from "./pages/ChangePassword";
import TeamsPage from "./pages/Teams";
import TeamDetailPage from "./pages/TeamDetail";
import TeamAcceptPage from "./pages/TeamAccept";
import ContactPage from "./pages/Contact";
import CookieConsent from "./components/CookieConsent";

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <ToastProvider>
            <Layout>
              <Routes>
                {/* Public pages */}
                <Route path="/" element={<LandingPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="/terms" element={<TermsPage />} />
                <Route path="/contact" element={<ContactPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />
                <Route path="/verify-email" element={<VerifyEmailPage />} />
                <Route path="/forgot-password" element={<ForgotPasswordPage />} />
                <Route path="/reset-password" element={<ResetPasswordPage />} />

                {/* Protected pages */}
                <Route
                  path="/dashboard"
                  element={
                    <ProtectedRoute>
                      <DashboardPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/record"
                  element={
                    <ProtectedRoute>
                      <RecordPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/record/:wakeWord/add"
                  element={
                    <ProtectedRoute>
                      <AddSamplesPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/training/:jobId"
                  element={
                    <ProtectedRoute>
                      <TrainingStatusPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/model/:modelId/performance"
                  element={
                    <ProtectedRoute>
                      <ModelPerformancePage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/account/password"
                  element={
                    <ProtectedRoute>
                      <ChangePasswordPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/teams"
                  element={
                    <ProtectedRoute>
                      <TeamsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/teams/accept"
                  element={
                    <ProtectedRoute>
                      <TeamAcceptPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/teams/:teamId"
                  element={
                    <ProtectedRoute>
                      <TeamDetailPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="*"
                  element={<Navigate to="/" replace />}
                />
              </Routes>
            </Layout>
            <ToastContainer />
            <CookieConsent />
          </ToastProvider>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
