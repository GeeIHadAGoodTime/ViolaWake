import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export default function ProtectedRoute({
  children,
}: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="dashboard-loading">
        <div className="spinner" />
      </div>
    );
  }

  if (!isAuthenticated) {
    const returnPath = location.pathname + location.search;
    const loginUrl =
      returnPath && returnPath !== "/"
        ? `/login?return=${encodeURIComponent(returnPath)}`
        : "/login";
    return <Navigate to={loginUrl} replace />;
  }

  return <>{children}</>;
}
