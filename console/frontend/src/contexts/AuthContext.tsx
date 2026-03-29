import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { User } from "../types";
import * as api from "../api";

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  /** Alias for isLoading -- kept for backward compatibility with pages. */
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function getPostAuthRedirect(search: string): string {
  const params = new URLSearchParams(search);
  const returnPath = params.get("return");

  if (
    !returnPath ||
    !returnPath.startsWith("/") ||
    returnPath.startsWith("//")
  ) {
    return "/dashboard";
  }

  params.delete("return");
  const remainingQuery = params.toString();
  if (!remainingQuery) {
    return returnPath;
  }

  const separator = returnPath.includes("?") ? "&" : "?";
  return `${returnPath}${separator}${remainingQuery}`;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Validate existing token on mount
  useEffect(() => {
    if (!api.isAuthenticated()) {
      setIsLoading(false);
      return;
    }

    api
      .getMe()
      .then((userData) => {
        setUser(userData);
        setIsLoading(false);
      })
      .catch(() => {
        api.logout();
        setUser(null);
        setIsLoading(false);
      });
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const { user: userData } = await api.login(email, password);
        setUser(userData);
        setIsLoading(false);
        navigate(getPostAuthRedirect(location.search), { replace: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Login failed";
        setError(message);
        setIsLoading(false);
      }
    },
    [location.search, navigate],
  );

  const register = useCallback(
    async (email: string, password: string, name: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const { user: userData } = await api.register(email, password, name);
        setUser(userData);
        setIsLoading(false);
        navigate(getPostAuthRedirect(location.search), { replace: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Registration failed";
        setError(message);
        setIsLoading(false);
      }
    },
    [location.search, navigate],
  );

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
    setError(null);
    navigate("/login");
  }, [navigate]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: user !== null,
        isLoading,
        loading: isLoading,
        error,
        login,
        register,
        logout,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
