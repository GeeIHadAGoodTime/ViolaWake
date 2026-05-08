import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { trackEvent } from "./analytics";
import "./styles/global.css";

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

if (
  window.location.pathname === "/billing" &&
  new URLSearchParams(window.location.search).has("session_id")
) {
  trackEvent("checkout_completed");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
