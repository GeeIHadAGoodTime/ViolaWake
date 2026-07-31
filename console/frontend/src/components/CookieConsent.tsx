import CookieConsent from "react-cookie-consent";

export default function CookieBanner() {
  return (
    <CookieConsent
      location="bottom"
      buttonText="Accept"
      style={{ background: "#1a1a2e", borderTop: "1px solid #333" }}
      buttonStyle={{ background: "#5b4bd8", color: "#fff", borderRadius: "6px", padding: "0.5rem 1.25rem", fontWeight: 600 }}
      expires={365}
    >
      We use cookies for authentication.{" "}
      <a href="/privacy" style={{ color: "#a29bfe", textDecoration: "underline", textUnderlineOffset: "0.15em" }}>Privacy Policy</a>
    </CookieConsent>
  );
}
