import { Link } from "react-router-dom";

export default function ContactPage() {
  return (
    <div className="contact-page">
      <div className="contact-header">
        <h1 className="page-title">Contact</h1>
        <p className="page-subtitle">
          We&apos;re a small team. Email lands in a real human inbox.
        </p>
      </div>

      <div className="contact-grid">
        <section className="contact-card">
          <h2>General questions</h2>
          <p>
            Product questions, sales, demos, partnership requests.
          </p>
          <a className="contact-email" href="mailto:hello@violawake.com">
            hello@violawake.com
          </a>
        </section>

        <section className="contact-card">
          <h2>Enterprise</h2>
          <p>
            Volume licensing, custom training configurations, on-prem deployments.
          </p>
          <a
            className="contact-email"
            href="mailto:enterprise@violawake.com"
          >
            enterprise@violawake.com
          </a>
        </section>

        <section className="contact-card">
          <h2>Security</h2>
          <p>
            Vulnerability reports. Please use responsible disclosure — we
            respond within one business day.
          </p>
          <a className="contact-email" href="mailto:security@violawake.com">
            security@violawake.com
          </a>
        </section>

        <section className="contact-card">
          <h2>Privacy &amp; legal</h2>
          <p>
            Data deletion requests, GDPR / CCPA inquiries, terms questions.
          </p>
          <a className="contact-email" href="mailto:privacy@violawake.com">
            privacy@violawake.com
          </a>
          <a className="contact-email" href="mailto:legal@violawake.com">
            legal@violawake.com
          </a>
        </section>
      </div>

      <div className="contact-aux">
        <h2>Other ways to reach us</h2>
        <ul className="contact-links">
          <li>
            <strong>GitHub</strong> —{" "}
            <a
              href="https://github.com/GeeIHadAGoodTime/ViolaWake/issues"
              target="_blank"
              rel="noopener noreferrer"
            >
              file an issue
            </a>{" "}
            for SDK bugs or feature requests.
          </li>
          <li>
            <strong>Pricing</strong> —{" "}
            <Link to="/pricing">see plans</Link> if you&apos;re weighing
            Free vs Developer vs Business.
          </li>
          <li>
            <strong>Account &amp; billing</strong> —{" "}
            <Link to="/billing">manage your subscription</Link> from the
            console (logged-in users).
          </li>
        </ul>
      </div>
    </div>
  );
}
