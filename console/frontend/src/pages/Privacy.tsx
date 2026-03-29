import { Link } from "react-router-dom";

export default function PrivacyPage() {
  return (
    <div className="legal-page">
      <div className="legal-content">
        <h1 className="legal-title">Privacy Policy</h1>
        <p className="legal-updated">Last updated: March 28, 2026</p>

        <section className="legal-section">
          <h2>1. Introduction</h2>
          <p>
            ViolaWake (&quot;we&quot;, &quot;us&quot;, &quot;our&quot;) operates the
            ViolaWake Console web application and the ViolaWake SDK. This
            Privacy Policy explains what information we collect, how we use it,
            and what controls you have when you use those services.
          </p>
        </section>

        <section className="legal-section">
          <h2>2. Information We Collect</h2>

          <h3>2.1 Account Information</h3>
          <p>
            When you create a Console account, we collect your email address,
            display name, and a hashed version of your password. We do not store
            passwords in plaintext.
          </p>

          <h3>2.2 Voice Recordings</h3>
          <p>
            When you record samples in the Console, those recordings are
            uploaded to the Console server and stored on the server filesystem
            so they can be used for the training job you requested. You can
            delete your recordings from the dashboard at any time. Automatic
            retention cleanup is now active: recordings are deleted after 90
            days by default. Recordings linked to an in-progress training job
            are never deleted until that job completes.
          </p>

          <h3>2.3 Trained Models</h3>
          <p>
            Models produced for your account are stored with your account until
            you delete them or delete your account.
          </p>

          <h3>2.4 Billing Information</h3>
          <p>
            Payment processing is handled by Stripe. We do not store full card
            numbers or CVV values. We store only the subscription and billing
            metadata needed to manage your Console account.
          </p>
        </section>

        <section className="legal-section">
          <h2>3. How We Use Information</h2>
          <ul>
            <li>Provide and maintain the ViolaWake Console</li>
            <li>Train the wake word model you requested</li>
            <li>Process subscriptions and billing events</li>
            <li>
              Send service emails such as verification, password reset, and
              training completion messages
            </li>
            <li>Respond to support requests and investigate abuse</li>
          </ul>
          <p>
            We do not sell your personal information. We do not use your voice
            recordings for advertising.
          </p>
        </section>

        <section className="legal-section">
          <h2>4. The ViolaWake SDK</h2>
          <p>
            The ViolaWake SDK performs wake word detection on your device. The
            SDK does not send inference audio to our servers and does not
            include built-in analytics or telemetry.
          </p>
        </section>

        <section className="legal-section">
          <h2>5. Data Security</h2>
          <p>
            We use reasonable technical measures to protect the Console,
            including TLS for data in transit, password hashing, and
            authenticated API access. Recordings and trained model artifacts are
            currently stored on the server filesystem used by the Console
            deployment rather than encrypted object storage.
          </p>
        </section>

        <section className="legal-section">
          <h2>6. Training Infrastructure</h2>
          <p>
            Training jobs currently run on the Console server CPU. The current
            implementation does not use Modal GPU workers or a separate managed
            training platform.
          </p>
        </section>

        <section className="legal-section">
          <h2>7. Data Retention</h2>
          <ul>
            <li>
              Account information is retained while your account remains active.
            </li>
            <li>
              Recordings remain stored until you delete them or delete your
              account.
            </li>
            <li>
              Trained models remain stored until you delete them or delete your
              account.
            </li>
            <li>
              Recordings are automatically deleted after 90 days. Recordings
              tied to an active training job are retained until that job
              completes.
            </li>
            <li>
              Trained models are automatically deleted after 365 days.
            </li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>8. Third-Party Services</h2>
          <ul>
            <li>
              <strong>Stripe</strong> processes payments and subscription events.
              Stripe&apos;s privacy policy is available at{" "}
              <a
                href="https://stripe.com/privacy"
                target="_blank"
                rel="noopener noreferrer"
              >
                stripe.com/privacy
              </a>
              .
            </li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>9. Your Rights</h2>
          <ul>
            <li>Access the account information we hold about you</li>
            <li>Correct inaccurate account details</li>
            <li>Delete recordings, trained models, or your account</li>
          </ul>
          <p>
            You can delete your account from the service API, and you can also
            contact{" "}
            <a href="mailto:privacy@violawake.com">privacy@violawake.com</a> for
            privacy-related requests.
          </p>
        </section>

        <section className="legal-section">
          <h2>10. Changes to This Policy</h2>
          <p>
            We may update this Privacy Policy from time to time. When we make
            material changes, we will update the date above and publish the
            revised version through the Console or website.
          </p>
        </section>

        <section className="legal-section">
          <h2>11. Contact</h2>
          <p>
            For questions about this Privacy Policy or our data practices,
            contact us at{" "}
            <a href="mailto:privacy@violawake.com">privacy@violawake.com</a>.
          </p>
        </section>

        <div className="legal-footer-nav">
          <Link to="/terms">Terms of Service</Link>
          <Link to="/">Back to Home</Link>
        </div>
      </div>
    </div>
  );
}
