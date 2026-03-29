import { Link } from "react-router-dom";

export default function TermsPage() {
  return (
    <div className="legal-page">
      <div className="legal-content">
        <h1 className="legal-title">Terms of Service</h1>
        <p className="legal-updated">Last updated: March 28, 2026</p>

        <section className="legal-section">
          <h2>1. Service Description</h2>
          <p>
            ViolaWake provides two products: (a) the ViolaWake SDK, an
            open-source Python library for on-device wake word detection,
            licensed under Apache License 2.0; and (b) the ViolaWake Console, a
            web application for recording voice samples, submitting training
            jobs, and managing trained models. These Terms govern your use of
            the Console. Use of the SDK is governed by the Apache License 2.0.
          </p>
          <p>
            The current Console implementation stores recordings and model
            artifacts on the server filesystem used by the Console deployment,
            and training jobs run on that server&apos;s CPU. We may change the
            underlying infrastructure over time, but these Terms do not promise
            any specific cloud provider, GPU platform, or storage vendor.
          </p>
        </section>

        <section className="legal-section">
          <h2>2. Account Registration</h2>
          <p>
            To use the Console, you must create an account with a valid email
            address and password. You are responsible for maintaining the
            confidentiality of your login credentials and for all activity that
            occurs under your account.
          </p>
          <p>
            You must be at least 16 years old to create an account. By
            registering, you represent that you meet this requirement.
          </p>
        </section>

        <section className="legal-section">
          <h2>3. Acceptable Use</h2>
          <p>You agree not to use ViolaWake to:</p>
          <ul>
            <li>
              Record or upload voice samples of any person without that
              person&apos;s consent
            </li>
            <li>
              Train wake word models for surveillance, unauthorized monitoring,
              or any unlawful purpose
            </li>
            <li>
              Attempt to reverse-engineer, exploit, or disrupt the Console or
              its supporting infrastructure
            </li>
            <li>
              Submit automated, synthetic, or bot-generated recordings through
              the Console when the workflow requires real user samples
            </li>
            <li>
              Attempt to overload training queues, evade quotas, or bypass rate
              limits
            </li>
            <li>
              Use the Console to create models for threats, harassment, or hate
              speech
            </li>
            <li>Resell Console access or share account credentials</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>4. Intellectual Property</h2>

          <h3>4.1 Your Content</h3>
          <p>
            You retain ownership of the recordings you upload and the trained
            models produced from them. You grant us a limited license to store
            and process that content solely to provide the Console service you
            requested.
          </p>

          <h3>4.2 The ViolaWake SDK</h3>
          <p>
            The ViolaWake SDK is released under the Apache License 2.0. You may
            use, modify, and distribute it in accordance with that license.
          </p>

          <h3>4.3 The ViolaWake Console</h3>
          <p>
            The Console application, backend service, and related proprietary
            code remain the property of ViolaWake. Your subscription grants you
            a non-exclusive, non-transferable right to access and use the
            Console during the term of your subscription.
          </p>
        </section>

        <section className="legal-section">
          <h2>5. Payments and Billing</h2>

          <h3>5.1 Free Tier</h3>
          <p>
            The Free tier allows up to 3 model training jobs per calendar month
            at no charge.
          </p>

          <h3>5.2 Paid Subscriptions</h3>
          <p>
            Developer and Business subscriptions are billed through Stripe on a
            recurring basis until cancelled.
          </p>

          <h3>5.3 Refunds</h3>
          <p>
            If a service problem on our side materially prevents you from using
            a paid allocation, contact{" "}
            <a href="mailto:billing@violawake.com">billing@violawake.com</a>.
            Credits or refunds are provided at our discretion.
          </p>
        </section>

        <section className="legal-section">
          <h2>6. Service Availability</h2>
          <p>
            We aim to keep the Console available, but we do not guarantee
            uninterrupted service. The Console may be unavailable because of
            maintenance, capacity limits, upstream provider outages, or events
            outside our control.
          </p>
        </section>

        <section className="legal-section">
          <h2>7. Limitation of Liability</h2>
          <p>
            TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, VIOLAWAKE AND ITS
            AFFILIATES WILL NOT BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL,
            CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS, LOST
            DATA, BUSINESS INTERRUPTION, OR LOSS OF GOODWILL ARISING OUT OF OR
            RELATED TO YOUR USE OF THE SERVICE.
          </p>
          <p>
            OUR TOTAL LIABILITY FOR ANY CLAIM ARISING FROM OR RELATED TO THESE
            TERMS OR THE SERVICE WILL NOT EXCEED THE AMOUNT YOU PAID US IN THE
            12 MONTHS BEFORE THE CLAIM, OR $100, WHICHEVER IS GREATER.
          </p>
        </section>

        <section className="legal-section">
          <h2>8. Indemnification</h2>
          <p>
            You agree to indemnify and hold ViolaWake harmless from claims,
            damages, losses, and expenses arising from your use of the service,
            your violation of these Terms, or your violation of another
            party&apos;s rights.
          </p>
        </section>

        <section className="legal-section">
          <h2>9. Termination</h2>

          <h3>9.1 By You</h3>
          <p>
            You may close your account by using the available Console/API
            account-deletion flow or by contacting{" "}
            <a href="mailto:hello@violawake.com">hello@violawake.com</a>. If
            you want to keep trained models, download them before deleting your
            account.
          </p>

          <h3>9.2 By Us</h3>
          <p>
            We may suspend or terminate your account if you violate these Terms,
            abuse the service, or if we are required to do so by law.
          </p>

          <h3>9.3 Effect of Termination</h3>
          <p>
            When an account is deleted, access to the Console ends and the
            account data we still store for that account is removed through our
            deletion workflow. Models you already downloaded remain yours.
          </p>
        </section>

        <section className="legal-section">
          <h2>10. Changes to These Terms</h2>
          <p>
            We may update these Terms from time to time. When changes are
            material, we will update the date above and publish the revised
            Terms through the Console or website.
          </p>
        </section>

        <section className="legal-section">
          <h2>11. Governing Law</h2>
          <p>
            These Terms are governed by the laws of the State of Delaware,
            United States, without regard to conflict of law principles.
          </p>
        </section>

        <section className="legal-section">
          <h2>12. Contact</h2>
          <p>
            For questions about these Terms, contact{" "}
            <a href="mailto:legal@violawake.com">legal@violawake.com</a>.
          </p>
        </section>

        <div className="legal-footer-nav">
          <Link to="/privacy">Privacy Policy</Link>
          <Link to="/">Back to Home</Link>
        </div>
      </div>
    </div>
  );
}
