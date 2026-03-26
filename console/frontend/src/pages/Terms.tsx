import { Link } from "react-router-dom";

export default function TermsPage() {
  return (
    <div className="legal-page">
      <div className="legal-content">
        <h1 className="legal-title">Terms of Service</h1>
        <p className="legal-updated">Last updated: March 26, 2026</p>

        <section className="legal-section">
          <h2>1. Service Description</h2>
          <p>
            ViolaWake provides two products: (a) the ViolaWake SDK, an
            open-source Python library for on-device wake word detection,
            licensed under Apache License 2.0; and (b) the ViolaWake Console, a
            web application for recording voice samples, training custom wake
            word models, and managing trained models. These Terms of Service
            govern your use of the ViolaWake Console. Use of the SDK is governed
            by the Apache License 2.0.
          </p>
        </section>

        <section className="legal-section">
          <h2>2. Account Registration</h2>
          <p>
            To use the Console, you must create an account with a valid email
            address and password. You are responsible for maintaining the
            confidentiality of your login credentials and for all activities that
            occur under your account. You must notify us immediately of any
            unauthorized use of your account.
          </p>
          <p>
            You must be at least 16 years of age to create an account. By
            registering, you represent that you meet this age requirement.
          </p>
        </section>

        <section className="legal-section">
          <h2>3. Acceptable Use</h2>
          <p>You agree not to use ViolaWake to:</p>
          <ul>
            <li>
              Record or upload voice samples of any person without their
              explicit consent
            </li>
            <li>
              Train wake word models for surveillance, unauthorized monitoring,
              or any purpose that violates applicable law
            </li>
            <li>
              Attempt to reverse-engineer, decompile, or extract proprietary
              training infrastructure code (the SDK itself is open source; the
              Console backend is not)
            </li>
            <li>
              Submit automated, bot-generated, or synthetic audio recordings
              through the Console recording interface
            </li>
            <li>
              Interfere with the service infrastructure, including attempting to
              overload training queues, bypass rate limits, or exploit
              vulnerabilities
            </li>
            <li>
              Use the Console to train models that detect words or phrases
              constituting hate speech, threats, or harassment
            </li>
            <li>
              Resell Console access or share account credentials with third
              parties
            </li>
          </ul>
          <p>
            We reserve the right to suspend or terminate accounts that violate
            these terms, with or without notice, depending on the severity of
            the violation.
          </p>
        </section>

        <section className="legal-section">
          <h2>4. Intellectual Property</h2>

          <h3>4.1 Your Content</h3>
          <p>
            You retain full ownership of the voice recordings you upload and the
            wake word models trained from them. We do not claim any intellectual
            property rights over your content. You grant us a limited,
            non-exclusive license to process your recordings solely for the
            purpose of training the model you requested.
          </p>

          <h3>4.2 The ViolaWake SDK</h3>
          <p>
            The ViolaWake SDK is released under the Apache License 2.0. You may
            use, modify, and distribute it in accordance with that license. The
            full license text is available in the SDK repository.
          </p>

          <h3>4.3 The ViolaWake Console</h3>
          <p>
            The Console application, its backend infrastructure, training
            pipeline, and associated proprietary code are owned by ViolaWake.
            Your subscription grants you a non-exclusive, non-transferable right
            to access and use the Console during the term of your subscription.
          </p>

          <h3>4.4 Trained Models</h3>
          <p>
            ONNX model files produced by the Console are your property. You may
            use them in any application, commercial or otherwise, without
            ongoing license fees. This right survives termination of your
            account for models downloaded before termination.
          </p>
        </section>

        <section className="legal-section">
          <h2>5. Payments and Billing</h2>

          <h3>5.1 Free Tier</h3>
          <p>
            The Free tier allows up to 3 model training jobs per calendar month
            at no charge. No credit card is required for the Free tier.
          </p>

          <h3>5.2 Paid Subscriptions</h3>
          <p>
            Developer ($29/month) and Business ($99/month) subscriptions are
            billed monthly through Stripe. Your subscription begins on the date
            of your first payment and renews automatically each month until
            canceled.
          </p>

          <h3>5.3 Enterprise Plans</h3>
          <p>
            Enterprise pricing is negotiated individually. Contact{" "}
            <a href="mailto:enterprise@violawake.com">
              enterprise@violawake.com
            </a>{" "}
            for details.
          </p>

          <h3>5.4 Cancellation</h3>
          <p>
            You may cancel your subscription at any time through the Console
            settings page. Cancellation takes effect at the end of the current
            billing period. No prorated refunds are issued for partial months.
            After cancellation, your account reverts to the Free tier.
          </p>

          <h3>5.5 Refunds</h3>
          <p>
            If you experience a service outage or training failure caused by our
            infrastructure that prevents you from using your paid allocation for
            a billing period, contact{" "}
            <a href="mailto:billing@violawake.com">billing@violawake.com</a>{" "}
            and we will issue a credit or refund at our discretion.
          </p>
        </section>

        <section className="legal-section">
          <h2>6. Service Availability</h2>
          <p>
            We strive to maintain high availability but do not guarantee
            uninterrupted access to the Console. We may perform scheduled
            maintenance with reasonable advance notice. We are not liable for
            downtime caused by factors outside our control, including
            third-party service outages, network failures, or force majeure
            events.
          </p>
        </section>

        <section className="legal-section">
          <h2>7. Limitation of Liability</h2>
          <p>
            TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, VIOLAWAKE AND ITS
            OFFICERS, EMPLOYEES, AND AFFILIATES SHALL NOT BE LIABLE FOR ANY
            INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES,
            INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, BUSINESS
            OPPORTUNITIES, OR GOODWILL, ARISING OUT OF OR RELATED TO YOUR USE OF
            THE SERVICE.
          </p>
          <p>
            OUR TOTAL LIABILITY FOR ANY CLAIM ARISING FROM OR RELATED TO THESE
            TERMS OR THE SERVICE SHALL NOT EXCEED THE AMOUNT YOU PAID US IN THE
            12 MONTHS PRECEDING THE CLAIM, OR $100, WHICHEVER IS GREATER.
          </p>
          <p>
            The ViolaWake SDK is provided "AS IS" under the Apache License 2.0
            without warranties of any kind. See the Apache License 2.0 for the
            full warranty disclaimer.
          </p>
        </section>

        <section className="legal-section">
          <h2>8. Indemnification</h2>
          <p>
            You agree to indemnify and hold ViolaWake harmless from any claims,
            damages, losses, or expenses (including reasonable attorney's fees)
            arising from (a) your use of the service, (b) your violation of
            these Terms, (c) your violation of any third party's rights,
            including intellectual property or privacy rights, or (d) content
            you upload to the service, including voice recordings.
          </p>
        </section>

        <section className="legal-section">
          <h2>9. Termination</h2>

          <h3>9.1 By You</h3>
          <p>
            You may close your account at any time by contacting us at{" "}
            <a href="mailto:hello@violawake.com">hello@violawake.com</a>. Upon
            account closure, we will delete your account data and voice
            recordings within 30 days. You should download any trained models
            you wish to keep before requesting account closure.
          </p>

          <h3>9.2 By Us</h3>
          <p>
            We may suspend or terminate your account if you violate these Terms,
            engage in abusive behavior, or if we are required to do so by law.
            For non-urgent violations, we will provide 7 days notice and an
            opportunity to cure the violation before termination.
          </p>

          <h3>9.3 Effect of Termination</h3>
          <p>
            Upon termination, your right to access the Console ceases. Models
            you have already downloaded remain yours. Sections 4 (Intellectual
            Property), 7 (Limitation of Liability), 8 (Indemnification), and 11
            (Governing Law) survive termination.
          </p>
        </section>

        <section className="legal-section">
          <h2>10. Changes to These Terms</h2>
          <p>
            We may modify these Terms from time to time. When we make material
            changes, we will notify you by email or by posting a notice in the
            Console dashboard at least 14 days before the changes take effect.
            Your continued use of the service after the effective date
            constitutes acceptance of the updated Terms. If you disagree with
            the changes, you may close your account before they take effect.
          </p>
        </section>

        <section className="legal-section">
          <h2>11. Governing Law</h2>
          <p>
            These Terms are governed by and construed in accordance with the
            laws of the State of Delaware, United States, without regard to
            conflict of law principles. Any disputes arising from these Terms or
            your use of the service shall be resolved in the state or federal
            courts located in Delaware.
          </p>
        </section>

        <section className="legal-section">
          <h2>12. Contact</h2>
          <p>
            For questions about these Terms of Service, contact us at:
          </p>
          <p>
            <strong>Email:</strong>{" "}
            <a href="mailto:legal@violawake.com">legal@violawake.com</a>
            <br />
            <strong>General inquiries:</strong>{" "}
            <a href="mailto:hello@violawake.com">hello@violawake.com</a>
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
