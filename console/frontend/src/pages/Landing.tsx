import { Link, Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export default function LandingPage() {
  const { isAuthenticated } = useAuth();

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="landing-page">
      {/* Hero */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            Custom Wake Words.<br />
            Open Source. $0 to Start.
          </h1>
          <p className="hero-subtitle">
            Train a personal wake word detector from 10 voice samples.
            Production-tested, Apache 2.0 licensed.
            The open alternative to Picovoice.
          </p>
          <div className="hero-actions">
            <Link to="/register" className="btn btn-primary btn-large">
              Create free account
            </Link>
            <a
              href="https://github.com/GeeIHadAGoodTime/ViolaWake"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost btn-large"
            >
              View SDK on GitHub
            </a>
          </div>
          <p className="hero-fineprint">
            Account required to record and train. SDK is free and open
            source — works without an account.
          </p>
        </div>
        <div className="hero-code">
          <div className="code-window">
            <div className="code-titlebar">
              <span className="code-dot code-dot-red" />
              <span className="code-dot code-dot-yellow" />
              <span className="code-dot code-dot-green" />
              <span className="code-filename">detect.py</span>
            </div>
            <pre className="code-block">
              <code>{`from violawake_sdk import WakeDetector

detector = WakeDetector(model="my_word.onnx")
for frame in mic_stream():
    if detector.detect(frame):
        print("Wake word detected!")`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Comparison */}
      <section className="section comparison-section">
        <h2 className="section-title">How we compare</h2>
        <p className="section-subtitle">
          Production-grade accuracy without the enterprise price tag.
        </p>
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th className="comparison-highlight">ViolaWake</th>
                <th>Picovoice</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>SDK License</td>
                <td className="comparison-highlight">Apache 2.0 (forever free)</td>
                <td>Proprietary</td>
              </tr>
              <tr>
                <td>Training</td>
                <td className="comparison-highlight">Open (your data stays yours)</td>
                <td>Black box</td>
              </tr>
              <tr>
                <td>Console Pricing</td>
                <td className="comparison-highlight">Free / $29 / $99</td>
                <td>Start Free / Talk to Sales</td>
              </tr>
              <tr>
                <td>Enterprise</td>
                <td className="comparison-highlight">From $99/mo</td>
                <td>Contact sales</td>
              </tr>
              <tr>
                <td>Accuracy disclosure</td>
                <td className="comparison-highlight">0.8% EER &middot; 0.001 FA/hr (production reference model); 5.49% EER (adversarial benchmark v2)</td>
                <td>No comparable public FAR/recall benchmark</td>
              </tr>
              <tr>
                <td>Training Samples</td>
                <td className="comparison-highlight">10</td>
                <td>Text-only (0 samples)</td>
              </tr>
              <tr>
                <td>Model Format</td>
                <td className="comparison-highlight">ONNX (portable)</td>
                <td>Proprietary binary</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* How It Works */}
      <section className="section how-it-works-section">
        <h2 className="section-title">How it works</h2>
        <p className="section-subtitle">
          From sign-up to a deployable wake word in under 5 minutes. A free
          account is required to record and train — the SDK itself is free
          and open source forever.
        </p>
        <div className="steps-grid steps-grid-four">
          <div className="step-card">
            <div className="step-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
            <div className="step-number">1</div>
            <h3 className="step-title">Sign up free</h3>
            <p className="step-desc">
              Create a free account — no credit card. The free tier includes
              3 trained models per month so you can try the full pipeline.
            </p>
          </div>
          <div className="step-card">
            <div className="step-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </div>
            <div className="step-number">2</div>
            <h3 className="step-title">Record</h3>
            <p className="step-desc">
              Say your wake word 10 times in our browser-based recorder. No
              downloads, no setup.
            </p>
          </div>
          <div className="step-card">
            <div className="step-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="16 18 22 12 16 6" />
                <polyline points="8 6 2 12 8 18" />
              </svg>
            </div>
            <div className="step-number">3</div>
            <h3 className="step-title">Train</h3>
            <p className="step-desc">
              Our ML pipeline trains a custom temporal head on OpenWakeWord embeddings.
              Real-time progress via SSE.
            </p>
          </div>
          <div className="step-card">
            <div className="step-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </div>
            <div className="step-number">4</div>
            <h3 className="step-title">Deploy</h3>
            <p className="step-desc">
              Download your 102KB wake head. About 1.4MB total runtime with the
              shared OpenWakeWord backbone. Runs anywhere Python runs.
            </p>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="section proof-section">
        <div className="proof-card">
          <div className="proof-stats">
            <div className="proof-stat">
              <span className="proof-stat-value">0.8%</span>
              <span className="proof-stat-label">EER (equal error rate)</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">&lt;5ms</span>
              <span className="proof-stat-label">inference latency</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">10</span>
              <span className="proof-stat-label">samples to train</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">102KB</span>
              <span className="proof-stat-label">wake head</span>
            </div>
          </div>
          <p className="proof-text">
            Built by the makers of Viola, an AI voice assistant. Powered by the
            same wake word technology running 24/7 in production. Metrics from
            our TemporalCNN(96,9) production reference model evaluated on 25,409
            parameters. <em>Your custom model&apos;s accuracy varies with sample
            quantity and quality;</em> the figures above are the upper bound we&apos;ve
            measured on a fully-tuned production model.
          </p>
        </div>
      </section>

      {/* Pricing Preview */}
      <section className="section pricing-preview-section">
        <h2 className="section-title">Simple, honest pricing</h2>
        <p className="section-subtitle">
          The SDK is always free. Pay only for Console training when you need it.
        </p>
        <div className="pricing-preview-grid">
          <div className="pricing-preview-card">
            <h3>Free</h3>
            <p className="pricing-preview-price">$0<span>/mo</span></p>
            <p className="pricing-preview-desc">3 models per month. Perfect for experimentation.</p>
          </div>
          <div className="pricing-preview-card pricing-preview-popular">
            <h3>Developer</h3>
            <p className="pricing-preview-price">$29<span>/mo</span></p>
            <p className="pricing-preview-desc">20 models, priority queue. For serious projects.</p>
          </div>
          <div className="pricing-preview-card">
            <h3>Business</h3>
            <p className="pricing-preview-price">$99<span>/mo</span></p>
            <p className="pricing-preview-desc">Unlimited models, accelerated training. Ship at scale.</p>
          </div>
        </div>
        <div className="pricing-preview-cta">
          <Link to="/pricing" className="btn btn-ghost btn-large">
            See full pricing details
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-links">
            <div className="footer-col">
              <h4>Product</h4>
              <Link to="/pricing">Pricing</Link>
              <a
                href="https://github.com/GeeIHadAGoodTime/ViolaWake"
                target="_blank"
                rel="noopener noreferrer"
              >
                Documentation
              </a>
              <a
                href="https://github.com/GeeIHadAGoodTime/ViolaWake"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
            <div className="footer-col">
              <h4>Company</h4>
              <Link to="/privacy">Privacy Policy</Link>
              <Link to="/terms">Terms of Service</Link>
              <a href="mailto:hello@violawake.com">Contact</a>
            </div>
          </div>
          <div className="footer-bottom">
            <span>&copy; 2026 ViolaWake. Apache 2.0 Licensed.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
