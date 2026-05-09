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
            Custom wake words.<br />
            Open source. Yours forever.
          </h1>
          <p className="hero-subtitle">
            Train a personal wake-word detector from your voice samples.
            Apache 2.0 SDK, ONNX models, no API keys. Works offline forever.
          </p>
          <div className="hero-actions">
            <Link to="/register" className="btn btn-primary btn-large">
              Get started free
            </Link>
            <a
              href="https://github.com/GeeIHadAGoodTime/ViolaWake"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost btn-large"
            >
              View on GitHub
            </a>
          </div>
          <p className="hero-fineprint">
            Apache 2.0 licensed. No runtime API keys. No phone-home.
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
          Factual product differences, verified 2026-05-08.
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
                <td className="comparison-highlight">Apache 2.0 SDK</td>
                <td>Proprietary SDK and service terms<sup>1</sup></td>
              </tr>
              <tr>
                <td>Runtime key</td>
                <td className="comparison-highlight">No API key or phone-home</td>
                <td>AccessKey required by Picovoice docs<sup>2</sup></td>
              </tr>
              <tr>
                <td>Console Pricing</td>
                <td className="comparison-highlight">Free / $29 / $99</td>
                <td>Verify current Picovoice terms directly<sup>3</sup></td>
              </tr>
              <tr>
                <td>Accuracy disclosure</td>
                <td className="comparison-highlight">0.8% EER on production reference model; user-trained accuracy varies</td>
                <td>FAQ claims 97%+ detection and &lt;1 false alarm in 10 hours<sup>4</sup></td>
              </tr>
              <tr>
                <td>Training Samples</td>
                <td className="comparison-highlight">10+ to start; more for production</td>
                <td>Text-to-wake-word Console flow<sup>1</sup></td>
              </tr>
              <tr>
                <td>Model Format</td>
                <td className="comparison-highlight">ONNX wake head</td>
                <td>Picovoice .ppn/.pv assets<sup>1</sup></td>
              </tr>
            </tbody>
          </table>
          <p className="hero-fineprint">
            Sources as of 2026-05-08: 1.{" "}
            <a href="https://picovoice.ai/docs/porcupine/" target="_blank" rel="noopener noreferrer">
              Picovoice Porcupine docs
            </a>
            ; 2.{" "}
            <a href="https://picovoice.ai/docs/quick-start/porcupine-python/" target="_blank" rel="noopener noreferrer">
              Python Quick Start
            </a>
            ; 3.{" "}
            <a href="https://picovoice.ai/pricing/" target="_blank" rel="noopener noreferrer">
              Picovoice pricing
            </a>
            ; 4.{" "}
            <a href="https://picovoice.ai/docs/faq/porcupine/" target="_blank" rel="noopener noreferrer">
              Porcupine FAQ
            </a>
            .
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="section how-it-works-section">
        <h2 className="section-title">How it works</h2>
        <p className="section-subtitle">
          Sign up, record 10+ samples, train, deploy.
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
              Create an account and open the browser recorder. No credit card
              is required for the free tier.
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
              Record or upload at least 10 wake-word samples. Add more samples
              before production.
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
              Train a custom TemporalCNN head on OpenWakeWord embeddings and
              review the metrics.
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
              Download the ONNX wake head and run it locally with the Apache
              2.0 SDK.
            </p>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="section proof-section">
        <div className="proof-card">
          <div className="proof-stats">
            <div className="proof-stat">
              <span className="proof-stat-value">0</span>
              <span className="proof-stat-label">runtime API keys</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">0</span>
              <span className="proof-stat-label">phone-home calls</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">100%</span>
              <span className="proof-stat-label">Apache 2.0 SDK</span>
            </div>
            <div className="proof-stat">
              <span className="proof-stat-value">102KB</span>
              <span className="proof-stat-label">wake head</span>
            </div>
          </div>
          <p className="proof-text">
            TemporalCNN(96, 9), 25,409 parameters, ONNX runtime inference.
            Production reference model: 0.8% EER and d-prime 8.58 on a curated
            benchmark. <em>Your trained model&apos;s accuracy depends on sample
            quantity, sample quality, microphones, rooms, negatives, and
            threshold tuning.</em>
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
