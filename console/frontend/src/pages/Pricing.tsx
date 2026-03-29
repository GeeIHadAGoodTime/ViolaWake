import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { isAuthenticated, createCheckout } from "../api";

interface FaqItem {
  question: string;
  answer: string;
}

const faqs: FaqItem[] = [
  {
    question: "Is the SDK really free?",
    answer:
      "Yes. The ViolaWake SDK is licensed under Apache 2.0 and will remain free forever. Detection runs entirely on your hardware with zero API calls. The Console is the paid product — it provides the web-based training interface.",
  },
  {
    question: "Can I train models without the Console?",
    answer:
      "Yes. The CLI training tool is free and open source. The Console adds convenience — browser-based recording, managed training infrastructure, real-time progress, and model management. But the training code itself is Apache 2.0.",
  },
  {
    question: "What happens if I cancel?",
    answer:
      "Your trained models are yours forever. Download them before canceling and use them in perpetuity with the free SDK. Canceling only stops your ability to train new models through the Console.",
  },
  {
    question: "Do you store my voice recordings?",
    answer:
      "Recordings are stored securely and can be deleted from your dashboard at any time. We never share your audio data. See our Privacy Policy for full details.",
  },
  {
    question: "How does this compare to Picovoice?",
    answer:
      "Picovoice charges $6,000+/year for enterprise access with a proprietary SDK. ViolaWake gives you an Apache 2.0 SDK (free forever) and Console training from $0/mo. We publish our evaluation methodology and provide violawake-eval so you can benchmark on your own data. See the comparison on our homepage.",
  },
  {
    question: "What model format do I get?",
    answer:
      "All models are delivered as standard ONNX files. ONNX runs on any platform with an ONNX Runtime — Python, C++, JavaScript, mobile, and edge devices. No vendor lock-in.",
  },
  {
    question: "How long does training take?",
    answer:
      "Training typically completes in 2-5 minutes depending on your tier. Free tier uses standard CPU training. Developer and Business tiers get priority queue access for faster processing.",
  },
];

interface TierFeature {
  text: string;
  included: boolean;
}

interface PricingTier {
  name: string;
  price: string;
  period: string;
  description: string;
  features: TierFeature[];
  cta: string;
  ctaLink: string;
  popular?: boolean;
  external?: boolean;
}

const tiers: PricingTier[] = [
  {
    name: "Free",
    price: "$0",
    period: "/mo",
    description: "Get started with wake word training at no cost.",
    features: [
      { text: "3 models per month", included: true },
      { text: "Standard training (CPU)", included: true },
      { text: "Community support", included: true },
      { text: "Apache 2.0 SDK included", included: true },
      { text: "Priority training queue", included: false },
      { text: "Accelerated training", included: false },
    ],
    cta: "Get Started",
    ctaLink: "/register",
  },
  {
    name: "Developer",
    price: "$29",
    period: "/mo",
    description: "For developers shipping real products.",
    features: [
      { text: "20 models per month", included: true },
      { text: "Priority training queue (2x faster processing)", included: true },
      { text: "Email support", included: true },
      { text: "Apache 2.0 SDK included", included: true },
      { text: "Faster training times", included: true },
      { text: "Accelerated training", included: false },
    ],
    cta: "Get Started",
    ctaLink: "/register",
    popular: true,
  },
  {
    name: "Business",
    price: "$99",
    period: "/mo",
    description: "Unlimited training for teams shipping at scale.",
    features: [
      { text: "Unlimited models", included: true },
      { text: "Accelerated training (up to 2x faster)", included: true },
      { text: "Priority email support", included: true },
      { text: "Apache 2.0 SDK included", included: true },
      { text: "Fastest training times", included: true },
      { text: "Team management", included: true },
    ],
    cta: "Get Started",
    ctaLink: "/register",
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "",
    description: "For organizations with specific requirements.",
    features: [
      { text: "Unlimited models", included: true },
      { text: "Unlimited team members", included: true },
      { text: "Priority support", included: true },
      { text: "Custom training configurations", included: true },
      { text: "Volume licensing", included: true },
    ],
    cta: "Contact Sales",
    ctaLink: "mailto:enterprise@violawake.com",
    external: true,
  },
];

function getTierKey(name: string): string {
  return name.toLowerCase();
}

function buildPricingAuthLink(tier: string): string {
  const params = new URLSearchParams({
    return: "/pricing",
    tier,
  });
  return `/register?${params.toString()}`;
}

export default function PricingPage() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);
  const [checkoutLoading, setCheckoutLoading] = useState<string | null>(null);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const selectedTier = searchParams.get("tier")?.toLowerCase() ?? null;
  const selectedTierName =
    tiers.find((tier) => getTierKey(tier.name) === selectedTier)?.name ?? null;

  function toggleFaq(index: number) {
    setOpenFaq(openFaq === index ? null : index);
  }

  async function handleCheckout(tier: string) {
    if (!isAuthenticated()) {
      navigate(buildPricingAuthLink(tier));
      return;
    }

    setCheckoutLoading(tier);
    setCheckoutError(null);
    try {
      const { checkout_url } = await createCheckout(tier);
      window.location.href = checkout_url;
    } catch (err) {
      setCheckoutError(
        err instanceof Error ? err.message : "Failed to start checkout. Please try again.",
      );
    } finally {
      setCheckoutLoading(null);
    }
  }

  function renderCtaButton(tier: PricingTier) {
    // Enterprise: external mailto link
    if (tier.external) {
      return (
        <a
          href={tier.ctaLink}
          className={`btn btn-full ${tier.popular ? "btn-primary" : "btn-ghost"}`}
        >
          {tier.cta}
        </a>
      );
    }

    // Free tier: navigate directly to register
    if (tier.name === "Free") {
      return (
        <Link
          to="/register"
          className={`btn btn-full ${tier.popular ? "btn-primary" : "btn-ghost"}`}
        >
          {tier.cta}
        </Link>
      );
    }

    // Paid tiers (Developer, Business): checkout flow
    const tierKey = getTierKey(tier.name);
    const loading = checkoutLoading === tierKey;
    return (
      <button
        onClick={() => handleCheckout(tierKey)}
        disabled={loading}
        className={`btn btn-full ${tier.popular ? "btn-primary" : "btn-ghost"}`}
      >
        {loading ? "Redirecting..." : tier.cta}
      </button>
    );
  }

  return (
    <div className="pricing-page">
      <div className="pricing-header">
        <h1 className="page-title">Pricing</h1>
        <p className="pricing-header-sub">
          The SDK is always free and open source. Pay only for the Console
          training infrastructure you need.
        </p>
      </div>

      {selectedTierName && (
        <div className="pricing-selection-banner" role="status">
          Continuing with the {selectedTierName} plan.
        </div>
      )}

      {checkoutError && (
        <div className="pricing-error" role="alert">
          {checkoutError}
        </div>
      )}

      {/* Pricing Grid */}
      <div className="pricing-grid">
        {tiers.map((tier) => {
          const tierKey = getTierKey(tier.name);
          const isSelected = selectedTier === tierKey;
          return (
          <div
            key={tier.name}
            className={`pricing-card ${tier.popular ? "pricing-card-popular" : ""} ${isSelected ? "pricing-card-selected" : ""}`}
          >
            {tier.popular && (
              <div className="pricing-popular-badge">POPULAR</div>
            )}
            <h3 className="pricing-card-name">{tier.name}</h3>
            <div className="pricing-card-price">
              <span className="pricing-amount">{tier.price}</span>
              {tier.period && (
                <span className="pricing-period">{tier.period}</span>
              )}
            </div>
            <p className="pricing-card-desc">{tier.description}</p>
            <ul className="pricing-features">
              {tier.features.map((feature, i) => (
                <li
                  key={i}
                  className={`pricing-feature ${feature.included ? "" : "pricing-feature-excluded"}`}
                >
                  <span className="pricing-feature-icon">
                    {feature.included ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                    )}
                  </span>
                  {feature.text}
                </li>
              ))}
            </ul>
            {renderCtaButton(tier)}
          </div>
          );
        })}
      </div>

      {/* FAQ */}
      <div className="pricing-faq">
        <h2 className="section-title">Frequently asked questions</h2>
        <div className="faq-list">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className={`faq-item ${openFaq === index ? "faq-item-open" : ""}`}
            >
              <button
                className="faq-question"
                onClick={() => toggleFaq(index)}
                aria-expanded={openFaq === index}
              >
                <span>{faq.question}</span>
                <span className="faq-chevron">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </span>
              </button>
              {openFaq === index && (
                <div className="faq-answer">
                  <p>{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="pricing-bottom-cta">
        <p>Not sure? Start with the free tier. No credit card required.</p>
        <Link to="/register" className="btn btn-primary btn-large">
          Get Started Free
        </Link>
      </div>
    </div>
  );
}
