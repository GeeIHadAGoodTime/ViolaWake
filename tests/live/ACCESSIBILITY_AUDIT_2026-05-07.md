# ViolaWake Public Page Accessibility Audit - 2026-05-07

## Executive Summary

Result: **Fail for automated WCAG-AA readiness**.

Lighthouse CI completed successfully against all requested public pages on the live deployment:

- `https://violawake.com/`: accessibility score 95, 1 failed audit
- `https://violawake.com/pricing`: accessibility score 93, 2 failed audits
- `https://violawake.com/register`: accessibility score 95, 1 failed audit
- `https://violawake.com/login`: accessibility score 95, 1 failed audit
- `https://violawake.com/privacy`: accessibility score 90, 2 failed audits
- `https://violawake.com/terms`: accessibility score 90, 2 failed audits

The live site cannot be represented as WCAG-AA compliant until the automated failures are remediated and re-tested. The failures are concentrated in low-contrast accent text, low-contrast cookie banner controls, legal links that rely on color alone, and one pricing page heading-order issue.

Automated checks passed for page language (`<html lang="en">`), visible form labels on login/register, link/button accessible names, ARIA validity, and heading order outside `/pricing`. Lighthouse reported no applicable failures for image alt text because the audited public pages did not expose image elements in the tested render.

Focus order was not fully certified by the automated run. Lighthouse did not flag `tabindex` or focusable-control issues, and source review shows the visible baseline flows use native links/buttons/inputs in DOM order. A manual keyboard and screen-reader pass should still be run after remediation before claiming full WCAG-AA conformance.

## Tooling

- Node: `v22.18.0`
- npx: `10.9.3`
- Auditor: Lighthouse CI CLI `0.15.1`
- Command:

```powershell
npx --yes @lhci/cli@latest collect --url=https://violawake.com/ --url=https://violawake.com/pricing --url=https://violawake.com/register --url=https://violawake.com/login --url=https://violawake.com/privacy --url=https://violawake.com/terms --numberOfRuns=1 --settings.onlyCategories=accessibility
```

## Raw Auditor Output - Head 100

The Lighthouse CI collection output was shorter than 100 lines:

```text
Running Lighthouse 1 time(s) on https://violawake.com/
Run #1...done.
Running Lighthouse 1 time(s) on https://violawake.com/pricing
Run #1...done.
Running Lighthouse 1 time(s) on https://violawake.com/register
Run #1...done.
Running Lighthouse 1 time(s) on https://violawake.com/login
Run #1...done.
Running Lighthouse 1 time(s) on https://violawake.com/privacy
Run #1...done.
Running Lighthouse 1 time(s) on https://violawake.com/terms
Run #1...done.
Done running Lighthouse!
```

## Top Issues by Severity

1. **High - Site-wide accent text contrast fails WCAG AA.** Accent purple text fails against dark page/card backgrounds with ratios such as 2.91:1, 3.27:1, 3.51:1, and 3.95:1 where 4.5:1 is required for normal text.
2. **High - Cookie banner controls fail contrast on most audited pages.** The privacy link is 3.95:1 and the `Accept` button is 4.31:1.
3. **High - Legal page links rely on color only.** `/privacy` and `/terms` fail `link-in-text-block` for inline links such as contact emails and external provider policy links.
4. **Medium - Pricing cards skip heading levels.** `/pricing` goes from the page `h1` to plan-card `h3` headings before the FAQ `h2`.
5. **Medium - Error-state auth messages use clickable non-semantic containers in source.** `Login.tsx` and `Register.tsx` use `div.auth-error` with `onClick={clearError}`. This is not present in the default no-error live render, but it should be corrected before a broader manual keyboard/screen-reader signoff.

## Per-Page Findings

### `/`

Score: 95. Failed audits: 1.

- **High - Color contrast:** `color-contrast` failed for `table.comparison-table` cells using `.comparison-highlight`. Lighthouse reported examples including:
  - `VIOLAWAKE`: foreground `#6c5ce7`, background `#16213e`, ratio 3.27:1.
  - Comparison table body values: foreground `#6c5ce7`, background `#1a1a2e`, ratio 3.51:1.
- **Passed/not applicable:** Heading order passed; page language passed; link/button names passed; image alt was not applicable.

Suggested fix:

- In `console/frontend/src/styles/global.css:2`, add a separate text-link token such as `--link-accent: #a29bfe`.
- In `console/frontend/src/styles/global.css:43`, use `color: var(--link-accent)` for normal text links.
- In `console/frontend/src/styles/global.css:1306` and `console/frontend/src/styles/global.css:1310`, change `.comparison-highlight` text color from `var(--accent)` to `var(--link-accent)`.

### `/pricing`

Score: 93. Failed audits: 2.

- **High - Color contrast:** `color-contrast` failed for excluded pricing features and cookie banner controls.
  - `.pricing-feature-excluded`: effective foreground `#5f657d`, background `#1e2a4a`, ratio 2.45:1.
  - Cookie `Privacy Policy`: foreground `#6c63ff`, background `#1a1a2e`, ratio 3.95:1.
  - Cookie `Accept`: white text on `#6c63ff`, ratio 4.31:1.
- **Medium - Heading order:** `heading-order` failed because plan cards use `h3.pricing-card-name` directly after the page `h1`.
- **Passed/not applicable:** Page language passed; link/button names passed; ARIA checks passed; form labels and image alt were not applicable.

Suggested fix:

- In `console/frontend/src/styles/global.css:1635`, remove the `opacity: 0.5` treatment from `.pricing-feature-excluded` and rely on an accessible visual treatment such as line-through or muted text without opacity.
- In `console/frontend/src/components/CookieConsent.tsx:9`, darken the cookie button background to an AA-compliant value such as `#5b4bd8`.
- In `console/frontend/src/components/CookieConsent.tsx:13`, change the cookie privacy link to an AA-compliant color such as `#a29bfe` and underline it.
- In `console/frontend/src/pages/Pricing.tsx:253`, change the plan name heading from `<h3 className="pricing-card-name">` to `<h2 className="pricing-card-name">`.

### `/register`

Score: 95. Failed audits: 1.

- **High - Color contrast:** `color-contrast` failed for the auth footer `Sign in` link and cookie banner controls.
  - `.auth-link`: foreground `#6c5ce7`, background `#1e2a4a`, ratio 2.91:1.
  - Cookie `Privacy Policy`: ratio 3.95:1.
  - Cookie `Accept`: ratio 4.31:1.
- **Medium - Error-state semantics from source review:** `console/frontend/src/pages/Register.tsx:62` renders `div.auth-error` with `onClick={clearError}` in error states. Use a semantic dismiss button or remove the click handler and announce the error with `role="alert"`.
- **Passed/not applicable:** Form labels passed for name, email, and password. Page language, heading order, link/button names, and ARIA checks passed.

Suggested fix:

- In `console/frontend/src/styles/global.css:317`, set `.auth-link` to an AA-compliant link color such as `var(--link-accent)`.
- Apply the cookie banner fixes from `/pricing`.
- Replace the error-state clickable `div` at `console/frontend/src/pages/Register.tsx:62` with a semantic alert plus a real `button` if dismiss behavior is retained.

### `/login`

Score: 95. Failed audits: 1.

- **High - Color contrast:** `color-contrast` failed for auth footer links and cookie banner controls.
  - `Forgot password?` and `Register`: foreground `#6c5ce7`, background `#1e2a4a`, ratio 2.91:1.
  - Cookie `Privacy Policy`: ratio 3.95:1.
  - Cookie `Accept`: ratio 4.31:1.
- **Medium - Error/status semantics from source review:** `console/frontend/src/pages/Login.tsx:61`, `console/frontend/src/pages/Login.tsx:66`, and `console/frontend/src/pages/Login.tsx:71` render auth messages without `role="alert"` or a semantic dismiss control for the clickable error state.
- **Passed/not applicable:** Form labels passed for email and password. Page language, heading order, link/button names, and ARIA checks passed.

Suggested fix:

- In `console/frontend/src/styles/global.css:317`, set `.auth-link` to `var(--link-accent)`.
- Apply the cookie banner fixes from `/pricing`.
- Add `role="alert"` to auth error/status messages and replace the clickable error `div` with a real dismiss `button` if dismiss behavior is required.

### `/privacy`

Score: 90. Failed audits: 2.

- **High - Color contrast:** `color-contrast` failed for inline legal links, footer legal links, and cookie banner controls.
  - Legal links use foreground `#6c5ce7` on `#1a1a2e`, ratio 3.51:1.
  - Cookie banner controls fail as described above.
- **High - Link distinguishability:** `link-in-text-block` failed for inline links including `stripe.com/privacy` and `privacy@violawake.com`; links rely on color alone.
- **Passed/not applicable:** Heading order passed; page language passed; link names passed; image alt and forms were not applicable.

Suggested fix:

- In `console/frontend/src/styles/global.css:1802`, change `.legal-section a` to `color: var(--link-accent)` and add persistent `text-decoration: underline`.
- In `console/frontend/src/styles/global.css:1819`, change `.legal-footer-nav a` to `color: var(--link-accent)` and add persistent underline or another non-color affordance.
- The affected source links are in `console/frontend/src/pages/Privacy.tsx:136`, `console/frontend/src/pages/Privacy.tsx:155`, `console/frontend/src/pages/Privacy.tsx:177`, `console/frontend/src/pages/Privacy.tsx:196`, and footer links at `console/frontend/src/pages/Privacy.tsx:201`.
- Apply the cookie banner fixes from `/pricing`.

### `/terms`

Score: 90. Failed audits: 2.

- **High - Color contrast:** `color-contrast` failed for inline legal links, footer legal links, and cookie banner controls.
  - Legal links use foreground `#6c5ce7` on `#1a1a2e`, ratio 3.51:1.
  - Cookie banner controls fail as described above.
- **High - Link distinguishability:** `link-in-text-block` failed for inline links including `billing@violawake.com`, `hello@violawake.com`, and `legal@violawake.com`; links rely on color alone.
- **Passed/not applicable:** Heading order passed; page language passed; link names passed; image alt and forms were not applicable.

Suggested fix:

- Apply the `.legal-section a` and `.legal-footer-nav a` CSS fixes from `/privacy`.
- The affected source links are in `console/frontend/src/pages/Terms.tsx:124`, `console/frontend/src/pages/Terms.tsx:138`, `console/frontend/src/pages/Terms.tsx:149`, `console/frontend/src/pages/Terms.tsx:225`, and footer links near the end of `Terms.tsx`.
- Apply the cookie banner fixes from `/pricing`.

## Remediation Notes

The automated failures are mostly suitable for a small follow-up patch:

- Introduce separate tokens for text links/highlights versus filled button backgrounds. A light link token fixes dark-background text contrast while preserving a darker filled button background for white text.
- Remove opacity-based muting from excluded pricing features.
- Underline legal and cookie links persistently.
- Fix the `/pricing` heading order by using `h2` for plan card headings.

After deployment, re-run the same Lighthouse CI command and then complete a manual keyboard and screen-reader smoke test covering tab order, focus visibility, form error announcements, and cookie-banner interaction.
