# ViolaWake SEO Audit and Architecture Decision

Last updated: 2026-05-08

## Current crawler surface before this work

Raw fetches of `https://violawake.com/` with `Mozilla/5.0`, `Googlebot/2.1`, `GPTBot`, `ChatGPT-User/1.0`, `ClaudeBot/1.0`, `PerplexityBot/1.0`, and `Bingbot/2.0` all returned the same SPA shell:

- Size: about 1.5 KB.
- Title: `ViolaWake Console`.
- Body: `<div id="root"></div>` plus Vite bundle.
- JSON-LD: absent.
- Real landing copy: hidden behind JavaScript.

Existing crawler files:

- `/robots.txt`: present, minimal.
- `/sitemap.xml`: present, short, included login/register.
- `/llms.txt`: fell through to SPA HTML.
- `/manifest.webmanifest`: fell through to SPA HTML.
- `/site.webmanifest`: fell through to SPA HTML.

Rendered DOM did contain marketing copy after JavaScript loaded, including the hero, comparison table, pricing preview, and footer. That helps human users but does not solve raw crawler visibility.

## Existing frontend structure

Public marketing pages before this pass:

- `console/frontend/src/pages/Landing.tsx`
- `console/frontend/src/pages/Pricing.tsx`
- `console/frontend/src/pages/Privacy.tsx`
- `console/frontend/src/pages/Terms.tsx`
- `console/frontend/src/pages/Contact.tsx`

Authenticated/app routes stay inside the SPA:

- `/dashboard`
- `/record`
- `/record/:wakeWord/add`
- `/training/:jobId`
- `/billing`
- `/model/:modelId/performance`
- `/account/password`
- `/teams`
- `/teams/accept`
- `/teams/:teamId`

The build previously used plain Vite React with no SSR, no SSG, and no prerender plugin.

## Competitor facts checked

Picovoice Porcupine:

- Product page title: "Porcupine Wake Word: On-Device Keyword Spotting for Enterprises".
- H1: "Fast, accurate, and lightweight custom wake word detection".
- Product page claims: custom wake words in seconds, no training data required, embedded/mobile/web/desktop/server deployment, 3.8% single-core CPU utilization on Raspberry Pi 3, 97.1% accuracy at 1 false alarm per 10 hours, about 250K custom wake words trained and deployed in 2025.
- Product calls to action: Start Free, Talk to Sales, Contact Sales.
- Public `/pricing/` URL redirected to contact in the crawl rather than exposing a static price table.
- Terms of Use updated March 18, 2026 with effective date March 30, 2026.

OpenWakeWord:

- GitHub README describes it as an open-source wake word library for voice-enabled applications and interfaces.
- License file is Apache License 2.0.
- README includes model training and Raspberry Pi relevant material.
- ViolaWake uses OpenWakeWord as a frozen embedding backbone and must say that plainly.

Snowboy:

- GitHub README says KITT.AI planned to shut down Snowboy, NLU, and Chatflow by December 31, 2020.
- README says official websites/APIs would be taken down and repositories would remain open with community support.
- License file governs source, libraries, resource files, and the bundled `snowboy.umdl`; other hotword models have their own licenses.

Google KWS:

- `google-research/kws_streaming` is an Apache-2.0 research reference for streaming keyword spotting models.
- It is useful as technical background, not as a direct SaaS competitor.

## Search landscape

Runtime Bing results for several exact unquoted queries were weak and sometimes lexically confused. Examples from the fetched Bing SERP:

| Query | Top result pattern observed |
|---|---|
| `picovoice alternative` | Picovoice official pages, Picovoice GitHub, general tool/profile pages |
| `picovoice pricing` | Picovoice official pages and third-party profile/review pages |
| `openwakeword vs picovoice` | OpenWakeWord GitHub, openwakeword.com, Home Assistant docs, Hugging Face, PyPI |
| `custom wake word python` | Bing raw SERP was polluted by dictionary results for "custom" |
| `porcupine wake word alternative` | Bing raw SERP was polluted by animal results for "porcupine" |
| `snowboy replacement` | Bing raw SERP was polluted by unrelated "advanced yoga" results |
| `wake word detection apache 2.0` | Bing raw SERP was polluted by dictionary results for "wake" |

Actionable conclusion: the keyword landscape is under-served by clear long-form pages with exact phrases in titles, headings, quick answers, tables, FAQ schema, and markdown mirrors. The best target pages are comparison and replacement pages, not only a generic landing page.

## Architecture decision

Chosen: keep Vite for the authenticated app and add a post-build static marketing generator.

Why:

- Lowest risk to the signed-in Console.
- No runtime SSR or Edge Worker required.
- No Astro migration needed.
- No dependency on browser-based prerendering during build.
- Raw HTML and markdown can be generated deterministically from one content module.
- Cloudflare Pages can serve marketing files directly and rewrite authenticated routes to `/app.html`.

Rejected:

- Full Astro migration: higher route risk and larger refactor for little benefit.
- Vite prerender plugin: likely needs browser automation and can trip over auth redirects/localStorage state.
- Runtime SSR: conflicts with static-first and local Docker backend constraints.

## Implemented static route plan

The build now runs:

```bash
tsc && vite build && node scripts/generate-marketing.mjs
```

The generator copies the Vite SPA shell to `dist/app.html`, then writes static HTML and markdown mirrors for marketing routes. `_redirects` maps auth/app routes to `app.html`; marketing routes are served as static files.
