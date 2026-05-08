# ViolaWake SEO Runbook

Last updated: 2026-05-08

This runbook covers the human-operated steps that cannot be completed from the local repository without account ownership. The build now emits static HTML, markdown mirrors, JSON-LD, `robots.txt`, `sitemap.xml`, `llms.txt`, Open Graph images, favicons, and an SPA app shell.

## Build artifact contract

Run:

```bash
cd console/frontend
npm run build
```

Expected static marketing routes:

- `/`
- `/pricing`
- `/privacy`
- `/terms`
- `/contact`
- `/compare/picovoice`
- `/compare/openwakeword`
- `/compare/snowboy`
- `/docs`
- `/blog`
- `/blog/how-we-trained-wake-word-08-eer-25k-parameters`
- `/blog/open-source-vs-proprietary-wake-word-detection-2026`
- `/blog/raspberry-pi-voice-assistant-violawake`
- `/faq`
- `/about`

Expected crawler files:

- `/robots.txt`
- `/sitemap.xml`
- `/llms.txt`
- `/manifest.webmanifest`
- `/*.md` and nested markdown mirrors for every marketing page

The authenticated app remains the Vite SPA shell at `/app.html`. Cloudflare Pages `_redirects` rewrites app routes such as `/login`, `/register`, `/dashboard`, `/record`, `/billing`, `/teams`, and `/training/*` to `/app.html`.

## Google Search Console

1. Open `https://search.google.com/search-console`.
2. Add property: `violawake.com`.
3. Preferred verification: Domain property with DNS TXT record.
4. In Cloudflare DNS, add the TXT record exactly as Google provides it.
5. Wait for propagation and click Verify in Search Console.
6. Open Sitemaps.
7. Submit: `https://violawake.com/sitemap.xml`.
8. After deploy, use URL Inspection for:
   - `https://violawake.com/`
   - `https://violawake.com/compare/picovoice`
   - `https://violawake.com/compare/openwakeword`
   - `https://violawake.com/compare/snowboy`
   - `https://violawake.com/docs`
9. Request indexing for the comparison pages first because they target the highest-intent discovery searches.

## Bing Webmaster Tools

1. Open `https://www.bing.com/webmasters/`.
2. Add site: `https://violawake.com`.
3. If Google Search Console is already verified, use Bing's import flow.
4. If not importing, verify with DNS TXT in Cloudflare DNS.
5. Submit sitemap: `https://violawake.com/sitemap.xml`.
6. Use URL Submission for the same priority URLs:
   - `/`
   - `/compare/picovoice`
   - `/compare/openwakeword`
   - `/compare/snowboy`
   - `/docs`
   - `/faq`

## IndexNow

IndexNow helps Bing, Yandex, Seznam, and Naver discover changed URLs quickly.

1. Generate a key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

2. Create a text file at the site root named `<key>.txt` containing only the key.
3. In this repo, place that file in `console/frontend/public/<key>.txt`.
4. Build and deploy.
5. Submit changed URLs:

```bash
curl "https://api.indexnow.org/indexnow?url=https://violawake.com/compare/picovoice&key=<key>"
curl "https://api.indexnow.org/indexnow?url=https://violawake.com/compare/openwakeword&key=<key>"
curl "https://api.indexnow.org/indexnow?url=https://violawake.com/compare/snowboy&key=<key>"
```

For bulk submission:

```bash
curl -X POST "https://api.indexnow.org/indexnow" \
  -H "Content-Type: application/json" \
  -d '{"host":"violawake.com","key":"<key>","urlList":["https://violawake.com/","https://violawake.com/compare/picovoice","https://violawake.com/docs","https://violawake.com/faq"]}'
```

## Cloudflare Crawler Hints

1. Open Cloudflare dashboard.
2. Select the `violawake.com` zone.
3. Go to Caching, then Configuration.
4. Enable Crawler Hints.
5. Keep the sitemap URL public at `https://violawake.com/sitemap.xml`.

## Plausible analytics

The frontend includes:

```html
<script defer data-domain="violawake.com" src="https://plausible.io/js/script.file-downloads.outbound-links.tagged-events.js"></script>
```

Create the site in Plausible with domain `violawake.com`. The CSP allows `https://plausible.io` in `script-src` and `connect-src`.

### Key events

Implemented or defined events:

| Event | Source |
|---|---|
| `signup` | Register API success and static CTA links |
| `first_recording` | First successful recording upload per browser session |
| `training_started` | Training start API success |
| `training_completed` | Training progress completion |
| `plan_clicked` | Static pricing CTA links |
| `checkout_started` | Billing checkout API success |
| `checkout_completed` | `/billing?session_id=...` app load |

### Reading funnel data

1. Open Plausible for `violawake.com`.
2. Use Goals to create a funnel-like view from these events:
   - `signup`
   - `first_recording`
   - `training_started`
   - `training_completed`
   - `checkout_started`
   - `checkout_completed`
3. Segment by entry page. Watch whether `/compare/picovoice`, `/compare/openwakeword`, and `/blog/raspberry-pi-voice-assistant-violawake` lead to signups.
4. Segment by referrer. Track Google, Bing, Perplexity, ChatGPT, Claude, GitHub, Hacker News, Reddit, and dev.to.

## Post-deploy crawler checks

Run:

```bash
curl -L https://violawake.com/ | findstr /C:"Custom wake words"
curl -L https://violawake.com/compare/picovoice | findstr /C:"ViolaWake vs Picovoice"
curl -L https://violawake.com/compare/picovoice.md | findstr /C:"Quick answer"
curl -L https://violawake.com/llms.txt | findstr /C:"ViolaWake vs Picovoice"
curl -L https://violawake.com/sitemap.xml | findstr /C:"compare/picovoice"
```

If those pass, raw HTML and markdown are visible to crawlers.
