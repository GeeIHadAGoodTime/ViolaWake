"""ViolaWake Launch Deployment — Interactive Playwright Helper.

Opens each service to the exact page you need. Follow the on-screen
instructions at each step. Press Enter to advance to the next step.

Usage:
    python scripts/deploy_launch.py
"""

import subprocess
import sys
import time

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing playwright...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.sync_api import sync_playwright


# Pre-computed deployment values
PROD_SECRET_KEY = "yUmy1p8mDkfRiDbJMUR5YVz5ZQG21fUFHBCFNraKCgM"

RAILWAY_ENV_VARS = {
    "VIOLAWAKE_ENV": "production",
    "VIOLAWAKE_PORT": "8000",
    "VIOLAWAKE_SECRET_KEY": PROD_SECRET_KEY,
    "VIOLAWAKE_ALGORITHM": "HS256",
    "VIOLAWAKE_ACCESS_TOKEN_EXPIRE_HOURS": "24",
    "VIOLAWAKE_CORS_ORIGINS": "https://violawake.com",
    "VIOLAWAKE_TRAINING_TIMEOUT": "1800",
    "VIOLAWAKE_MAX_CONCURRENT_JOBS": "2",
    "VIOLAWAKE_CONSOLE_BASE_URL": "https://violawake.com",
    # These come from existing .env — user copies from Stripe dashboard:
    # VIOLAWAKE_STRIPE_SECRET_KEY = (live mode key from Stripe)
    # VIOLAWAKE_STRIPE_WEBHOOK_SECRET = (from Stripe webhook setup)
    # VIOLAWAKE_STRIPE_PRICE_DEVELOPER = (from Stripe product setup)
    # VIOLAWAKE_STRIPE_PRICE_BUSINESS = (from Stripe product setup)
    # VIOLAWAKE_RESEND_API_KEY = (already have)
    # VIOLAWAKE_DB_URL = (from Railway PostgreSQL addon)
}

DNS_RECORDS = [
    {"type": "CNAME", "name": "api", "target": "<railway-domain>.up.railway.app"},
    {"type": "CNAME", "name": "@", "target": "<cloudflare-pages-domain>.pages.dev"},
]


def wait_for_user(message: str) -> None:
    print(f"\n{'='*60}")
    print(message)
    print(f"{'='*60}")
    input("\n>>> Press Enter when done to continue...")


def step_1_domain(page):
    """Register violawake.com domain."""
    print("\n" + "="*60)
    print("STEP 1: Register violawake.com")
    print("="*60)
    print("""
We need to register violawake.com. Cloudflare Registrar is cheapest
and gives free DNS + CDN.

I'm opening Cloudflare's domain registration page.

WHAT TO DO:
1. Log in to Cloudflare (or create an account)
2. Search for 'violawake.com'
3. Register the domain (~$10/year for .com)
4. Cloudflare will auto-configure DNS management
""")
    page.goto("https://dash.cloudflare.com/?to=/:account/domains/register")
    wait_for_user("Register violawake.com on Cloudflare, then press Enter")


def step_2_cloudflare_pages(page):
    """Set up Cloudflare Pages for frontend."""
    print("\n" + "="*60)
    print("STEP 2: Cloudflare Pages (Frontend)")
    print("="*60)
    print("""
Deploy the React frontend to Cloudflare Pages.

I'm opening the Cloudflare Pages creation page.

WHAT TO DO:
1. Click "Create a project" → "Connect to Git"
2. Select the GeeIHadAGoodTime/ViolaWake repository
3. Configure build settings:
   - Project name: violawake
   - Production branch: master
   - Framework preset: None (or Vite)
   - Build command: cd console/frontend && npm ci && npm run build
   - Build output directory: console/frontend/dist
   - Root directory: (leave empty — build command handles it)
4. Add environment variable:
   - VITE_API_URL = https://api.violawake.com/api
5. Click "Save and Deploy"
""")
    page.goto("https://dash.cloudflare.com/?to=/:account/pages/new/provider/github")
    wait_for_user("Create Cloudflare Pages project, then press Enter")


def step_3_cloudflare_dns(page):
    """Configure DNS records."""
    print("\n" + "="*60)
    print("STEP 3: DNS Records")
    print("="*60)
    print("""
After Pages deploys, set up the custom domain.

I'm opening the Cloudflare Pages settings.

WHAT TO DO:
1. Go to Pages → violawake project → Custom Domains
2. Add 'violawake.com' as a custom domain
3. Cloudflare will auto-configure the DNS records
4. Wait for SSL certificate (usually <2 minutes)

For the API subdomain (after Railway is set up):
5. Go to DNS → Records
6. Add CNAME record:
   - Name: api
   - Target: <your-railway-domain>.up.railway.app
   - Proxy: OFF (orange cloud → grey cloud)
     Railway needs direct connection for WebSocket/SSE
""")
    page.goto("https://dash.cloudflare.com/?to=/:account/pages")
    wait_for_user("Configure custom domain for Pages, then press Enter")


def step_4_railway(page):
    """Set up Railway for backend."""
    print("\n" + "="*60)
    print("STEP 4: Railway (Backend)")
    print("="*60)
    print("""
Deploy the FastAPI backend to Railway.

I'm opening Railway's new project page.

WHAT TO DO:
1. Log in / create account (GitHub auth is easiest)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select GeeIHadAGoodTime/ViolaWake
4. Railway will auto-detect railway.toml and Dockerfile
5. Before first deploy, add a PostgreSQL plugin:
   - Click "+ New" → "Database" → "PostgreSQL"
   - Copy the DATABASE_URL (starts with postgresql://...)
""")
    page.goto("https://railway.com/new")
    wait_for_user("Create Railway project + PostgreSQL addon, then press Enter")


def step_5_railway_env(page):
    """Configure Railway environment variables."""
    print("\n" + "="*60)
    print("STEP 5: Railway Environment Variables")
    print("="*60)
    print("""
Add all required environment variables to the Railway service.

I'm opening Railway dashboard. Navigate to your service → Variables.

COPY THESE VARIABLES (one by one or use Railway's bulk editor):
""")
    for key, value in RAILWAY_ENV_VARS.items():
        print(f"  {key} = {value}")

    print("""
ALSO ADD THESE (copy from your .env or Stripe dashboard):
  VIOLAWAKE_DB_URL = <PostgreSQL URL from Railway addon — change 'postgres://' to 'postgresql+asyncpg://'>
  VIOLAWAKE_STRIPE_SECRET_KEY = <from Stripe dashboard — use LIVE key for production>
  VIOLAWAKE_STRIPE_WEBHOOK_SECRET = <create webhook at Step 6>
  VIOLAWAKE_STRIPE_PRICE_DEVELOPER = <Stripe Price ID for $29/mo plan>
  VIOLAWAKE_STRIPE_PRICE_BUSINESS = <Stripe Price ID for $99/mo plan>
  VIOLAWAKE_RESEND_API_KEY = <already have in .env>

After adding all variables:
1. Click "Deploy" to trigger a redeploy
2. Wait for health check to pass (/api/health)
3. Note your Railway domain: xxx.up.railway.app
""")
    page.goto("https://railway.com/dashboard")
    wait_for_user("Add all env vars and deploy, then press Enter")


def step_6_stripe_webhook(page):
    """Set up Stripe webhook for production."""
    print("\n" + "="*60)
    print("STEP 6: Stripe Webhook (Production)")
    print("="*60)
    print("""
Create a webhook endpoint for the production API.

I'm opening the Stripe webhook configuration page.

WHAT TO DO:
1. Click "Add endpoint"
2. Endpoint URL: https://api.violawake.com/api/billing/webhook
3. Select events:
   - checkout.session.completed
   - customer.subscription.updated
   - customer.subscription.deleted
   - invoice.payment_failed
4. Click "Add endpoint"
5. Copy the webhook signing secret (starts with whsec_)
6. Add it to Railway as VIOLAWAKE_STRIPE_WEBHOOK_SECRET

If you want to use LIVE mode (real payments):
- Toggle to "Live mode" in Stripe dashboard
- Create the same products/prices in live mode
- Use the live secret key + price IDs in Railway

For soft launch, test mode is fine — users can use Stripe test cards.
""")
    page.goto("https://dashboard.stripe.com/webhooks/create")
    wait_for_user("Create Stripe webhook, then press Enter")


def step_7_railway_domain(page):
    """Configure custom domain on Railway."""
    print("\n" + "="*60)
    print("STEP 7: Railway Custom Domain")
    print("="*60)
    print("""
Add api.violawake.com as a custom domain on Railway.

I'm opening Railway dashboard.

WHAT TO DO:
1. Go to your service → Settings → Networking → Custom Domain
2. Add: api.violawake.com
3. Railway will show a CNAME target
4. Go back to Cloudflare DNS and add:
   - Type: CNAME
   - Name: api
   - Target: <railway-provided-target>
   - Proxy: OFF (grey cloud — important for WebSocket/SSE)
5. Wait for Railway to verify the domain
""")
    page.goto("https://railway.com/dashboard")
    wait_for_user("Configure api.violawake.com on Railway, then press Enter")


def step_8_verify(page):
    """Verify everything works."""
    print("\n" + "="*60)
    print("STEP 8: Verification")
    print("="*60)
    print("""
Let's verify everything is working.

I'm opening the key URLs. Check each one:

1. https://violawake.com — frontend loads
2. https://api.violawake.com/api/health — returns 200 JSON
3. https://violawake.com/pricing — pricing page loads
4. https://violawake.com/privacy — privacy policy loads
5. Register a test account → record 10 samples → train a model
""")
    page.goto("https://api.violawake.com/api/health")
    time.sleep(3)
    page.goto("https://violawake.com")
    wait_for_user("Verify all URLs work, then press Enter")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         ViolaWake Launch Deployment Helper               ║
║                                                          ║
║  This script opens each service to the exact page you    ║
║  need. Follow instructions, then press Enter to advance. ║
║                                                          ║
║  Steps:                                                  ║
║    1. Register violawake.com (Cloudflare)                ║
║    2. Cloudflare Pages (frontend)                        ║
║    3. DNS configuration                                  ║
║    4. Railway project (backend)                          ║
║    5. Railway environment variables                      ║
║    6. Stripe webhook                                     ║
║    7. Railway custom domain                              ║
║    8. Verification                                       ║
╚══════════════════════════════════════════════════════════╝
""")

    input("Press Enter to start...")

    with sync_playwright() as p:
        # Use persistent context so user stays logged in
        context = p.chromium.launch_persistent_context(
            user_data_dir="",  # fresh context
            headless=False,
            args=["--start-maximized"],
            no_viewport=True,
        )
        page = context.new_page()

        step_1_domain(page)
        step_2_cloudflare_pages(page)
        step_3_cloudflare_dns(page)
        step_4_railway(page)
        step_5_railway_env(page)
        step_6_stripe_webhook(page)
        step_7_railway_domain(page)
        step_8_verify(page)

        print("\n" + "="*60)
        print("DEPLOYMENT COMPLETE!")
        print("="*60)
        print("""
ViolaWake is now live at:
  Frontend: https://violawake.com
  API:      https://api.violawake.com
  PyPI:     https://pypi.org/project/violawake/
  GitHub:   https://github.com/GeeIHadAGoodTime/ViolaWake

Next: Post on Hacker News (Show HN draft is at docs/SHOW_HN_DRAFT.md)
""")
        context.close()


if __name__ == "__main__":
    main()
