"""Take screenshots of every ViolaWake Console page for visual audit."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

FRONTEND = "http://localhost:5173"
BACKEND = "http://localhost:8000"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)

PASSWORD = "AuditPass123!"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # 1. Landing page
        page.goto(FRONTEND)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "01_landing_hero.png"), full_page=False)
        # Scroll to comparison
        page.evaluate("window.scrollTo(0, 700)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "02_landing_comparison.png"), full_page=False)
        # Scroll to how it works
        page.evaluate("window.scrollTo(0, 1400)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "03_landing_howitworks.png"), full_page=False)
        # Scroll to stats
        page.evaluate("window.scrollTo(0, 2100)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "04_landing_stats.png"), full_page=False)
        # Scroll to pricing preview
        page.evaluate("window.scrollTo(0, 2800)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "05_landing_pricing_preview.png"), full_page=False)

        # 2. Register page
        page.goto(f"{FRONTEND}/register")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "06_register.png"), full_page=False)

        # 3. Login page
        page.goto(f"{FRONTEND}/login")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "07_login.png"), full_page=False)

        # 4. Pricing page
        page.goto(f"{FRONTEND}/pricing")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "08_pricing_top.png"), full_page=False)
        page.evaluate("window.scrollTo(0, 600)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "09_pricing_faq.png"), full_page=False)

        # 5. Register a real user and go to dashboard
        email = f"audit_{int(time.time())}@violawake.dev"
        import requests
        resp = requests.post(
            f"{BACKEND}/api/auth/register",
            json={"email": email, "password": PASSWORD, "name": "Audit User"},
            timeout=10,
        )
        token = resp.json()["token"]

        # Set token and go to dashboard
        page.goto(f"{FRONTEND}/login")
        page.evaluate(f"localStorage.setItem('token', '{token}')")
        page.goto(f"{FRONTEND}/dashboard")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "10_dashboard_empty.png"), full_page=False)

        # 6. Record page
        page.goto(f"{FRONTEND}/record")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "11_record_setup.png"), full_page=False)

        # Fill in wake word
        ww_input = page.locator("#wakeword")
        if ww_input.is_visible():
            ww_input.fill("hey viola")
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / "12_record_ready.png"), full_page=False)

        browser.close()

    print("Screenshots saved to:", OUT)
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    run()
