// Off-box ViolaWake prod down-detector (Cloudflare Worker, cron-triggered).
// Pings api.violawake.com/api/health from Cloudflare's edge and pages the
// operator's red-alert Telegram bot (the SAME channel as Viola's uptime
// detector) when ViolaWake is unreachable or reports itself down. Runs OFF the
// prod host (Docker Desktop on Jay's machine), so it still fires when that host
// and its in-container canaries are down. Bot creds are Worker secrets.
//
// Mirror of NOVVIOLA cloudflare/uptime-worker retargeted at ViolaWake. Part of
// generalizing ops infra across all three live businesses, not just Viola.
export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCheck(env));
  },
  async fetch(request, env, ctx) {
    // GET /?test=down sends ONE test page to prove the alert path end-to-end
    // (clearly labelled as a test); GET / runs the real probe and returns JSON.
    const u = new URL(request.url);
    if (u.searchParams.get("test") === "down") {
      await pageTelegram(env, env.HEALTH_URL || "(test)", "TEST PAGE — ViolaWake alert path verification, prod is fine");
      return new Response(JSON.stringify({ test_page_sent: true, ts: new Date().toISOString() }, null, 2), { headers: { "content-type": "application/json" } });
    }
    const result = await runCheck(env, { manual: true });
    return new Response(JSON.stringify(result, null, 2), { headers: { "content-type": "application/json" } });
  },
};

async function runCheck(env, opts = {}) {
  const url = env.HEALTH_URL || "https://api.violawake.com/api/health";
  let status = "down", detail = "", httpCode = 0;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const resp = await fetch(url, { signal: controller.signal, cf: { cacheTtl: 0 } });
    clearTimeout(timer);
    httpCode = resp.status;
    const body = await resp.text();
    if (resp.ok) {
      let overall = "";
      try { overall = String(JSON.parse(body).status || "").toLowerCase(); } catch (e) { overall = ""; }
      // ViolaWake /api/health returns {"status":"ok","ready":true,...}. Only an
      // explicit down/unhealthy overall status pages; anything else is UP.
      if (overall === "down" || overall === "unhealthy") { status = "down"; detail = "overall status=" + overall; }
      else { status = "up"; detail = overall || "ok"; }
    } else {
      status = "down"; detail = "http " + httpCode;
    }
  } catch (e) {
    const reason = (e && e.name === "AbortError") ? "timeout" : ((e && e.message) || String(e));
    status = "down"; detail = "unreachable: " + reason;
  }
  if (status === "down") { await pageTelegram(env, url, detail); }
  return { status, detail, httpCode, manual: !!opts.manual, ts: new Date().toISOString() };
}

async function pageTelegram(env, url, detail) {
  const token = env.VIOLA_ALERT_TELEGRAM_BOT_TOKEN, chat = env.VIOLA_ALERT_TELEGRAM_CHAT_ID;
  if (!token || !chat) return;
  const text =
    "\u{1F534} *OUTAGE* — ViolaWake production is DOWN\n" +
    "*Metric:* `violawake_prod_down`\n" +
    "*Detail:* " + detail + " (" + url + ")\n" +
    "_Detected from the Cloudflare edge, off the prod host._\n\n" +
    "*What to do:* confirm Docker Desktop is running and `docker ps` shows wakeword-backend-1 + wakeword-tunnel-1 up; the containers now carry restart=always so a Docker/host bounce should auto-recover them.";
  await fetch("https://api.telegram.org/bot" + token + "/sendMessage", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ chat_id: chat, text, parse_mode: "Markdown" }),
  });
}
