import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  allMarketingPaths,
  blogIndex,
  blogPosts,
  pages,
  site,
} from "./marketing-content.mjs";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.join(frontendRoot, "dist");
const today = "2026-05-08";
// MUST be the canonical directory form ("/app/"), never a file path.
// Cloudflare Pages 308-canonicalizes any ".html" rewrite target to its clean
// URL (/app.html -> /app, /app/index.html -> /app/) BEFORE serving, turning
// the 200 rewrite into a redirect that strips the original SPA path — which
// broke /verify-email and /reset-password deep links for every user from
// 2026-05-08 to 2026-07-03. Gate: spa-deep-link-rewrite-target.
const appShellRedirectTarget = "/app/";

function toPosix(value) {
  return value.replace(/\\/g, "/");
}

function canonicalUrl(route) {
  return route === "/" ? site.origin : `${site.origin}${route}`;
}

function routeHtmlPath(route) {
  if (route === "/") return path.join(distRoot, "index.html");
  return path.join(distRoot, route.slice(1), "index.html");
}

function routeMarkdownPath(route) {
  if (route === "/") return path.join(distRoot, "index.md");
  return path.join(distRoot, `${route.slice(1)}.md`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function inlineMarkdown(value) {
  let html = escapeHtml(value);
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    (_match, label, href) => {
      const eventClass = href.startsWith("/register")
        ? href.includes("tier=") || href.includes("plan=")
          ? " plausible-event-name=plan_clicked"
          : " plausible-event-name=signup"
        : "";
      const classAttr = eventClass ? ` class="${eventClass.trim()}"` : "";
      return `<a${classAttr} href="${escapeHtml(href)}">${escapeHtml(label)}</a>`;
    },
  );
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html;
}

function markdownToHtml(markdown) {
  const lines = markdown.trim().split(/\r?\n/);
  const html = [];
  let paragraph = [];
  let list = [];
  let table = [];
  let inCode = false;
  let codeLang = "";
  let codeLines = [];

  function flushParagraph() {
    if (paragraph.length === 0) return;
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (list.length === 0) return;
    html.push(
      `<ul>${list
        .map((item) => `<li>${inlineMarkdown(item)}</li>`)
        .join("")}</ul>`,
    );
    list = [];
  }

  function flushTable() {
    if (table.length < 2) {
      table = [];
      return;
    }
    const rows = table
      .filter((line) => !/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line))
      .map((line) =>
        line
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((cell) => cell.trim()),
      );
    const [header, ...body] = rows;
    html.push(
      `<div class="static-table-wrap"><table><thead><tr>${header
        .map((cell) => `<th>${inlineMarkdown(cell)}</th>`)
        .join("")}</tr></thead><tbody>${body
        .map(
          (row) =>
            `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`,
        )
        .join("")}</tbody></table></div>`,
    );
    table = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.startsWith("~~~")) {
      if (inCode) {
        html.push(
          `<pre class="static-code"><code class="language-${escapeHtml(codeLang)}">${escapeHtml(
            codeLines.join("\n"),
          )}</code></pre>`,
        );
        inCode = false;
        codeLang = "";
        codeLines = [];
      } else {
        flushParagraph();
        flushList();
        flushTable();
        inCode = true;
        codeLang = line.slice(3).trim();
      }
      continue;
    }

    if (inCode) {
      codeLines.push(rawLine);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      flushTable();
      continue;
    }

    if (/^\|.*\|$/.test(line)) {
      flushParagraph();
      flushList();
      table.push(line);
      continue;
    }

    flushTable();

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      html.push(`<h${level} id="${id}">${inlineMarkdown(text)}</h${level}>`);
      continue;
    }

    const bullet = /^-\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }

    const numbered = /^\d+\.\s+(.+)$/.exec(line);
    if (numbered) {
      flushParagraph();
      list.push(numbered[1]);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushTable();
  return html.join("\n");
}

async function findCssHref() {
  const assetsDir = path.join(distRoot, "assets");
  if (!existsSync(assetsDir)) return null;
  const files = await readdir(assetsDir);
  const css = files.find((file) => file.endsWith(".css"));
  return css ? `/assets/${css}` : null;
}

function breadcrumbItems(route) {
  if (route === "/") return [];
  const parts = route.split("/").filter(Boolean);
  const items = [{ name: "Home", url: site.origin }];
  let current = "";
  for (const part of parts) {
    current += `/${part}`;
    items.push({
      name: part
        .split("-")
        .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
        .join(" "),
      url: canonicalUrl(current),
    });
  }
  return items;
}

function buildJsonLd(page) {
  const graph = [];
  const image = `${site.origin}${page.ogImage}`;

  if (page.schema?.includes("Organization")) {
    graph.push({
      "@type": "Organization",
      "@id": `${site.origin}/#organization`,
      name: site.name,
      url: site.origin,
      logo: `${site.origin}/apple-touch-icon-180.png`,
      sameAs: [site.github],
      contactPoint: [
        {
          "@type": "ContactPoint",
          email: site.contactEmail,
          contactType: "customer support",
        },
      ],
    });
  }

  if (page.schema?.includes("WebSite")) {
    graph.push({
      "@type": "WebSite",
      "@id": `${site.origin}/#website`,
      url: site.origin,
      name: site.name,
      description: site.description,
      publisher: { "@id": `${site.origin}/#organization` },
    });
  }

  if (page.schema?.includes("SoftwareApplication")) {
    graph.push({
      "@type": "SoftwareApplication",
      name: "ViolaWake",
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Windows, macOS, Linux, Raspberry Pi",
      softwareVersion: "0.2.6",
      license: "https://www.apache.org/licenses/LICENSE-2.0",
      url: site.origin,
      description: site.description,
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
    });
  }

  if (page.schema?.includes("Product")) {
    graph.push({
      "@type": "Product",
      name: "ViolaWake Console",
      description: "Hosted custom wake word recording, training, and model management.",
      brand: { "@type": "Brand", name: "ViolaWake" },
      offers: [
        { "@type": "Offer", name: "Free", price: "0", priceCurrency: "USD" },
        { "@type": "Offer", name: "Developer", price: "29", priceCurrency: "USD" },
        { "@type": "Offer", name: "Business", price: "99", priceCurrency: "USD" },
      ],
    });
  }

  if (page.schema?.includes("FAQPage") && page.faqs?.length) {
    graph.push({
      "@type": "FAQPage",
      mainEntity: page.faqs.map((item) => ({
        "@type": "Question",
        name: item.q,
        acceptedAnswer: {
          "@type": "Answer",
          text: item.a,
        },
      })),
    });
  }

  if (page.schema?.includes("BreadcrumbList")) {
    const items = breadcrumbItems(page.path);
    if (items.length) {
      graph.push({
        "@type": "BreadcrumbList",
        itemListElement: items.map((item, index) => ({
          "@type": "ListItem",
          position: index + 1,
          name: item.name,
          item: item.url,
        })),
      });
    }
  }

  if (page.schema?.includes("Article")) {
    graph.push({
      "@type": "Article",
      headline: page.title,
      description: page.description,
      datePublished: page.date,
      dateModified: today,
      image,
      author: {
        "@type": "Organization",
        name: "ViolaWake",
        url: site.origin,
      },
      publisher: {
        "@type": "Organization",
        name: "ViolaWake",
        logo: {
          "@type": "ImageObject",
          url: `${site.origin}/apple-touch-icon-180.png`,
        },
      },
      mainEntityOfPage: canonicalUrl(page.path),
    });
  }

  if (page.schema?.includes("HowTo")) {
    graph.push({
      "@type": "HowTo",
      name: "Build a Raspberry Pi voice assistant with ViolaWake",
      totalTime: "PT30M",
      tool: ["Raspberry Pi", "USB microphone", "Python 3.10", "ViolaWake SDK"],
      step: [
        "Install the ViolaWake SDK.",
        "Copy a trained ONNX model to the Raspberry Pi.",
        "Run WakeDetector against microphone frames.",
        "Tune the wake word threshold.",
        "Test in the deployment room.",
      ].map((text) => ({ "@type": "HowToStep", text })),
    });
  }

  if (graph.length === 0) {
    graph.push({
      "@type": "WebPage",
      name: page.title,
      description: page.description,
      url: canonicalUrl(page.path),
    });
  }

  return {
    "@context": "https://schema.org",
    "@graph": graph,
  };
}

function renderNav(page) {
  const links = page.nav ?? [];
  return `<nav class="static-nav" aria-label="Primary">
    <a class="static-brand" href="/"><span class="brand-icon">W</span><span>ViolaWake</span></a>
    <div class="static-nav-links">
      ${links
        .map((link) => `<a href="${link.href}">${escapeHtml(link.label)}</a>`)
        .join("")}
      <a href="/login">Login</a>
      <a class="btn btn-primary btn-nav plausible-event-name=signup" href="/register">Get Started</a>
    </div>
  </nav>`;
}

function renderFooter() {
  return `<footer class="static-footer">
    <div>
      <strong>ViolaWake</strong>
      <p>Open-source custom wake word training and on-device detection.</p>
    </div>
    <div class="static-footer-links">
      <a href="/pricing">Pricing</a>
      <a href="/docs">Docs</a>
      <a href="/compare/picovoice">Picovoice alternative</a>
      <a href="/privacy">Privacy</a>
      <a href="/terms">Terms</a>
      <a href="${site.github}">GitHub</a>
    </div>
  </footer>`;
}

function renderShell(page, bodyHtml, cssHref) {
  const canonical = canonicalUrl(page.path);
  const ogImage = `${site.origin}${page.ogImage}`;
  const cssLink = cssHref ? `<link rel="stylesheet" href="${cssHref}" />` : "";
  const jsonLd = JSON.stringify(buildJsonLd(page));
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(page.title)}</title>
    <meta name="description" content="${escapeHtml(page.description)}" />
    <link rel="canonical" href="${canonical}" />
    <meta name="robots" content="index,follow,max-snippet:-1,max-image-preview:large,max-video-preview:-1" />
    <meta property="og:title" content="${escapeHtml(page.title)}" />
    <meta property="og:description" content="${escapeHtml(page.description)}" />
    <meta property="og:type" content="${page.schema?.includes("Article") ? "article" : "website"}" />
    <meta property="og:url" content="${canonical}" />
    <meta property="og:image" content="${ogImage}" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${escapeHtml(page.title)}" />
    <meta name="twitter:description" content="${escapeHtml(page.description)}" />
    <meta name="twitter:image" content="${ogImage}" />
    <link rel="icon" href="/favicon.ico" sizes="any" />
    <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png" />
    <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png" />
    <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon-180.png" />
    <link rel="mask-icon" href="/mask-icon.svg" color="#7e6cff" />
    <link rel="manifest" href="/manifest.webmanifest" />
    ${cssLink}
    <script defer data-domain="violawake.com" src="https://plausible.io/js/script.file-downloads.outbound-links.tagged-events.js"></script>
    <script type="application/ld+json">${jsonLd}</script>
  </head>
  <body class="static-marketing">
    <div class="static-viola-banner">
      <div class="static-viola-banner-inner">
        <span>ViolaWake is the open wake word engine behind Viola, a voice assistant.</span>
        <a href="https://useviola.com">Come meet Viola &rarr;</a>
      </div>
    </div>
    ${renderNav(page)}
    <main class="static-main">
      <article class="static-prose">
        ${bodyHtml}
      </article>
    </main>
    ${renderFooter()}
    <script defer src="/bug-report.js"></script>
  </body>
</html>
`;
}

function blogIndexMarkdown() {
  return `# ViolaWake Blog

Technical notes on wake word training, open-source voice activation, deployment, and comparison research.

${blogPosts
  .map(
    (post) => `## [${post.title}](${post.path})

${post.description}

Published ${post.date}.
`,
  )
  .join("\n")}
`;
}

async function writeRoute(page, cssHref) {
  const htmlPath = routeHtmlPath(page.path);
  const mdPath = routeMarkdownPath(page.path);
  await mkdir(path.dirname(htmlPath), { recursive: true });
  await mkdir(path.dirname(mdPath), { recursive: true });
  const html = renderShell(page, markdownToHtml(page.markdown), cssHref);
  await writeFile(htmlPath, html, "utf8");
  await writeFile(mdPath, `${page.markdown.trim()}\n`, "utf8");
}

async function writeRobots() {
  const protectedRoutes = [
    "/api/",
    "/login",
    "/register",
    "/verify-email",
    "/forgot-password",
    "/reset-password",
    "/dashboard",
    "/record",
    "/training",
    "/billing",
    "/model",
    "/account",
    "/teams",
  ];
  const bots = [
    "Googlebot",
    "Bingbot",
    "GPTBot",
    "ChatGPT-User",
    "ClaudeBot",
    "PerplexityBot",
    "GoogleOther",
    "DuckDuckBot",
    "*",
  ];
  const sections = bots
    .map(
      (bot) => `User-agent: ${bot}
Allow: /
${protectedRoutes.map((route) => `Disallow: ${route}`).join("\n")}`,
    )
    .join("\n\n");
  await writeFile(
    path.join(distRoot, "robots.txt"),
    `${sections}\n\nSitemap: ${site.origin}/sitemap.xml\n`,
    "utf8",
  );
}

async function writeSitemap() {
  const priorityByPath = new Map(
    [
      ...pages,
      blogIndex,
      ...blogPosts,
    ].map((page) => [page.path, page.priority ?? "0.7"]),
  );
  const urls = allMarketingPaths
    .map(
      (route) => `  <url>
    <loc>${canonicalUrl(route)}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${route.startsWith("/blog") ? "monthly" : "weekly"}</changefreq>
    <priority>${priorityByPath.get(route) ?? "0.7"}</priority>
  </url>`,
    )
    .join("\n");
  await writeFile(
    path.join(distRoot, "sitemap.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`,
    "utf8",
  );
}

async function writeLlmsTxt() {
  const lines = [
    "# ViolaWake",
    "",
    "> Open-source custom wake word training, on-device ONNX inference, and an Apache 2.0 Python SDK.",
    "",
    "## Start here",
    "",
    "- [Home](https://violawake.com/): Product overview and quick answer.",
    "- [Docs](https://violawake.com/docs): SDK quickstart, API docs, and training links.",
    "- [Pricing](https://violawake.com/pricing): Free, Developer, Business, and Enterprise Console plans.",
    "- [FAQ](https://violawake.com/faq): Offline inference, Raspberry Pi, privacy, and licensing answers.",
    "",
    "## Comparisons",
    "",
    "- [ViolaWake vs Picovoice](https://violawake.com/compare/picovoice): Open-source alternative to Picovoice Porcupine.",
    "- [ViolaWake vs OpenWakeWord](https://violawake.com/compare/openwakeword): Honest upstream relationship and value-add.",
    "- [ViolaWake vs Snowboy](https://violawake.com/compare/snowboy): Replacement path for deprecated Snowboy projects.",
    "",
    "## Blog",
    "",
    ...blogPosts.map(
      (post) => `- [${post.title}](https://violawake.com${post.path}): ${post.description}`,
    ),
    "",
    "## Markdown mirrors",
    "",
    ...allMarketingPaths.map((route) => {
      const mdPath = route === "/" ? "/index.md" : `${route}.md`;
      return `- [${mdPath}](https://violawake.com${mdPath})`;
    }),
    "",
    "## Repository",
    "",
    `- [GitHub](${site.github}): SDK, Console, training code, benchmarks, and docs.`,
  ];
  await writeFile(path.join(distRoot, "llms.txt"), `${lines.join("\n")}\n`, "utf8");
}

async function writeManifest() {
  const manifest = {
    name: "ViolaWake",
    short_name: "ViolaWake",
    description: site.description,
    start_url: "/",
    display: "standalone",
    background_color: "#0f1117",
    theme_color: "#7e6cff",
    icons: [
      {
        src: "/favicon-16.png",
        sizes: "16x16",
        type: "image/png",
      },
      {
        src: "/favicon-32.png",
        sizes: "32x32",
        type: "image/png",
      },
      {
        src: "/apple-touch-icon-180.png",
        sizes: "180x180",
        type: "image/png",
      },
    ],
  };
  await writeFile(
    path.join(distRoot, "manifest.webmanifest"),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
}

async function writeRedirects() {
  const marketingLines = allMarketingPaths
    .filter((route) => route !== "/")
    .map((route) => `${route} ${route}/index.html 200`);
  const routes = [
    "/login",
    "/register",
    "/verify-email",
    "/forgot-password",
    "/reset-password",
    "/dashboard",
    "/record",
    "/record/*",
    "/training/*",
    "/billing",
    "/model/*",
    "/account/*",
    "/teams",
    "/teams/*",
  ];
  const lines = [
    ...marketingLines,
    ...routes.map((route) => `${route} ${appShellRedirectTarget} 200`),
  ];
  lines.push("/* /index.html 200");
  await writeFile(path.join(distRoot, "_redirects"), `${lines.join("\n")}\n`, "utf8");
}

async function copyAppShell() {
  const indexPath = path.join(distRoot, "index.html");
  let appShell = await readFile(indexPath, "utf8");
  if (!appShell.includes('name="robots"')) {
    appShell = appShell.replace(
      "<head>",
      '<head>\n    <meta name="robots" content="noindex,follow" />',
    );
  }
  await mkdir(path.join(distRoot, "app"), { recursive: true });
  // app/index.html backs the "/app/" rewrite target (the canonical directory
  // form — see appShellRedirectTarget above for why it must not be a file path).
  await writeFile(path.join(distRoot, "app", "index.html"), appShell, "utf8");
  await writeFile(path.join(distRoot, "app.html"), appShell, "utf8");
}

async function main() {
  if (!existsSync(distRoot)) {
    throw new Error("dist directory not found. Run vite build first.");
  }

  await copyAppShell();
  const cssHref = await findCssHref();

  const blogIndexPage = {
    ...blogIndex,
    markdown: blogIndexMarkdown(),
  };

  for (const page of [...pages, blogIndexPage, ...blogPosts]) {
    await writeRoute(page, cssHref);
  }

  await writeRobots();
  await writeSitemap();
  await writeLlmsTxt();
  await writeManifest();
  await writeRedirects();

  const staleSiteManifest = path.join(distRoot, "site.webmanifest");
  if (existsSync(staleSiteManifest)) {
    await rm(staleSiteManifest, { force: true });
  }

  console.log(
    `Generated ${pages.length + 1 + blogPosts.length} marketing routes and markdown mirrors in ${toPosix(distRoot)}`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
