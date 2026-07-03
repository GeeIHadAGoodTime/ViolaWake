// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

// Wrangler generates POLICY_AUD and TEAM_DOMAIN as literal "" type because they
// are empty in wrangler.jsonc.  We need to widen them to `string` for production
// use without touching the generated file.  The Omit trick drops the literal keys
// from Cloudflare.Env before re-adding them as plain strings.
export type Env = Omit<Cloudflare.Env, "POLICY_AUD" | "TEAM_DOMAIN"> & {
	/** Cloudflare Access audience tag for this app. Required in production. */
	POLICY_AUD: string;
	/** Cloudflare Access team domain, e.g. "https://myteam.cloudflareaccess.com". */
	TEAM_DOMAIN: string;
};
