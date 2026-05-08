import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(scriptDir, "..");

const recordPage = readFileSync(resolve(root, "src/pages/Record.tsx"), "utf8");
const api = readFileSync(resolve(root, "src/api.ts"), "utf8");

const checks = [
  ["record tab is present", /Record\s*<\/button>/.test(recordPage)],
  ["upload tab is present", /Upload files/.test(recordPage)],
  ["file picker accept attribute is wired", /accept=\{ACCEPTED_AUDIO\}/.test(recordPage)],
  ["record page calls bulk upload helper", /bulkUploadRecordings/.test(recordPage)],
  ["bulk upload helper posts to endpoint", /\/recordings\/bulk-upload/.test(api)],
];

const failures = checks.filter(([, passed]) => !passed);

if (failures.length > 0) {
  for (const [name] of failures) {
    console.error(`FAIL: ${name}`);
  }
  process.exit(1);
}

for (const [name] of checks) {
  console.log(`PASS: ${name}`);
}
