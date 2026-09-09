// The one command. Serves the artifacts, drives the page in whatever browsers
// are installed, and fails if any posterior has drifted from nutpie's.

import { chromium, firefox, webkit } from "playwright";
import { server } from "./serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const wanted = (process.env.BROWSERS ?? "chromium,firefox,webkit").split(",");
const engines = { chromium, firefox, webkit };

let failed = false;
for (const name of wanted) {
  const launcher = engines[name.trim()];
  if (!launcher) continue;
  let browser;
  try {
    browser = await launcher.launch();
  } catch (e) {
    // A missing engine is not a failing result; say so and move on.
    console.log(`${name.padEnd(9)} not installed (npx playwright install ${name})`);
    continue;
  }
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${PORT}/`);

  const results = await page.evaluate(async () => {
    const { MODELS, compare } = await import("/demo/app.js");
    const out = [];
    for (const m of MODELS) out.push(await compare(m));
    return out;
  });
  await browser.close();

  if (errors.length) {
    console.log(`${name.padEnd(9)} page errors: ${errors.join("; ")}`);
    failed = true;
    continue;
  }
  const worst = Math.max(...results.map((r) => r.worst));
  const slow = results.reduce((a, r) => a + r.ms, 0);
  console.log(
    `${name.padEnd(9)} ${results.length} models, worst ${worst.toFixed(3)} sd, ` +
    `${slow.toFixed(0)} ms total`,
  );
  for (const r of results) {
    if (r.worst > 0.3) {
      console.log(`    ${r.name}: ${r.worst.toFixed(3)} sd`);
      failed = true;
    }
  }
}

server.close();
process.exit(failed ? 1 : 0);
