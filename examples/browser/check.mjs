// The one command. Serves the artifacts, drives the page in whatever browsers
// are installed, and fails if any posterior has drifted from nutpie's.

import { chromium, firefox, webkit } from "playwright";
import { server } from "../../serve.mjs";

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
    const { MODELS, compare } = await import("/examples/browser/app.js");
    const out = [];
    for (const m of MODELS) out.push(await compare(m));
    return out;
  });
  // ArviZ over the four chains, through posteriorwasm; Pyodide comes from jsDelivr.
  const diag = await page.evaluate(async () => {
    const { sample } = await import("/examples/browser/app.js");
    const { createAnalyzer } = await import("/posteriorwasm/index.js");
    const r = await sample("linear_regression");
    const analyzer = createAnalyzer();
    const res = await analyzer.analyze({ names: r.meta.paramNames, chains: r.chains, diverging: r.diverging });
    analyzer.terminate();
    return res.summary;
  });
  await browser.close();

  const worstRhat = Math.max(...diag.map((s) => s.rHat));
  const leastEss = Math.min(...diag.map((s) => s.essBulk));
  console.log(`${name.padEnd(9)} ArviZ on linear_regression: r_hat ≤ ${worstRhat.toFixed(3)}, ess_bulk ≥ ${leastEss.toFixed(0)}`);
  if (!(worstRhat < 1.01)) failed = true;

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
