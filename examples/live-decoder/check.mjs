// Drives the page in a real browser: clicks train, waits for it to finish,
// and fails if the fit lands far worse than the decoder reliably reaches.

import { chromium, firefox, webkit } from "playwright";
import { server } from "../../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const wanted = (process.env.BROWSERS ?? "chromium,firefox,webkit").split(",");
const engines = { chromium, firefox, webkit };

// Measured range across runs is ~0.020-0.026; the linear PPCA this replaced
// never got below ~0.047, so this is a generous ceiling, not a tight bound.
const MAX_MSE = 0.035;

let failed = false;
for (const name of wanted) {
  const launcher = engines[name.trim()];
  if (!launcher) continue;
  let browser;
  try {
    browser = await launcher.launch();
  } catch {
    console.log(`${name.padEnd(9)} not installed (npx playwright install ${name})`);
    continue;
  }
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${PORT}/examples/live-decoder/`);

  const t0 = Date.now();
  await page.click("#go");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.startsWith("done in"),
    null,
    { timeout: 60_000 },
  );
  const wallMs = Date.now() - t0;
  const status = await page.textContent("#status");
  const thumbCount = await page.locator(".thumb").count();
  await browser.close();

  if (errors.length) {
    console.log(`${name.padEnd(9)} page errors: ${errors.join("; ")}`);
    failed = true;
    continue;
  }

  const mseMatch = status.match(/reconstruction MSE ([\d.]+)/);
  const mse = mseMatch ? parseFloat(mseMatch[1]) : NaN;
  const ok = Number.isFinite(mse) && mse < MAX_MSE && thumbCount === 32;
  console.log(
    `${name.padEnd(9)} ${ok ? "ok" : "FAIL"} mse=${mse} thumbs=${thumbCount} wall=${wallMs}ms`,
  );
  if (!ok) failed = true;
}

server.close();
process.exit(failed ? 1 : 0);
