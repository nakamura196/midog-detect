// 画面を開いて標本 1 枚を最後まで推論し、予測の点を JSON に書き出す（Python の結果と突き合わせるため）。
// 使い方: node scripts/check.mjs [URL] [出力] [--wasm]   別の標本は URL に ?slide=345
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";

const url = process.argv[2] ?? "http://localhost:5190/";
const out = process.argv[3] ?? "../out/web-points-006.json";
const wasm = process.argv.includes("--wasm");
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU"] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
page.on("console", (m) => console.log("[console]", m.text()));
const u = new URL(url);
if (wasm) u.searchParams.set("backend", "wasm");
await page.goto(u.toString(), { waitUntil: "commit" });
const tick = setInterval(async () => console.log("[status]", await page.textContent("#status").catch(() => "")), 10_000);
await page.waitForFunction(() => !document.getElementById("run").disabled || /エラー/.test(document.getElementById("status").textContent), null, { timeout: 600_000 });
console.log("backend:", await page.textContent("#backend"));
await page.click("#run");
await page.waitForFunction(() => window.__midog.done.size > 0 && !window.__midog.running, null, { timeout: 1_800_000, polling: 2000 });
console.log(await page.textContent("#status"));
console.log((await page.textContent("#result")).replace(/\s+/g, " "));
const preds = await page.evaluate(() => window.__midog.preds);
writeFileSync(out, JSON.stringify(preds));
await page.screenshot({ path: out.replace(/\.json$/, ".png") });
clearInterval(tick);
await browser.close();
