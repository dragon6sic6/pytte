// app.js: the three panels. All inference happens in worker.js; this file only moves text around.
const SIZES = [
  { key: "0.3m", label: "0,3 M", params: "0,33 M", mb: "0,4" },
  { key: "1m", label: "1 M", params: "1,3 M", mb: "1,4" },
  { key: "2m", label: "2 M", params: "2,2 M", mb: "2,4" },
  { key: "3m", label: "3 M", params: "3,4 M", mb: "3,6" },
  { key: "7m", label: "7 M", params: "7,2 M", mb: "7,3" },
  { key: "15m", label: "15 M", params: "15,7 M", mb: "16" },
];
const baseFile = (s) => `pytte-${s.key}.bin`;
const chatFile = (s) => `pytte-${s.key}-chat.bin`;

const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js", { type: "module" });
let nextId = 1;
const handlers = new Map();          // request id -> {onToken, onDone, onError}
const progressBars = new Map();      // file -> progress element waiting for it
const loaded = new Set();

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") {
    const bar = progressBars.get(m.file);
    if (bar) { bar.hidden = false; bar.firstElementChild.style.width = m.total ? `${100 * m.got / m.total}%` : "50%"; }
    return;
  }
  if (m.type === "loaded") { loaded.add(m.file); const bar = progressBars.get(m.file); if (bar) bar.hidden = true; return; }
  const h = handlers.get(m.id);
  if (!h) return;
  if (m.type === "token") h.onToken(m.text);
  else if (m.type === "done") { handlers.delete(m.id); h.onDone(m); }
  else if (m.type === "cancelled") { handlers.delete(m.id); if (h.onCancel) h.onCancel(); }
  else if (m.type === "error") { handlers.delete(m.id); h.onError(m.message); }
  else if (m.type === "tokens" || m.type === "probs") { handlers.delete(m.id); h.onDone(m); }
};

function request(msg, h) {
  const id = nextId++;
  handlers.set(id, h);
  worker.postMessage({ ...msg, id });
  return id;
}
function cancelAll() { worker.postMessage({ type: "cancel" }); handlers.clear(); }
function speed(m) { return m.secs > 0 ? `${Math.round(m.tokens / m.secs)} token/s` : ""; }

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.hidden = p.id !== `tab-${b.dataset.tab}`);
}));

// size pickers (chat and inside)
function sizePicker(el, initial, onChange) {
  let current = initial;
  SIZES.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("role", "radio"); b.textContent = s.label; b.setAttribute("aria-checked", s.key === current);
    b.addEventListener("click", () => { current = s.key; el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", x === b)); onChange(s); });
    el.appendChild(b);
  });
  return () => SIZES.find((s) => s.key === current);
}

// ------------------------------------------------------------------ 1. the slider
const growOut = $("grow-out"), growStatus = $("grow-status"), growTicks = $("grow-ticks"), growSize = $("grow-size");
let growSeed = Math.floor(Math.random() * 1e6);
SIZES.forEach((s) => { const t = document.createElement("span"); t.textContent = s.label; growTicks.appendChild(t); });
function markTick() { [...growTicks.children].forEach((t, i) => t.classList.toggle("on", i === +growSize.value)); }
markTick();

function growRun() {
  cancelAll();
  const s = SIZES[+growSize.value], prompt = $("grow-prompt").value;
  const file = baseFile(s);
  progressBars.set(file, $("grow-progress"));
  growOut.innerHTML = "";
  const p = document.createElement("span"); p.className = "prompt"; p.textContent = prompt; growOut.appendChild(p);
  const body = document.createElement("span"); growOut.appendChild(body);
  const cur = document.createElement("span"); cur.className = "cursor"; growOut.appendChild(cur);
  growStatus.textContent = `pytte-${s.key} · ${s.params} parametrar · ${s.mb} MB${loaded.has(file) ? "" : " · laddar…"}`;
  request({ type: "generate", file, prompt, seed: growSeed, temp: 0.8, topP: 0.95, maxNew: 250 }, {
    onToken: (t) => { body.textContent += t; },
    onDone: (m) => { cur.remove(); growStatus.textContent = `pytte-${s.key} · ${s.params} parametrar · ${s.mb} MB · ${m.tokens} token · ${speed(m)} · frö ${growSeed}`; },
    onCancel: () => cur.remove(),
    onError: (msg) => { cur.remove(); growStatus.textContent = `Fel: ${msg}`; },
  });
}
growSize.addEventListener("input", () => { markTick(); growRun(); });
$("grow-run").addEventListener("click", growRun);
$("grow-reseed").addEventListener("click", () => { growSeed = Math.floor(Math.random() * 1e6); growRun(); });
$("grow-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") growRun(); });

// ------------------------------------------------------------------ 2. chat
const chatLog = $("chat-log"), chatStatus = $("chat-status");
let turns = [];
const chatSize = sizePicker($("chat-sizes"), "7m", () => { chatStatus.textContent = ""; });
function bubble(cls, text) {
  const d = document.createElement("div"); d.className = `msg ${cls}`;
  if (cls === "bot") { const w = document.createElement("span"); w.className = "who"; w.textContent = "Pytte"; d.appendChild(w); }
  const t = document.createElement("span"); t.textContent = text; d.appendChild(t);
  chatLog.appendChild(d); chatLog.scrollTop = chatLog.scrollHeight;
  return t;
}
$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("chat-input").value.trim();
  if (!q) return;
  $("chat-input").value = "";
  cancelAll();
  if (turns.length % 2 === 1) turns.pop();          // a reply that was cut short is dropped
  turns.push(q); bubble("user", q);
  const s = chatSize(), file = chatFile(s);
  progressBars.set(file, $("chat-progress"));
  const span = bubble("bot", "");
  chatStatus.textContent = `pytte-${s.key}-chat${loaded.has(file) ? "" : " · laddar…"}`;
  let reply = "";
  request({ type: "generate", file, turns, seed: Math.floor(Math.random() * 1e6), temp: 0.8, topP: 0.95, maxNew: 300, stop: "\nDu:" }, {
    onToken: (t) => { reply += t; span.textContent = reply.trimStart(); chatLog.scrollTop = chatLog.scrollHeight; },
    onDone: (m) => { turns.push(reply.trim()); chatStatus.textContent = `pytte-${s.key}-chat · ${m.tokens} token · ${speed(m)}`; },
    onCancel: () => {},
    onError: (msg) => { chatStatus.textContent = `Fel: ${msg}`; },
  });
});
$("chat-reset").addEventListener("click", () => { cancelAll(); turns = []; chatLog.innerHTML = ""; chatStatus.textContent = ""; });

// ------------------------------------------------------------------ 3. under the hood
const insideTokens = $("inside-tokens"), insideProbs = $("inside-probs"), insideStatus = $("inside-status");
const insideSize = sizePicker($("inside-sizes"), "3m", () => insideRun());
let insideTimer = null;
const PALETTE = ["#FBE9B8", "#DCE8F7", "#E5F2DC", "#F7DDDC", "#EADDF7", "#F7EAD6"];
function insideRun() {
  const s = insideSize(), file = baseFile(s), text = $("inside-text").value;
  progressBars.set(file, $("inside-progress"));
  insideStatus.textContent = `pytte-${s.key}${loaded.has(file) ? "" : " · laddar…"}`;
  request({ type: "probs", file, text }, {
    onDone: (m) => {
      insideTokens.innerHTML = "";
      m.tokens.forEach((t, i) => {
        const d = document.createElement("span"); d.className = "tok"; d.style.background = PALETTE[i % PALETTE.length];
        d.textContent = t.text.replace(/ /g, "␣").replace(/\n/g, "⏎");
        const id = document.createElement("small"); id.textContent = `#${t.id}`; d.appendChild(id);
        insideTokens.appendChild(d);
      });
      insideProbs.innerHTML = "";
      m.top.forEach((t) => {
        const a = document.createElement("span"); a.className = "t"; a.textContent = t.text.replace(/ /g, "␣").replace(/\n/g, "⏎");
        const b = document.createElement("div"); b.className = "bar"; const f = document.createElement("div"); f.style.width = `${Math.max(1, 100 * t.p)}%`; b.appendChild(f);
        const c = document.createElement("span"); c.className = "p"; c.textContent = `${(100 * t.p).toFixed(1).replace(".", ",")} %`;
        insideProbs.append(a, b, c);
      });
      insideStatus.textContent = `pytte-${s.key} · ${m.tokens.length} token`;
    },
    onToken: () => {}, onError: (msg) => { insideStatus.textContent = `Fel: ${msg}`; },
  });
}
$("inside-text").addEventListener("input", () => { clearTimeout(insideTimer); insideTimer = setTimeout(insideRun, 250); });
document.querySelector('[data-tab="inside"]').addEventListener("click", () => { if (!insideTokens.childElementCount) insideRun(); });

// start with a story
growRun();
