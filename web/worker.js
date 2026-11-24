// worker.js: keeps the model off the UI thread. Messages in:
//   {type:"load", file}                               -> progress + loaded
//   {type:"generate", id, file, prompt|turns, seed, temp, topP, topK, maxNew, stop}
//                                                     -> token ... done / cancelled
//   {type:"cancel"}
//   {type:"tokenize", id, file, text}                 -> tokens [{id, text}]
//   {type:"probs", id, file, text}                    -> tokens + top next-token probabilities
import { Pytte, makeRng, chatPromptIds } from "./pytte.js";

const models = new Map();
let cancelled = false;

async function get(file) {
  if (models.has(file)) return models.get(file);
  const url = `models/${file}`;
  const model = await Pytte.load(url, (got, total) => postMessage({ type: "progress", file, got, total }));
  models.set(file, model);
  postMessage({ type: "loaded", file, nParams: model.nParams, bytes: 0 });
  return model;
}

function tokenList(model, ids) {
  return ids.map((id) => ({ id, text: new TextDecoder().decode(model.tokenBytes(id)) }));
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "cancel") { cancelled = true; return; }
    if (m.type === "load") { await get(m.file); return; }
    const model = await get(m.file);
    if (m.type === "tokenize") {
      postMessage({ type: "tokens", id: m.id, tokens: tokenList(model, model.encode(m.text)) });
      return;
    }
    if (m.type === "probs") {
      const ids = [model.eot, ...model.encode(m.text)];
      for (let i = 0; i < ids.length && i < model.maxSeq; i++) model.forward(ids[i], i);
      const top = model.topProbs(12).map((t) => ({ ...t, text: new TextDecoder().decode(model.tokenBytes(t.id)) }));
      postMessage({ type: "probs", id: m.id, tokens: tokenList(model, ids.slice(1)), top });
      return;
    }
    if (m.type === "generate") {
      cancelled = false;
      const ids = m.turns ? chatPromptIds(model, m.turns) : (m.prompt ? model.encode(m.prompt) : [model.eot]);
      if (ids.length >= model.maxSeq) { postMessage({ type: "error", id: m.id, message: "texten är längre än modellens kontext" }); return; }
      const rng = makeRng(m.seed ?? 0);
      const t0 = performance.now();
      let pos = 0;
      for (const id of ids) model.forward(id, pos++);
      const decoder = new TextDecoder();
      let n = 0, pending = "";
      const stop = m.stop || null;
      while (n < (m.maxNew ?? 300) && pos < model.maxSeq) {
        const id = model.sample(m.temp ?? 0.8, m.topP ?? 0.95, m.topK ?? 0, rng);
        if (id === model.eot) break;
        n++;
        let text = decoder.decode(model.tokenBytes(id), { stream: true });
        if (stop) {   // hold back output that may be the start of the stop string
          pending += text; text = "";
          if (pending.includes(stop)) { text = pending.slice(0, pending.indexOf(stop)); pending = ""; postMessage({ type: "token", id: m.id, text }); break; }
          let hold = 0;
          for (let k = Math.min(stop.length - 1, pending.length); k > 0; k--) if (stop.startsWith(pending.slice(-k))) { hold = k; break; }
          text = pending.slice(0, pending.length - hold); pending = pending.slice(pending.length - hold);
        }
        if (text) postMessage({ type: "token", id: m.id, text });
        model.forward(id, pos++);
        if (n % 4 === 0) await new Promise((r) => setTimeout(r, 0));   // let a cancel message through
        if (cancelled) { postMessage({ type: "cancelled", id: m.id }); return; }
      }
      if (pending) postMessage({ type: "token", id: m.id, text: pending });
      const secs = (performance.now() - t0) / 1000;
      postMessage({ type: "done", id: m.id, tokens: n, secs, promptTokens: ids.length });
    }
  } catch (err) {
    postMessage({ type: "error", id: m.id, message: String(err.message || err) });
  }
};
