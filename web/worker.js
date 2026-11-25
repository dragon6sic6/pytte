// worker.js: keeps the model off the UI thread. Messages in:
//   {type:"load", file}                               -> progress + loaded
//   {type:"generate", id, file, prompt|turns, seed, temp, topP, topK, maxNew, stop}
//                                                     -> token ... done / cancelled
//   {type:"cancel"}
//   {type:"tokenize", id, file, text}                 -> tokens [{id, text}]
//   {type:"probs", id, file, text}                    -> tokens + top next-token probabilities
//   {type:"trace", id, file, text|ids, vocab}         -> everything the model computes for a short text (explainer page)
//   {type:"neighbors", id, file, token, k}            -> the tokens with the most similar embedding rows
import { Pytte, makeRng, chatPromptIds } from "./pytte.js?v=4";

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

const tokText = (model, id) => new TextDecoder().decode(model.tokenBytes(id));

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
    if (m.type === "neighbors") {
      postMessage({ type: "neighbors", id: m.id, token: m.token, list: model.neighbors(m.token, m.k ?? 8).map((t) => ({ ...t, text: tokText(model, t.id) })) });
      return;
    }
    if (m.type === "trace") {
      const MAXT = 40, L = model.nLayers, H = model.nHeads, D = model.dim;
      const ids = [model.eot, ...(m.ids ?? model.encode(m.text))].slice(0, MAXT), T = ids.length;
      const emb = new Float32Array(T * D), att = new Float32Array(L * H * T * T), upd = new Float32Array(T * L * 2);
      let last = null;
      for (let pos = 0; pos < T; pos++) {
        model.tap = {};
        model.forward(ids[pos], pos);
        const tap = model.tap; model.tap = null;
        emb.set(tap.emb, pos * D); upd.set(tap.upd, pos * L * 2);
        for (let lh = 0; lh < L * H; lh++) att.set(tap.att[lh], (lh * T + pos) * T);
        last = tap;
      }
      const named = (list) => list.map((t) => ({ ...t, text: tokText(model, t.id) }));
      const lens = [named(Pytte.top(model.lensLogits(last.emb), 5)), ...last.resid.map((x) => named(Pytte.top(model.lensLogits(x), 5)))];
      const logits = model.logits.slice();
      const out = { type: "trace", id: m.id, T, L, H, D, vocabSize: model.vocab, eot: model.eot, tokens: tokenList(model, ids), ids: ids.slice(1),
                    text: new TextDecoder().decode(Uint8Array.from(ids.slice(1).flatMap((i) => [...model.tokenBytes(i)]))), emb, att, upd, lens, logits };
      if (m.vocab) out.vocab = Array.from({ length: model.vocab }, (_, i) => tokText(model, i));
      postMessage(out, [emb.buffer, att.buffer, upd.buffer, logits.buffer]);
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
