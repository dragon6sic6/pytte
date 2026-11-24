// pytte.js: run a Pytte .bin (see pytte/export.py) in the browser. Plain JavaScript, no
// dependencies. Mirrors run.c step by step: byte-level BPE tokenizer with GPT-2's
// pre-tokenization, Llama-style forward pass with a KV cache (int8 weights and int8
// activations, int32 accumulation), and temperature / top-k / top-p sampling.

const PRE_TOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Pytte {
  // ---------------------------------------------------------------- loading
  static async load(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`kunde inte hämta ${url}: ${res.status}`);
    const total = +res.headers.get("content-length") || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (onProgress) onProgress(got, total);
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    return new Pytte(buf.buffer);
  }

  constructor(buffer) {
    const dv = new DataView(buffer);
    let o = 0;
    const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
    const f32 = () => { const v = dv.getFloat32(o, true); o += 4; return v; };
    if (String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "PYTT") throw new Error("inte en Pytte-fil");
    o = 4;
    if (i32() !== 1) throw new Error("okänd filversion");
    this.dim = i32(); this.nLayers = i32(); this.nHeads = i32(); this.nKvHeads = i32(); this.ffDim = i32();
    this.vocab = i32(); this.maxSeq = i32(); this.quant = i32(); this.tie = i32();
    this.ropeTheta = f32(); this.normEps = f32();
    this.headDim = this.dim / this.nHeads;
    // tokenizer
    const nVocab = i32();
    this.tokBytes = new Array(nVocab);
    this.byteId = new Int32Array(256).fill(-1);
    const u8 = new Uint8Array(buffer);
    for (let i = 0; i < nVocab; i++) {
      const len = u8[o++];
      this.tokBytes[i] = u8.slice(o, o + len); o += len;
      if (len === 1) this.byteId[this.tokBytes[i][0]] = i;
    }
    this.eot = i32();
    const nMerges = i32();
    this.merges = new Map();   // key a*vocab+b -> [rank, result]
    for (let r = 0; r < nMerges; r++) { const a = i32(), b = i32(), c = i32(); this.merges.set(a * nVocab + b, [r, c]); }
    // weights: copy the rest into a fresh buffer so that every f32 view is 4-byte aligned
    const w = buffer.slice(o);
    let p = 0;
    const D = this.dim, H = this.nHeads, KV = this.nKvHeads, HD = this.headDim, FF = this.ffDim;
    const mat = (rows, cols) => {
      const m = { rows, cols };
      if (this.quant) {
        m.q = new Int8Array(w, p, rows * cols); p += rows * cols;
        m.s = new Float32Array(w, p, rows); p += rows * 4;
      } else { m.f = new Float32Array(w, p, rows * cols); p += rows * cols * 4; }
      return m;
    };
    const vec = (n) => { const v = new Float32Array(w, p, n); p += n * 4; return v; };
    this.embed = mat(this.vocab, D);
    this.layers = [];
    for (let l = 0; l < this.nLayers; l++) {
      this.layers.push({
        attnNorm: vec(D), wq: mat(H * HD, D), wk: mat(KV * HD, D), wv: mat(KV * HD, D), wo: mat(D, H * HD),
        mlpNorm: vec(D), w1: mat(FF, D), w3: mat(FF, D), w2: mat(D, FF),
      });
    }
    this.norm = vec(D);
    if (p !== w.byteLength) throw new Error("filen har fel längd");
    this.nParams = this.vocab * D + this.nLayers * (2 * D + D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * FF) + D;
    // state
    const KVD = KV * HD;
    this.x = new Float32Array(D); this.h = new Float32Array(D); this.q = new Float32Array(H * HD);
    this.k = new Float32Array(KVD); this.v = new Float32Array(KVD); this.att = new Float32Array(this.maxSeq);
    this.o = new Float32Array(H * HD); this.g = new Float32Array(FF); this.u = new Float32Array(FF);
    this.logits = new Float32Array(this.vocab);
    this.kc = []; this.vc = [];
    for (let l = 0; l < this.nLayers; l++) { this.kc.push(new Float32Array(this.maxSeq * KVD)); this.vc.push(new Float32Array(this.maxSeq * KVD)); }
    this.xq = new Int8Array(Math.max(D, FF, H * HD));
    this.order = new Int32Array(this.vocab); this.probs = new Float32Array(this.vocab);
    this.encoder = new TextEncoder();
  }

  // ---------------------------------------------------------------- tokenizer
  encodePiece(bytes, out) {
    let ids = Array.from(bytes, (b) => this.byteId[b]);
    for (;;) {
      let best = -1, bestRank = Infinity, bestRes = 0;
      for (let i = 0; i + 1 < ids.length; i++) {
        const m = this.merges.get(ids[i] * this.vocab + ids[i + 1]);
        if (m && m[0] < bestRank) { best = i; bestRank = m[0]; bestRes = m[1]; }
      }
      if (best < 0) break;
      ids.splice(best, 2, bestRes);
    }
    for (const id of ids) out.push(id);
  }

  encode(text) {
    const out = [];
    for (const m of text.matchAll(PRE_TOKENIZE)) this.encodePiece(this.encoder.encode(m[0]), out);
    return out;
  }

  // token -> its raw bytes (a token can be a partial UTF-8 character; use a streaming TextDecoder)
  tokenBytes(id) { return this.tokBytes[id]; }

  decode(ids) {
    const parts = ids.map((i) => this.tokBytes[i]);
    const n = parts.reduce((a, b) => a + b.length, 0);
    const buf = new Uint8Array(n); let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return new TextDecoder().decode(buf);
  }

  // ---------------------------------------------------------------- forward pass
  matvec(out, W, x) {
    const R = W.rows, C = W.cols;
    if (W.q) {
      let amax = 0;
      for (let c = 0; c < C; c++) { const a = Math.abs(x[c]); if (a > amax) amax = a; }
      const xs = amax > 0 ? amax / 127 : 1, inv = 1 / xs, xq = this.xq, q = W.q, s = W.s;
      for (let c = 0; c < C; c++) xq[c] = Math.round(x[c] * inv);
      for (let r = 0; r < R; r++) {
        const base = r * C; let acc = 0;
        for (let c = 0; c < C; c++) acc += q[base + c] * xq[c];
        out[r] = acc * s[r] * xs;
      }
    } else {
      const f = W.f;
      for (let r = 0; r < R; r++) {
        const base = r * C; let acc = 0;
        for (let c = 0; c < C; c++) acc += f[base + c] * x[c];
        out[r] = acc;
      }
    }
  }

  embedRow(id, out) {
    const W = this.embed, C = W.cols;
    if (W.q) { const s = W.s[id], base = id * C; for (let c = 0; c < C; c++) out[c] = W.q[base + c] * s; }
    else out.set(W.f.subarray(id * C, id * C + C));
  }

  rmsnorm(out, x, g) {
    let ss = 0; const n = x.length;
    for (let i = 0; i < n; i++) ss += x[i] * x[i];
    const r = 1 / Math.sqrt(ss / n + this.normEps);
    for (let i = 0; i < n; i++) out[i] = x[i] * r * g[i];
  }

  rope(v, nHeads, pos) {
    const hd = this.headDim, half = hd / 2;
    for (let i = 0; i < half; i++) {
      const a = pos * Math.pow(this.ropeTheta, -2 * i / hd), ca = Math.cos(a), sa = Math.sin(a);
      for (let h = 0; h < nHeads; h++) {
        const p = h * hd, x1 = v[p + i], x2 = v[p + i + half];
        v[p + i] = x1 * ca - x2 * sa; v[p + i + half] = x1 * sa + x2 * ca;
      }
    }
  }

  // one token at position pos; leaves the next-token logits in this.logits
  forward(token, pos) {
    const D = this.dim, H = this.nHeads, KV = this.nKvHeads, HD = this.headDim, KVD = KV * HD, FF = this.ffDim;
    const scale = 1 / Math.sqrt(HD), x = this.x, h = this.h;
    this.embedRow(token, x);
    for (let l = 0; l < this.nLayers; l++) {
      const L = this.layers[l], kc = this.kc[l], vc = this.vc[l];
      this.rmsnorm(h, x, L.attnNorm);
      this.matvec(this.q, L.wq, h); this.matvec(this.k, L.wk, h); this.matvec(this.v, L.wv, h);
      this.rope(this.q, H, pos); this.rope(this.k, KV, pos);
      kc.set(this.k, pos * KVD); vc.set(this.v, pos * KVD);
      const att = this.att, o = this.o, q = this.q;
      for (let hh = 0; hh < H; hh++) {
        const qb = hh * HD, kvh = Math.floor(hh / (H / KV)) * HD;
        let mx = -1e30;
        for (let t = 0; t <= pos; t++) {
          const kb = t * KVD + kvh; let sc = 0;
          for (let i = 0; i < HD; i++) sc += q[qb + i] * kc[kb + i];
          sc *= scale; att[t] = sc; if (sc > mx) mx = sc;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { att[t] = Math.exp(att[t] - mx); sum += att[t]; }
        o.fill(0, qb, qb + HD);
        for (let t = 0; t <= pos; t++) {
          const vb = t * KVD + kvh, p = att[t] / sum;
          for (let i = 0; i < HD; i++) o[qb + i] += p * vc[vb + i];
        }
      }
      this.matvec(h, L.wo, o);
      for (let i = 0; i < D; i++) x[i] += h[i];
      this.rmsnorm(h, x, L.mlpNorm);
      this.matvec(this.g, L.w1, h); this.matvec(this.u, L.w3, h);
      const g = this.g, u = this.u;
      for (let i = 0; i < FF; i++) { const a = g[i]; g[i] = a / (1 + Math.exp(-a)) * u[i]; }
      this.matvec(h, L.w2, g);
      for (let i = 0; i < D; i++) x[i] += h[i];
    }
    this.rmsnorm(h, x, this.norm);
    this.matvec(this.logits, this.embed, h);
  }

  // ---------------------------------------------------------------- sampling
  sample(temp, topP, topK, rng) {
    const logits = this.logits, n = this.vocab;
    if (temp <= 0) { let b = 0; for (let i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
    const order = this.order, probs = this.probs;
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => logits[b] - logits[a]);
    let keep = topK > 0 && topK < n ? topK : n;
    const mx = logits[order[0]]; let sum = 0;
    for (let i = 0; i < keep; i++) { probs[i] = Math.exp((logits[order[i]] - mx) / temp); sum += probs[i]; }
    if (topP < 1) {
      let cum = 0, k = 0;
      for (; k < keep; k++) { const p = probs[k] / sum; if (cum >= topP) break; cum += p; }
      keep = Math.max(k, 1); sum = 0;
      for (let i = 0; i < keep; i++) sum += probs[i];
    }
    const r = rng() * sum; let acc = 0;
    for (let i = 0; i < keep; i++) { acc += probs[i]; if (r < acc) return order[i]; }
    return order[keep - 1];
  }

  // probabilities of the most likely next tokens after the current forward() call
  topProbs(k = 10) {
    const logits = this.logits, n = this.vocab, order = this.order;
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => logits[b] - logits[a]);
    const mx = logits[order[0]]; let z = 0;
    for (let i = 0; i < n; i++) z += Math.exp(logits[i] - mx);
    const out = [];
    for (let i = 0; i < k; i++) out.push({ id: order[i], p: Math.exp(logits[order[i]] - mx) / z });
    return out;
  }
}

// xorshift64* with the same seeding as run.c, so a seed gives the same story on both
export function makeRng(seed) {
  let s = (BigInt(seed) * 0x9E3779B97F4A7C15n + 0x2545F4914F6CDD1Dn) & 0xFFFFFFFFFFFFFFFFn;
  const M = 0xFFFFFFFFFFFFFFFFn;
  return () => {
    s ^= s >> 12n; s ^= (s << 25n) & M; s ^= s >> 27n;
    const v = ((s * 2685821657736338717n) & M) >> 40n;
    return Number(v) / 16777216;
  };
}

// the chat format of pytte.prepare_sft: <|endoftext|>Du: ...\nPytte: ...<|endoftext|>
export function chatPromptIds(model, turns) {
  const ids = [model.eot];
  turns.forEach((t, i) => {
    if (i % 2 === 0) ids.push(...model.encode(`Du: ${t}\nPytte:`));
    else { ids.push(...model.encode(" " + t)); ids.push(model.eot); }
  });
  return ids;
}
