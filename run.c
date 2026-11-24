/* run.c: run a Pytte model from one .bin file. No dependencies, one file.

   cc -O2 -o run run.c -lm
   ./run pytte-7m-chat.bin -p "Det var en gång"        continue a text
   ./run pytte-7m-chat.bin                             a story from nothing
   ./run pytte-7m-chat.bin -chat                       talk to a chat-tuned model (tom rad avslutar, /ny börjar om)
   ./run pytte-7m-chat.bin -chat -p "Berätta en saga om en älg."
   ./run pytte-7m-chat.bin -tok "midsommarafton"       show the tokens
   ./run pytte-7m-chat.bin -score "Hon såg en älg."    log-probability of a text (for the grammar test)
   options: -n max_new (300) -t temperature (0.8) -top_p (0.95) -k top_k (0) -s seed (time) -v (print tok/s)

   The whole forward pass of a Llama-style decoder is here, in order:
   byte-level BPE tokenizer -> embedding -> N x [RMSNorm -> attention with RoPE and a KV cache
   -> residual -> RMSNorm -> SwiGLU -> residual] -> RMSNorm -> tied LM head -> sampling.
   File format: see pytte/export.py. Weights are int8 per-row (or f32 with --f32 export). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ model file */
typedef struct { int8_t *q; float *s; float *f; int rows, cols; } Mat;   /* q+s (int8) or f (f32) */

typedef struct {
    int dim, n_layers, n_heads, n_kv_heads, ff_dim, vocab, max_seq, quant, tie;
    float rope_theta, norm_eps;
    int head_dim;
    /* tokenizer */
    int n_vocab, eot, n_merges;
    unsigned char **tok_bytes; unsigned char *tok_len;
    int byte_id[256];
    int *pair_keys; int *pair_rank; int *pair_res; int pair_cap;   /* hash: (a*vocab+b) -> rank, result */
    /* weights */
    Mat embed; float *norm;
    float **attn_norm, **mlp_norm;
    Mat *wq, *wk, *wv, *wo, *w1, *w3, *w2;
} Model;

static void die(const char *m) { fprintf(stderr, "run: %s\n", m); exit(1); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }
static void rd(void *p, size_t n, FILE *f) { if (fread(p, 1, n, f) != n) die("truncated model file"); }
static int rdi(FILE *f) { int32_t v; rd(&v, 4, f); return v; }
static float rdf(FILE *f) { float v; rd(&v, 4, f); return v; }

static void read_mat(Model *m, Mat *w, int rows, int cols, FILE *f) {
    w->rows = rows; w->cols = cols; w->q = NULL; w->s = NULL; w->f = NULL;
    if (m->quant) {
        w->q = xmalloc((size_t)rows * cols); rd(w->q, (size_t)rows * cols, f);
        w->s = xmalloc((size_t)rows * 4);    rd(w->s, (size_t)rows * 4, f);
    } else {
        w->f = xmalloc((size_t)rows * cols * 4); rd(w->f, (size_t)rows * cols * 4, f);
    }
}
static float *read_vec(int n, FILE *f) { float *v = xmalloc((size_t)n * 4); rd(v, (size_t)n * 4, f); return v; }

static void pair_put(Model *m, int a, int b, int rank, int res) {
    uint64_t key = (uint64_t)a * m->n_vocab + b;
    unsigned h = (unsigned)(key * 2654435761u) & (m->pair_cap - 1);
    while (m->pair_keys[h] >= 0) h = (h + 1) & (m->pair_cap - 1);
    m->pair_keys[h] = (int)key; m->pair_rank[h] = rank; m->pair_res[h] = res;
}
static int pair_get(Model *m, int a, int b, int *res) {   /* returns rank or -1 */
    uint64_t key = (uint64_t)a * m->n_vocab + b;
    unsigned h = (unsigned)(key * 2654435761u) & (m->pair_cap - 1);
    while (m->pair_keys[h] >= 0) {
        if (m->pair_keys[h] == (int)key) { *res = m->pair_res[h]; return m->pair_rank[h]; }
        h = (h + 1) & (m->pair_cap - 1);
    }
    return -1;
}

static void load_model(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open model file");
    char magic[4]; rd(magic, 4, f);
    if (memcmp(magic, "PYTT", 4)) die("not a Pytte .bin file");
    if (rdi(f) != 1) die("unsupported file version");
    m->dim = rdi(f); m->n_layers = rdi(f); m->n_heads = rdi(f); m->n_kv_heads = rdi(f); m->ff_dim = rdi(f);
    m->vocab = rdi(f); m->max_seq = rdi(f); m->quant = rdi(f); m->tie = rdi(f);
    m->rope_theta = rdf(f); m->norm_eps = rdf(f);
    m->head_dim = m->dim / m->n_heads;
    if (!m->tie) die("untied embeddings are not supported");
    /* tokenizer */
    m->n_vocab = rdi(f);
    m->tok_bytes = xmalloc(sizeof(char *) * m->n_vocab);
    m->tok_len = xmalloc(m->n_vocab);
    for (int i = 0; i < 256; i++) m->byte_id[i] = -1;
    for (int i = 0; i < m->n_vocab; i++) {
        unsigned char len; rd(&len, 1, f);
        m->tok_len[i] = len;
        m->tok_bytes[i] = xmalloc(len + 1); rd(m->tok_bytes[i], len, f); m->tok_bytes[i][len] = 0;
        if (len == 1) m->byte_id[m->tok_bytes[i][0]] = i;
    }
    m->eot = rdi(f);
    m->n_merges = rdi(f);
    m->pair_cap = 1; while (m->pair_cap < 4 * m->n_merges) m->pair_cap <<= 1;
    m->pair_keys = xmalloc(4 * m->pair_cap); m->pair_rank = xmalloc(4 * m->pair_cap); m->pair_res = xmalloc(4 * m->pair_cap);
    for (int i = 0; i < m->pair_cap; i++) m->pair_keys[i] = -1;
    for (int r = 0; r < m->n_merges; r++) { int a = rdi(f), b = rdi(f), c = rdi(f); pair_put(m, a, b, r, c); }
    /* weights */
    int L = m->n_layers, D = m->dim, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim, FF = m->ff_dim;
    read_mat(m, &m->embed, m->vocab, D, f);
    m->attn_norm = xmalloc(sizeof(float *) * L); m->mlp_norm = xmalloc(sizeof(float *) * L);
    m->wq = xmalloc(sizeof(Mat) * L); m->wk = xmalloc(sizeof(Mat) * L); m->wv = xmalloc(sizeof(Mat) * L);
    m->wo = xmalloc(sizeof(Mat) * L); m->w1 = xmalloc(sizeof(Mat) * L); m->w3 = xmalloc(sizeof(Mat) * L); m->w2 = xmalloc(sizeof(Mat) * L);
    for (int l = 0; l < L; l++) {
        m->attn_norm[l] = read_vec(D, f);
        read_mat(m, &m->wq[l], H * HD, D, f);
        read_mat(m, &m->wk[l], KV * HD, D, f);
        read_mat(m, &m->wv[l], KV * HD, D, f);
        read_mat(m, &m->wo[l], D, H * HD, f);
        m->mlp_norm[l] = read_vec(D, f);
        read_mat(m, &m->w1[l], FF, D, f);
        read_mat(m, &m->w3[l], FF, D, f);
        read_mat(m, &m->w2[l], D, FF, f);
    }
    m->norm = read_vec(D, f);
    unsigned char extra; if (fread(&extra, 1, 1, f) == 1) die("trailing bytes in model file");
    fclose(f);
}

/* ------------------------------------------------------------------ tokenizer */
/* GPT-2 style pre-tokenization: 's|'t|'re|'ve|'m|'ll|'d | ?letters+ | ?digits+ | ?other+ | ws+(?!\S) | ws+ */
static int utf8_decode(const unsigned char *s, int n, int *cp) {
    if (n <= 0) return 0;
    unsigned c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 6 && n >= 2) { *cp = ((c & 31) << 6) | (s[1] & 63); return 2; }
    if ((c >> 4) == 14 && n >= 3) { *cp = ((c & 15) << 12) | ((s[1] & 63) << 6) | (s[2] & 63); return 3; }
    if ((c >> 3) == 30 && n >= 4) { *cp = ((c & 7) << 18) | ((s[1] & 63) << 12) | ((s[2] & 63) << 6) | (s[3] & 63); return 4; }
    *cp = c; return 1;   /* invalid byte: treat as one symbol */
}
static int is_space(int c) { return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x2028 || c == 0x2029 || c == 0x3000; }
static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_letter(int c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (c < 0x80) return 0;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return 1;
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;      /* Latin-1, Latin Extended A/B */
    if (c >= 0x250 && c <= 0x2AF) return 1;                          /* IPA */
    if (c >= 0x370 && c <= 0x1FFF) return c != 0x37E && c != 0x387;  /* Greek, Cyrillic, ... (approximation) */
    if (c >= 0x3040 && c <= 0x9FFF) return 1;                        /* CJK (approximation) */
    return 0;
}
static int cls(int c) { return is_space(c) ? 1 : is_letter(c) ? 2 : is_digit(c) ? 3 : 4; }

/* BPE-encode one pre-token (bytes s[0..n)) into ids; returns count */
static int bpe_piece(Model *m, const unsigned char *s, int n, int *ids) {
    int k = 0;
    for (int i = 0; i < n; i++) { int id = m->byte_id[s[i]]; if (id < 0) die("byte without a token"); ids[k++] = id; }
    for (;;) {
        int best = -1, best_rank = 0x7fffffff, best_res = 0;
        for (int i = 0; i + 1 < k; i++) {
            int res, r = pair_get(m, ids[i], ids[i + 1], &res);
            if (r >= 0 && r < best_rank) { best = i; best_rank = r; best_res = res; }
        }
        if (best < 0) break;
        ids[best] = best_res;
        memmove(&ids[best + 1], &ids[best + 2], sizeof(int) * (k - best - 2));
        k--;
    }
    return k;
}

static int encode(Model *m, const char *text, int *ids, int max_ids) {
    const unsigned char *s = (const unsigned char *)text;
    int n = (int)strlen(text), i = 0, k = 0, cp, len;
    while (i < n) {
        int start = i;
        /* contractions */
        static const char *con[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        int matched = 0;
        for (int c = 0; c < 7; c++) {
            int cl = (int)strlen(con[c]);
            if (i + cl <= n && !memcmp(s + i, con[c], cl)) { i += cl; matched = 1; break; }
        }
        if (!matched) {
            int j = i;
            if (s[j] == ' ') j++;
            len = utf8_decode(s + j, n - j, &cp);
            int c = len ? cls(cp) : 1;
            if (len && c != 1) {           /* ?letters+ | ?digits+ | ?other+ */
                while (j < n) { len = utf8_decode(s + j, n - j, &cp); if (cls(cp) != c) break; j += len; }
                i = j;
            } else {                       /* whitespace run */
                j = i; int last = i;
                while (j < n) { len = utf8_decode(s + j, n - j, &cp); if (!is_space(cp)) break; last = j; j += len; }
                if (j < n && last > i) i = last;        /* ws+(?!\S): leave the last space for the next word */
                else if (j < n) i = j;                  /* single space before a non-space that nothing else took */
                else i = j;                             /* trailing whitespace */
            }
        }
        if (k + (i - start) > max_ids) die("prompt too long");
        k += bpe_piece(m, s + start, i - start, ids + k);
    }
    return k;
}

/* ------------------------------------------------------------------ forward pass */
typedef struct {
    float *x, *h, *q, *k, *v, *att, *o, *g, *u, *logits;
    float *kc, *vc;      /* KV cache [layer][pos][kv_heads*head_dim] */
} State;

static void state_alloc(Model *m, State *s) {
    int D = m->dim, KVD = m->n_kv_heads * m->head_dim;
    s->x = xmalloc(4 * D); s->h = xmalloc(4 * D); s->q = xmalloc(4 * m->n_heads * m->head_dim);
    s->k = xmalloc(4 * KVD); s->v = xmalloc(4 * KVD); s->att = xmalloc(4 * m->max_seq);
    s->o = xmalloc(4 * m->n_heads * m->head_dim); s->g = xmalloc(4 * m->ff_dim); s->u = xmalloc(4 * m->ff_dim);
    s->logits = xmalloc(4 * m->vocab);
    s->kc = xmalloc(4 * (size_t)m->n_layers * m->max_seq * KVD);
    s->vc = xmalloc(4 * (size_t)m->n_layers * m->max_seq * KVD);
}

static void matvec(float *out, const Mat *w, const float *x) {   /* out[rows] = W[rows,cols] x */
    int R = w->rows, C = w->cols;
    if (w->q) {
        /* Quantise the activation to int8 once (symmetric, one scale), then every row is an
           int8 x int8 dot product accumulated in int32, which the compiler vectorises; the
           float version of this loop did not vectorise and ran 14x slower. */
        static int8_t xq[8192];
        if (C > 8192) die("activation too wide");
        float amax = 0; for (int c = 0; c < C; c++) { float a = fabsf(x[c]); if (a > amax) amax = a; }
        float xs = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / xs;
        for (int c = 0; c < C; c++) xq[c] = (int8_t)lrintf(x[c] * inv);
        for (int r = 0; r < R; r++) {
            const int8_t *row = w->q + (size_t)r * C; int32_t acc = 0;
            for (int c = 0; c < C; c++) acc += (int32_t)row[c] * (int32_t)xq[c];
            out[r] = (float)acc * w->s[r] * xs;
        }
    } else {
        for (int r = 0; r < R; r++) {
            const float *row = w->f + (size_t)r * C; float acc = 0;
            for (int c = 0; c < C; c++) acc += row[c] * x[c];
            out[r] = acc;
        }
    }
}
static void embed_row(const Mat *w, int id, float *out) {
    int C = w->cols;
    if (w->q) for (int c = 0; c < C; c++) out[c] = w->q[(size_t)id * C + c] * w->s[id];
    else memcpy(out, w->f + (size_t)id * C, 4 * C);
}
static void rmsnorm(float *out, const float *x, const float *g, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * g[i];
}
static void rope(float *v, int n_heads, int hd, int pos, float theta) {   /* MLX nn.RoPE, traditional=False */
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(theta, -2.0f * i / hd), a = pos * freq, ca = cosf(a), sa = sinf(a);
        for (int h = 0; h < n_heads; h++) {
            float *p = v + h * hd, x1 = p[i], x2 = p[i + half];
            p[i] = x1 * ca - x2 * sa; p[i + half] = x1 * sa + x2 * ca;
        }
    }
}

/* run one token at position pos; leaves next-token logits in s->logits */
static void forward(Model *m, State *s, int token, int pos) {
    int D = m->dim, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim, KVD = KV * HD, FF = m->ff_dim;
    float scale = 1.0f / sqrtf((float)HD);
    embed_row(&m->embed, token, s->x);
    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(s->h, s->x, m->attn_norm[l], D, m->norm_eps);
        matvec(s->q, &m->wq[l], s->h); matvec(s->k, &m->wk[l], s->h); matvec(s->v, &m->wv[l], s->h);
        rope(s->q, H, HD, pos, m->rope_theta); rope(s->k, KV, HD, pos, m->rope_theta);
        float *kc = s->kc + ((size_t)l * m->max_seq + pos) * KVD, *vc = s->vc + ((size_t)l * m->max_seq + pos) * KVD;
        memcpy(kc, s->k, 4 * KVD); memcpy(vc, s->v, 4 * KVD);
        for (int h = 0; h < H; h++) {
            float *q = s->q + h * HD; int kvh = h / (H / KV);
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                float *kt = s->kc + ((size_t)l * m->max_seq + t) * KVD + kvh * HD, sc = 0;
                for (int i = 0; i < HD; i++) sc += q[i] * kt[i];
                s->att[t] = sc * scale; if (s->att[t] > mx) mx = s->att[t];
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { s->att[t] = expf(s->att[t] - mx); sum += s->att[t]; }
            float *o = s->o + h * HD; memset(o, 0, 4 * HD);
            for (int t = 0; t <= pos; t++) {
                float *vt = s->vc + ((size_t)l * m->max_seq + t) * KVD + kvh * HD, p = s->att[t] / sum;
                for (int i = 0; i < HD; i++) o[i] += p * vt[i];
            }
        }
        matvec(s->h, &m->wo[l], s->o);
        for (int i = 0; i < D; i++) s->x[i] += s->h[i];
        rmsnorm(s->h, s->x, m->mlp_norm[l], D, m->norm_eps);
        matvec(s->g, &m->w1[l], s->h); matvec(s->u, &m->w3[l], s->h);
        for (int i = 0; i < FF; i++) { float g = s->g[i]; s->g[i] = g / (1.0f + expf(-g)) * s->u[i]; }
        matvec(s->h, &m->w2[l], s->g);
        for (int i = 0; i < D; i++) s->x[i] += s->h[i];
    }
    rmsnorm(s->h, s->x, m->norm, D, m->norm_eps);
    matvec(s->logits, &m->embed, s->h);
}

/* ------------------------------------------------------------------ sampling */
static uint64_t rng_state = 88172645463325252ull;
static float rng_float(void) {   /* xorshift64*, uniform [0,1) */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (float)((rng_state * 2685821657736338717ull) >> 40) / (float)(1ull << 24);
}
static const float *g_sort_logits;   /* qsort_r differs between macOS and glibc; a static is portable */
static int cmp_desc(const void *a, const void *b) {
    float d = g_sort_logits[*(const int *)b] - g_sort_logits[*(const int *)a];
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}
static int sample(float *logits, int n, float temp, float top_p, int top_k, int *order, float *probs) {
    if (temp <= 0) { int best = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i; return best; }
    for (int i = 0; i < n; i++) order[i] = i;
    g_sort_logits = logits; qsort(order, n, sizeof(int), cmp_desc);
    int keep = (top_k > 0 && top_k < n) ? top_k : n;
    float mx = logits[order[0]], sum = 0;
    for (int i = 0; i < keep; i++) { probs[i] = expf((logits[order[i]] - mx) / temp); sum += probs[i]; }
    if (top_p < 1.0f) {   /* nucleus: keep tokens until the mass passes top_p (same rule as pytte.sample) */
        float cum = 0; int k = 0;
        for (; k < keep; k++) { float p = probs[k] / sum; if (cum >= top_p) break; cum += p; }
        keep = k > 0 ? k : 1;
        sum = 0; for (int i = 0; i < keep; i++) sum += probs[i];
    }
    float r = rng_float() * sum, acc = 0;
    for (int i = 0; i < keep; i++) { acc += probs[i]; if (r < acc) return order[i]; }
    return order[keep - 1];
}

/* ------------------------------------------------------------------ generation */
typedef struct { const char *stop; int matched; } Stopper;   /* holds back output that may be the stop string */

static void emit(Model *m, int id, Stopper *st) {
    unsigned char *b = m->tok_bytes[id]; int n = m->tok_len[id];
    if (!st->stop) { fwrite(b, 1, n, stdout); fflush(stdout); return; }
    int L = (int)strlen(st->stop);
    for (int i = 0; i < n; i++) {
        if (b[i] == (unsigned char)st->stop[st->matched]) { st->matched++; if (st->matched == L) return; }
        else {
            if (st->matched) { fwrite(st->stop, 1, st->matched, stdout); st->matched = 0; }
            if (b[i] == (unsigned char)st->stop[0]) st->matched = 1; else fputc(b[i], stdout);
        }
    }
    fflush(stdout);
}

/* feeds prompt ids, then samples up to max_new tokens; stops at EOT or the stop string.
   Returns the number of generated tokens; appends them to out (if non-NULL). */
static int generate(Model *m, State *s, int *prompt, int n_prompt, int max_new, float temp, float top_p, int top_k,
                    const char *stop, int *out, double *elapsed) {
    int *order = xmalloc(4 * m->vocab); float *probs = xmalloc(4 * m->vocab);
    Stopper st = {stop, 0};
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int pos = 0;
    for (int i = 0; i < n_prompt; i++) forward(m, s, prompt[i], pos++);
    int n = 0;
    while (n < max_new && pos < m->max_seq) {
        int id = sample(s->logits, m->vocab, temp, top_p, top_k, order, probs);
        if (id == m->eot) break;
        if (out) out[n] = id;
        n++;
        emit(m, id, &st);
        if (stop && st.matched == (int)strlen(stop)) break;
        forward(m, s, id, pos++);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    free(order); free(probs);
    return n;
}

/* ------------------------------------------------------------------ main */
static void usage(void) {
    fprintf(stderr, "usage: run <model.bin> [-p prompt] [-chat] [-tok text] [-score text] [-n max_new] [-t temp] [-top_p p] [-k top_k] [-s seed] [-v]\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    const char *path = argv[1], *prompt = NULL, *tok_text = NULL, *score_text = NULL;
    int chat = 0, verbose = 0, max_new = 300, top_k = 0; float temp = 0.8f, top_p = 0.95f; uint64_t seed = (uint64_t)time(NULL);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-chat")) chat = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (i + 1 >= argc) usage();
        else if (!strcmp(argv[i], "-score")) score_text = argv[++i];
        else if (!strcmp(argv[i], "-p")) prompt = argv[++i];
        else if (!strcmp(argv[i], "-tok")) tok_text = argv[++i];
        else if (!strcmp(argv[i], "-n")) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-top_p")) top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-k")) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")) seed = (uint64_t)atoll(argv[++i]);
        else usage();
    }
    Model m; memset(&m, 0, sizeof m); load_model(&m, path);
    rng_state = seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
    int cap = m.max_seq + 1024;
    int *ids = xmalloc(4 * cap);

    if (tok_text) {
        int n = encode(&m, tok_text, ids, cap);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
        for (int i = 0; i < n; i++) { printf("[%.*s]", m.tok_len[ids[i]], m.tok_bytes[ids[i]]); } printf("  (%d tokens)\n", n);
        return 0;
    }
    State s; state_alloc(&m, &s);
    double el;

    if (score_text) {   /* sum of log p(token | previous), text preceded by <|endoftext|> as in pytte.eval */
        int n = encode(&m, score_text, ids + 1, cap - 1) + 1; ids[0] = m.eot;
        if (n > m.max_seq) die("text longer than the context");
        double total = 0;
        for (int i = 0; i + 1 < n; i++) {
            forward(&m, &s, ids[i], i);
            float mx = s.logits[0]; for (int j = 1; j < m.vocab; j++) if (s.logits[j] > mx) mx = s.logits[j];
            double z = 0; for (int j = 0; j < m.vocab; j++) z += exp((double)s.logits[j] - mx);
            total += (double)s.logits[ids[i + 1]] - mx - log(z);
        }
        printf("%.6f %d\n", total, n - 1);
        return 0;
    }

    if (!chat) {
        int n = 0;
        if (prompt && *prompt) { n = encode(&m, prompt, ids, cap); fputs(prompt, stdout); }
        else ids[n++] = m.eot;
        if (n >= m.max_seq) die("prompt longer than the context");
        int g = generate(&m, &s, ids, n, max_new, temp, top_p, top_k, NULL, NULL, &el);
        printf("\n"); if (verbose) fprintf(stderr, "[%d tokens, %.0f tok/s]\n", g, g / (el > 0 ? el : 1e-9));
        return 0;
    }

    /* chat: <|endoftext|>Du: <request>\nPytte: <reply><|endoftext|>Du: ... ; history is re-encoded each turn */
    char **turns = xmalloc(sizeof(char *) * 256); int n_turns = 0;   /* alternating user, assistant */
    char line[4096], buf[8192];
    int *reply = xmalloc(4 * (max_new + 1));
    for (;;) {
        const char *req;
        if (prompt) req = prompt;
        else {
            fputs("Du: ", stdout); fflush(stdout);
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!line[0]) break;
            if (!strcmp(line, "/ny")) { for (int i = 0; i < n_turns; i++) free(turns[i]); n_turns = 0; continue; }
            req = line;
        }
        turns[n_turns++] = strdup(req);
        /* encode the history, dropping the oldest pairs until it fits with room for the reply */
        int first = 0, n;
        for (;;) {
            n = 0; ids[n++] = m.eot;
            for (int i = first; i < n_turns; i++) {
                if (i % 2 == 0) snprintf(buf, sizeof buf, "Du: %s\nPytte:", turns[i]);
                else snprintf(buf, sizeof buf, " %s", turns[i]);
                n += encode(&m, buf, ids + n, cap - n);
                if (i % 2 == 1) ids[n++] = m.eot;
            }
            if (n + max_new <= m.max_seq || first + 2 >= n_turns) break;
            first += 2;
        }
        if (n >= m.max_seq) die("request longer than the context");
        fputs("Pytte:", stdout);
        int g = generate(&m, &s, ids, n, max_new, temp, top_p, top_k, "\nDu:", reply, &el);
        printf("\n"); if (verbose) fprintf(stderr, "[%d tokens, %.0f tok/s]\n", g, g / (el > 0 ? el : 1e-9));
        /* store the reply text (without a trailing stop string) as the assistant turn */
        int len = 0; for (int i = 0; i < g; i++) len += m.tok_len[reply[i]];
        char *a = xmalloc(len + 1); len = 0;
        for (int i = 0; i < g; i++) { memcpy(a + len, m.tok_bytes[reply[i]], m.tok_len[reply[i]]); len += m.tok_len[reply[i]]; }
        a[len] = 0;
        char *cut = strstr(a, "\nDu:"); if (cut) *cut = 0;
        while (len > 0 && (a[len - 1] == ' ' || a[len - 1] == '\n')) a[--len] = 0;
        char *start = a; while (*start == ' ') start++;
        turns[n_turns++] = strdup(start); free(a);
        if (prompt) break;
        if (n_turns >= 254) { for (int i = 0; i < n_turns; i++) free(turns[i]); n_turns = 0; }
    }
    return 0;
}
