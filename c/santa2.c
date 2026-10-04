/* Traveling Santa - CVRP solver.
 *
 * Instance: 10000 customers on a sphere (R = 6378 km), depot at Korvatunturi,
 * vehicle capacity 10,000,000 g. Minimise total great-circle distance of all
 * routes (each route starts and ends at the depot).
 *
 * Method: Clarke-Wright savings for the initial solution, then SISR
 * (Slack Induction by String Removals, Christiaens & Vanden Berghe 2020)
 * ruin-and-recreate with simulated-annealing acceptance, plus periodic
 * intra-route polishing (2-opt + Or-opt). Runs P independent annealing
 * chains in parallel that periodically re-seed from the global best.
 *
 * Usage: santa <seconds> <threads> <out.csv> [in.csv]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#define R_EARTH   6378.0
#define CAPACITY  10000000LL
#define MAXN      10005
#define MAXR      1200          /* route slots (need ~540)                 */
#define MAXL      512           /* max customers on one route (bound: 428) */
#define KNN       40

/* ------------------------------------------------------------------ data */

static int      N;                       /* number of customers            */
static double   PX[MAXN], PY[MAXN], PZ[MAXN];
static double   DDEP[MAXN];              /* great-circle distance to depot  */
static long long DEM[MAXN];
static int      CID[MAXN];               /* external id                     */
static int      NB[MAXN][KNN];           /* k nearest customers             */

/* SISR parameters (overridable through the environment for tuning) */
static double P_CBAR  = 10.0;   /* average number of customers removed  */
static double P_LMAX  = 10.0;   /* max string cardinality               */
static double P_ALPHA = 0.01;   /* split-string preserve continuation   */
static double P_BLINK = 0.01;   /* blink rate during greedy insertion   */
static double P_T0    = 45.0;   /* SA start temperature (km)            */
static double P_T1    = 0.6;    /* SA end temperature (km)              */

static inline double dist(int a, int b)
{
    double dx = PX[a] - PX[b], dy = PY[a] - PY[b], dz = PZ[a] - PZ[b];
    double c  = 0.5 * sqrt(dx * dx + dy * dy + dz * dz);
    if (c > 1.0) c = 1.0;
    return (2.0 * R_EARTH) * asin(c);
}

/* distance where either endpoint may be the depot (index 0) */
static inline double dst(int a, int b)
{
    if (a == 0) return DDEP[b];
    if (b == 0) return DDEP[a];
    return dist(a, b);
}

/* ------------------------------------------------------------------- rng */

typedef struct { uint64_t s[4]; } RNG;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rnext(RNG *r)
{
    uint64_t res = rotl(r->s[1] * 5, 7) * 9;
    uint64_t t   = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3]  = rotl(r->s[3], 45);
    return res;
}
static inline double rnd01(RNG *r) { return (double)(rnext(r) >> 11) * 0x1.0p-53; }
static inline int    rndi(RNG *r, int n) { return (int)(rnd01(r) * n); }

static void rseed(RNG *r, uint64_t seed)
{
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* -------------------------------------------------------------- solution */

typedef struct {
    int      *node;            /* MAXR * MAXL                    */
    int       len[MAXR];
    long long load[MAXR];
    int       nr;              /* high-water mark of route slots */
    int       rof[MAXN];       /* route of customer, -1 = absent */
    int       pos[MAXN];       /* index inside its route         */
    double    cost;
} Sol;

#define RN(s, r) ((s)->node + (size_t)(r) * MAXL)

static void sol_alloc(Sol *s)
{
    s->node = malloc(sizeof(int) * (size_t)MAXR * MAXL);
    if (!s->node) { fprintf(stderr, "oom\n"); exit(1); }
}

static void sol_copy(Sol *d, const Sol *s)
{
    memcpy(d->len, s->len, sizeof(s->len));
    memcpy(d->load, s->load, sizeof(s->load));
    memcpy(d->rof, s->rof, sizeof(s->rof));
    memcpy(d->pos, s->pos, sizeof(s->pos));
    d->nr = s->nr;
    d->cost = s->cost;
    for (int r = 0; r < s->nr; r++)
        if (s->len[r]) memcpy(RN(d, r), RN(s, r), sizeof(int) * (size_t)s->len[r]);
}

static double route_cost(const Sol *s, int r)
{
    int L = s->len[r];
    if (!L) return 0.0;
    const int *a = RN(s, r);
    double c = DDEP[a[0]] + DDEP[a[L - 1]];
    for (int i = 0; i + 1 < L; i++) c += dist(a[i], a[i + 1]);
    return c;
}

static double total_cost(const Sol *s)
{
    double c = 0.0;
    for (int r = 0; r < s->nr; r++) c += route_cost(s, r);
    return c;
}

/* cost delta of inserting customer c at position p (0..len) of route r */
static inline double ins_delta(const Sol *s, int r, int p, int c)
{
    int L = s->len[r];
    if (L == 0) return 2.0 * DDEP[c];
    const int *a = RN(s, r);
    if (p == 0) return DDEP[c] + dist(c, a[0]) - DDEP[a[0]];
    if (p == L) return dist(a[L - 1], c) + DDEP[c] - DDEP[a[L - 1]];
    return dist(a[p - 1], c) + dist(c, a[p]) - dist(a[p - 1], a[p]);
}

static void ins_at(Sol *s, int r, int p, int c)
{
    int *a = RN(s, r);
    int  L = s->len[r];
    s->cost += ins_delta(s, r, p, c);
    memmove(a + p + 1, a + p, sizeof(int) * (size_t)(L - p));
    a[p] = c;
    s->len[r] = L + 1;
    s->load[r] += DEM[c];
    s->rof[c] = r;
    for (int i = p; i <= L; i++) s->pos[a[i]] = i;
}

static void rem_at(Sol *s, int r, int p)
{
    int *a = RN(s, r);
    int  L = s->len[r];
    int  c = a[p];
    double before, after;
    if (L == 1)            { before = 2.0 * DDEP[c];                        after = 0.0; }
    else if (p == 0)       { before = DDEP[c] + dist(c, a[1]);              after = DDEP[a[1]]; }
    else if (p == L - 1)   { before = dist(a[L - 2], c) + DDEP[c];          after = DDEP[a[L - 2]]; }
    else                   { before = dist(a[p - 1], c) + dist(c, a[p + 1]);after = dist(a[p - 1], a[p + 1]); }
    s->cost += after - before;
    memmove(a + p, a + p + 1, sizeof(int) * (size_t)(L - p - 1));
    s->len[r] = L - 1;
    s->load[r] -= DEM[c];
    s->rof[c] = -1;
    for (int i = p; i < L - 1; i++) s->pos[a[i]] = i;
}

/* ------------------------------------------------- intra-route polishing */

/* 2-opt + Or-opt(1..3, both orientations) to a local optimum. */
static __thread int po_tmp[MAXL];

static void polish_route(Sol *s, int r)
{
    int L = s->len[r];
    if (L < 3) return;
    int *a = RN(s, r);
    int seg[3];

    for (int round = 0; round < 60; round++) {
        int improved = 0;

        /* 2-opt: reverse a[i..j] */
        for (int i = 0; i < L - 1; i++) {
            int pi = (i == 0) ? 0 : a[i - 1];
            for (int j = i + 1; j < L; j++) {
                int nj = (j == L - 1) ? 0 : a[j + 1];
                if (pi == 0 && nj == 0) continue;          /* whole-route flip */
                double old = dst(pi, a[i]) + dst(a[j], nj);
                double nw  = dst(pi, a[j]) + dst(a[i], nj);
                if (nw < old - 1e-9) {
                    for (int x = i, y = j; x < y; x++, y--) { int t = a[x]; a[x] = a[y]; a[y] = t; }
                    s->cost += nw - old;
                    improved = 1;
                }
            }
        }

        /* Or-opt: relocate a segment of 1..3 nodes, either orientation */
        for (int sl = 1; sl <= 3 && sl < L; sl++) {
            for (int i = 0; i + sl <= L; i++) {
                int pv = (i == 0)      ? 0 : a[i - 1];
                int nx = (i + sl >= L) ? 0 : a[i + sl];
                double gain = dst(pv, a[i]) + dst(a[i + sl - 1], nx) - dst(pv, nx);
                if (gain <= 1e-9) continue;
                for (int k = 0; k < sl; k++) seg[k] = a[i + k];

                int m = 0;                                  /* route without the segment */
                for (int k = 0; k < L; k++) if (k < i || k >= i + sl) po_tmp[m++] = a[k];

                int bestp = -1, bestrev = 0;
                double bestg = 1e-9;
                for (int p = 0; p <= m; p++) {
                    int u = (p == 0) ? 0 : po_tmp[p - 1];
                    int v = (p == m) ? 0 : po_tmp[p];
                    double base = dst(u, v);
                    if (p != i) {
                        double f = dst(u, seg[0]) + dst(seg[sl - 1], v) - base;
                        if (gain - f > bestg) { bestg = gain - f; bestp = p; bestrev = 0; }
                    }
                    if (sl > 1) {
                        double b = dst(u, seg[sl - 1]) + dst(seg[0], v) - base;
                        if (gain - b > bestg) { bestg = gain - b; bestp = p; bestrev = 1; }
                    }
                }
                if (bestp >= 0) {
                    int w = 0;
                    for (int k = 0; k < bestp; k++) a[w++] = po_tmp[k];
                    for (int k = 0; k < sl; k++)    a[w++] = bestrev ? seg[sl - 1 - k] : seg[k];
                    for (int k = bestp; k < m; k++) a[w++] = po_tmp[k];
                    s->cost -= bestg;
                    improved = 1;
                }
            }
        }
        if (!improved) break;
    }
    for (int i = 0; i < L; i++) s->pos[a[i]] = i;
}

static void polish_all(Sol *s)
{
    for (int r = 0; r < s->nr; r++) if (s->len[r] > 2) polish_route(s, r);
}

/* ------------------------------------------------- inter-route local opt */

/* Relocate / swap / 2-opt* over k-nearest neighbours with don't-look bits.
 * Returns 1 if anything improved. */
static int inter_opt(Sol *s, const int *cust, int ncust, int *dl_stack, unsigned char *dl_in)
{
    int top = 0, any = 0;
    for (int i = 0; i < ncust; i++) { int c = cust[i]; dl_stack[top++] = c; dl_in[c] = 1; }

    while (top > 0) {
        int v = dl_stack[--top];
        dl_in[v] = 0;
        int rv = s->rof[v];
        if (rv < 0) continue;
        int pv = s->pos[v];
        int Lv = s->len[rv];
        int *av = RN(s, rv);
        int prv = (pv == 0) ? 0 : av[pv - 1];
        int nxv = (pv == Lv - 1) ? 0 : av[pv + 1];
        double gain_rm = dst(prv, v) + dst(v, nxv) - dst(prv, nxv);
        long long lv1 = 0;
        for (int i = 0; i <= pv; i++) lv1 += DEM[av[i]];

        int    bmove = -1, br = -1, bp = -1, bu = -1;
        double bdel = -1e-7;

        for (int k = 0; k < KNN; k++) {
            int u = NB[v][k];
            int ru = s->rof[u];
            if (ru < 0) continue;

            /* relocate v next to u */
            if (ru != rv && s->load[ru] + DEM[v] <= CAPACITY) {
                int pu = s->pos[u];
                for (int t = 0; t < 2; t++) {
                    int p = pu + t;
                    double d = ins_delta(s, ru, p, v) - gain_rm;
                    if (d < bdel) { bdel = d; bmove = 0; br = ru; bp = p; }
                }
            }
            /* swap v and u */
            if (ru != rv && u != v) {
                if (s->load[ru] - DEM[u] + DEM[v] <= CAPACITY &&
                    s->load[rv] - DEM[v] + DEM[u] <= CAPACITY) {
                    int pu = s->pos[u], Lu = s->len[ru];
                    int *au = RN(s, ru);
                    int pru = (pu == 0) ? 0 : au[pu - 1];
                    int nxu = (pu == Lu - 1) ? 0 : au[pu + 1];
                    double old = dst(prv, v) + dst(v, nxv) + dst(pru, u) + dst(u, nxu);
                    double nw  = dst(prv, u) + dst(u, nxv) + dst(pru, v) + dst(v, nxu);
                    double d = nw - old;
                    if (d < bdel) { bdel = d; bmove = 1; br = ru; bu = u; }
                }
            }
            /* 2-opt*: join v's prefix with u's suffix and vice versa */
            if (ru != rv) {
                int pu = s->pos[u], Lu = s->len[ru];
                int *au = RN(s, ru);
                long long lu1 = 0;
                for (int i = 0; i <= pu; i++) lu1 += DEM[au[i]];
                long long lv2 = s->load[rv] - lv1, lu2 = s->load[ru] - lu1;
                if (lv1 + lu2 <= CAPACITY && lu1 + lv2 <= CAPACITY) {
                    int an = (pv == Lv - 1) ? 0 : av[pv + 1];
                    int bn = (pu == Lu - 1) ? 0 : au[pu + 1];
                    double old = dst(v, an) + dst(u, bn);
                    double nw  = dst(v, bn) + dst(u, an);
                    double d = nw - old;
                    if (d < bdel) { bdel = d; bmove = 2; br = ru; bu = u; }
                }
            }
        }

        if (bmove < 0) continue;
        any = 1;

        if (bmove == 0) {
            int adj = (br == rv && bp > pv) ? 1 : 0;
            rem_at(s, rv, pv);
            ins_at(s, br, bp - adj, v);
        } else if (bmove == 1) {
            int u = bu, ru = br;
            int pu = s->pos[u];
            /* replace in place */
            int *av2 = RN(s, rv), *au2 = RN(s, ru);
            double d0 = route_cost(s, rv) + route_cost(s, ru);
            av2[pv] = u; au2[pu] = v;
            s->rof[u] = rv; s->pos[u] = pv;
            s->rof[v] = ru; s->pos[v] = pu;
            s->load[rv] += DEM[u] - DEM[v];
            s->load[ru] += DEM[v] - DEM[u];
            s->cost += route_cost(s, rv) + route_cost(s, ru) - d0;
        } else {
            int ru = br, pu = s->pos[bu];
            int Lu = s->len[ru];
            static __thread int tv[MAXL], tu[MAXL];
            int *av2 = RN(s, rv), *au2 = RN(s, ru);
            double d0 = route_cost(s, rv) + route_cost(s, ru);
            int nv = 0, nu = 0;
            for (int i = 0; i <= pv; i++) tv[nv++] = av2[i];
            for (int i = pu + 1; i < Lu; i++) tv[nv++] = au2[i];
            for (int i = 0; i <= pu; i++) tu[nu++] = au2[i];
            for (int i = pv + 1; i < Lv; i++) tu[nu++] = av2[i];
            long long wv = 0, wu = 0;
            for (int i = 0; i < nv; i++) { av2[i] = tv[i]; s->rof[tv[i]] = rv; s->pos[tv[i]] = i; wv += DEM[tv[i]]; }
            for (int i = 0; i < nu; i++) { au2[i] = tu[i]; s->rof[tu[i]] = ru; s->pos[tu[i]] = i; wu += DEM[tu[i]]; }
            s->len[rv] = nv; s->load[rv] = wv;
            s->len[ru] = nu; s->load[ru] = wu;
            s->cost += route_cost(s, rv) + route_cost(s, ru) - d0;
        }

        /* wake affected neighbourhoods */
        int wake[8], nw2 = 0;
        wake[nw2++] = v;
        if (bu > 0) wake[nw2++] = bu;
        if (prv) wake[nw2++] = prv;
        if (nxv) wake[nw2++] = nxv;
        for (int i = 0; i < nw2; i++) {
            int x = wake[i];
            if (!dl_in[x]) { dl_in[x] = 1; dl_stack[top++] = x; }
            for (int k = 0; k < 8; k++) {
                int y = NB[x][k];
                if (!dl_in[y]) { dl_in[y] = 1; dl_stack[top++] = y; }
            }
        }
    }
    return any;
}

/* --------------------------------------------------------- initial: CW */

static int uf[MAXN];
static int uf_find(int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }

typedef struct { float sv; int i, j; } Sav;
static int sav_cmp(const void *a, const void *b)
{
    float x = ((const Sav *)a)->sv, y = ((const Sav *)b)->sv;
    return (x < y) - (x > y);          /* descending */
}

static void build_savings(Sol *s)
{
    static int adj[MAXN][2];
    static long long rload[MAXN];
    for (int i = 1; i <= N; i++) { uf[i] = i; adj[i][0] = adj[i][1] = 0; rload[i] = DEM[i]; }

    int  cap = N * KNN;
    Sav *sv  = malloc(sizeof(Sav) * (size_t)cap);
    int  m = 0;
    for (int i = 1; i <= N; i++)
        for (int k = 0; k < KNN; k++) {
            int j = NB[i][k];
            if (j > i) { sv[m].sv = (float)(DDEP[i] + DDEP[j] - dist(i, j)); sv[m].i = i; sv[m].j = j; m++; }
        }
    qsort(sv, (size_t)m, sizeof(Sav), sav_cmp);

    for (int t = 0; t < m; t++) {
        int i = sv[t].i, j = sv[t].j;
        if (adj[i][1] || adj[j][1]) continue;             /* not an endpoint */
        int ri = uf_find(i), rj = uf_find(j);
        if (ri == rj) continue;
        if (rload[ri] + rload[rj] > CAPACITY) continue;
        if (adj[i][0] == 0) adj[i][0] = j; else adj[i][1] = j;
        if (adj[j][0] == 0) adj[j][0] = i; else adj[j][1] = i;
        uf[ri] = rj;
        rload[rj] += rload[ri];
    }
    free(sv);

    /* walk the paths out into routes */
    static unsigned char seen[MAXN];
    memset(seen, 0, sizeof(seen));
    s->nr = 0;
    for (int i = 1; i <= N; i++) s->rof[i] = -1;
    for (int i = 1; i <= N; i++) {
        if (seen[i] || adj[i][1]) continue;               /* start at endpoints */
        int r = s->nr++;
        int *a = RN(s, r);
        int L = 0;
        long long w = 0;
        int cur = i, prev = 0;
        while (cur) {
            a[L] = cur; s->rof[cur] = r; s->pos[cur] = L; L++;
            w += DEM[cur]; seen[cur] = 1;
            int nxt = (adj[cur][0] != prev) ? adj[cur][0] : adj[cur][1];
            prev = cur; cur = nxt;
        }
        s->len[r] = L; s->load[r] = w;
    }
    for (int i = 1; i <= N; i++) {                        /* isolated cycles (shouldn't happen) */
        if (seen[i]) continue;
        int r = s->nr++;
        int *a = RN(s, r);
        int L = 0; long long w = 0;
        int cur = i, prev = 0;
        while (cur && !seen[cur]) {
            a[L] = cur; s->rof[cur] = r; s->pos[cur] = L; L++;
            w += DEM[cur]; seen[cur] = 1;
            int nxt = (adj[cur][0] != prev) ? adj[cur][0] : adj[cur][1];
            prev = cur; cur = nxt;
        }
        s->len[r] = L; s->load[r] = w;
    }
    s->cost = total_cost(s);
}

/* ------------------------------------------------------------- SISR core */

#define UNDO_CAP 256

typedef struct {
    int   touched[UNDO_CAP];
    int   ntouch;
    int   sat;                 /* backup overflowed -> rollback impossible */
    int   stamp[MAXR];
    int   cur_stamp;
    int   bk_node[UNDO_CAP][MAXL];
    int   bk_len[UNDO_CAP];
    long long bk_load[UNDO_CAP];
    int   bk_nr;
    double bk_cost;
} Undo;

static void undo_begin(Undo *u, Sol *s)
{
    u->ntouch = 0;
    u->sat = 0;
    u->cur_stamp++;
    u->bk_nr = s->nr;
    u->bk_cost = s->cost;
}

static inline void undo_mark(Undo *u, Sol *s, int r)
{
    if (u->stamp[r] == u->cur_stamp) return;
    if (u->ntouch >= UNDO_CAP) { u->sat = 1; return; }
    u->stamp[r] = u->cur_stamp;
    int t = u->ntouch++;
    u->touched[t] = r;
    u->bk_len[t] = s->len[r];
    u->bk_load[t] = s->load[r];
    if (s->len[r]) memcpy(u->bk_node[t], RN(s, r), sizeof(int) * (size_t)s->len[r]);
}

static void undo_rollback(Undo *u, Sol *s)
{
    for (int t = 0; t < u->ntouch; t++) {
        int r = u->touched[t];
        s->len[r] = u->bk_len[t];
        s->load[r] = u->bk_load[t];
        int *a = RN(s, r);
        if (u->bk_len[t]) memcpy(a, u->bk_node[t], sizeof(int) * (size_t)u->bk_len[t]);
        for (int i = 0; i < u->bk_len[t]; i++) { s->rof[a[i]] = r; s->pos[a[i]] = i; }
    }
    s->nr = u->bk_nr;
    s->cost = u->bk_cost;
}

static int new_route(Sol *s)
{
    for (int r = 0; r < s->nr; r++) if (s->len[r] == 0) return r;
    if (s->nr >= MAXR) return -1;
    int r = s->nr++;
    s->len[r] = 0; s->load[r] = 0;
    return r;
}

static int cmp_int_desc(const void *a, const void *b) { return *(const int *)b - *(const int *)a; }

/* A sub-instance: a subset of the routes/customers of the global solution,
 * held in its own Sol so that threads never share mutable state. */
typedef struct {
    Sol   *s;
    int   *cust;          /* customers belonging to this sub-instance */
    int    ncust;
    RNG    rng;
    Undo  *u;
    int   *dls;
    unsigned char *dli;
    Sol   *best;
    Sol   *result;      /* what main merges back after the round */
    double t_hi, t_lo, seconds;
    int    use_final;      /* return the annealed state, not the best seen */
    long   iters;
} Ctx;

static int sisr_iteration(Ctx *ctx, double T)
{
    Sol *s = ctx->s;
    Undo *u = ctx->u;
    RNG *rng = &ctx->rng;
    undo_begin(u, s);

    int nonempty = 0;
    for (int r = 0; r < s->nr; r++) if (s->len[r]) nonempty++;
    if (!nonempty) return 0;
    double avg   = (double)ctx->ncust / (double)nonempty;
    double lsmax = P_LMAX < avg ? P_LMAX : avg;
    double ksmax = (4.0 * P_CBAR) / (1.0 + lsmax) - 1.0;
    if (ksmax < 1.0) ksmax = 1.0;
    int ks = 1 + rndi(rng, (int)ksmax + 1);

    int removed[512], nrem = 0;
    int rseen[16], nrs = 0;
    int seed = ctx->cust[rndi(rng, ctx->ncust)];
    int idxbuf[MAXL];

    for (int k = -1; k < KNN && nrs < ks; k++) {
        int j = (k < 0) ? seed : NB[seed][k];
        int r = s->rof[j];
        if (r < 0) continue;
        int dup = 0;
        for (int t = 0; t < nrs; t++) if (rseen[t] == r) { dup = 1; break; }
        if (dup) continue;
        if (nrs < 16) rseen[nrs++] = r;

        int L = s->len[r];
        int posj = s->pos[j];
        int lmax = (int)lsmax; if (lmax > L) lmax = L; if (lmax < 1) lmax = 1;
        int l = 1 + rndi(rng, lmax);
        int nsel = 0;

        if (rnd01(rng) < 0.5 || l >= L) {
            int lo = posj - l + 1; if (lo < 0) lo = 0;
            int hi = posj;         if (hi > L - l) hi = L - l;
            int st = lo + rndi(rng, hi - lo + 1);
            for (int x = 0; x < l; x++) idxbuf[nsel++] = st + x;
        } else {
            int m = 1;
            while (rnd01(rng) > P_ALPHA && l + m < L) m++;
            int tot = l + m;
            int lo = posj - tot + 1; if (lo < 0) lo = 0;
            int hi = posj;           if (hi > L - tot) hi = L - tot;
            if (hi < lo) {
                int lo2 = posj - l + 1; if (lo2 < 0) lo2 = 0;
                int hi2 = posj;         if (hi2 > L - l) hi2 = L - l;
                int st = lo2 + rndi(rng, hi2 - lo2 + 1);
                for (int x = 0; x < l; x++) idxbuf[nsel++] = st + x;
            } else {
                int st = lo + rndi(rng, hi - lo + 1);
                int ps = -1;
                for (int tries = 0; tries < 8; tries++) {
                    int cand = st + rndi(rng, tot - m + 1);
                    if (posj < cand || posj > cand + m - 1) { ps = cand; break; }
                }
                if (ps < 0) ps = (posj - st < tot - m) ? st + tot - m : st;
                for (int x = st; x < st + tot; x++)
                    if (x < ps || x >= ps + m) idxbuf[nsel++] = x;
            }
        }

        undo_mark(u, s, r);
        qsort(idxbuf, (size_t)nsel, sizeof(int), cmp_int_desc);
        for (int x = 0; x < nsel; x++) {
            if (nrem >= 512) break;
            removed[nrem++] = RN(s, r)[idxbuf[x]];
            rem_at(s, r, idxbuf[x]);
        }
    }

    if (nrem == 0) return 0;

    /* order the removed customers */
    int mode = rndi(rng, 4);
    for (int i = nrem - 1; i > 0; i--) { int k = rndi(rng, i + 1); int t = removed[i]; removed[i] = removed[k]; removed[k] = t; }
    if (mode == 1) {          /* demand descending */
        for (int i = 1; i < nrem; i++) { int v = removed[i], j2 = i - 1;
            while (j2 >= 0 && DEM[removed[j2]] < DEM[v]) { removed[j2 + 1] = removed[j2]; j2--; } removed[j2 + 1] = v; }
    } else if (mode == 2) {   /* far from depot first */
        for (int i = 1; i < nrem; i++) { int v = removed[i], j2 = i - 1;
            while (j2 >= 0 && DDEP[removed[j2]] < DDEP[v]) { removed[j2 + 1] = removed[j2]; j2--; } removed[j2 + 1] = v; }
    } else if (mode == 3) {   /* close to depot first */
        for (int i = 1; i < nrem; i++) { int v = removed[i], j2 = i - 1;
            while (j2 >= 0 && DDEP[removed[j2]] > DDEP[v]) { removed[j2 + 1] = removed[j2]; j2--; } removed[j2 + 1] = v; }
    }

    for (int i = 0; i < nrem; i++) {
        int c = removed[i];
        double bd = 1e18;
        int br = -1, bp = -1;
        for (int k = 0; k < KNN; k++) {
            int j = NB[c][k];
            int r = s->rof[j];
            if (r < 0) continue;
            if (s->load[r] + DEM[c] > CAPACITY) continue;
            int p0 = s->pos[j];
            for (int t = 0; t < 2; t++) {
                if (rnd01(rng) < P_BLINK) continue;
                double d = ins_delta(s, r, p0 + t, c);
                if (d < bd) { bd = d; br = r; bp = p0 + t; }
            }
        }
        double dnew = 2.0 * DDEP[c];
        if (br < 0 || dnew < bd) {
            int r = new_route(s);
            if (r >= 0) { undo_mark(u, s, r); br = r; bp = 0; }
        }
        if (br < 0) { undo_rollback(u, s); return 0; }
        undo_mark(u, s, br);
        ins_at(s, br, bp, c);
    }

    /* polish the routes we disturbed */
    for (int t = 0; t < u->ntouch; t++) {
        int r = u->touched[t];
        if (s->len[r] > 2) polish_route(s, r);
    }

    double delta = s->cost - u->bk_cost;
    if (u->sat || delta < 0.0 || rnd01(rng) < exp(-delta / T)) return 1;
    undo_rollback(u, s);
    return 0;
}

/* ---------------------------------------------------------------- driver */

static Sol   gsol;                 /* the one solution everybody improves   */
static double t_start, t_limit;
static int    NTHREADS;
static Sol   *seed_sol;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* copy only what belongs to this sub-instance (cheap) */
static void subsol_copy(Sol *d, const Sol *s, const int *cust, int ncust)
{
    memcpy(d->len, s->len, sizeof(s->len));
    memcpy(d->load, s->load, sizeof(s->load));
    d->nr = s->nr;
    d->cost = s->cost;
    for (int r = 0; r < s->nr; r++)
        if (s->len[r]) memcpy(RN(d, r), RN(s, r), sizeof(int) * (size_t)s->len[r]);
    for (int i = 0; i < ncust; i++) { int c = cust[i]; d->rof[c] = s->rof[c]; d->pos[c] = s->pos[c]; }
}

static void *worker(void *arg)
{
    Ctx *ctx = (Ctx *)arg;
    Sol *cur = ctx->s, *bst = ctx->best;

    polish_all(cur);
    inter_opt(cur, ctx->cust, ctx->ncust, ctx->dls, ctx->dli);
    polish_all(cur);
    cur->cost = total_cost(cur);
    subsol_copy(bst, cur, ctx->cust, ctx->ncust);

    double t0 = now(), tend = t0 + ctx->seconds;
    long it = 0;
    for (;;) {
        double t = now();
        if (t >= tend) break;
        double frac = (t - t0) / ctx->seconds;
        double T = ctx->t_hi * pow(ctx->t_lo / ctx->t_hi, frac);

        for (int rep = 0; rep < 64; rep++) {
            sisr_iteration(ctx, T);
            it++;
            if (cur->cost < bst->cost - 1e-9) subsol_copy(bst, cur, ctx->cust, ctx->ncust);
        }
        if ((it & 0x1FFFF) == 0) {
            polish_all(cur);
            inter_opt(cur, ctx->cust, ctx->ncust, ctx->dls, ctx->dli);
            polish_all(cur);
            if (cur->cost < bst->cost - 1e-9) subsol_copy(bst, cur, ctx->cust, ctx->ncust);
        }
    }

    ctx->iters = it;
    Sol *ret = ctx->use_final ? cur : bst;
    polish_all(ret);
    inter_opt(ret, ctx->cust, ctx->ncust, ctx->dls, ctx->dli);
    polish_all(ret);
    ret->cost = total_cost(ret);
    ctx->result = ret;
    return NULL;
}

/* ------------------------------------------------------ decomposition */

static double *proj_key;
static int proj_cmp(const void *a, const void *b)
{
    double x = proj_key[*(const int *)a], y = proj_key[*(const int *)b];
    return (x > y) - (x < y);
}

/* Recursive median split of route centroids into `groups` compact cells. */
static void split_rec(int *idx, int n, int groups, int gstart, int *out,
                      const double *cx, const double *cy, const double *cz, RNG *rng)
{
    if (groups <= 1 || n <= 1) {
        for (int i = 0; i < n; i++) out[idx[i]] = gstart;
        return;
    }
    /* pick, out of a few random directions, the one that spreads the cell most */
    double bu[3] = {1, 0, 0}, bvar = -1;
    for (int t = 0; t < 5; t++) {
        double u[3];
        for (int k = 0; k < 3; k++) {
            double a = rnd01(rng), b = rnd01(rng);
            u[k] = sqrt(-2.0 * log(a + 1e-12)) * cos(2.0 * M_PI * b);
        }
        double nn = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-12;
        u[0] /= nn; u[1] /= nn; u[2] /= nn;
        double m = 0, m2 = 0;
        for (int i = 0; i < n; i++) {
            int r = idx[i];
            double p = u[0] * cx[r] + u[1] * cy[r] + u[2] * cz[r];
            m += p; m2 += p * p;
        }
        m /= n; double var = m2 / n - m * m;
        if (var > bvar) { bvar = var; bu[0] = u[0]; bu[1] = u[1]; bu[2] = u[2]; }
    }
    for (int i = 0; i < n; i++) {
        int r = idx[i];
        proj_key[r] = bu[0] * cx[r] + bu[1] * cy[r] + bu[2] * cz[r];
    }
    qsort(idx, (size_t)n, sizeof(int), proj_cmp);

    int g1 = groups / 2, g2 = groups - g1;
    int n1 = (int)((double)n * g1 / groups + 0.5);
    if (n1 < 1) n1 = 1;
    if (n1 > n - 1) n1 = n - 1;
    split_rec(idx, n1, g1, gstart, out, cx, cy, cz, rng);
    split_rec(idx + n1, n - n1, g2, gstart + g1, out, cx, cy, cz, rng);
}

/* ------------------------------------------------------------------ knn */

typedef struct { int lo, hi; } KRange;
static void *knn_worker(void *arg)
{
    KRange *k = (KRange *)arg;
    double bd[KNN]; int bi[KNN];
    for (int i = k->lo; i <= k->hi; i++) {
        for (int t = 0; t < KNN; t++) { bd[t] = 1e18; bi[t] = 1; }
        for (int j = 1; j <= N; j++) {
            if (j == i) continue;
            double dx = PX[i] - PX[j], dy = PY[i] - PY[j], dz = PZ[i] - PZ[j];
            double q = dx * dx + dy * dy + dz * dz;
            if (q >= bd[KNN - 1]) continue;
            int p = KNN - 1;
            while (p > 0 && bd[p - 1] > q) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
            bd[p] = q; bi[p] = j;
        }
        for (int t = 0; t < KNN; t++) NB[i][t] = bi[t];
    }
    return NULL;
}

/* ------------------------------------------------------------------ main */

static int id2idx[20005];

int main(int argc, char **argv)
{
    double secs = (argc > 1) ? atof(argv[1]) : 60.0;
    NTHREADS     = (argc > 2) ? atoi(argv[2]) : 8;
    const char *outf = (argc > 3) ? argv[3] : "out/solution.csv";
    const char *inf  = (argc > 4) ? argv[4] : NULL;

    const char *e;
    if ((e = getenv("SANTA_CBAR")))  P_CBAR  = atof(e);
    if ((e = getenv("SANTA_LMAX")))  P_LMAX  = atof(e);
    if ((e = getenv("SANTA_ALPHA"))) P_ALPHA = atof(e);
    if ((e = getenv("SANTA_BLINK"))) P_BLINK = atof(e);
    if ((e = getenv("SANTA_T0")))    P_T0    = atof(e);
    if ((e = getenv("SANTA_T1")))    P_T1    = atof(e);

    /* depot */
    {
        double la = 68.073611 * M_PI / 180.0, lo = 29.315278 * M_PI / 180.0;
        PX[0] = cos(la) * cos(lo); PY[0] = cos(la) * sin(lo); PZ[0] = sin(la);
        DEM[0] = 0; DDEP[0] = 0.0; CID[0] = 0;
    }

    FILE *f = fopen("nicelist.txt", "r");
    if (!f) { fprintf(stderr, "cannot open nicelist.txt\n"); return 1; }
    char line[256];
    N = 0;
    while (fgets(line, sizeof(line), f)) {
        int id; double la, lo; long long w;
        if (sscanf(line, "%d;%lf;%lf;%lld", &id, &la, &lo, &w) != 4) continue;
        N++;
        CID[N] = id;
        id2idx[id] = N;
        la *= M_PI / 180.0; lo *= M_PI / 180.0;
        PX[N] = cos(la) * cos(lo); PY[N] = cos(la) * sin(lo); PZ[N] = sin(la);
        DEM[N] = w;
    }
    fclose(f);
    for (int i = 1; i <= N; i++) DDEP[i] = dist(0, i);
    fprintf(stderr, "loaded N=%d\n", N);

    /* knn */
    {
        pthread_t th[64]; KRange kr[64];
        int nt = NTHREADS > 64 ? 64 : NTHREADS;
        int chunk = (N + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            kr[t].lo = 1 + t * chunk;
            kr[t].hi = (t == nt - 1) ? N : (t + 1) * chunk;
            if (kr[t].lo > N) { kr[t].lo = 1; kr[t].hi = 0; }
            pthread_create(&th[t], NULL, knn_worker, &kr[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    }

    sol_alloc(&gsol);
    seed_sol = malloc(sizeof(Sol)); sol_alloc(seed_sol);

    if (inf) {
        FILE *g = fopen(inf, "r");
        if (!g) { fprintf(stderr, "cannot open %s\n", inf); return 1; }
        seed_sol->nr = 0;
        for (int i = 1; i <= N; i++) seed_sol->rof[i] = -1;
        char buf[8192];
        while (fgets(buf, sizeof(buf), g)) {
            int r = seed_sol->nr;
            int *a = RN(seed_sol, r);
            int L = 0; long long w = 0;
            char *p = buf;
            while (*p) {
                while (*p && (*p == ' ' || *p == ';' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
                if (!*p) break;
                int id = (int)strtol(p, &p, 10);
                int v = id2idx[id];
                if (v) { a[L] = v; seed_sol->rof[v] = r; seed_sol->pos[v] = L; L++; w += DEM[v]; }
            }
            if (L) { seed_sol->len[r] = L; seed_sol->load[r] = w; seed_sol->nr++; }
        }
        fclose(g);
        int miss = 0;
        for (int i = 1; i <= N; i++) if (seed_sol->rof[i] < 0) miss++;
        if (miss) { fprintf(stderr, "seed missing %d customers -> using savings\n", miss); inf = NULL; }
        else { seed_sol->cost = total_cost(seed_sol); fprintf(stderr, "seed cost %.3f km (%d routes)\n", seed_sol->cost, seed_sol->nr); }
    }
    if (!inf) {
        build_savings(seed_sol);
        fprintf(stderr, "savings cost %.3f km (%d routes)\n", seed_sol->cost, seed_sol->nr);
    }

    sol_copy(&gsol, seed_sol);
    gsol.cost = total_cost(&gsol);

    double round_len = (getenv("SANTA_ROUND")) ? atof(getenv("SANTA_ROUND")) : 8.0;
    int use_final = (getenv("SANTA_FINAL")) ? atoi(getenv("SANTA_FINAL")) : 0;
    double cycle  = (getenv("SANTA_CYCLE")) ? atof(getenv("SANTA_CYCLE")) : 0.0;
    int G = NTHREADS;
    Ctx *ctxs = calloc((size_t)G, sizeof(Ctx));
    for (int g = 0; g < G; g++) {
        ctxs[g].s    = malloc(sizeof(Sol)); sol_alloc(ctxs[g].s);
        ctxs[g].best = malloc(sizeof(Sol)); sol_alloc(ctxs[g].best);
        ctxs[g].u    = calloc(1, sizeof(Undo));
        ctxs[g].dls  = malloc(sizeof(int) * (MAXN + 64));
        ctxs[g].dli  = calloc(MAXN, 1);
        ctxs[g].cust = malloc(sizeof(int) * MAXN);
        ctxs[g].ncust = 0;
        for (int i = 0; i <= N; i++) { ctxs[g].s->rof[i] = -1; ctxs[g].best->rof[i] = -1; }
        rseed(&ctxs[g].rng, 0x9E3779B9ULL * (g + 1) + (uint64_t)(now() * 1e6));
    }

    int    *ridx   = malloc(sizeof(int) * MAXR);
    int    *rgroup = malloc(sizeof(int) * MAXR);
    double *cx = malloc(sizeof(double) * MAXR);
    double *cy = malloc(sizeof(double) * MAXR);
    double *cz = malloc(sizeof(double) * MAXR);
    proj_key   = malloc(sizeof(double) * MAXR);
    int *allc = malloc(sizeof(int) * (N + 1));
    for (int i = 1; i <= N; i++) allc[i - 1] = i;
    int *gdls = malloc(sizeof(int) * (MAXN + 64));
    unsigned char *gdli = calloc(MAXN, 1);
    RNG mrng; rseed(&mrng, 0xC0FFEEULL + (uint64_t)(now() * 1e6));

    Sol bestever; sol_alloc(&bestever); sol_copy(&bestever, &gsol);
    pthread_t th[64];
    t_start = now(); t_limit = secs;
    int round = 0;

    while (now() - t_start < t_limit) {
        double el = now() - t_start;
        /* cyclic reheating: with monotone merging, a single long cooldown
         * spends most of its budget at useless temperatures */
        double frac = (cycle > 0.0) ? fmod(el, cycle) / cycle : el / t_limit;
        double Tg = P_T0 * pow(P_T1 / P_T0, frac);

        int nre = 0;
        for (int r = 0; r < gsol.nr; r++) {
            if (!gsol.len[r]) continue;
            double sx = 0, sy = 0, sz = 0;
            int *a = RN(&gsol, r);
            for (int i = 0; i < gsol.len[r]; i++) { sx += PX[a[i]]; sy += PY[a[i]]; sz += PZ[a[i]]; }
            double nn = sqrt(sx * sx + sy * sy + sz * sz) + 1e-12;
            cx[r] = sx / nn; cy[r] = sy / nn; cz[r] = sz / nn;
            ridx[nre++] = r;
        }
        split_rec(ridx, nre, G, 0, rgroup, cx, cy, cz, &mrng);

        for (int g = 0; g < G; g++) {
            for (int i = 0; i < ctxs[g].ncust; i++) {
                int c = ctxs[g].cust[i];
                ctxs[g].s->rof[c] = -1; ctxs[g].best->rof[c] = -1;
            }
            ctxs[g].ncust = 0; ctxs[g].s->nr = 0;
        }
        for (int r = 0; r < gsol.nr; r++) {
            if (!gsol.len[r]) continue;
            Ctx *c = &ctxs[rgroup[r]];
            Sol *ss = c->s;
            int q = ss->nr++;
            int L = gsol.len[r];
            int *a = RN(ss, q);
            memcpy(a, RN(&gsol, r), sizeof(int) * (size_t)L);
            ss->len[q] = L; ss->load[q] = gsol.load[r];
            for (int i = 0; i < L; i++) { ss->rof[a[i]] = q; ss->pos[a[i]] = i; c->cust[c->ncust++] = a[i]; }
        }
        for (int g = 0; g < G; g++) {
            ctxs[g].s->cost = total_cost(ctxs[g].s);
            ctxs[g].t_hi = Tg; ctxs[g].t_lo = Tg * 0.25;
            ctxs[g].seconds = round_len;
            ctxs[g].use_final = use_final;
            ctxs[g].result = ctxs[g].best;
            pthread_create(&th[g], NULL, worker, &ctxs[g]);
        }
        for (int g = 0; g < G; g++) pthread_join(th[g], NULL);

        gsol.nr = 0;
        for (int g = 0; g < G; g++) {
            Sol *bs = ctxs[g].result;
            for (int r = 0; r < bs->nr; r++) {
                int L = bs->len[r];
                if (!L) continue;
                int q = gsol.nr++;
                int *a = RN(&gsol, q);
                memcpy(a, RN(bs, r), sizeof(int) * (size_t)L);
                gsol.len[q] = L; gsol.load[q] = bs->load[r];
                for (int i = 0; i < L; i++) { gsol.rof[a[i]] = q; gsol.pos[a[i]] = i; }
            }
        }
        gsol.cost = total_cost(&gsol);

        round++;
        if (round % 4 == 0) {
            polish_all(&gsol);
            inter_opt(&gsol, allc, N, gdls, gdli);
            polish_all(&gsol);
            gsol.cost = total_cost(&gsol);
        }

        if (gsol.cost < bestever.cost - 1e-9) {
            sol_copy(&bestever, &gsol);
            FILE *o = fopen(outf, "w");
            for (int r = 0; r < bestever.nr; r++) {
                int L = bestever.len[r];
                if (!L) continue;
                int *a = RN(&bestever, r);
                for (int i = 0; i < L; i++) fprintf(o, "%s%d", i ? "; " : "", CID[a[i]]);
                fputc('\n', o);
            }
            fclose(o);
        }
        int ne = 0;
        for (int r = 0; r < gsol.nr; r++) if (gsol.len[r]) ne++;
        fprintf(stderr, "round %3d  t=%6.0fs  T=%7.3f  cost=%.0f m  routes=%d\n",
                round, now() - t_start, Tg, gsol.cost * 1000.0, ne);
        fflush(stderr);
    }

    printf("%.6f\n", bestever.cost * 1000.0);
    return 0;
}
