/*
 * backend_simd.c -- hand-written SIMD forward backend.
 *
 * Kernels: outer-product register blocks (BLIS style). Per layer, W is
 * packed at create into column panels of PW = 16 output units: panel p
 * holds W[p*16 .. p*16+15][k] for k = 0..K-1, one 64-byte line per k
 * (aligned, pad units zero). A micro-kernel computes MR rows x NV
 * vectors of VL units: per k it loads NV weight vectors, broadcasts
 * x[r][k] and does MR*NV FMAs into register accumulators; bias +
 * activation run in the epilogue on whole vectors. No horizontal sums.
 *
 * Sparse k: a kernel walks a list of the k where some row of its tile
 * is nonzero instead of 0..K-1 (exact: skipped terms are 0*w). The prod
 * nets have ~14% nonzero inputs and 17-28% nonzero relu outputs, so this
 * cuts the work 2x (tiles of 4-6 rows) to 5x (one row). Lists are built
 * per tile and layer (layer 0 when sparse1, every layer for tiles of
 * <= sparse_mr rows) and used when they drop at least 1/8 of K.
 *
 * Rows are processed in chunks of CHUNK; a chunk is split into balanced
 * tiles of <= mrmax rows; per layer the loop is vector group outer, tile
 * inner, so a weight group stays in L2 across the tiles of the chunk.
 * nv per tile height is the widest built (NVMAX_INIT); a group takes the
 * smallest nv over the chunk's tiles.
 *
 * ISA: AVX-512F (VL 16, mr <= 6 by default, <= 12 built), AVX2+FMA
 * (VL 8, mr <= 4, <= 6 built) or generic vectors (VL 4), chosen at
 * create with __builtin_cpu_supports; the kernels carry target
 * attributes, so a build without -march=native still runs them. The ISA
 * pass is compiled three times by including this file with SIMD_PASS
 * set. List kernels of <= 2 rows prefetch the weight rows PFD list
 * entries ahead on AVX2 (+18% at n = 1 on Zen 3; slower on Zen 5, off).
 *
 * Env (read at create): BGC_SIMD_ISA=avx2|scalar caps the ISA;
 * BGC_SIMD_SPARSE1=0|1 (layer-0 lists, default 1); BGC_SIMD_SPARSE_MR=m
 * (every-layer lists for tiles of <= m rows, default all, 0 = off);
 * BGC_SIMD_MR=m tile height (up to the built maximum); BGC_SIMD_NV=v
 * caps the vector group width.
 */

#ifndef SIMD_PASS

#include "puretd_backend.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK 96            /* rows per internal chunk */
#define MRCAP 12            /* largest tile height over all ISAs */
#define PW 16               /* packed panel width (floats): one cache line per k */
#define NVSLOTS 5           /* nv in {1, 2, 4, 8, 16} -> slot log2(nv) */
#define PF_PAD 16           /* list tail pad >= any pass's PFD */
#define PF_MR 2             /* list kernels with mr <= PF_MR prefetch the weight
                               rows PFD entries ahead (PFD per ISA pass, 0 = off) */

enum { ACT_RELU, ACT_LEAKY, ACT_HARDSIG, ACT_LINEAR };
enum { ISA_SCALAR, ISA_AVX2, ISA_AVX512 };

/* x rows (stride ldx) x NV vectors of W (panel stride ps floats) -> o
 * rows (stride ldo). kl/nk: k list, or kl == NULL for k = 0..K-1. */
typedef void (*kern_fn)(const float *x, int ldx, const float *W, size_t ps,
                        int K, const int *kl, int nk, const float *bias,
                        float *o, int ldo, int act);
/* k with any of mr rows nonzero, ascending, k < K. Returns the count. */
typedef int (*list_fn)(const float *x, int ldx, int mr, int K, int *kl);
/* Layer 0 of n rows from a parent row's pre-activation p0 (forward_delta):
 * o[r] = act(p0 + sum over the k < K, ascending, where x[r][k] != par[k]
 * of W[:, k] * (x[r][k] - par[k])), nvec output vectors. WT: layer 0
 * transposed, row k = the nvec * VL weights of input k. kl/dv: scratch of
 * K + VL entries. Adds the differing k count to *nd. */
typedef void (*delta_fn)(const float *x, int ldx, int n, const float *par, const float *p0,
                         const float *WT, int K, int nvec, float *o, int ldo,
                         int act, int *kl, float *dv, long *nd);

typedef struct {
    int vl, mrmax, mrlim;           /* default and largest tile height */
    const char *name;
    kern_fn const (*kern)[NVSLOTS]; /* [mr][slot], NULL where not built */
    const int *nvmax;               /* [mr] widest nv built */
    list_fn list;
    delta_fn delta0;
} isa_ops;

typedef struct {
    int K, M, npan;                 /* npan = ceil(M / PW) */
    float *W;                       /* npan panels of K*PW */
    float *bias;                    /* npan*PW, pad zero */
} layer;

struct bgc_backend {
    isa_ops ops;
    int nl;                         /* hidden + output */
    int activation, equity, out_dim;
    int sparse1, sparse_mr;
    int nvcap;                      /* widest panel group */
    layer L[NN_MAX_LAYERS + 1];
    int ld;                         /* scratch row stride (floats) */
    float *buf[2];                  /* CHUNK x ld each */
    int *klist;                     /* per tile: (kmax + 16) ints */
    int kstride;
    int delta0;                     /* forward_delta computes layer 0 by deltas */
    float *p0;                      /* forward_delta: the parent's layer-0 pre-activation */
    float *W0T;                     /* forward_delta: layer 0 as K rows of npan * PW */
    float *dv;                      /* forward_delta: per-row deltas, kstride floats */
#ifdef BGC_BF16
    int bf16;                       /* bf16 path usable (built, CPU has it, relu, not BGC_BF16=0) */
    uint32_t *Wb[NN_MAX_LAYERS + 1];/* per layer: npan panels x kp k-pairs x 16 lanes (lo = even k) */
    int kp[NN_MAX_LAYERS + 1];      /* k pairs per layer: ceil(K / 2) */
    uint16_t *bb[3];                /* bf16 activations, CHUNK x ldb: [2] = layer-0 input */
    int ldb;                        /* bf16 row stride (uint16, multiple of 32) */
    float *bout;                    /* output layer, CHUNK x PW fp32 */
#endif
    char desc[160];
};

static size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

#ifdef BGC_PROFILE
#include <time.h>
/* Profile build only (CFLAGS_EXTRA=-DBGC_PROFILE): per layer, seconds,
   rows, and k entries walked per row (sum over tiles of k-list length x
   tile rows; K x rows for dense tiles). bgc_simd_prof_read copies
   [NN_MAX_LAYERS + 1] of each; bgc_simd_prof_reset clears. */
static double sprof_t[NN_MAX_LAYERS + 1];
static long sprof_rows[NN_MAX_LAYERS + 1], sprof_kent[NN_MAX_LAYERS + 1];
static double sprof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
__attribute__((visibility("default")))
int bgc_simd_prof_read(double *t, long *rows, long *kent) {
    for (int l = 0; l < NN_MAX_LAYERS + 1; l++) {
        t[l] = sprof_t[l]; rows[l] = sprof_rows[l]; kent[l] = sprof_kent[l];
    }
    return NN_MAX_LAYERS + 1;
}
__attribute__((visibility("default")))
void bgc_simd_prof_reset(void) {
    memset(sprof_t, 0, sizeof sprof_t);
    memset(sprof_rows, 0, sizeof sprof_rows);
    memset(sprof_kent, 0, sizeof sprof_kent);
}
#endif

static void *alloc64(size_t bytes) {
    bytes = round_up(bytes ? bytes : 64, 64);
    void *p = aligned_alloc(64, bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

#define SLOT(nv) ((nv) >= 16 ? 4 : (nv) >= 8 ? 3 : (nv) >= 4 ? 2 : (nv) >= 2 ? 1 : 0)
static int slot_of(int nv) { return SLOT(nv); }

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

/* ---- ISA passes ---- */

#pragma GCC push_options
#pragma GCC target("avx512f,avx2,fma")
#define SIMD_PASS 3
#define FN(x) CAT(x, _avx512)
#define VL 16
#define MRMAX 12
#define VEC __m512
#define VZERO() _mm512_setzero_ps()
#define VLOAD(p) _mm512_load_ps(p)
#define VSTORE(p, v) _mm512_store_ps((p), (v))
#define VBCAST(s) _mm512_set1_ps(s)
#define VFMA(a, b, c) _mm512_fmadd_ps((a), (b), (c))
#define VADD(a, b) _mm512_add_ps((a), (b))
#define VMUL(a, b) _mm512_mul_ps((a), (b))
#define VDIV(a, b) _mm512_div_ps((a), (b))
#define VMAX(a, b) _mm512_max_ps((a), (b))
#define VMIN(a, b) _mm512_min_ps((a), (b))
#define VNZMASK(p) ((unsigned)_mm512_cmp_ps_mask(_mm512_loadu_ps(p), _mm512_setzero_ps(), _CMP_NEQ_UQ))
#define VNEQMASK(p, q) ((unsigned)_mm512_cmp_ps_mask(_mm512_loadu_ps(p), _mm512_loadu_ps(q), _CMP_NEQ_UQ))
#define INSTANTIATE \
    KDEF(1,1) KDEF(1,2) KDEF(1,4) KDEF(1,8) KDEF(1,16) \
    KDEF(2,1) KDEF(2,2) KDEF(2,4) KDEF(2,8) \
    KDEF(3,1) KDEF(3,2) KDEF(3,4) KDEF(3,8) \
    KDEF(4,1) KDEF(4,2) KDEF(4,4) KDEF(5,1) KDEF(5,2) KDEF(5,4) \
    KDEF(6,1) KDEF(6,2) KDEF(6,4) KDEF(7,1) KDEF(7,2) KDEF(8,1) KDEF(8,2) \
    KDEF(9,1) KDEF(9,2) KDEF(10,1) KDEF(10,2) KDEF(11,1) KDEF(11,2) \
    KDEF(12,1) KDEF(12,2)
#define NVMAX_INIT {0, 8, 8, 8, 4, 4, 4, 2, 2, 2, 2, 2, 2}
#define PFD 0
#define MRDEF 6
#include "backend_simd.c"
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx2,fma")
#define SIMD_PASS 2
#define FN(x) CAT(x, _avx2)
#define VL 8
#define MRMAX 6
#define VEC __m256
#define VZERO() _mm256_setzero_ps()
#define VLOAD(p) _mm256_load_ps(p)
#define VSTORE(p, v) _mm256_store_ps((p), (v))
#define VBCAST(s) _mm256_set1_ps(s)
#define VFMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define VADD(a, b) _mm256_add_ps((a), (b))
#define VMUL(a, b) _mm256_mul_ps((a), (b))
#define VDIV(a, b) _mm256_div_ps((a), (b))
#define VMAX(a, b) _mm256_max_ps((a), (b))
#define VMIN(a, b) _mm256_min_ps((a), (b))
#define VNZMASK(p) ((unsigned)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(p), _mm256_setzero_ps(), _CMP_NEQ_UQ)))
#define VNEQMASK(p, q) ((unsigned)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(q), _CMP_NEQ_UQ)))
#define INSTANTIATE \
    KDEF(1,1) KDEF(1,2) KDEF(1,4) KDEF(1,8) \
    KDEF(2,1) KDEF(2,2) KDEF(2,4) KDEF(3,1) KDEF(3,2) KDEF(3,4) \
    KDEF(4,1) KDEF(4,2) KDEF(5,1) KDEF(5,2) KDEF(6,1) KDEF(6,2)
#define NVMAX_INIT {0, 8, 4, 4, 2, 2, 2}
#define PFD 8
#define MRDEF 4
#include "backend_simd.c"
#pragma GCC pop_options

/* Generic-vector fallback: plain C, the compiler picks the baseline ISA. */
typedef float v4sf __attribute__((vector_size(16)));
static inline v4sf v4_max(v4sf a, v4sf b) { v4sf r; for (int i = 0; i < 4; i++) r[i] = a[i] > b[i] ? a[i] : b[i]; return r; }
static inline v4sf v4_min(v4sf a, v4sf b) { v4sf r; for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? a[i] : b[i]; return r; }
static inline unsigned v4_nzmask(const float *p) { unsigned m = 0; for (int i = 0; i < 4; i++) m |= (unsigned)(p[i] != 0.0f) << i; return m; }
static inline unsigned v4_neqmask(const float *p, const float *q) { unsigned m = 0; for (int i = 0; i < 4; i++) m |= (unsigned)(p[i] != q[i]) << i; return m; }
#define SIMD_PASS 1
#define FN(x) CAT(x, _scalar)
#define VL 4
#define MRMAX 4
#define VEC v4sf
#define VZERO() ((v4sf){0})
#define VLOAD(p) (*(const v4sf *)(p))
#define VSTORE(p, v) (*(v4sf *)(p) = (v))
#define VBCAST(s) ((v4sf){0} + (s))
#define VFMA(a, b, c) ((a) * (b) + (c))
#define VADD(a, b) ((a) + (b))
#define VMUL(a, b) ((a) * (b))
#define VDIV(a, b) ((a) / (b))
#define VMAX(a, b) v4_max((a), (b))
#define VMIN(a, b) v4_min((a), (b))
#define VNZMASK(p) v4_nzmask(p)
#define VNEQMASK(p, q) v4_neqmask((p), (q))
#define INSTANTIATE KDEF(1,1) KDEF(1,2) KDEF(1,4) KDEF(2,1) KDEF(2,2) KDEF(2,4) \
    KDEF(3,1) KDEF(3,2) KDEF(4,1) KDEF(4,2)
#define NVMAX_INIT {0, 4, 4, 2, 2}
#define PFD 8
#define MRDEF 4
#include "backend_simd.c"

/* ---- driver ---- */

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    return (s && *s) ? atoi(s) : dflt;
}

static int pick_isa(void) {
    const char *cap = getenv("BGC_SIMD_ISA");
    int best = ISA_SCALAR;
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
    if (best == ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
    if (cap && !strcmp(cap, "scalar")) best = ISA_SCALAR;
    else if (cap && !strcmp(cap, "avx2") && best > ISA_AVX2) best = ISA_AVX2;
    return best;
}

#ifdef BGC_BF16
/* ---- bf16 forward (build with CFLAGS_EXTRA=-DBGC_BF16; OFF by default) ----
 * VDPBF16PS (AVX512_BF16): per 32-bit lane acc += a.lo * b.lo + a.hi * b.hi
 * over bf16 pairs, i.e. two k per instruction: 2x the MACs of an fp32 FMA.
 * Weights and activations are rounded to bf16 (RNE: the inputs and the
 * relu outputs between layers), accumulation and bias stay fp32. Outside
 * the parity bounds (relative ~2^-9 per term): for the 1-ply target rows
 * of a build that asks for it, behind its own A/B gate. The fp32 path's
 * tile driver, k lists over pairs. */

static void forward_chunk_bf16(bgc_backend *b, const float *x, int ldx, int n, float *out,
                               const float *par);

static uint16_t bf16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static int bf16_create(bgc_backend *b, const NNModel *m) {
    __builtin_cpu_init();
    const char *env = getenv("BGC_BF16");
    b->bf16 = __builtin_cpu_supports("avx512bf16") && b->activation == NN_ACTIVATION_RELU
              && b->nl >= 2 && b->L[b->nl - 1].npan == 1 && !(env && !strcmp(env, "0"));
    int wmax = 0;
    for (int l = 0; l < b->nl; l++) {
        const layer *y = &b->L[l];
        const int kp = (y->K + 1) / 2;
        b->kp[l] = kp;
        if (y->npan * PW > wmax) wmax = y->npan * PW;
        if (y->K > wmax) wmax = y->K;
        b->Wb[l] = alloc64((size_t)y->npan * kp * PW * sizeof(uint32_t));
        if (!b->Wb[l]) return -1;
        for (int j = 0; j < y->M; j++)
            for (int k = 0; k < y->K; k++) {
                uint32_t *w = b->Wb[l] + ((size_t)(j / PW) * kp + k / 2) * PW + j % PW;
                const uint32_t h = bf16_rne(m->weight[l][(size_t)j * y->K + k]);
                *w |= (k & 1) ? h << 16 : h;
            }
    }
    b->ldb = (int)round_up((size_t)wmax, 32) + 32;
    for (int i = 0; i < 3; i++)
        if (!(b->bb[i] = alloc64((size_t)CHUNK * b->ldb * sizeof(uint16_t)))) return -1;
    if (!(b->bout = alloc64((size_t)CHUNK * PW * sizeof(float)))) return -1;
    return 0;
}

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bf16,avx512bw,avx512vl,avx2,fma")

/* MR rows x NV panels over the pair list ql[0..nq): x rows of bf16 pairs
 * (uint32, stride ldx), W panels ps uint32 apart; bias + relu, stored as
 * bf16 at ob (stride ldob uint16), or (ob NULL) as fp32 at of (the last
 * hidden layer, whose output feeds the fp32 output layer). */
static inline __attribute__((always_inline)) void
kbf_body(const int MR, const int NV, const uint32_t *x, int ldx, const uint32_t *W, size_t ps,
         const int *ql, int nq, const float *bias, uint16_t *ob, int ldob, float *of, int ldof)
{
    __m512 acc[6][4];
#pragma GCC unroll 6
    for (int r = 0; r < MR; r++)
#pragma GCC unroll 4
        for (int v = 0; v < NV; v++) acc[r][v] = _mm512_setzero_ps();
    for (int i = 0; i < nq; i++) {
        const int q = ql[i];
        __m512bh wv[4];
#pragma GCC unroll 4
        for (int v = 0; v < NV; v++)
            wv[v] = (__m512bh)_mm512_load_si512((const void *)(W + v * ps + (size_t)q * PW));
#pragma GCC unroll 6
        for (int r = 0; r < MR; r++) {
            const __m512bh xb = (__m512bh)_mm512_set1_epi32((int)x[(size_t)r * ldx + q]);
#pragma GCC unroll 4
            for (int v = 0; v < NV; v++) acc[r][v] = _mm512_dpbf16_ps(acc[r][v], wv[v], xb);
        }
    }
    const __m512 zero = _mm512_setzero_ps();
#pragma GCC unroll 4
    for (int v = 0; v < NV; v++) {
        const __m512 bv = _mm512_load_ps(bias + v * PW);
#pragma GCC unroll 6
        for (int r = 0; r < MR; r++) {
            const __m512 y = _mm512_max_ps(_mm512_add_ps(acc[r][v], bv), zero);
            if (ob)
                _mm256_store_si256((__m256i *)(ob + (size_t)r * ldob + v * PW),
                                   (__m256i)_mm512_cvtneps_pbh(y));
            else
                _mm512_store_ps(of + (size_t)r * ldof + v * PW, y);
        }
    }
}

typedef void (*kbf_fn)(const uint32_t *, int, const uint32_t *, size_t, const int *, int,
                       const float *, uint16_t *, int, float *, int);
#define KBF(MR, NV) \
    static void kbf_##MR##_##NV(const uint32_t *x, int ldx, const uint32_t *W, size_t ps, \
                                const int *ql, int nq, const float *bias, uint16_t *ob, \
                                int ldob, float *of, int ldof) { \
        kbf_body(MR, NV, x, ldx, W, ps, ql, nq, bias, ob, ldob, of, ldof); }
KBF(1,1) KBF(1,2) KBF(1,4) KBF(2,1) KBF(2,2) KBF(2,4) KBF(3,1) KBF(3,2) KBF(3,4)
KBF(4,1) KBF(4,2) KBF(4,4) KBF(5,1) KBF(5,2) KBF(5,4) KBF(6,1) KBF(6,2) KBF(6,4)
#undef KBF
static kbf_fn const kbf_tab[7][3] = {
    {0, 0, 0}, {kbf_1_1, kbf_1_2, kbf_1_4}, {kbf_2_1, kbf_2_2, kbf_2_4},
    {kbf_3_1, kbf_3_2, kbf_3_4}, {kbf_4_1, kbf_4_2, kbf_4_4}, {kbf_5_1, kbf_5_2, kbf_5_4},
    {kbf_6_1, kbf_6_2, kbf_6_4}};

/* Pairs q < kp with any of mr rows nonzero (rows of uint32 pairs, zero
   padded to a multiple of 16 pairs). */
static int bf_list(const uint32_t *x, int ldx, int mr, int kp, int *ql) {
    int c = 0;
    for (int q0 = 0; q0 < kp; q0 += 16) {
        __mmask16 m = 0;
        for (int r = 0; r < mr; r++)
            m |= _mm512_test_epi32_mask(_mm512_loadu_si512((const void *)(x + (size_t)r * ldx + q0)),
                                        _mm512_set1_epi32(-1));
        unsigned mm = m;
        while (mm) {
            const int q = q0 + __builtin_ctz(mm);
            if (q < kp) ql[c++] = q;
            mm &= mm - 1;
        }
    }
    return c;
}

/* par: NULL (every layer in bf16), or the rows' parent: layer 0 in fp32 by
 * deltas (delta0, b->p0 set), its relu output rounded to bf16 for layer 1. */
static void forward_chunk_bf16(bgc_backend *b, const float *x, int ldx, int n, float *out,
                               const float *par) {
    const int mrmax = 6;
    const int nt = (n + mrmax - 1) / mrmax;
    int t0[CHUNK + 1], nk[CHUNK];
    for (int t = 0; t <= nt; t++) t0[t] = (int)((long)t * n / nt);
    const float *src = x;                     /* fp32 rows to round to bf16 */
    int lds = ldx, K0 = b->L[0].K, l0 = 0;
    if (par) {
        const layer *y = &b->L[0];
        long nd = 0;
        b->ops.delta0(x, ldx, n, par, b->p0, b->W0T, y->K, y->npan * (PW / b->ops.vl),
                      b->buf[0], b->ld, ACT_RELU, b->klist, b->dv, &nd);
        src = b->buf[0];
        lds = b->ld;
        K0 = y->npan * PW;
        l0 = 1;
    }
    uint16_t *dst = b->bb[l0 ? 0 : 2];
    for (int r = 0; r < n; r++) {
        uint16_t *row = dst + (size_t)r * b->ldb;
        int k = 0;
        for (; k + 16 <= K0; k += 16)
            _mm256_storeu_si256((__m256i *)(row + k),
                                (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + (size_t)r * lds + k)));
        for (; k < K0; k++) row[k] = bf16_rne(src[(size_t)r * lds + k]);
        for (; k < b->ldb; k++) row[k] = 0;
    }
    const uint16_t *in = dst;
    const int ldi32 = b->ldb / 2;
    /* bf16 for the hidden layers from l0; the output layer stays fp32 (its
       weights' rounding is the bias of an all-bf16 forward: 2.3e-3 -> 6e-4
       mean shift on the cubeless prod net, emulated). */
    for (int l = l0; l < b->nl - 1; l++) {
#ifdef BGC_PROFILE
        double sp_t0 = sprof_now();
        long kent = 0;
#endif
        const layer *y = &b->L[l];
        const int last = (l == b->nl - 2);       /* fp32 output for the output layer */
        uint16_t *o = last ? NULL : b->bb[l & 1];
        const int kp = b->kp[l];
        for (int t = 0; t < nt; t++) {
            nk[t] = bf_list((const uint32_t *)in + (size_t)t0[t] * ldi32, ldi32,
                            t0[t + 1] - t0[t], kp, b->klist + (size_t)t * b->kstride);
#ifdef BGC_PROFILE
            kent += 2L * nk[t] * (t0[t + 1] - t0[t]);
#endif
        }
        const size_t ps = (size_t)kp * PW;
        for (int p = 0; p < y->npan;) {
            const int nv = (p % 4 == 0 && p + 4 <= y->npan) ? 4
                         : (p % 2 == 0 && p + 2 <= y->npan) ? 2 : 1;
            const int sl = nv == 4 ? 2 : nv == 2 ? 1 : 0;
            for (int t = 0; t < nt; t++) {
                const int r0 = t0[t], mr = t0[t + 1] - r0;
                kbf_tab[mr][sl]((const uint32_t *)in + (size_t)r0 * ldi32, ldi32,
                                b->Wb[l] + (size_t)p * ps, ps,
                                b->klist + (size_t)t * b->kstride, nk[t], y->bias + (size_t)p * PW,
                                o ? o + (size_t)r0 * b->ldb + (size_t)p * PW : NULL, b->ldb,
                                b->buf[1] + (size_t)r0 * b->ld + (size_t)p * PW, b->ld);
            }
            p += nv;
        }
        if (!last) {                          /* zero the pad past the layer's outputs */
            for (int r = 0; r < n; r++) {
                uint16_t *row = o + (size_t)r * b->ldb;
                for (int k = y->npan * PW; k < b->ldb; k++) row[k] = 0;
            }
            in = o;
        }
#ifdef BGC_PROFILE
        sprof_t[l] += sprof_now() - sp_t0;
        sprof_rows[l] += n;
        sprof_kent[l] += kent;
#endif
    }
    {   /* output layer, fp32 dense over the last hidden layer's fp32 relu */
        const layer *y = &b->L[b->nl - 1];
        const float *h = b->buf[1];           /* nl >= 2 (bf16_create) */
        const int ldh = b->ld;
        for (int t = 0; t < nt; t++) {
            const int r0 = t0[t], mr = t0[t + 1] - r0;
            b->ops.kern[mr][0](h + (size_t)r0 * ldh, ldh, y->W, (size_t)y->K * PW, y->K, NULL, 0,
                               y->bias, b->bout + (size_t)r0 * PW, PW, ACT_LINEAR);
        }
    }
    const int od = b->out_dim;
    for (int r = 0; r < n; r++)
        for (int j = 0; j < od; j++) {
            const float v = b->bout[(size_t)r * PW + j];
            out[(size_t)r * od + j] = b->equity ? v : 1.0f / (1.0f + expf(-v));
        }
}
#pragma GCC pop_options

int bgc_backend_bf16(const bgc_backend *b) { return b->bf16; }

static void parent_p0(bgc_backend *b, const float *parent, int ldx);

/* The bf16 forward when usable, else the fp32 one. parent (nullable): as
 * bgc_backend_forward_delta, layer 0 then stays fp32 (by deltas). */
void bgc_backend_forward_bf16(bgc_backend *b, const float *parent, const float *x, int ldx,
                              int n, float *out) {
    if (!b->bf16) {
        if (parent) bgc_backend_forward_delta(b, parent, x, ldx, n, out);
        else bgc_backend_forward(b, x, ldx, n, out);
        return;
    }
    if (parent && !b->delta0) parent = NULL;
    if (parent) parent_p0(b, parent, ldx);
    for (int r = 0; r < n; r += CHUNK) {
        const int c = (n - r < CHUNK) ? n - r : CHUNK;
        forward_chunk_bf16(b, x + (size_t)r * ldx, ldx, c, out + (size_t)r * b->out_dim, parent);
    }
}
#endif /* BGC_BF16 */

bgc_backend *bgc_backend_create(const NNModel *m) {
    bgc_backend *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    const int isa = pick_isa();
    b->ops = isa == ISA_AVX512 ? ops_avx512 : isa == ISA_AVX2 ? ops_avx2 : ops_scalar;
    int mr_cap = env_int("BGC_SIMD_MR", 0);
    if (mr_cap > 0 && mr_cap <= b->ops.mrlim) b->ops.mrmax = mr_cap;
    b->sparse1 = env_int("BGC_SIMD_SPARSE1", 1);
    b->sparse_mr = env_int("BGC_SIMD_SPARSE_MR", MRCAP);
    b->nvcap = 1;
    for (int cap = env_int("BGC_SIMD_NV", 16); b->nvcap * 2 <= cap && b->nvcap < 16;) b->nvcap *= 2;
    b->nl = m->num_hidden + 1;
    b->activation = m->activation;
    b->equity = (m->output_mode == NN_OUTPUT_EQUITY);
    b->out_dim = m->layer_out[m->num_hidden];
    const int vl = b->ops.vl;
    int kmax = 0, wmax = 0;
    for (int l = 0; l < b->nl; l++) {
        layer *y = &b->L[l];
        y->K = m->layer_in[l];
        y->M = m->layer_out[l];
        y->npan = (y->M + PW - 1) / PW;
        if (y->K > kmax) kmax = y->K;
        if (y->npan * PW > wmax) wmax = y->npan * PW;
        y->W = alloc64((size_t)y->npan * y->K * PW * sizeof(float));
        y->bias = alloc64((size_t)y->npan * PW * sizeof(float));
        if (!y->W || !y->bias) goto fail;
        for (int j = 0; j < y->M; j++) {
            float *dst = y->W + (size_t)(j / PW) * y->K * PW + j % PW;
            const float *src = m->weight[l] + (size_t)j * y->K;
            for (int k = 0; k < y->K; k++) dst[(size_t)k * PW] = src[k];
            y->bias[j] = m->bias[l][j];
        }
    }
    /* +16: breaks 4 KiB aliasing between the rows of a tile. */
    b->ld = (int)round_up((size_t)wmax, 16) + 16;
    b->kstride = kmax + PF_PAD + 16;
    for (int i = 0; i < 2; i++)
        if (!(b->buf[i] = alloc64((size_t)CHUNK * b->ld * sizeof(float)))) goto fail;
    if (!(b->klist = malloc((size_t)CHUNK * b->kstride * sizeof(int)))) goto fail;
    b->delta0 = env_int("BGC_SIMD_DELTA0", 1) != 0;
    if (!(b->p0 = alloc64((size_t)b->L[0].npan * PW * sizeof(float)))) goto fail;
    if (!(b->dv = alloc64((size_t)b->kstride * sizeof(float)))) goto fail;
    {
        const layer *y0 = &b->L[0];
        const size_t mp = (size_t)y0->npan * PW;
        if (!(b->W0T = alloc64((size_t)y0->K * mp * sizeof(float)))) goto fail;
        for (int j = 0; j < y0->M; j++)
            for (int k = 0; k < y0->K; k++)
                b->W0T[(size_t)k * mp + j] = m->weight[0][(size_t)j * y0->K + k];
    }
#ifdef BGC_BF16
    if (bf16_create(b, m)) goto fail;
#endif
    snprintf(b->desc, sizeof b->desc,
             "simd %s VL%d mr<=%d nv%d (mr1 nv%d) sparse1=%d sparse_mr=%d delta0=%d%s",
             b->ops.name, vl, b->ops.mrmax, b->ops.nvmax[b->ops.mrmax],
             b->ops.nvmax[1], b->sparse1, b->sparse_mr, b->delta0,
#ifdef BGC_BF16
             b->bf16 ? " bf16" : " bf16-unavailable"
#else
             ""
#endif
             );
    return b;
fail:
    fprintf(stderr, "backend_simd: out of memory\n");
    bgc_backend_destroy(b);
    return NULL;
}

void bgc_backend_destroy(bgc_backend *b) {
    if (!b) return;
    for (int l = 0; l < NN_MAX_LAYERS + 1; l++) { free(b->L[l].W); free(b->L[l].bias); }
    free(b->buf[0]);
    free(b->buf[1]);
    free(b->klist);
    free(b->p0);
    free(b->dv);
    free(b->W0T);
#ifdef BGC_BF16
    for (int l = 0; l < NN_MAX_LAYERS + 1; l++) free(b->Wb[l]);
    for (int i = 0; i < 3; i++) free(b->bb[i]);
    free(b->bout);
#endif
    free(b);
}

static inline float sigmoidf_(float v) { return 1.0f / (1.0f + expf(-v)); }

/* One chunk of n <= CHUNK rows. par: NULL, or layer 0 by deltas from that
 * parent row (b->p0 holds its pre-activation). */
static void forward_chunk(bgc_backend *b, const float *x, int ldx, int n, float *out,
                          const float *par) {
    const isa_ops *ops = &b->ops;
    const int vl = ops->vl;
    const int nt = (n + ops->mrmax - 1) / ops->mrmax;
    int t0[CHUNK + 1];                          /* tile t = rows t0[t] .. t0[t+1]-1 */
    for (int t = 0; t <= nt; t++) t0[t] = (int)((long)t * n / nt);
    int nvg = b->nvcap;                         /* panel group width: min over tiles */
    for (int t = 0; t < nt; t++) {
        int nv = ops->nvmax[t0[t + 1] - t0[t]];
        if (nv < nvg) nvg = nv;
    }
    int nk[CHUNK];
    const float *in = x;
    int ldi = ldx;
    for (int l = 0; l < b->nl; l++) {
#ifdef BGC_PROFILE
        double sp_t0 = sprof_now();
#endif
        const layer *y = &b->L[l];
        const int hidden = (l < b->nl - 1);
        float *o = b->buf[l & 1];
        const int ldo = b->ld;
        int act = ACT_LINEAR;
        if (hidden) {
            switch (b->activation) {
                case NN_ACTIVATION_RELU: act = ACT_RELU; break;
                case NN_ACTIVATION_LEAKY_RELU: act = ACT_LEAKY; break;
                case NN_ACTIVATION_HARDSIGMOID: act = ACT_HARDSIG; break;
                case NN_ACTIVATION_SIGMOID: case NN_ACTIVATION_TANH: act = ACT_LINEAR; break;
                default: act = ACT_RELU;
            }
        }
        if (l == 0 && par) {
            long nd = 0;
            ops->delta0(in, ldi, n, par, b->p0, b->W0T, y->K,
                        y->npan * (PW / vl), o, ldo, act, b->klist, b->dv, &nd);
#ifdef BGC_PROFILE
            sprof_kent[0] += nd;
            sprof_rows[0] += n;
            sprof_t[0] += sprof_now() - sp_t0;
            sp_t0 = sprof_now();
#endif
        }
        for (int t = 0; t < nt && !(l == 0 && par); t++) {
            const int mr = t0[t + 1] - t0[t];
            nk[t] = -1;
            if ((l == 0 && b->sparse1) || mr <= b->sparse_mr) {
                int c = ops->list(in + (size_t)t0[t] * ldi, ldi, mr, y->K,
                                  b->klist + (size_t)t * b->kstride);
                if (8 * c <= 7 * y->K) nk[t] = c;
            }
        }
        /* p counts vectors; a panel holds PW / vl of them. A group of nv > 1
           vectors starts on a multiple of nv, so on a panel boundary. */
        const size_t ps = (size_t)y->K * PW;
        const int nvec = y->npan * (PW / vl);
        for (int p = (l == 0 && par) ? nvec : 0; p < nvec;) {
            const int nv = (p % nvg == 0 && p + nvg <= nvec) ? nvg : 1;
            const float *Wp = y->W + (size_t)(p / (PW / vl)) * ps + (size_t)(p % (PW / vl)) * vl;
            const int s = slot_of(nv);
            for (int t = 0; t < nt; t++) {
                const int r0 = t0[t], mr = t0[t + 1] - r0;
                const int *kl = nk[t] >= 0 ? b->klist + (size_t)t * b->kstride : NULL;
                ops->kern[mr][s](in + (size_t)r0 * ldi, ldi, Wp, ps, y->K,
                                 kl, nk[t], y->bias + (size_t)p * vl,
                                 o + (size_t)r0 * ldo + (size_t)p * vl, ldo, act);
            }
            p += nv;
        }
        if (hidden && (b->activation == NN_ACTIVATION_SIGMOID ||
                       b->activation == NN_ACTIVATION_TANH)) {
            const int sig = (b->activation == NN_ACTIVATION_SIGMOID);
            for (int r = 0; r < n; r++) {
                float *row = o + (size_t)r * ldo;
                for (int j = 0; j < y->M; j++) row[j] = sig ? sigmoidf_(row[j]) : tanhf(row[j]);
            }
        }
        in = o;
        ldi = ldo;
#ifdef BGC_PROFILE
        if (!(l == 0 && par)) {
            sprof_t[l] += sprof_now() - sp_t0;
            sprof_rows[l] += n;
            for (int t = 0; t < nt; t++)
                sprof_kent[l] += (long)(nk[t] >= 0 ? nk[t] : y->K) * (t0[t + 1] - t0[t]);
        } else {
            sprof_t[l] += sprof_now() - sp_t0;    /* the sigmoid/tanh pass, if any */
        }
#endif
    }
    const int od = b->out_dim;
    for (int r = 0; r < n; r++) {
        const float *row = in + (size_t)r * ldi;
        for (int j = 0; j < od; j++)
            out[(size_t)r * od + j] = b->equity ? row[j] : sigmoidf_(row[j]);
    }
}

void bgc_backend_forward(bgc_backend *b, const float *x, int ldx, int n, float *out) {
    for (int r = 0; r < n; r += CHUNK) {
        const int c = (n - r < CHUNK) ? n - r : CHUNK;
        forward_chunk(b, x + (size_t)r * ldx, ldx, c, out + (size_t)r * b->out_dim, NULL);
    }
}

/* Layer 0 by deltas from `parent` (delta0 on, the default; BGC_SIMD_DELTA0=0
 * at create: the plain forward). p0 = the parent's pre-activation, its k
 * ascending as a one-row tile computes it (bias in the epilogue, linear);
 * a row's layer-0 pre-activation is then p0 plus its differing k's terms,
 * ascending: it depends on the row and the parent only. */
static void parent_p0(bgc_backend *b, const float *parent, int ldx) {
    const isa_ops *ops = &b->ops;
    const layer *y = &b->L[0];
    const int nk = ops->list(parent, ldx, 1, y->K, b->klist);
    const size_t ps = (size_t)y->K * PW;
    const int nvec = y->npan * (PW / ops->vl);
    for (int p = 0; p < nvec; p++) {
        const float *Wp = y->W + (size_t)(p / (PW / ops->vl)) * ps
                          + (size_t)(p % (PW / ops->vl)) * ops->vl;
        ops->kern[1][0](parent, ldx, Wp, ps, y->K, b->klist, nk, y->bias + (size_t)p * ops->vl,
                        b->p0 + (size_t)p * ops->vl, 0, ACT_LINEAR);
    }
}

void bgc_backend_forward_delta(bgc_backend *b, const float *parent, const float *x, int ldx,
                               int n, float *out) {
    if (!b->delta0) { bgc_backend_forward(b, x, ldx, n, out); return; }
    parent_p0(b, parent, ldx);
    for (int r = 0; r < n; r += CHUNK) {
        const int c = (n - r < CHUNK) ? n - r : CHUNK;
        forward_chunk(b, x + (size_t)r * ldx, ldx, c, out + (size_t)r * b->out_dim, parent);
    }
}

const char *bgc_backend_name(void) { return "simd"; }

/* A row's sums run over k ascending in every tile shape; the terms a
   tile adds for another row's nonzero k are exact zeros. */
int bgc_backend_row_exact(void) { return 1; }

const char *bgc_backend_describe(const bgc_backend *b) { return b->desc; }

#else /* SIMD_PASS: one ISA's kernels. */

/* Offset of vector v of a group from the group's first vector. */
#define VOFF(v) ((size_t)((v) / (PW / VL)) * ps + (size_t)((v) % (PW / VL)) * VL)

static inline __attribute__((always_inline)) void
FN(kbody)(const int MR, const int NV, const int LIST, const float *x, int ldx,
          const float *W, size_t ps, int K, const int *kl, int nk,
          const float *bias, float *o, int ldo, int act)
{
    VEC acc[MRMAX][16];
#pragma GCC unroll 16
    for (int r = 0; r < MR; r++)
#pragma GCC unroll 16
        for (int v = 0; v < NV; v++) acc[r][v] = VZERO();
    const int n = LIST ? nk : K;
#pragma GCC unroll 2
    for (int i = 0; i < n; i++) {
        const int k = LIST ? kl[i] : i;
        const float *w = W + (size_t)k * PW;
        if (LIST && PFD && MR <= PF_MR) {
            const float *wp = W + (size_t)kl[i + PFD] * PW;
#pragma GCC unroll 16
            for (int v = 0; v < NV; v += PW / VL) __builtin_prefetch(wp + VOFF(v), 0, 3);
        }
        VEC wv[16];
#pragma GCC unroll 16
        for (int v = 0; v < NV; v++) wv[v] = VLOAD(w + VOFF(v));
#pragma GCC unroll 16
        for (int r = 0; r < MR; r++) {
            const VEC xb = VBCAST(x[(size_t)r * ldx + k]);
#pragma GCC unroll 16
            for (int v = 0; v < NV; v++) acc[r][v] = VFMA(wv[v], xb, acc[r][v]);
        }
    }
    const VEC zero = VZERO();
#pragma GCC unroll 16
    for (int v = 0; v < NV; v++) {
        const VEC bv = VLOAD(bias + v * VL);
#pragma GCC unroll 16
        for (int r = 0; r < MR; r++) {
            VEC y = VADD(acc[r][v], bv);
            if (act == ACT_RELU) y = VMAX(y, zero);
            else if (act == ACT_LEAKY) y = VMAX(y, VMUL(y, VBCAST(0.01f)));
            else if (act == ACT_HARDSIG)
                y = VMIN(VMAX(VADD(VDIV(y, VBCAST(6.0f)), VBCAST(0.5f)), zero), VBCAST(1.0f));
            VSTORE(o + (size_t)r * ldo + v * VL, y);
        }
    }
}

#define KDEF(MR, NV)                                                              \
    static void FN(k_##MR##_##NV)(const float *x, int ldx, const float *W,        \
                                  size_t ps, int K, const int *kl, int nk,        \
                                  const float *bias, float *o, int ldo, int act) { \
        if (kl) FN(kbody)(MR, NV, 1, x, ldx, W, ps, K, kl, nk, bias, o, ldo, act); \
        else    FN(kbody)(MR, NV, 0, x, ldx, W, ps, K, kl, nk, bias, o, ldo, act); \
    }
INSTANTIATE
#undef KDEF

#define DG (VL == 16 ? 16 : 8)    /* output vectors per delta0 group */
static inline __attribute__((always_inline)) void
FN(dgroup)(const int G, const float *p0, const float *WTv, size_t ldw, const int *kl,
           const float *dv, int c, float *o, int act)
{
    VEC acc[16];
#pragma GCC unroll 16
    for (int v = 0; v < G; v++) acc[v] = VLOAD(p0 + v * VL);
    for (int i = 0; i < c; i++) {
        const float *w = WTv + (size_t)kl[i] * ldw;
        const VEC xb = VBCAST(dv[i]);
#pragma GCC unroll 16
        for (int v = 0; v < G; v++) acc[v] = VFMA(VLOAD(w + v * VL), xb, acc[v]);
    }
    const VEC zero = VZERO();
#pragma GCC unroll 16
    for (int v = 0; v < G; v++) {
        VEC y = acc[v];
        if (act == ACT_RELU) y = VMAX(y, zero);
        else if (act == ACT_LEAKY) y = VMAX(y, VMUL(y, VBCAST(0.01f)));
        else if (act == ACT_HARDSIG)
            y = VMIN(VMAX(VADD(VDIV(y, VBCAST(6.0f)), VBCAST(0.5f)), zero), VBCAST(1.0f));
        VSTORE(o + v * VL, y);
    }
}

static void FN(delta0)(const float *x, int ldx, int n, const float *par, const float *p0,
                       const float *WT, int K, int nvec, float *o, int ldo,
                       int act, int *kl, float *dv, long *nd) {
    const size_t ldw = (size_t)nvec * VL;
    for (int r = 0; r < n; r++) {
        const float *xr = x + (size_t)r * ldx;
        int c = 0, k0 = 0;
        for (; k0 + VL <= K; k0 += VL) {
            unsigned m = VNEQMASK(xr + k0, par + k0);
            while (m) {
                const int k = k0 + __builtin_ctz(m);
                kl[c] = k;
                dv[c++] = xr[k] - par[k];
                m &= m - 1;
            }
        }
        for (; k0 < K; k0++)
            if (xr[k0] != par[k0]) { kl[c] = k0; dv[c++] = xr[k0] - par[k0]; }
        *nd += c;
        float *orow = o + (size_t)r * ldo;
        int p = 0;
        for (; p + DG <= nvec; p += DG)
            FN(dgroup)(DG, p0 + (size_t)p * VL, WT + (size_t)p * VL, ldw, kl, dv, c,
                       orow + (size_t)p * VL, act);
        for (; p < nvec; p++)
            FN(dgroup)(1, p0 + (size_t)p * VL, WT + (size_t)p * VL, ldw, kl, dv, c,
                       orow + (size_t)p * VL, act);
    }
}
#undef DG

static int FN(klist)(const float *x, int ldx, int mr, int K, int *kl) {
    int c = 0, k0 = 0;
    for (; k0 + VL <= K; k0 += VL) {
        unsigned m = 0;
        for (int r = 0; r < mr; r++) m |= VNZMASK(x + (size_t)r * ldx + k0);
        while (m) { kl[c++] = k0 + __builtin_ctz(m); m &= m - 1; }
    }
    for (; k0 < K; k0++) {
        int nz = 0;
        for (int r = 0; r < mr; r++) nz |= (x[(size_t)r * ldx + k0] != 0.0f);
        if (nz) kl[c++] = k0;
    }
    for (int i = 0; i < PF_PAD; i++) kl[c + i] = c ? kl[c - 1] : 0;   /* prefetch pad */
    return c;
}

#define KDEF(MR, NV) [MR][SLOT(NV)] = FN(k_##MR##_##NV),
static kern_fn const FN(kern_tab)[MRMAX + 1][NVSLOTS] = { INSTANTIATE };
#undef KDEF
static const int FN(nvmax_tab)[MRMAX + 1] = NVMAX_INIT;

static const isa_ops FN(ops) = {
    VL, MRDEF, MRMAX,
#if SIMD_PASS == 3
    "avx512",
#elif SIMD_PASS == 2
    "avx2",
#else
    "scalar",
#endif
    FN(kern_tab), FN(nvmax_tab), FN(klist), FN(delta0)};

#undef SIMD_PASS
#undef FN
#undef VL
#undef MRMAX
#undef VEC
#undef VZERO
#undef VLOAD
#undef VSTORE
#undef VBCAST
#undef VFMA
#undef VADD
#undef VMUL
#undef VDIV
#undef VMAX
#undef VMIN
#undef VNZMASK
#undef VNEQMASK
#undef VOFF
#undef INSTANTIATE
#undef NVMAX_INIT
#undef PFD
#undef MRDEF

#endif /* SIMD_PASS */
