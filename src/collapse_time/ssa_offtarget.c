/*
 * Exact SSA (Gillespie direct method) for the four-channel SimpleQS network,
 * counts (X, W), population scale N, c and r given on the command line.
 *
 *   division   X -> X+1   rate X * (b0 + b1 * hill(W/N))
 *   death      X -> X-1   rate X * (d0 + c + theta * X/N)
 *   production W -> W+1   rate r * aC * X
 *   removal    W -> W-1   rate r * kappa * W
 *
 * Each trajectory starts at (X0, W0) and is continued past the first
 * threshold entry X <= xthr*N until it enters the SMALL off rectangle
 * (X <= as*N and W <= bs*N), which lies inside the LARGE one
 * (X <= al*N and W <= bl*N), or until the horizon H (right censoring).
 * Mode "thr" instead stops at the first threshold entry (validation only).
 *
 * Attempts: an attempt starts at a threshold entry made after the process has
 * been in the on-box B_on = {|X/N-xon|<=dx, |W/N-won|<=dw} (the initial state
 * is in B_on). It ends with outcome 0 if B_on is re-entered before the large
 * rectangle, 1 if the large rectangle is entered first, 2 if censored.
 *
 * Output (binary, little endian, to argv file):
 *   per trajectory record of 12 doubles:
 *     tau_thr, Xthr, Wthr, tau_off, Xoff, Woff, tau_small, tau_ext,
 *     n_attempts, n_events, tau_last_attempt_start, t_end
 *   (times are +inf, and states -1, if the event did not occur by H)
 *   then per attempt record of 5 doubles, appended to a second file:
 *     traj_index, t_start, X, W, outcome
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t next_u64(void) { /* xoshiro256** */
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static uint64_t splitmix(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static inline double unif(void) { /* (0,1] */
    return ((next_u64() >> 11) + 1) * 0x1.0p-53;
}

int main(int argc, char **argv) {
    if (argc < 20) {
        fprintf(stderr, "usage: ssa mode N r c X0 W0 xthr H nrep seed1 seed2 seed3 "
                        "al bl as bs xon won dx dw out_traj out_att\n");
        return 1;
    }
    int k = 1;
    const char *mode = argv[k++];
    int N = atoi(argv[k++]);
    double r = atof(argv[k++]), c = atof(argv[k++]);
    long X0 = atol(argv[k++]), W0 = atol(argv[k++]);
    double xthr = atof(argv[k++]), H = atof(argv[k++]);
    long nrep = atol(argv[k++]);
    uint64_t sd1 = strtoull(argv[k++], 0, 10), sd2 = strtoull(argv[k++], 0, 10),
             sd3 = strtoull(argv[k++], 0, 10);
    double al = atof(argv[k++]), bl = atof(argv[k++]), as = atof(argv[k++]),
           bs = atof(argv[k++]);
    double xon = atof(argv[k++]), won = atof(argv[k++]), dx = atof(argv[k++]),
           dw = atof(argv[k++]);
    const char *out_traj = argv[k++];
    const char *out_att = argv[k++];
    int thr_only = strcmp(mode, "thr") == 0;

    const double b0 = 0.5, b1 = 2.0, Kh = 1.0, d0 = 0.8, theta = 1.0, aC = 2.0,
                 kappa = 1.0;
    const double thr = xthr * N; /* same float comparison as the existing code */
    long XL = (long)floor(al * N + 1e-9), WL = (long)floor(bl * N + 1e-9);
    long XS = (long)floor(as * N + 1e-9), WS = (long)floor(bs * N + 1e-9);
    double onxlo = (xon - dx) * N, onxhi = (xon + dx) * N;
    double onwlo = (won - dw) * N, onwhi = (won + dw) * N;

    uint64_t seed = sd1 * 1000003ULL ^ (sd2 * 7919ULL + 0x1234567ULL) ^ (sd3 << 32);
    for (int i = 0; i < 4; i++) s[i] = splitmix(&seed);

    FILE *ft = fopen(out_traj, "wb"), *fa = fopen(out_att, "wb");
    if (!ft || !fa) { perror("fopen"); return 2; }

    /* Hill table for W up to a generous bound; computed on the fly beyond. */
    long WMAX = 20L * N + 1000;
    double *btab = malloc(sizeof(double) * (WMAX + 1));
    for (long w = 0; w <= WMAX; w++) {
        double q = (double)w / N, q4 = q * q * q * q;
        btab[w] = b0 + b1 * q4 / (Kh * Kh * Kh * Kh + q4);
    }

    for (long rep = 0; rep < nrep; rep++) {
        long X = X0, W = W0;
        double t = 0.0;
        double tau_thr = INFINITY, Xthr = -1, Wthr = -1;
        double tau_off = INFINITY, Xoff = -1, Woff = -1;
        double tau_small = INFINITY, tau_ext = INFINITY;
        double natt = 0, nev = 0, t_att = INFINITY;
        int armed = 1;       /* in/after B_on, so a threshold entry starts an attempt */
        int in_attempt = 0;
        double att_t = 0, att_X = 0, att_W = 0;
        for (;;) {
            double bw = (W <= WMAX) ? btab[W]
                                    : b0 + b1 * pow((double)W / N, 4) /
                                               (1.0 + pow((double)W / N, 4));
            double a0 = X * bw;
            double a1 = X * (d0 + c + theta * (double)X / N);
            double a2 = r * aC * X;
            double a3 = r * kappa * W;
            double tot = a0 + a1 + a2 + a3;
            if (tot <= 0) { t = INFINITY; break; } /* only at (0,0) */
            double tn = t - log(unif()) / tot;
            if (tn > H) break; /* censored */
            t = tn;
            double u = (1.0 - unif()) * tot; /* [0,tot) */
            if (u < a0) X++;
            else if (u < a0 + a1) X--;
            else if (u < a0 + a1 + a2) W++;
            else W--;
            nev += 1;

            if (X == 0 && isinf(tau_ext)) tau_ext = t;
            if (X <= thr) {
                if (isinf(tau_thr)) { tau_thr = t; Xthr = X; Wthr = W; }
                if (thr_only) break;
                if (armed && isinf(tau_off)) {
                    armed = 0; in_attempt = 1; natt += 1;
                    att_t = t; att_X = X; att_W = W; t_att = t;
                }
            }
            if (isinf(tau_off) && X <= XL && W <= WL) {
                tau_off = t; Xoff = X; Woff = W;
                if (in_attempt) {
                    double rec[5] = {rep, att_t, att_X, att_W, 1};
                    fwrite(rec, sizeof(double), 5, fa);
                    in_attempt = 0;
                }
            }
            if (X <= XS && W <= WS) { tau_small = t; break; }
            if (isinf(tau_off) && !armed && X >= onxlo && X <= onxhi &&
                W >= onwlo && W <= onwhi) {
                armed = 1;
                if (in_attempt) {
                    double rec[5] = {rep, att_t, att_X, att_W, 0};
                    fwrite(rec, sizeof(double), 5, fa);
                    in_attempt = 0;
                }
            }
        }
        if (in_attempt) {
            double rec[5] = {rep, att_t, att_X, att_W, 2};
            fwrite(rec, sizeof(double), 5, fa);
        }
        double out[12] = {tau_thr, Xthr, Wthr, tau_off, Xoff, Woff, tau_small,
                          tau_ext, natt, nev, t_att, t};
        fwrite(out, sizeof(double), 12, ft);
    }
    fclose(ft); fclose(fa);
    free(btab);
    return 0;
}
