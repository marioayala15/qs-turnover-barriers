/*
 * Committor from near the saddle for the four-channel network.
 * Exact SSA (Gillespie direct method), same rates and RNG as
 * src/collapse_time/ssa_offtarget.c.
 *
 * Each trajectory starts at (X0, W0) and stops at the first of:
 *   entry into R_off = {X <= al*N, W <= bl*N}                  -> outcome 1
 *   entry into B_on  = {|X/N-xon| <= dx, |W/N-won| <= dw}      -> outcome 0
 *   time H                                                     -> outcome 2
 * Output: per trajectory 3 doubles (outcome, stopping time, n_events).
 *
 * usage: ssa_commit N r c X0 W0 al bl xon won dx dw H nrep seed1 seed2 seed3 out
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t next_u64(void) {
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
static inline double unif(void) { return ((next_u64() >> 11) + 1) * 0x1.0p-53; }

int main(int argc, char **argv) {
    if (argc != 18) {
        fprintf(stderr, "usage: ssa_commit N r c X0 W0 al bl xon won dx dw H nrep "
                        "seed1 seed2 seed3 out\n");
        return 1;
    }
    int k = 1;
    int N = atoi(argv[k++]);
    double r = atof(argv[k++]), c = atof(argv[k++]);
    long X0 = atol(argv[k++]), W0 = atol(argv[k++]);
    double al = atof(argv[k++]), bl = atof(argv[k++]);
    double xon = atof(argv[k++]), won = atof(argv[k++]);
    double dx = atof(argv[k++]), dw = atof(argv[k++]);
    double H = atof(argv[k++]);
    long nrep = atol(argv[k++]);
    uint64_t sd1 = strtoull(argv[k++], 0, 10), sd2 = strtoull(argv[k++], 0, 10),
             sd3 = strtoull(argv[k++], 0, 10);
    const char *out = argv[k++];

    const double b0 = 0.5, b1 = 2.0, Kh = 1.0, d0 = 0.8, theta = 1.0, aC = 2.0,
                 kappa = 1.0;
    long XL = (long)floor(al * N + 1e-9), WL = (long)floor(bl * N + 1e-9);
    double onxlo = (xon - dx) * N, onxhi = (xon + dx) * N;
    double onwlo = (won - dw) * N, onwhi = (won + dw) * N;

    uint64_t seed = sd1 * 1000003ULL ^ (sd2 * 7919ULL + 0x1234567ULL) ^ (sd3 << 32);
    for (int i = 0; i < 4; i++) s[i] = splitmix(&seed);

    FILE *fo = fopen(out, "wb");
    if (!fo) { perror("fopen"); return 2; }

    for (long rep = 0; rep < nrep; rep++) {
        long X = X0, W = W0;
        double t = 0.0, nev = 0, outcome = 2;
        for (;;) {
            if (X <= XL && W <= WL) { outcome = 1; break; }
            if (X >= onxlo && X <= onxhi && W >= onwlo && W <= onwhi) { outcome = 0; break; }
            double q = (double)W / N, q4 = q * q * q * q;
            double bw = b0 + b1 * q4 / (Kh * Kh * Kh * Kh + q4);
            double a0 = X * bw;
            double a1 = X * (d0 + c + theta * (double)X / N);
            double a2 = r * aC * X;
            double a3 = r * kappa * W;
            double tot = a0 + a1 + a2 + a3;
            if (tot <= 0) { outcome = 1; break; } /* (0,0) lies in R_off anyway */
            double tn = t - log(unif()) / tot;
            if (tn > H) { t = H; break; }
            t = tn;
            double u = (1.0 - unif()) * tot;
            if (u < a0) X++;
            else if (u < a0 + a1) X--;
            else if (u < a0 + a1 + a2) W++;
            else W--;
            nev += 1;
        }
        double rec[3] = {outcome, t, nev};
        fwrite(rec, sizeof(double), 3, fo);
    }
    fclose(fo);
    return 0;
}
