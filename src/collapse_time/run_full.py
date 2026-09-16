"""Full collapse-time campaign: extend the pilot to the complete grid of the
existing threshold campaign (r = 0.5, 1, 2, 4, 8 at c = 0.36).

Reuses run_pilot.py unchanged: points already in data/collapse_time/raw are skipped, so the 20
pilot points are kept. New points use 30 chunks (50 replicas each) so that the
most expensive point, r = 0.5 N = 110, spreads over all workers.

Usage:  python run_full.py [--workers 10]
"""
import run_pilot

run_pilot.PLAN = {
    0.5: [40, 50, 60, 70, 80, 90, 100, 110],
    1.0: [40, 47, 55, 62, 70, 77, 85, 92, 100],
    2.0: [50, 65, 80, 95, 110],
    4.0: [50, 60, 70, 80, 90, 100, 110],
    8.0: [60, 80, 100, 120, 140],
}
run_pilot.N_CHUNK = 30

if __name__ == "__main__":
    run_pilot.main()
