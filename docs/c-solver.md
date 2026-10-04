# The C solver — how it was solved

> Written by Claude Code at the end of the August 2026 sessions in which it
> implemented this solver; "I" below is Claude. Edited only to fix paths for this
> repository and to bring the leaderboard up to date.

**Final: 7,581,357,706 m.** First place on the leaderboard as "jarski & claude",
29,943,482 m (0.39%) ahead of the next entry.

| # | Entry | Total distance (m) | vs. lower bound |
|---|---|---:|---:|
| — | *radial lower bound — provably unreachable* | *7,167,812,591* | — |
| **1** | **jarski & claude (this solver)** | **7,581,357,706** | **+5.77%** |
| 2 | noahiscool13 | 7,611,301,188 | +6.19% |
| 3 | Claude Opus 4.6 | 7,654,321,000 | +6.79% |
| 4 | Mathijs | 7,666,666,666 | +6.96% |

Leaderboard as of 2026-10-04. Entries 3 and 4 are suspiciously round numbers.

Every number here was produced by [`tools/score.py`](../tools/score.py), an independent validator that
re-derives all distances from `nicelist.txt` and checks feasibility. The solver's own
incremental arithmetic is never reported as a result.

![The solved network: 533 gift runs](figs/fig1-map.svg)

---

## 1. The problem

Deliver 10,000 presents from Korvatunturi (68.073611N, 29.315278E). The sleigh carries
10 t per trip; each trip leaves and returns to the depot. Earth is a sphere of radius
6378 km. Minimise total great-circle distance.

This is a **capacitated vehicle routing problem** — NP-hard, and at n = 10,000 firmly in
metaheuristic territory.

## 2. First: read the instance, don't write code

This step decided everything else.

**Demand fixes the route count.** Total demand is 5,308,227.5 kg, so **≥ 531 routes** are
required. Weights are near-uniform 0–1 t (median 503 kg) with a thin tail to 9.16 t →
about 19 stops per route. Weight and distance-from-depot are uncorrelated (r = −0.014),
so there is no "heavy presents live nearby" structure to exploit.

**A lower bound reframes the goal.** Every route's load must be hauled out and back, so

```
LB = 2 · Σ dᵢwᵢ / Q  =  7,167,812,591 m
```

is valid. The leaderboard's best sat only **6.19% above it**. That one number changed the
plan: the instance is *tight*, radial travel dominates, and therefore **fill rate is the
lever, not elegant tours** — 1% of unused capacity costs ≈ 73,000 km.

**A contradiction caught a bad assumption.** Modelling the children as uniform over the
sphere predicted an optimum near 8.8 Gm — *worse* than scores already on the board. Rather
than distrust the board, I measured the geometry: median nearest-neighbour distance is
**5.0 km**. The children are population-clustered, so local tours are cheap and the radial
term dominates even harder than assumed. The uniform model was simply wrong.

![Instance diagnosis: clustering and packing](figs/fig5-instance.svg)

The left panel is why tours are cheap; the right panel is the solver's response to that —
it packs the sleigh to a median of 99.81%.

## 3. The solver

C, pthreads, no dependencies. Two binaries share a core; `santa2` is production.

**Representation.** Points are 3D unit vectors; distance is `2R·asin(chord/2)`. There is
deliberately **no distance matrix**: at 10,000² a float matrix is 400 MB, and a random
cache miss into it costs more than recomputing the `asin`. A solution is a flat
`routes × stops` array plus `rof`/`pos` inverse indices and an incrementally maintained
cost; insert/remove update cost in O(1) from edge deltas.

**Construction — Clarke-Wright savings** over k-nearest-neighbour pairs, with routes held
as *undirected paths* (`adj[i][0..1]` + union-find) so a merge never has to reverse a
segment. → 7,894,733,532 m.

**Improvement — SISR** (Slack Induction by String Removals, Christiaens & Vanden Berghe
2020). Each iteration removes a few contiguous *strings* of customers around a random
seed — including "split strings" that preserve an interior chunk — then reinserts them
greedily next to a near neighbour, with a small "blink" chance of skipping a candidate
position. Simulated-annealing acceptance. Supporting local search: 2-opt + Or-opt(1–3,
both orientations) on disturbed routes, and relocate / swap / 2-opt\* with don't-look bits
periodically.

**Undo is a journal of touched routes only.** Copying the whole solution to revert a
rejected move would have dominated runtime; only the ~15 routes an iteration actually
modifies get snapshotted.

## 4. What was tried, in order

![How the score fell, and what moved it](figs/fig2-progress.svg)

**Savings → SISR (−253 M m).** The savings solution is 10.1% over the bound, with obvious
defects: routes that wander and capacity left on the table. SISR was chosen over a
genetic method (HGS) because it scales far better to n = 10,000 and handles the capacity
constraint natively through its insertion step.

**v1: 14 independent annealing chains (the obvious parallelisation, and the wrong one).**
Each thread ran its own SISR chain on the full instance, occasionally re-seeding from the
global best. It worked, but 14 cores bought nowhere near 14 cores' worth of search — the
chains largely rediscover the same improvements.

**v2: spatial decomposition (−7 to −19 M m at equal time).** Every round, route centroids
are split by recursive median cuts into one compact geographic cell per thread. Each
thread copies its cell into a private solution and runs SISR on that sub-instance alone,
so all 14 cores improve **one** solution. Re-randomising the partition each round is what
lets improvements cross cell boundaries. This was the single biggest algorithmic win and
it was consistent — v2 beat v1 at both temperatures tested.

**Monotone merging.** Cells return their *best* state, not their final annealed state.
Global cost is then non-increasing by construction, which buys two things: seeding a run
from a previous solution can never regress, and a high exploration temperature is free
because bad excursions are never merged. `SANTA_FINAL=1` switches to final-state merging;
it did not test better.

**Cyclic reheating — rejected.** Since merging is monotone, a single long cooldown seemed
wasteful: most of the schedule sits at temperatures that are either too hot to keep
anything or too cold to escape anything. A sawtooth should have been better. It was
worse, by 5.2 M m — the only rejected idea that clears the noise floor.

## 5. Tuning — and the mistake worth reading

Start temperature is the parameter that visibly moves the result, so I swept it: 15, 45,
100, 250, 500, 1000, 2500. The curve had a clean optimum at 500, and I adopted it.

Then, writing this document, I noticed I had recorded "round length 10 s beats both 5 s
and 20 s" — but the two comparisons had been run at *different* temperatures, and at
T0=250 the 5 s round had actually **won**. That prompted the check I should have run
first: **four repeats of one identical configuration.**

![The tuning curve, and how much of it is real](figs/fig3-temperature.svg)

Four identical runs of v2 / T0=500 / 300 s returned 7,592.33, 7,594.02, 7,590.05 and
7,592.93 M m — **spread 3.96 M m, sd 1.67**. The single run that had won the sweep scored
7,589.94, *better than all four repeats of the very same setting*. I had partly tuned to
luck.

Re-reading the eleven single-run experiments against that floor:

![Five plausible improvements, measured](figs/fig4-tried.svg)

| Claim I made | Measured Δ | Verdict against a 3.96 M m floor |
|---|---:|---|
| v2 decomposition beats v1 | 7.5 and 19.3 M m | **holds** — large, and reproduced at two temperatures |
| T0 = 100 and 2500 are bad | 6.6 and 6.9 M m | **holds** |
| Cyclic reheating is worse | 5.2 M m | **probably holds** (n = 1 per arm) |
| T0 = 500 is *the* optimum | ~3.1 M m over T0=250 | **not resolved** — 250–1000 is a plateau |
| Round 20 s worse than 10 s | 2.5 M m | **not resolved** |
| Final-state merging is worse | 1.0 M m | **not resolved** |
| cbar 20 worse than 10 | 0.2 M m | **not resolved** |
| Round 5 s worse than 10 s | −1.2 M m | **false as written** — 5 s was ahead |

Four of my eight "conclusions" did not survive contact with the noise floor, and one was
backwards. The shape of the temperature response is real; the location of its peak is not.

**This does not affect the result.** The final number came from a monotone run and was
independently validated — the parameters were merely somewhere on a plateau rather than at
a proven optimum.

## 6. The final run

![Convergence of the final three-hour run](figs/fig6-convergence.svg)

Three hours, 1079 rounds, seeded from the 600 s solution: 7,588.61 → 7,581.36 M m. A gain
of 7.2 M m (0.10%), with the last 1000 s contributing about 30 k m. Returns had fallen by
two orders of magnitude — more runtime on one machine is not where the next gain is.

Final solution: 533 routes, median fill 99.81%, two routes above the theoretical minimum.

## 7. Key learnings

1. **Compute a bound before writing a solver.** The radial bound turned "optimise routes"
   into the far more specific "maximise fill rate", and made *is this good?* answerable at
   every later moment.

2. **When the model contradicts observed reality, suspect the model.** A uniform-sphere
   estimate predicted worse-than-achieved scores. Measuring the actual nearest-neighbour
   distribution found the clustering and corrected the whole picture.

3. **Measure the noise floor before trusting any tuning result.** This is the big one. I
   ran eleven configurations at n = 1 and wrote up conclusions from differences of
   0.2–2.5 M m, when repeated runs of a *single* configuration span 3.96 M m. Four
   repeats — twenty minutes — would have prevented several false conclusions. Do it before
   the sweep, not after.

4. **N cores is not N× search.** Independent parallel chains scale badly on one large
   instance. Decomposing so every core improves the *same* solution beat every parameter
   change combined.

5. **Design for monotonicity when the structure allows it.** Merging per-cell bests made
   the search non-regressing, which made run-continuation and aggressive exploration free.
   An architectural property paid more than tuning did.

6. **Micro-decisions follow from hardware, not habit.** Skipping the 400 MB distance matrix
   and journalling only touched routes for undo were each worth more than they look.

7. **Validate with an independent implementation.** A solver maintaining its own
   incremental cost will cheerfully report a number that has drifted. The separate scorer
   confirmed exact agreement and checked feasibility.

8. **Know when to stop, and say why.** Returns had flattened by 100×. Stopping is a
   result, not a surrender.

## 8. What I would do next

Not more runtime. The remaining 5.77% over the bound is local tour cost plus two routes of
capacity slack, which needs a stronger move set:

- **Exact TSP per route.** Routes average 19 stops — Held-Karp is tractable at that size
  and would close whatever 2-opt/Or-opt leaves behind.
- **Capacity-repair operators** aimed specifically at the six routes below 98% fill and the
  two routes above the 531 minimum.
- **Proper experiment design**: 3–5 seeds per configuration, and a paired comparison, so
  differences of 1 M m can actually be resolved.

## 9. Files

| Path | Purpose |
|---|---|
| [`c/santa2.c`](../c/santa2.c) | production solver — SISR + spatial decomposition |
| [`c/santa.c`](../c/santa.c) | v1 reference — independent parallel chains |
| [`c/run.sh`](../c/run.sh) | run with the final configuration, then score |
| [`tools/score.py`](../tools/score.py) | independent validator + scorer |
| [`tools/gen_figs.py`](../tools/gen_figs.py) | generated every figure here from the run logs (logs not included) |

```bash
tools/fetch_data.sh
make -C c
cd c && mkdir -p out && SANTA_T0=500 SANTA_T1=1.0 SANTA_ROUND=10 ./santa2 600 14 out/best.csv
python3 ../tools/score.py out/best.csv
# continue from a previous solution (never worse than the seed):
./santa2 3600 14 out/best2.csv out/best.csv
```

Total search: ~4.7 core-hours on an M4 Pro (14 cores).
