# Reaktor Traveling Santa

My solvers for Reaktor's [Traveling Santa](https://traveling-santa.reaktor.com/) challenge,
from a Python ant colony in December 2018 to the solver that has held first place on the
[leaderboard](https://traveling-santa.reaktor.com/leaderboard) since August 2026. The 2026
code was written by AI coding assistants working to my direction. The table under
[Who did what](#who-did-what) says which parts.

**Result: 7,581,357,706 m, 1st of all entries, 0.39% shorter than the next one.**

![The final solution: 533 sleigh runs out of Korvatunturi](docs/figs/fig1-map.svg)

## The problem

Santa has to deliver 10,000 presents from Korvatunturi, Finland. The sleigh carries 10 t
per trip and every trip starts and ends at Korvatunturi. Earth is a sphere of radius
6,378 km and Santa flies great circles. Minimise the total distance.

This is a capacitated vehicle routing problem (CVRP) with 10,000 customers, too large for
exact methods. Two numbers frame it:

- Total demand is 5,308 t, so at least **531 trips** are needed. Good solutions use 533.
- There is a simple lower bound on the total distance:

  ```
  LB = 2 · Σᵢ dᵢ · wᵢ / Q  =  7,167,812,591 m
  ```

  where `dᵢ` is the great-circle distance from Korvatunturi to child `i`, `wᵢ` is the
  weight of child `i`'s present, and `Q` = 10,000,000 g is the sleigh's capacity.

  Why it holds: a trip has to reach its farthest child and come back, so it is at
  least `2 · max dᵢ` long. A weighted average never exceeds the maximum, and the trip
  carries at most `Q`, so `max dᵢ ≥ Σ dᵢwᵢ / Σ wᵢ ≥ Σ dᵢwᵢ / Q` over the trip's children.
  Adding this up over all trips gives `LB`. Put simply, every gram has to be flown out to
  its child and the sleigh has to come back, and one trip can share that cost among at
  most 10 t.

  The best solutions sit about 6% above the bound. Most of the distance is the radial
  flight out and back, so packing the sleigh full matters more than the order of stops
  within a trip.

## Results

All distances below were re-scored from the saved solution files with
[`tools/score.py`](tools/score.py), which re-derives every distance from the input data
and checks feasibility.

| When | Solver | Distance (m) | Over lower bound |
|---|---|---:|---:|
| 2018-12-23 – 2019-01-05 | Python ant colony ([`python/`](python/)) | 8,810,640,121 | 22.9% |
| 2026-03 | Rust port of the ant colony ([`rust/`](rust/), `SOLVER=aco`) | 8,646,801,234 | 20.6% |
| 2026-08-05 | Rust ant colony + 2-opt/Or-opt local search | 8,296,882,600 | 15.8% |
| 2026-08-06 | Rust simulated annealing on one solution (SA/ILS) | 7,912,338,223 | 10.4% |
| 2026-08-10 | Rust SA/ILS + ruin-and-recreate, ten chained 1 h runs | **7,784,637,555** | 8.6% |
| 2026-08-10 | C: Clarke-Wright savings construction only ([`c/`](c/)) | 7,894,733,532 | 10.1% |
| 2026-08-11 | C: SISR + spatial decomposition, 600 s | 7,588,607,409 | 5.9% |
| 2026-08-11 | C: same, continued for 3 h | **7,581,357,706** | 5.8% |

The two bold rows are on the leaderboard (2026-10-04):

| Rank | Entry | Distance (m) |
|---:|---|---:|
| 1 | jarski & claude (C solver) | 7,581,357,706 |
| 2 | noahiscool13 | 7,611,301,188 |
| 9 | jarski (Rust SA/ILS) | 7,784,637,555 |

A few entries between 2 and 9 have patterned scores (7,654,321,000; 7,666,666,666;
7,777,777,777), so I don't treat them as real baselines. I don't think the 2018 result
was ever submitted.

## The story

**December 2018.** I wrote an ant colony optimiser in Python over the Christmas holidays.
Each "ant" builds a full solution: start a trip at a random child, keep flying to the
nearest child that still fits in the sleigh (and sometimes to the strongest pheromone
trail instead), and go home when nothing fits. The best solutions then reinforce the
pheromone on their edges. It reached 8.81 Gm. The code is kept as it was, apart from a
few fixes that let it run from a clean clone (see the git log).

**March 2026.** I ported the ant colony to Rust with an AI assistant, adding candidate
lists, Max-Min pheromone bounds and parallel ants with rayon. Faster, but only slightly
better: 8.65 Gm.

**August 2026.** I came back with Claude Code. In the Rust solver, local search (2-opt,
Or-opt) and then simulated annealing on a single solution, with periodic ruin-and-recreate,
took it to 7.78 Gm. That entry is still 9th. The switch from the ant colony paid off
because the ants rebuilt about 100 solutions per second from scratch, while annealing
evaluated over 100 million cheap moves per second against the current best. The module
docs in
[`rust/src/ils.rs`](rust/src/ils.rs) record two ideas that were measured and reverted.

Then I started over in C with a different design: Clarke-Wright savings for the initial
solution, then SISR (Slack Induction by String Removals, Christiaens & Vanden Berghe 2020)
ruin-and-recreate under simulated annealing. Claude Code implemented it. The biggest
win was parallelism: instead of 14 independent annealing chains, every round splits the
map into one geographic cell per core, so all cores improve the same solution, and each
cell returns its best state, so the global cost never goes up. 600 seconds reached
7,588.6 Mm and a 3-hour continuation reached 7,581.4 Mm. On this machine (M4 Pro,
14 cores) a 60-second run from scratch already gives about 7,619 Mm.

![How the C solver's score fell](docs/figs/fig2-progress.svg)

The full write-up of the C solver, including a tuning mistake caught by measuring the
noise floor, is in [`docs/c-solver.md`](docs/c-solver.md).

## Who did what

| Part | Design | Code |
|---|---|---|
| Python ant colony, 2018 | me | me |
| Rust ant colony port, March 2026 | me | me, with an AI assistant |
| Rust SA/ILS and ruin-and-recreate, August 2026 | me directing | Claude Code (Sonnet 5, Opus 5) |
| C solver, August 2026 | me (Clarke-Wright + SISR + annealing) | Claude Code |
| [`docs/c-solver.md`](docs/c-solver.md), figures | | Claude Code |
| This repository | me reviewing | assembled by Claude Code from the original project folders |

The 2026 work is an experiment in directing AI agents on a hard numerical problem: I
chose the methods, set the measurement rules (an independent scorer, repeated runs before
trusting a tuning result) and decided what to keep. The agents wrote the code.

## Running it

You need the challenge data first. It is Reaktor's, so it is not in this repository:

```sh
tools/fetch_data.sh    # downloads data/nicelist.txt, checks its sha256, links it into each solver dir
```

**C solver** (any C compiler with pthreads):

```sh
make -C c
c/run.sh 60                       # 60 s on all cores, writes c/out/best.csv and scores it
c/run.sh 3600 c/out/best.csv      # continue from a previous solution; never gets worse
```

**Rust solver** (stable Rust):

```sh
cd rust
cargo build --release && cargo test --release
ILS_BUDGET=60 ./target/release/traveling_santa                  # SA/ILS for 60 s
SOLVER=aco ACO_ITERATIONS=20 ./target/release/traveling_santa   # ant colony
```

**Python solver** (Python 3 with numpy; needs about 3 GB of RAM for the distance and
pheromone matrices):

```sh
cd python
python3 -m unittest TestDistance TestReadingFile
python3 runit.py 10               # 10 iterations of 4 ants
```

Rust and Python write each improving solution to `results/<km>.txt` in their directory
and resume from the best one on the next run.

**Score any solution:**

```sh
python3 tools/score.py c/out/best.csv
python3 tools/score.py rust/results/<file>.txt
```

## Layout

```
python/   2018 ant colony
rust/     ant colony and SA/ILS (2026)
c/        Clarke-Wright + SISR + spatial decomposition (2026, the winning solver)
tools/    fetch_data.sh, score.py, gen_figs.py
docs/     c-solver.md and figures
```

## License

[MIT](LICENSE). The challenge data belongs to Reaktor and is not covered.
