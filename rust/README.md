# Rust: ACO, then SA/ILS with ruin-and-recreate

Two solvers in one binary, picked with `SOLVER`:

| Mode | What it does | Best result |
|---|---|---:|
| `SOLVER=aco` | Max-Min Ant System with candidate lists, 2-opt and Or-opt local search, plateau kicks. The 2018 Python idea, rewritten and extended. | 8,296,882,600 m |
| default (`ils`) | One solution mutated by O(1)-delta moves under simulated annealing, 14 parallel trajectories, periodic ruin-and-recreate with regret-2 reinsertion. | 7,784,637,555 m |

The ILS result was 10 chained one-hour runs on 2026-08-10, each seeded from the previous best. It
is the "jarski" entry on the leaderboard.

```sh
../tools/fetch_data.sh                  # once, from anywhere in the repo
cargo build --release
cargo test --release

ILS_BUDGET=60 ./target/release/traveling_santa            # SA/ILS for 60 s
SOLVER=aco ACO_ITERATIONS=20 ./target/release/traveling_santa
python3 ../tools/score.py results/$(ls results | sort -n | head -1)
```

Both modes write every improving solution to `results/<km>.txt`, and both resume from the lowest
file there on the next run. ACO also persists its pheromone matrix to `pheromone_state.bin`
(about 800 MB); delete it to start ACO cold.

ILS knobs (env vars): `ILS_BUDGET` seconds (default 3600), `ILS_T0`, `ILS_REGRET`,
`ILS_RR_INTERVAL`, `ILS_RUIN_MIN`, `ILS_RUIN_MAX`, `ILS_POLISH`, `ILS_STAGNATION`.

The module docs in [`src/ils.rs`](src/ils.rs) record two ideas that were measured and reverted
(route elimination and capacity-aware move proposal), with the numbers.
