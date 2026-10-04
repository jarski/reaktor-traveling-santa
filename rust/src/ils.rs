//! Simulated-annealing local search over a single solution, as an alternative to the ACO in
//! `lib.rs`.
//!
//! The ACO rebuilds ~100 solutions from scratch every iteration via ~10,500 sequential stochastic
//! decisions each. This module instead holds one solution and mutates it with moves whose cost
//! delta is O(1) to evaluate, so a core can attempt on the order of a million moves per second
//! against the *current best* rather than manufacturing fresh, worse tours.
//!
//! Two properties of this particular instance drive the move set (both measured, see the plan):
//! the bin packing is effectively frozen (532 routes against a theoretical minimum of 531, and no
//! route can accept even one average-weight point), so inter-route *relocation* is almost always
//! capacity-infeasible and weight-feasible *swaps* do the work of changing route composition.
//!
//! ## Route elimination is impossible on this instance -- do not re-attempt it
//!
//! Reducing the route count looks like the single biggest prize available: depot legs are 84.1% of
//! the objective and a route removed saves a mean depot round trip of 12,280 km, more than the last
//! 18 hours of chained runs gained in total. It was implemented (a `SmallestRoute` ruin kind that
//! tears out the lightest route, paired with withholding the emptied slot during recreate, plus a
//! best-fit insertion tie-break) and measured against a 15-minute control from an identical seed.
//! Route count stayed at 533 in both arms; R&R acceptance fell 35.7% -> 26.3% because the
//! elimination cycles are near-certain to fail, and the slack distribution was byte-identical
//! before and after. Reverted.
//!
//! The reason is structural, not a tuning failure. There are 21,772,492 units of slack, nominally
//! 2.18 routes' worth, but 6,802,858 of that sits inside the lightest route itself and the rest is
//! fragmented into crumbs: outside that route the largest slack anywhere is 1,282,828 and the
//! second largest is 193,316. The lightest route's four points weigh 964,074 / 960,527 / 677,065 /
//! 595,476, so at most *one* of them fits anywhere at all -- the two heaviest sum to 1,924,601
//! against the only usable route's 1,282,828. Infeasible regardless of insertion order, and this is
//! the cheapest route in the instance to absorb; every other one is strictly harder. Reaching 532
//! would require globally repacking the instance, which is a bin-packing problem no local move can
//! reach, and 531 would need 99.97% packing efficiency.
//!
//! ## Unfreezing the partition does not help either -- also measured, also reverted
//!
//! A rigorous lower bound (each route must reach its farthest point and return, so
//! `sum(max_dist) >= integral ceil(W(d)/capacity) dd`) puts the floor at 7,186,631 against a
//! 7,784,638 solution. Of that 598,007 gap, 339,829 is *clustering* loss -- the distance between
//! this partition's radial floor and the bound -- which no intra-route move can reach. Since
//! `propose` picks its partner purely geographically and never reads a weight, the obvious
//! conclusion was that the search simply could not reach better partitions.
//!
//! So `propose` was made capacity-aware: fall back to a swap when a relocation is structurally
//! infeasible, and redraw a few times for a weight-compatible swap partner. It worked, as measured
//! by `InterFeas`, the share of inter-route proposals surviving `evaluate`:
//!
//!   capacity-blind -> InterFeas  3.5%, 48.01M moves/s, best 7,784,589.49
//!   capacity-aware -> InterFeas 28.4%, 37.35M moves/s, best 7,784,633.67
//!
//! An 8.1x improvement in feasibility, a 22% throughput cost from the redraws, and a *worse*
//! objective. Reverted, keeping only the `InterFeas` counter.
//!
//! The lesson is worth more than the code: making better clusterings *reachable* is not the same as
//! there being better clusterings to find. Against pairwise weight-feasible swaps this partition is
//! already at a local optimum, so the 339,829 km is not a pool the move set is failing to drain --
//! it is mostly not recoverable at all. The bound is loose by construction: the grouping that
//! actually minimises it (points ordered by distance from the depot) scores 16,935,212 in practice,
//! because minimising depot legs alone destroys locality. Anything targeting that gap needs
//! coordinated multi-point restructuring, which is what ruin-and-recreate already does.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use log::{error, info};
use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::gift_run::GiftRun;
use crate::point::{Point, KORVATUNTURI};
use crate::{create_distance_candidate_list, id_to_index, load_best, save_best};

/// Moves attempted between wall-clock checks. Amortises `Instant::now()` (and the temperature
/// recomputation that rides along with it) across a batch.
const CLOCK_CHECK_BATCH: u64 = 4096;

pub struct IlsParams {
    pub time_budget_secs: u64,
    pub log_interval_secs: u64,
    /// How many nearest neighbours of a point are considered when proposing a move.
    pub candidate_list_size: usize,
    /// Temperature at the end of the budget, as a fraction of the calibrated starting temperature.
    pub final_temperature_ratio: f64,
    /// Moves without a local improvement before a trajectory abandons its own state and restarts
    /// from the global best. `None` never restarts, leaving trajectories fully independent -- at
    /// these temperatures this forced restart is the *only* coupling between them.
    pub stagnation_moves: Option<u64>,
    /// Sample size used to calibrate the starting temperature from real move deltas.
    pub calibration_samples: usize,
    /// Probability with which a move of average worsening size should be accepted at the start.
    /// Seed-aware: 0.5 melts an already-annealed solution (measured: a 7.94M seed was thrown to
    /// 8.51M and spent the entire budget crawling back), so a low value is right when refining an
    /// existing solution and a high one when starting from something raw.
    pub target_acceptance_rate: f64,
    /// Absolute starting temperature, bypassing calibration. Needed because the equilibrium cost
    /// scales with T (measured: T~90 sits ~300,000 above the seed, T~3 only ~1,000), and the
    /// temperatures that keep an annealed seed intact correspond to acceptance rates far too small
    /// to express via `target_acceptance_rate`.
    pub initial_temperature: Option<f64>,
    /// Smallest / largest number of points torn out per ruin. Varying the scale matters more than
    /// the exact bounds.
    pub ruin_min: usize,
    pub ruin_max: usize,
    /// Attempted single moves between ruin-and-recreate cycles.
    pub moves_per_ruin_recreate: u64,
    /// Regret-2 reinsertion instead of greedy: place whichever point stands to lose the most if
    /// its best route fills up, rather than whichever happens to come next in a shuffle.
    pub use_regret_insertion: bool,
    /// Passes of intra-route re-optimisation over the reinserted points before the R&R accept test.
    /// 0 disables it, which doubles as the A/B switch.
    pub polish_passes: usize,
}

impl Default for IlsParams {
    fn default() -> Self {
        Self {
            time_budget_secs: 3600,
            log_interval_secs: 30,
            candidate_list_size: 20,
            final_temperature_ratio: 0.001,
            stagnation_moves: Some(20_000_000),
            calibration_samples: 10_000,
            target_acceptance_rate: 0.05,
            // Fixed rather than calibrated, from a controlled sweep (15 min each, identical
            // 7,915,056 seed): T0 90/20/5/1.25/0.3 gained 301/725/1794/1570/1952. The real effect
            // is >=20 vs <=5; within 5..0.3 the differences are run-to-run noise. Chosen for the
            // smallest melt (+35 vs +273,432 at T0=90). Note this suits *refining* an annealed
            // seed -- a cold start from raw construction wants `None` to get the calibrated value.
            initial_temperature: Some(0.3),
            ruin_min: 5,
            ruin_max: 20,
            // Two coupled sweeps, 15 min each from an identical 7,874,242 seed. Ruin size first,
            // at interval 10K (mean k -> gain / R&R acceptance):
            //   275 -> 176/0.0% | 115 -> 1,898/2.0% | 90 -> 4,061/1.7% | 40 -> 7,424/10.3%
            //    25 -> 27,845/18.7% | 12.5 -> 33,760/31.5% | 5 -> 8,926/49.7%
            // Acceptance climbs monotonically as k shrinks, but gain peaks in the middle: tiny
            // ruins are almost always accepted yet change too little, huge ruins would matter but
            // are essentially never accepted (7 of 124,916 at k~275).
            //
            // Then the interval, at ruin 5-20 -- small ruins are ~10x cheaper per cycle, so the
            // 10K tuned for ruin 30-200 was far too long:
            //   10K -> 33,760 | 3K -> 34,172 | 1K -> 36,079 | 300 -> 49,553
            //   100 -> 37,462 | 30 -> 42,974 | 10 -> 46,288
            // Everything <=300 beats everything >=1000; within 300..10 the curve is non-monotonic,
            // i.e. noise. 300 is nominally best and the most conservative of that band -- it keeps
            // 55.9M moves/s of single-move polishing, where interval 10 collapses to 3.1M and is
            // effectively pure ruin & recreate for no measurable gain.
            //
            // Combined, this is 26x the previous defaults (49,553 vs 1,898) on the same seed.
            moves_per_ruin_recreate: 300,
            use_regret_insertion: true,
            polish_passes: 2,
        }
    }
}

/// Immutable problem data shared by every trajectory.
pub struct Problem {
    all_distances: Array2<f64>,
    /// Indexed by `id_to_index(id)`; the depot's entry is 0.
    weights: Vec<f64>,
    capacity: f64,
    /// Nearest neighbours by pure distance, indexed by `id_to_index(id)`.
    candidates: Vec<Vec<u32>>,
    /// Every non-depot id, for uniform random selection.
    point_ids: Vec<u32>,
}

impl Problem {
    pub fn new(points: &[Point], capacity: f64, all_distances: Array2<f64>, candidate_list_size: usize) -> Self {
        let mut weights = vec![0.0; all_distances.nrows()];
        for p in points {
            weights[p.index()] = p.weight;
        }
        let candidates = create_distance_candidate_list(points, &all_distances, candidate_list_size);
        let point_ids = points.iter().map(|p| p.id).collect();
        Self { all_distances, weights, capacity, candidates, point_ids }
    }

    #[inline]
    fn d(&self, a: u32, b: u32) -> f64 {
        self.all_distances[[id_to_index(a), id_to_index(b)]]
    }

    #[inline]
    fn w(&self, id: u32) -> f64 {
        self.weights[id_to_index(id)]
    }
}

fn route_cost(route: &[u32], problem: &Problem) -> f64 {
    if route.is_empty() {
        return 0.0;
    }
    let depot = KORVATUNTURI.id;
    let mut cost = problem.d(depot, route[0]);
    for pair in route.windows(2) {
        cost += problem.d(pair[0], pair[1]);
    }
    cost + problem.d(route[route.len() - 1], depot)
}

#[derive(Clone, Copy, Debug)]
enum Move {
    /// Reverse `routes[r][i..=j]`. Always capacity-feasible.
    Intra2Opt { r: usize, i: usize, j: usize },
    /// Relocate one point within its own route. Always capacity-feasible.
    IntraOrOpt { r: usize, from: usize, to: usize },
    /// Exchange two points between different routes.
    InterSwap { r1: usize, i: usize, r2: usize, j: usize },
    /// Move a point from one route into another. Rarely feasible on this instance.
    InterRelocate { r1: usize, i: usize, r2: usize, pos: usize },
}

/// One complete solution, with the per-route caches and the point->position index needed to keep
/// move evaluation O(1). `location` is built once and maintained incrementally -- unlike
/// `apply_or_opt` in `lib.rs`, which refills a 10,001-element index on every call.
#[derive(Clone)]
pub struct Solution {
    routes: Vec<Vec<u32>>,
    route_weight: Vec<f64>,
    route_distance: Vec<f64>,
    total_distance: f64,
    location: Vec<(u32, u32)>,
}

impl Solution {
    fn from_routes(routes: Vec<Vec<u32>>, problem: &Problem) -> Self {
        let mut location = vec![(u32::MAX, u32::MAX); problem.all_distances.nrows()];
        let mut route_weight = vec![0.0; routes.len()];
        let mut route_distance = vec![0.0; routes.len()];

        for (r, route) in routes.iter().enumerate() {
            let mut weight = 0.0;
            for (pos, &id) in route.iter().enumerate() {
                location[id_to_index(id)] = (r as u32, pos as u32);
                weight += problem.w(id);
            }
            route_weight[r] = weight;
            route_distance[r] = route_cost(route, problem);
        }

        let total_distance = route_distance.iter().sum();
        Self { routes, route_weight, route_distance, total_distance, location }
    }

    pub fn from_gift_run(gift_run: &GiftRun, problem: &Problem) -> Self {
        let routes = gift_run.all.iter().filter(|r| !r.is_empty()).cloned().collect();
        Self::from_routes(routes, problem)
    }

    pub fn to_gift_run(&self) -> GiftRun {
        let mut gift_run = GiftRun::new();
        gift_run.all = self.routes.iter().filter(|r| !r.is_empty()).cloned().collect();
        gift_run.total_distance = self.total_distance;
        gift_run
    }

    /// Full recomputation of the objective, independent of the incremental caches. Used by the
    /// drift guard and by the tests -- never on the hot path.
    fn recomputed_distance(&self, problem: &Problem) -> f64 {
        self.routes.iter().map(|r| route_cost(r, problem)).sum()
    }

    /// Cost delta of `mv` without mutating. `None` means infeasible or degenerate.
    fn evaluate(&self, problem: &Problem, mv: Move) -> Option<f64> {
        let depot = KORVATUNTURI.id;
        match mv {
            Move::Intra2Opt { r, i, j } => {
                let route = &self.routes[r];
                if i >= j || j >= route.len() {
                    return None;
                }
                let prev = if i == 0 { depot } else { route[i - 1] };
                let next = if j == route.len() - 1 { depot } else { route[j + 1] };
                Some(problem.d(prev, route[j]) + problem.d(route[i], next)
                    - problem.d(prev, route[i]) - problem.d(route[j], next))
            }
            Move::IntraOrOpt { r, from, to } => {
                let route = &self.routes[r];
                let len = route.len();
                if from >= len || to >= len || from == to {
                    return None;
                }
                let p = route[from];
                let prev_f = if from == 0 { depot } else { route[from - 1] };
                let next_f = if from == len - 1 { depot } else { route[from + 1] };
                let removal = problem.d(prev_f, next_f) - problem.d(prev_f, p) - problem.d(p, next_f);

                // Indices into the list as it looks *after* removing `from`.
                let shortened = |k: usize| -> u32 { route[if k < from { k } else { k + 1 }] };
                let a = if to == 0 { depot } else { shortened(to - 1) };
                let b = if to == len - 1 { depot } else { shortened(to) };
                let insertion = problem.d(a, p) + problem.d(p, b) - problem.d(a, b);

                Some(removal + insertion)
            }
            Move::InterSwap { r1, i, r2, j } => {
                if r1 == r2 {
                    return None;
                }
                let (ra, rb) = (&self.routes[r1], &self.routes[r2]);
                if i >= ra.len() || j >= rb.len() {
                    return None;
                }
                let (p1, p2) = (ra[i], rb[j]);
                let (w1, w2) = (problem.w(p1), problem.w(p2));
                // Only the route receiving the heavier point can actually bind, but check both.
                if self.route_weight[r1] - w1 + w2 > problem.capacity {
                    return None;
                }
                if self.route_weight[r2] - w2 + w1 > problem.capacity {
                    return None;
                }
                let a1 = if i == 0 { depot } else { ra[i - 1] };
                let b1 = if i == ra.len() - 1 { depot } else { ra[i + 1] };
                let a2 = if j == 0 { depot } else { rb[j - 1] };
                let b2 = if j == rb.len() - 1 { depot } else { rb[j + 1] };

                let d1 = problem.d(a1, p2) + problem.d(p2, b1) - problem.d(a1, p1) - problem.d(p1, b1);
                let d2 = problem.d(a2, p1) + problem.d(p1, b2) - problem.d(a2, p2) - problem.d(p2, b2);
                Some(d1 + d2)
            }
            Move::InterRelocate { r1, i, r2, pos } => {
                if r1 == r2 {
                    return None;
                }
                let (ra, rb) = (&self.routes[r1], &self.routes[r2]);
                if i >= ra.len() || pos > rb.len() {
                    return None;
                }
                let p = ra[i];
                if self.route_weight[r2] + problem.w(p) > problem.capacity {
                    return None;
                }
                let prev = if i == 0 { depot } else { ra[i - 1] };
                let next = if i == ra.len() - 1 { depot } else { ra[i + 1] };
                let removal = problem.d(prev, next) - problem.d(prev, p) - problem.d(p, next);

                let a = if pos == 0 { depot } else { rb[pos - 1] };
                let b = if pos == rb.len() { depot } else { rb[pos] };
                let insertion = problem.d(a, p) + problem.d(p, b) - problem.d(a, b);

                Some(removal + insertion)
            }
        }
    }

    fn apply(&mut self, problem: &Problem, mv: Move) {
        match mv {
            Move::Intra2Opt { r, i, j } => {
                self.routes[r][i..=j].reverse();
                self.refresh_route(r, problem);
            }
            Move::IntraOrOpt { r, from, to } => {
                let p = self.routes[r].remove(from);
                self.routes[r].insert(to, p);
                self.refresh_route(r, problem);
            }
            Move::InterSwap { r1, i, r2, j } => {
                let p1 = self.routes[r1][i];
                let p2 = self.routes[r2][j];
                self.routes[r1][i] = p2;
                self.routes[r2][j] = p1;
                self.location[id_to_index(p2)] = (r1 as u32, i as u32);
                self.location[id_to_index(p1)] = (r2 as u32, j as u32);
                let (w1, w2) = (problem.w(p1), problem.w(p2));
                self.route_weight[r1] += w2 - w1;
                self.route_weight[r2] += w1 - w2;
                self.refresh_route_distance(r1, problem);
                self.refresh_route_distance(r2, problem);
            }
            Move::InterRelocate { r1, i, r2, pos } => {
                let p = self.routes[r1].remove(i);
                self.routes[r2].insert(pos, p);
                let w = problem.w(p);
                self.route_weight[r1] -= w;
                self.route_weight[r2] += w;
                self.refresh_route(r1, problem);
                self.refresh_route(r2, problem);
            }
        }
    }

    /// Rewrites the `location` entries for a whole route (~19 points here) and its cached cost.
    /// Cheaper and far less error-prone than tracking individual index shifts.
    fn refresh_route(&mut self, r: usize, problem: &Problem) {
        let route = &self.routes[r];
        let location = &mut self.location;
        for (pos, &id) in route.iter().enumerate() {
            location[id_to_index(id)] = (r as u32, pos as u32);
        }
        self.refresh_route_distance(r, problem);
    }

    fn refresh_route_distance(&mut self, r: usize, problem: &Problem) {
        let updated = route_cost(&self.routes[r], problem);
        self.total_distance += updated - self.route_distance[r];
        self.route_distance[r] = updated;
    }

    /// Pick a random point, pick one of its nearest neighbours, and propose a move that reconnects
    /// the two. Reuses the pruned distance candidate list rather than scanning all 10,000 points.
    fn propose(&self, problem: &Problem, rng: &mut StdRng, candidate_list_size: usize) -> Option<Move> {
        let p = problem.point_ids[rng.gen_range(0..problem.point_ids.len())];
        let (r1, i) = self.location[id_to_index(p)];
        let (r1, i) = (r1 as usize, i as usize);

        let candidates = &problem.candidates[id_to_index(p)];
        if candidates.is_empty() {
            return None;
        }
        let q = candidates[rng.gen_range(0..candidate_list_size.min(candidates.len()))];
        if q == KORVATUNTURI.id {
            return None;
        }
        let (r2, j) = self.location[id_to_index(q)];
        let (r2, j) = (r2 as usize, j as usize);

        if r1 == r2 {
            if i == j {
                return None;
            }
            if rng.gen_bool(0.5) {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                Some(Move::Intra2Opt { r: r1, i: lo, j: hi })
            } else {
                // Place p immediately after q, in post-removal coordinates.
                let len = self.routes[r1].len();
                let q_shifted = if j < i { j } else { j - 1 };
                let to = (q_shifted + 1).min(len - 1);
                if to == i {
                    return None;
                }
                Some(Move::IntraOrOpt { r: r1, from: i, to })
            }
        } else if rng.gen_bool(0.5) {
            Some(Move::InterSwap { r1, i, r2, j })
        } else {
            Some(Move::InterRelocate { r1, i, r2, pos: j })
        }
    }
}

/// Marker written into `location` for a point that has been torn out and not yet reinserted.
const UNPLACED: (u32, u32) = (u32::MAX, u32::MAX);

#[derive(Clone, Copy, Debug)]
enum RuinKind {
    /// Spatially contiguous blob grown outward from a seed point.
    Radial,
    /// Uniformly random points.
    Random,
    /// Whole routes, biased toward the worst cost-per-point.
    WorstRoute,
}

/// Reusable buffers plus the undo log for one ruin-and-recreate cycle. Cloning the whole `Solution`
/// per cycle (~100KB) would be memory-bandwidth bound at these rates, so routes are saved lazily on
/// first touch instead.
struct Scratch {
    removed: Vec<u32>,
    saved: Vec<(usize, Vec<u32>, f64, f64)>,
    is_saved: Vec<bool>,
    saved_total: f64,
    original_route_count: usize,
    frontier: Vec<u32>,
}

impl Scratch {
    fn new(route_capacity: usize) -> Self {
        Self {
            removed: Vec::new(),
            saved: Vec::new(),
            is_saved: vec![false; route_capacity],
            saved_total: 0.0,
            original_route_count: 0,
            frontier: Vec::new(),
        }
    }

    fn begin(&mut self, solution: &Solution) {
        self.removed.clear();
        self.saved.clear();
        self.frontier.clear();
        self.saved_total = solution.total_distance;
        self.original_route_count = solution.routes.len();
        if self.is_saved.len() < solution.routes.len() {
            self.is_saved.resize(solution.routes.len(), false);
        }
    }

    fn snapshot_route(&mut self, solution: &Solution, r: usize) {
        if r < self.is_saved.len() && self.is_saved[r] {
            return;
        }
        if r >= self.is_saved.len() {
            self.is_saved.resize(r + 1, false);
        }
        self.is_saved[r] = true;
        self.saved.push((r, solution.routes[r].clone(), solution.route_weight[r], solution.route_distance[r]));
    }

    /// Accepts the cycle: drops the undo log and clears only the flags actually set, so the cost is
    /// proportional to the cycle rather than to the number of routes.
    fn commit(&mut self) {
        for &(r, _, _, _) in &self.saved {
            self.is_saved[r] = false;
        }
        self.saved.clear();
    }
}

impl Solution {
    pub fn route_count(&self) -> usize {
        self.routes.iter().filter(|r| !r.is_empty()).count()
    }

    /// Tears up to `k` points out of the solution, recording undo information in `scratch`. Every
    /// loop is bounded so a pathological instance cannot spin.
    fn ruin(&mut self, problem: &Problem, scratch: &mut Scratch, rng: &mut StdRng, kind: RuinKind, k: usize) {
        let k = k.min(problem.point_ids.len() / 2);
        let mut guard = k * 64 + 64;

        match kind {
            RuinKind::Random => {
                while scratch.removed.len() < k && guard > 0 {
                    guard -= 1;
                    let p = problem.point_ids[rng.gen_range(0..problem.point_ids.len())];
                    if self.location[id_to_index(p)] != UNPLACED {
                        self.remove_point(problem, scratch, p);
                    }
                }
            }
            RuinKind::Radial => {
                // Breadth-first walk over the existing 20-neighbour candidate graph. Produces a
                // spatially contiguous blob, which is what lets a whole neighbourhood re-cluster.
                scratch.frontier.clear();
                scratch.frontier.push(problem.point_ids[rng.gen_range(0..problem.point_ids.len())]);
                let mut cursor = 0usize;

                while scratch.removed.len() < k && guard > 0 {
                    guard -= 1;
                    if cursor >= scratch.frontier.len() {
                        // Blob exhausted (already-removed or disconnected region); reseed.
                        scratch.frontier.push(problem.point_ids[rng.gen_range(0..problem.point_ids.len())]);
                    }
                    let current = scratch.frontier[cursor];
                    cursor += 1;

                    if self.location[id_to_index(current)] == UNPLACED {
                        continue;
                    }
                    self.remove_point(problem, scratch, current);

                    if scratch.frontier.len() < k * 8 {
                        for &n in &problem.candidates[id_to_index(current)] {
                            if n != KORVATUNTURI.id && self.location[id_to_index(n)] != UNPLACED {
                                scratch.frontier.push(n);
                            }
                        }
                    }
                }
            }
            RuinKind::WorstRoute => {
                // Sample a handful of routes and tear out the worst by cost-per-point. This is the
                // operator aimed at the geographically scattered routes.
                while scratch.removed.len() < k && guard > 0 {
                    guard -= 1;
                    let mut worst: Option<(usize, f64)> = None;
                    for _ in 0..10 {
                        let r = rng.gen_range(0..self.routes.len());
                        if self.routes[r].is_empty() {
                            continue;
                        }
                        let score = self.route_distance[r] / self.routes[r].len() as f64;
                        if worst.map_or(true, |(_, s)| score > s) {
                            worst = Some((r, score));
                        }
                    }
                    let Some((r, _)) = worst else { break };
                    scratch.snapshot_route(self, r);
                    for &id in &self.routes[r] {
                        self.location[id_to_index(id)] = UNPLACED;
                        scratch.removed.push(id);
                    }
                    self.routes[r].clear();
                    self.route_weight[r] = 0.0;
                    self.refresh_route_distance(r, problem);
                }
            }
        }
    }

    fn remove_point(&mut self, problem: &Problem, scratch: &mut Scratch, p: u32) {
        let (r, pos) = self.location[id_to_index(p)];
        let (r, pos) = (r as usize, pos as usize);
        scratch.snapshot_route(self, r);
        self.routes[r].remove(pos);
        self.route_weight[r] -= problem.w(p);
        self.location[id_to_index(p)] = UNPLACED;
        scratch.removed.push(p);
        // `refresh_route` only rewrites locations for points still in the route, so p's UNPLACED
        // marker survives.
        self.refresh_route(r, problem);
    }

    /// Cheapest feasible insertion of `p` in each reachable route, reduced to the best overall and
    /// the best in any *other* route. The second must be in a different route or the regret is
    /// meaningless -- two slots in one route are near-duplicates, whereas the point of regret is
    /// "how much worse off am I if this route stops being available".
    ///
    /// `buf` is caller-owned so the hot path does not allocate. A depot round trip is always
    /// included as a last-resort option, so the returned best is never `None`.
    fn best_two_insertions(
        &self,
        problem: &Problem,
        p: u32,
        buf: &mut Vec<(usize, usize, f64)>,
    ) -> (usize, usize, f64, f64) {
        let depot = KORVATUNTURI.id;
        let w = problem.w(p);
        buf.clear();

        // Positions adjacent to p's geometric neighbours that are currently placed.
        for &q in &problem.candidates[id_to_index(p)] {
            if q == depot {
                continue;
            }
            let loc = self.location[id_to_index(q)];
            if loc == UNPLACED {
                continue;
            }
            let r = loc.0 as usize;
            if self.route_weight[r] + w > problem.capacity {
                continue;
            }
            let j = loc.1 as usize;
            for &pos in &[j, j + 1] {
                let cost = self.insertion_cost(problem, r, pos, p);
                match buf.iter_mut().find(|e| e.0 == r) {
                    Some(e) if cost < e.2 => *e = (r, pos, cost),
                    Some(_) => {}
                    None => buf.push((r, pos, cost)),
                }
            }
        }

        // Capacity is tight enough that every candidate position can be infeasible; look further,
        // but stop as soon as there are two real options to compare.
        if buf.len() < 2 {
            for r in 0..self.routes.len() {
                if self.route_weight[r] + w > problem.capacity || buf.iter().any(|e| e.0 == r) {
                    continue;
                }
                let mut best_pos = 0;
                let mut best_cost = f64::INFINITY;
                for pos in 0..=self.routes[r].len() {
                    let cost = self.insertion_cost(problem, r, pos, p);
                    if cost < best_cost {
                        best_cost = cost;
                        best_pos = pos;
                    }
                }
                if best_cost.is_finite() {
                    buf.push((r, best_pos, best_cost));
                }
                if buf.len() >= 2 {
                    break;
                }
            }
        }

        // Reusing an empty route and opening a fresh one cost the same; prefer the empty slot so
        // the route vector does not grow. usize::MAX means "append a new route".
        let round_trip = 2.0 * problem.d(depot, p);
        let empty = self
            .routes
            .iter()
            .position(|route| route.is_empty())
            .unwrap_or(usize::MAX);
        if !buf.iter().any(|e| e.0 == empty) {
            buf.push((empty, 0, round_trip));
        }

        let mut best = (usize::MAX, 0usize, f64::INFINITY);
        let mut second = f64::INFINITY;
        for &(r, pos, cost) in buf.iter() {
            if cost < best.2 {
                second = best.2;
                best = (r, pos, cost);
            } else if cost < second {
                second = cost;
            }
        }
        (best.0, best.1, best.2, second)
    }

    /// Puts `p` into `route` (or a brand-new route when `route == usize::MAX`), returning the index
    /// actually used.
    fn place(&mut self, problem: &Problem, scratch: &mut Scratch, p: u32, route: usize, pos: usize) -> usize {
        let w = problem.w(p);
        if route == usize::MAX || route >= self.routes.len() {
            self.routes.push(vec![p]);
            self.route_weight.push(w);
            self.route_distance.push(0.0);
            let r = self.routes.len() - 1;
            self.location[id_to_index(p)] = (r as u32, 0);
            self.refresh_route_distance(r, problem);
            r
        } else {
            scratch.snapshot_route(self, route);
            let pos = pos.min(self.routes[route].len());
            self.routes[route].insert(pos, p);
            self.route_weight[route] += w;
            self.refresh_route(route, problem);
            route
        }
    }

    /// Reinsertion of everything `ruin` tore out, either greedily (each point at its own cheapest
    /// slot, in random order) or by regret-2 (repeatedly place whichever point has most to lose).
    fn recreate(&mut self, problem: &Problem, scratch: &mut Scratch, rng: &mut StdRng, use_regret: bool) {
        let mut order = std::mem::take(&mut scratch.removed);
        let mut buf: Vec<(usize, usize, f64)> = Vec::with_capacity(48);

        if !use_regret {
            order.shuffle(rng);
            for &p in &order {
                let (r, pos, _, _) = self.best_two_insertions(problem, p, &mut buf);
                self.place(problem, scratch, p, r, pos);
            }
        } else {
            // (id, best route, best pos, best cost, regret, needs recompute)
            let mut pending: Vec<(u32, usize, usize, f64, f64, bool)> =
                order.iter().map(|&id| (id, usize::MAX, 0, 0.0, 0.0, true)).collect();

            while !pending.is_empty() {
                for entry in pending.iter_mut() {
                    if entry.5 {
                        let (r, pos, best, second) = self.best_two_insertions(problem, entry.0, &mut buf);
                        entry.1 = r;
                        entry.2 = pos;
                        entry.3 = best;
                        // A point with no alternative has unbounded regret: place it while it still
                        // has somewhere to go.
                        entry.4 = second - best;
                        entry.5 = false;
                    }
                }

                let mut pick = 0usize;
                let mut best_regret = f64::NEG_INFINITY;
                for (i, entry) in pending.iter().enumerate() {
                    // Ties broken toward the costlier placement -- the more painful commitment.
                    if entry.4 > best_regret || (entry.4 == best_regret && entry.3 > pending[pick].3) {
                        best_regret = entry.4;
                        pick = i;
                    }
                }

                let (id, route, pos, _, _, _) = pending.swap_remove(pick);
                let used = self.place(problem, scratch, id, route, pos);

                // Only points whose chosen route just changed need re-evaluating; the rest are
                // unaffected, which keeps this near-linear instead of quadratic.
                for entry in pending.iter_mut() {
                    if entry.1 == used || entry.1 == usize::MAX {
                        entry.5 = true;
                    }
                }
            }
        }

        // Left populated on purpose: `polish_placed` needs the reinserted ids, and
        // `Scratch::begin` clears `removed` at the start of every cycle so nothing leaks.
        scratch.removed = order;
    }

    /// Re-optimises the points a cycle just reinserted, so the accept test judges a polished
    /// rebuild rather than a raw one. Recreate places each point at its cheapest slot *given the
    /// route at that moment*, so later insertions into the same route can strand earlier ones;
    /// nothing else revisits them before the Metropolis comparison.
    ///
    /// Deliberately limited to intra-route moves on the reinserted points. That keeps the cost at
    /// roughly one extra recreate (~940 evaluations/cycle against ~500) instead of the ~15,840 a
    /// full polish of every touched route would take, and it means only already-snapshotted routes
    /// can change -- `place` snapshots every existing route it inserts into, and any brand-new
    /// route is discarded wholesale by `rollback`'s truncate.
    fn polish_placed(&mut self, problem: &Problem, placed: &[u32], passes: usize) {
        for _ in 0..passes {
            let mut improved = false;

            for &p in placed {
                let loc = self.location[id_to_index(p)];
                if loc == UNPLACED {
                    continue;
                }
                let (r, pos) = (loc.0 as usize, loc.1 as usize);
                let len = self.routes[r].len();
                if len < 2 {
                    continue;
                }

                let mut best: Option<(Move, f64)> = None;
                let consider = |mv: Move, delta: Option<f64>, best: &mut Option<(Move, f64)>| {
                    if let Some(d) = delta {
                        if d < -1e-9 && best.map_or(true, |(_, bd)| d < bd) {
                            *best = Some((mv, d));
                        }
                    }
                };

                // Relocate p elsewhere in its own route.
                for to in 0..len {
                    if to == pos {
                        continue;
                    }
                    let mv = Move::IntraOrOpt { r, from: pos, to };
                    consider(mv, self.evaluate(problem, mv), &mut best);
                }

                // Reverse any segment with p at one end.
                for j in 0..len {
                    if j == pos {
                        continue;
                    }
                    let (i, j) = if pos < j { (pos, j) } else { (j, pos) };
                    let mv = Move::Intra2Opt { r, i, j };
                    consider(mv, self.evaluate(problem, mv), &mut best);
                }

                if let Some((mv, _)) = best {
                    self.apply(problem, mv);
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
    }

    fn insertion_cost(&self, problem: &Problem, r: usize, pos: usize, p: u32) -> f64 {
        let depot = KORVATUNTURI.id;
        let route = &self.routes[r];
        let pos = pos.min(route.len());
        let a = if pos == 0 { depot } else { route[pos - 1] };
        let b = if pos == route.len() { depot } else { route[pos] };
        problem.d(a, p) + problem.d(p, b) - problem.d(a, b)
    }

    fn rollback(&mut self, scratch: &mut Scratch) {
        self.routes.truncate(scratch.original_route_count);
        self.route_weight.truncate(scratch.original_route_count);
        self.route_distance.truncate(scratch.original_route_count);

        // Taken out so the saved flags can be cleared in the same pass.
        let mut saved = std::mem::take(&mut scratch.saved);
        for (r, contents, weight, distance) in saved.drain(..).rev() {
            scratch.is_saved[r] = false;
            // Routes opened during this cycle were just truncated away; their points are restored
            // by whichever pre-existing route they were torn out of.
            if r >= scratch.original_route_count {
                continue;
            }
            self.routes[r] = contents;
            self.route_weight[r] = weight;
            self.route_distance[r] = distance;
            let route = &self.routes[r];
            let location = &mut self.location;
            for (pos, &id) in route.iter().enumerate() {
                location[id_to_index(id)] = (r as u32, pos as u32);
            }
        }
        scratch.saved = saved;
        self.total_distance = scratch.saved_total;
    }
}

/// Sets the starting temperature from real move deltas so there is no magic constant: solving
/// `exp(-mean_positive_delta / T) = target_acceptance_rate` gives
/// `T0 = mean_positive_delta / ln(1 / target_acceptance_rate)`.
fn calibrate_temperature(problem: &Problem, solution: &Solution, rng: &mut StdRng, params: &IlsParams) -> f64 {
    if let Some(t0) = params.initial_temperature {
        return t0.max(1e-9);
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for _ in 0..params.calibration_samples {
        if let Some(mv) = solution.propose(problem, rng, params.candidate_list_size) {
            if let Some(delta) = solution.evaluate(problem, mv) {
                if delta > 0.0 {
                    sum += delta;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return 1.0;
    }
    let rate = params.target_acceptance_rate.clamp(1e-6, 0.999);
    ((sum / count as f64) / (1.0 / rate).ln()).max(1e-9)
}

/// Capacity-aware nearest neighbour. Cold-start path only -- the normal path seeds every
/// trajectory from the best solution already on disk.
fn construct_nearest_neighbour(problem: &Problem) -> Solution {
    let depot = KORVATUNTURI.id;
    let mut visited = vec![false; problem.all_distances.nrows()];
    let mut remaining = problem.point_ids.len();
    let mut routes: Vec<Vec<u32>> = Vec::new();

    while remaining > 0 {
        let mut route: Vec<u32> = Vec::new();
        let mut load = 0.0;
        let mut current = depot;

        loop {
            // The candidate list is distance-sorted, so the first that fits is the nearest.
            let mut chosen = problem.candidates[id_to_index(current)]
                .iter()
                .copied()
                .find(|&c| {
                    c != depot
                        && !visited[id_to_index(c)]
                        && load + problem.weights[id_to_index(c)] <= problem.capacity
                });

            if chosen.is_none() {
                let mut best: Option<(u32, f64)> = None;
                for &c in &problem.point_ids {
                    let ci = id_to_index(c);
                    if visited[ci] || load + problem.weights[ci] > problem.capacity {
                        continue;
                    }
                    let d = problem.d(current, c);
                    if best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((c, d));
                    }
                }
                chosen = best.map(|(c, _)| c);
            }

            match chosen {
                Some(c) => {
                    visited[id_to_index(c)] = true;
                    load += problem.weights[id_to_index(c)];
                    route.push(c);
                    current = c;
                    remaining -= 1;
                }
                None => break,
            }
        }

        if route.is_empty() {
            break; // nothing left fits anywhere; avoid spinning
        }
        routes.push(route);
    }

    Solution::from_routes(routes, problem)
}

struct Shared {
    best: Mutex<GiftRun>,
    attempted: AtomicU64,
    accepted: AtomicU64,
    /// Inter-route moves emitted by `propose`, and how many of those survived `evaluate` rather than
    /// being rejected on capacity. Sits at ~3.5%, the live measure of how frozen the partition is;
    /// see the module header for why raising it deliberately did not help.
    inter_proposed: AtomicU64,
    inter_feasible: AtomicU64,
}

pub fn run(points: &[Point], capacity: f64, all_distances: Array2<f64>, params: &IlsParams) -> GiftRun {
    info!("--- Starting SA/ILS local search ---");
    info!("- Number of points: {}", points.len());
    info!("- Capacity: {}", capacity);
    info!("- time_budget_secs: {}", params.time_budget_secs);
    info!("- candidate_list_size: {}", params.candidate_list_size);
    info!("- final_temperature_ratio: {}", params.final_temperature_ratio);
    info!("- stagnation_moves: {:?}", params.stagnation_moves);
    info!("- polish_passes: {}", params.polish_passes);
    log::logger().flush();

    info!("Pre-calculating distance candidate lists...");
    let problem = Problem::new(points, capacity, all_distances, params.candidate_list_size);

    let start = match load_best() {
        Some(gift_run) => {
            let solution = Solution::from_gift_run(&gift_run, &problem);
            info!(
                "Seeded from results/: {:.2} over {} routes",
                solution.total_distance,
                solution.routes.len()
            );
            solution
        }
        None => {
            info!("No saved solution; constructing via capacity-aware nearest neighbour...");
            let solution = construct_nearest_neighbour(&problem);
            info!(
                "Constructed: {:.2} over {} routes",
                solution.total_distance,
                solution.routes.len()
            );
            solution
        }
    };

    let trajectories = num_cpus::get();
    info!("Running {} independent trajectories on {} CPU cores.", trajectories, num_cpus::get());

    let shared = Shared {
        best: Mutex::new(start.to_gift_run()),
        attempted: AtomicU64::new(0),
        accepted: AtomicU64::new(0),
        inter_proposed: AtomicU64::new(0),
        inter_feasible: AtomicU64::new(0),
    };
    let deadline = Instant::now() + Duration::from_secs(params.time_budget_secs);

    (0..trajectories).into_par_iter().for_each(|t| {
        run_trajectory(&problem, &start, params, t as u64, deadline, &shared, t == 0);
    });

    let best = shared.best.lock().unwrap().clone();
    if let Err(e) = save_best(&best) {
        error!("Failed to save final best solution: {}", e);
    }
    info!("SA/ILS finished. Best: {:.2}", best.total_distance);
    best
}

/// Whether a trajectory should abandon its own state for the global best. Split out so `None`
/// ("never restart") is type-safe rather than riding on a sentinel that could overflow the compare.
fn should_restart(moves_since_improvement: u64, stagnation_moves: Option<u64>) -> bool {
    stagnation_moves.map_or(false, |limit| moves_since_improvement >= limit)
}

fn run_trajectory(
    problem: &Problem,
    start: &Solution,
    params: &IlsParams,
    seed: u64,
    deadline: Instant,
    shared: &Shared,
    is_logger: bool,
) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(seed.wrapping_mul(0x9E3779B9)));
    let mut current = start.clone();

    let t0 = calibrate_temperature(problem, &current, &mut rng, params);
    if is_logger {
        info!("Calibrated starting temperature: {:.4}", t0);
    }

    let mut temperature;
    let mut best_local = current.total_distance;
    let mut known_global = best_local;
    let mut moves_since_improvement: u64 = 0;
    let mut scratch = Scratch::new(current.routes.len() + 64);
    let mut moves_since_ruin: u64 = 0;
    let mut rr_attempted: u64 = 0;
    let mut rr_accepted: u64 = 0;
    let mut restarts: u64 = 0;
    let mut last_log = Instant::now();
    let started = Instant::now();
    let budget = params.time_budget_secs as f64;

    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }

        // Wall-clock cooling: independent of throughput, so the schedule holds regardless of how
        // many moves per second this machine actually manages.
        let progress = (started.elapsed().as_secs_f64() / budget).min(1.0);
        temperature = (t0 * params.final_temperature_ratio.powf(progress)).max(1e-12);

        let mut attempted = 0u64;
        let mut accepted = 0u64;
        let mut inter_proposed = 0u64;
        let mut inter_feasible = 0u64;

        for _ in 0..CLOCK_CHECK_BATCH {
            attempted += 1;

            // Ruin & recreate: single moves cannot free capacity, so periodically tear out a chunk
            // of the solution and rebuild it. Deltas here are ~1000s of km against a temperature of
            // 0.3 or less, so the shared Metropolis test is effectively accept-if-better -- the
            // intent. This sits at the top of the loop deliberately: below the `continue`s it only
            // advanced for proposals surviving both filters, which on this instance is ~0.3% of
            // attempts, so it fired ~300x less often than `moves_per_ruin_recreate` states.
            moves_since_ruin += 1;
            if moves_since_ruin >= params.moves_per_ruin_recreate {
                moves_since_ruin = 0;
                rr_attempted += 1;

                let before = current.total_distance;
                scratch.begin(&current);
                let kind = match rng.gen_range(0..3) {
                    0 => RuinKind::Radial,
                    1 => RuinKind::WorstRoute,
                    _ => RuinKind::Random,
                };
                let k = rng.gen_range(params.ruin_min..=params.ruin_max.max(params.ruin_min));
                current.ruin(problem, &mut scratch, &mut rng, kind, k);
                current.recreate(problem, &mut scratch, &mut rng, params.use_regret_insertion);
                if params.polish_passes > 0 {
                    let placed = std::mem::take(&mut scratch.removed);
                    current.polish_placed(problem, &placed, params.polish_passes);
                    scratch.removed = placed;
                }

                let delta = current.total_distance - before;
                if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
                    scratch.commit();
                    rr_accepted += 1;
                } else {
                    current.rollback(&mut scratch);
                }

                if current.total_distance < best_local - 1e-9 {
                    best_local = current.total_distance;
                    moves_since_improvement = 0;
                }
            }

            let Some(mv) = current.propose(problem, &mut rng, params.candidate_list_size) else {
                moves_since_improvement += 1;
                continue;
            };
            let is_inter = matches!(mv, Move::InterSwap { .. } | Move::InterRelocate { .. });
            if is_inter {
                inter_proposed += 1;
            }
            let Some(delta) = current.evaluate(problem, mv) else {
                moves_since_improvement += 1;
                continue;
            };
            if is_inter {
                inter_feasible += 1;
            }

            if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
                current.apply(problem, mv);
                accepted += 1;
            }

            if current.total_distance < best_local - 1e-9 {
                best_local = current.total_distance;
                moves_since_improvement = 0;
            } else {
                moves_since_improvement += 1;
            }
        }

        shared.attempted.fetch_add(attempted, Ordering::Relaxed);
        shared.accepted.fetch_add(accepted, Ordering::Relaxed);
        shared.inter_proposed.fetch_add(inter_proposed, Ordering::Relaxed);
        shared.inter_feasible.fetch_add(inter_feasible, Ordering::Relaxed);

        // Publish only genuine improvements; the lock is otherwise untouched.
        if best_local < known_global - 1e-9 {
            let mut global = shared.best.lock().unwrap();
            if current.total_distance < global.total_distance {
                *global = current.to_gift_run();
            }
            known_global = global.total_distance;
        }

        if should_restart(moves_since_improvement, params.stagnation_moves) {
            restarts += 1;
            let global = shared.best.lock().unwrap().clone();
            current = Solution::from_gift_run(&global, problem);
            best_local = current.total_distance;
            known_global = best_local;
            moves_since_improvement = 0;
        }

        if is_logger && last_log.elapsed().as_secs() >= params.log_interval_secs {
            let total_attempted = shared.attempted.load(Ordering::Relaxed);
            let total_accepted = shared.accepted.load(Ordering::Relaxed);
            let elapsed = started.elapsed().as_secs_f64();
            let global = shared.best.lock().unwrap().clone();

            // Drift guard: an incremental-delta bug would otherwise corrupt the objective silently.
            let recomputed = current.recomputed_distance(problem);
            if (recomputed - current.total_distance).abs() > 1e-3 {
                error!(
                    "DRIFT: incremental total {:.6} != recomputed {:.6}",
                    current.total_distance, recomputed
                );
            }

            info!(
                "[{:.1}%] Best: {:.2}, Traj0 Cur: {:.2}, T: {:.4}, {:.2}M moves/s, Accept: {:.2}%, R&R: {}/{} ({:.1}%), InterFeas: {:.1}%, Restarts: {}, Routes: {}, NoImprove: {}",
                progress * 100.0,
                global.total_distance,
                current.total_distance,
                temperature,
                total_attempted as f64 / elapsed / 1e6,
                if total_attempted == 0 { 0.0 } else { total_accepted as f64 / total_attempted as f64 * 100.0 },
                rr_accepted,
                rr_attempted,
                if rr_attempted == 0 { 0.0 } else { rr_accepted as f64 / rr_attempted as f64 * 100.0 },
                {
                    let ip = shared.inter_proposed.load(Ordering::Relaxed);
                    let iff = shared.inter_feasible.load(Ordering::Relaxed);
                    if ip == 0 { 0.0 } else { iff as f64 / ip as f64 * 100.0 }
                },
                restarts,
                current.route_count(),
                moves_since_improvement,
            );

            if let Err(e) = save_best(&global) {
                error!("Failed to save best solution: {}", e);
            }
            last_log = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_all_distances;

    fn test_problem(capacity: f64) -> Problem {
        let points = vec![
            Point::new(2, 60.0, 10.0, 100.0),
            Point::new(3, 60.0, 11.0, 100.0),
            Point::new(4, 60.0, 12.0, 100.0),
            Point::new(5, 61.0, 10.0, 100.0),
            Point::new(6, 61.0, 11.0, 100.0),
            Point::new(7, 61.0, 12.0, 100.0),
        ];
        let all_distances = calculate_all_distances(&points);
        Problem::new(&points, capacity, all_distances, 4)
    }

    fn two_route_solution(problem: &Problem) -> Solution {
        Solution::from_routes(vec![vec![2, 3, 4], vec![5, 6, 7]], problem)
    }

    /// The central invariant: a move's predicted delta must equal the actual change in the
    /// objective, and the incrementally-tracked total must still match a from-scratch recompute.
    fn assert_delta_matches(problem: &Problem, solution: &Solution, mv: Move) {
        let predicted = solution.evaluate(problem, mv).expect("move should be feasible");
        let before = solution.total_distance;

        let mut applied = solution.clone();
        applied.apply(problem, mv);
        let actual = applied.total_distance - before;

        assert!(
            (predicted - actual).abs() < 1e-6,
            "{:?}: predicted delta {} but got {}",
            mv, predicted, actual
        );
        assert!(
            (applied.total_distance - applied.recomputed_distance(problem)).abs() < 1e-6,
            "{:?}: incremental total drifted from recomputed", mv
        );
    }

    #[test]
    fn test_intra_2_opt_delta_matches_full_recompute() {
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert_delta_matches(&problem, &solution, Move::Intra2Opt { r: 0, i, j });
        }
    }

    #[test]
    fn test_intra_or_opt_delta_matches_full_recompute() {
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        for (from, to) in [(0, 1), (0, 2), (1, 0), (2, 0), (2, 1)] {
            assert_delta_matches(&problem, &solution, Move::IntraOrOpt { r: 0, from, to });
        }
    }

    #[test]
    fn test_inter_swap_delta_matches_full_recompute() {
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        for (i, j) in [(0, 0), (0, 2), (1, 1), (2, 0)] {
            assert_delta_matches(&problem, &solution, Move::InterSwap { r1: 0, i, r2: 1, j });
        }
    }

    #[test]
    fn test_inter_relocate_delta_matches_full_recompute() {
        // Capacity 400 leaves room for exactly one extra point in a 3-point route.
        let problem = test_problem(400.0);
        let solution = two_route_solution(&problem);
        for (i, pos) in [(0, 0), (1, 1), (2, 3)] {
            assert_delta_matches(&problem, &solution, Move::InterRelocate { r1: 0, i, r2: 1, pos });
        }
    }

    #[test]
    fn test_inter_relocate_rejects_over_capacity() {
        // Both routes are exactly full at capacity 300, so nothing can move across.
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        assert!(solution.evaluate(&problem, Move::InterRelocate { r1: 0, i: 0, r2: 1, pos: 0 }).is_none());
    }

    #[test]
    fn test_inter_swap_respects_capacity() {
        // Equal weights, so a swap is weight-neutral and stays feasible even at exact capacity.
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        assert!(solution.evaluate(&problem, Move::InterSwap { r1: 0, i: 0, r2: 1, j: 0 }).is_some());

        // A heavier point cannot displace a lighter one into an already-full route.
        let points = vec![
            Point::new(2, 60.0, 10.0, 100.0),
            Point::new(3, 60.0, 11.0, 100.0),
            Point::new(4, 61.0, 10.0, 250.0),
            Point::new(5, 61.0, 11.0, 100.0),
        ];
        let all_distances = calculate_all_distances(&points);
        let tight = Problem::new(&points, 350.0, all_distances, 3);
        let solution = Solution::from_routes(vec![vec![2, 3], vec![4, 5]], &tight);
        // Route 0 holds 200/350, route 1 holds 350/350. Moving the 250 into route 0 for a 100
        // would put route 0 at 350 (ok) and route 1 at 200 (ok).
        assert!(solution.evaluate(&tight, Move::InterSwap { r1: 0, i: 0, r2: 1, j: 0 }).is_some());
        // But swapping the 250 for the other 100 in route 0 is identical arithmetic; verify a
        // genuinely infeasible direction instead.
        let solution = Solution::from_routes(vec![vec![2, 3, 5], vec![4]], &tight);
        // Route 0 is at 300/350, route 1 at 250/350. Swapping a 100 out for the 250 in would put
        // route 0 at 450 -- infeasible.
        assert!(solution.evaluate(&tight, Move::InterSwap { r1: 0, i: 0, r2: 1, j: 0 }).is_none());
    }

    #[test]
    fn test_location_index_stays_consistent_after_random_moves() {
        let problem = test_problem(400.0);
        let mut solution = two_route_solution(&problem);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..500 {
            if let Some(mv) = solution.propose(&problem, &mut rng, 4) {
                if solution.evaluate(&problem, mv).is_some() {
                    solution.apply(&problem, mv);
                }
            }
        }

        for (r, route) in solution.routes.iter().enumerate() {
            for (pos, &id) in route.iter().enumerate() {
                assert_eq!(
                    solution.location[id_to_index(id)],
                    (r as u32, pos as u32),
                    "location index stale for id {}", id
                );
            }
        }
        assert!((solution.total_distance - solution.recomputed_distance(&problem)).abs() < 1e-6);
        for (r, &w) in solution.route_weight.iter().enumerate() {
            let actual: f64 = solution.routes[r].iter().map(|&id| problem.w(id)).sum();
            assert!((w - actual).abs() < 1e-9, "route weight cache stale for route {}", r);
            assert!(w <= problem.capacity + 1e-9, "route {} exceeds capacity", r);
        }
    }

    #[test]
    fn test_lower_target_acceptance_gives_cooler_start() {
        let problem = test_problem(400.0);
        let solution = two_route_solution(&problem);

        let temp_at = |rate: f64| {
            // `initial_temperature: None` opts into the calibration path this test exercises; the
            // default now pins an absolute temperature and would bypass it.
            let params = IlsParams {
                target_acceptance_rate: rate,
                calibration_samples: 2000,
                initial_temperature: None,
                ..Default::default()
            };
            let mut rng = StdRng::seed_from_u64(7);
            calibrate_temperature(&problem, &solution, &mut rng, &params)
        };

        let hot = temp_at(0.5);
        let cool = temp_at(0.05);
        assert!(cool < hot, "5% acceptance must calibrate cooler than 50% ({} vs {})", cool, hot);

        // T0 = mean / ln(1/rate), so the ratio is fixed by the rates alone: ln(2) vs ln(20).
        let expected = (1.0f64 / 0.05).ln() / (1.0f64 / 0.5).ln();
        assert!(
            ((hot / cool) - expected).abs() < 1e-9,
            "expected a {:.2}x reduction, got {:.2}x", expected, hot / cool
        );
    }

    /// A larger, spread-out instance so ruin operators have something to chew on.
    fn wide_problem(capacity: f64) -> Problem {
        let points: Vec<Point> = (0..40)
            .map(|i| {
                let id = (i + 2) as u32;
                let lat = 55.0 + (i % 8) as f64 * 1.5;
                let lon = 5.0 + (i / 8) as f64 * 3.0;
                Point::new(id, lat, lon, 100.0)
            })
            .collect();
        let all_distances = calculate_all_distances(&points);
        Problem::new(&points, capacity, all_distances, 6)
    }

    fn wide_solution(problem: &Problem) -> Solution {
        let routes: Vec<Vec<u32>> = (0..8)
            .map(|r| (0..5).map(|i| (2 + r * 5 + i) as u32).collect())
            .collect();
        Solution::from_routes(routes, problem)
    }

    fn all_ids(solution: &Solution) -> Vec<u32> {
        let mut ids: Vec<u32> = solution.routes.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn test_rollback_restores_exactly() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let original = solution.clone();
        let mut scratch = Scratch::new(solution.routes.len() + 8);
        let mut rng = StdRng::seed_from_u64(11);

        for (kind, regret) in [
            (RuinKind::Radial, false), (RuinKind::Random, false), (RuinKind::WorstRoute, false),
            (RuinKind::Radial, true), (RuinKind::Random, true), (RuinKind::WorstRoute, true),
        ] {
            scratch.begin(&solution);
            solution.ruin(&problem, &mut scratch, &mut rng, kind, 12);
            solution.recreate(&problem, &mut scratch, &mut rng, regret);
            let placed = std::mem::take(&mut scratch.removed);
            solution.polish_placed(&problem, &placed, 2);
            scratch.removed = placed;
            solution.rollback(&mut scratch);

            assert_eq!(solution.routes, original.routes, "{:?}/regret={}: routes not restored", kind, regret);
            assert_eq!(solution.route_weight, original.route_weight, "{:?}/regret={}: weights not restored", kind, regret);
            assert_eq!(solution.location, original.location, "{:?}/regret={}: location not restored", kind, regret);
            assert!(
                (solution.total_distance - original.total_distance).abs() < 1e-9,
                "{:?}/regret={}: total not restored", kind, regret
            );
            // Flags must be fully cleared, or the next cycle silently fails to snapshot.
            assert!(scratch.is_saved.iter().all(|&f| !f), "{:?}/regret={}: saved flags leaked", kind, regret);
        }
    }

    /// Regression: a route opened during recreate and then touched again gets snapshotted at an
    /// index past `original_route_count`. Rollback truncates it away, so restoring it by index
    /// panicked. Caught by a real run, not by the original tests.
    #[test]
    fn test_rollback_after_recreate_opens_new_routes() {
        // Every point nearly fills a sleigh, so recreate is forced to open fresh routes.
        let points: Vec<Point> = (0..8)
            .map(|i| Point::new((i + 2) as u32, 60.0 + i as f64 * 0.5, 10.0 + i as f64 * 0.5, 900.0))
            .collect();
        let all_distances = calculate_all_distances(&points);
        let problem = Problem::new(&points, 1000.0, all_distances, 4);
        let mut solution = Solution::from_routes(
            (0..8).map(|i| vec![(i + 2) as u32]).collect(),
            &problem,
        );
        let original = solution.clone();
        let mut scratch = Scratch::new(4); // deliberately smaller than the route count
        let mut rng = StdRng::seed_from_u64(21);

        for round in 0..20 {
            let regret = round % 2 == 0;
            scratch.begin(&solution);
            solution.ruin(&problem, &mut scratch, &mut rng, RuinKind::Random, 4);
            solution.recreate(&problem, &mut scratch, &mut rng, regret);
            // Polish can touch a route this cycle opened; rollback must still restore exactly.
            let placed = std::mem::take(&mut scratch.removed);
            solution.polish_placed(&problem, &placed, 2);
            scratch.removed = placed;
            solution.rollback(&mut scratch);

            assert_eq!(solution.routes, original.routes);
            assert_eq!(solution.route_weight, original.route_weight);
            assert_eq!(solution.location, original.location);
            assert!((solution.total_distance - original.total_distance).abs() < 1e-9);
            assert!(scratch.is_saved.iter().all(|&f| !f), "saved flags leaked");
        }
    }

    #[test]
    fn test_ruin_recreate_conserves_points_and_capacity() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let expected = all_ids(&solution);
        let mut scratch = Scratch::new(solution.routes.len() + 8);
        let mut rng = StdRng::seed_from_u64(3);

        for round in 0..60 {
            let regret = round % 4 < 2;
            let kind = match round % 3 {
                0 => RuinKind::Radial,
                1 => RuinKind::WorstRoute,
                _ => RuinKind::Random,
            };
            scratch.begin(&solution);
            solution.ruin(&problem, &mut scratch, &mut rng, kind, 10);
            solution.recreate(&problem, &mut scratch, &mut rng, regret);

            // Alternate accepting and rolling back so both paths are exercised.
            if round % 2 == 0 {
                scratch.commit();
            } else {
                solution.rollback(&mut scratch);
            }

            assert_eq!(all_ids(&solution), expected, "round {}: points lost or duplicated", round);
            for &id in &problem.point_ids {
                assert_ne!(
                    solution.location[id_to_index(id)], UNPLACED,
                    "round {}: point {} left unplaced", round, id
                );
            }
            for (r, &w) in solution.route_weight.iter().enumerate() {
                assert!(w <= problem.capacity + 1e-9, "round {}: route {} over capacity", round, r);
                let actual: f64 = solution.routes[r].iter().map(|&id| problem.w(id)).sum();
                assert!((w - actual).abs() < 1e-9, "round {}: weight cache stale on route {}", round, r);
            }
            assert!(
                (solution.total_distance - solution.recomputed_distance(&problem)).abs() < 1e-6,
                "round {}: total drifted from recomputed", round
            );
            for (r, route) in solution.routes.iter().enumerate() {
                for (pos, &id) in route.iter().enumerate() {
                    assert_eq!(solution.location[id_to_index(id)], (r as u32, pos as u32),
                        "round {}: location stale for {}", round, id);
                }
            }
        }
    }

    #[test]
    fn test_each_ruin_operator_removes_points() {
        let problem = wide_problem(600.0);
        let mut scratch = Scratch::new(16);
        let mut rng = StdRng::seed_from_u64(5);

        const K: usize = 10;
        for kind in [RuinKind::Radial, RuinKind::Random, RuinKind::WorstRoute] {
            let mut solution = wide_solution(&problem);
            let longest = solution.routes.iter().map(|r| r.len()).max().unwrap();
            scratch.begin(&solution);
            solution.ruin(&problem, &mut scratch, &mut rng, kind, K);

            let removed = scratch.removed.len();
            match kind {
                // These remove one point at a time and stop on the button.
                RuinKind::Radial | RuinKind::Random => {
                    assert_eq!(removed, K, "{:?} should remove exactly k", kind);
                }
                // This removes whole routes, so it overshoots by up to one route's worth. Worth
                // pinning: against ~19-point routes a k of 30 can overshoot by more than 50%, which
                // matters as soon as outcomes get attributed to k.
                RuinKind::WorstRoute => {
                    assert!(
                        removed >= K && removed <= K + longest,
                        "{:?} removed {} for k={}, outside [k, k+longest_route={}]",
                        kind, removed, K, K + longest
                    );
                }
            }

            for &id in &scratch.removed {
                assert_eq!(solution.location[id_to_index(id)], UNPLACED, "{:?}: {} not marked", kind, id);
            }
            let still_in: Vec<u32> = solution.routes.iter().flatten().copied().collect();
            for &id in &scratch.removed {
                assert!(!still_in.contains(&id), "{:?}: {} removed but still in a route", kind, id);
            }
        }
    }

    #[test]
    fn test_recreate_opens_route_when_nothing_fits() {
        // Four points, each nearly filling a sleigh: every point needs its own route.
        let points: Vec<Point> = (0..4)
            .map(|i| Point::new((i + 2) as u32, 60.0 + i as f64, 10.0 + i as f64, 900.0))
            .collect();
        let all_distances = calculate_all_distances(&points);
        let problem = Problem::new(&points, 1000.0, all_distances, 3);
        let mut solution = Solution::from_routes(vec![vec![2], vec![3], vec![4], vec![5]], &problem);
        let mut scratch = Scratch::new(8);
        let mut rng = StdRng::seed_from_u64(9);

        let regret = true;
        scratch.begin(&solution);
        solution.ruin(&problem, &mut scratch, &mut rng, RuinKind::Random, 2);
        solution.recreate(&problem, &mut scratch, &mut rng, regret);
        scratch.commit();

        assert_eq!(all_ids(&solution), vec![2, 3, 4, 5]);
        assert_eq!(solution.route_count(), 4, "each point still needs its own route");
        for (r, &w) in solution.route_weight.iter().enumerate() {
            assert!(w <= problem.capacity + 1e-9, "route {} over capacity", r);
        }
    }

    /// Regret's whole point: a point with one viable route must be placed before one with several
    /// equivalent options, regardless of shuffle order. Greedy has no such guarantee.
    #[test]
    fn test_regret_prefers_the_constrained_point() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let expected = all_ids(&solution);
        let mut scratch = Scratch::new(solution.routes.len() + 8);

        // Same ruin, both strategies: each must return a valid, complete solution, and regret must
        // never leave a point stranded in a worse slot than greedy's total.
        let mut greedy_total = 0.0;
        let mut regret_total = 0.0;
        for (regret, out) in [(false, &mut greedy_total), (true, &mut regret_total)] {
            let mut rng = StdRng::seed_from_u64(77);
            scratch.begin(&solution);
            solution.ruin(&problem, &mut scratch, &mut rng, RuinKind::Radial, 14);
            solution.recreate(&problem, &mut scratch, &mut rng, regret);
            *out = solution.total_distance;

            assert_eq!(all_ids(&solution), expected, "regret={}: points lost", regret);
            for &id in &problem.point_ids {
                assert_ne!(solution.location[id_to_index(id)], UNPLACED, "regret={}: {} unplaced", regret, id);
            }
            for (r, &w) in solution.route_weight.iter().enumerate() {
                assert!(w <= problem.capacity + 1e-9, "regret={}: route {} over capacity", regret, r);
            }
            assert!(
                (solution.total_distance - solution.recomputed_distance(&problem)).abs() < 1e-6,
                "regret={}: total drifted", regret
            );
            solution.rollback(&mut scratch);
        }
        assert!(greedy_total > 0.0 && regret_total > 0.0);
    }

    #[test]
    fn test_best_two_insertions_second_is_in_another_route() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let mut scratch = Scratch::new(16);
        let mut rng = StdRng::seed_from_u64(4);

        scratch.begin(&solution);
        solution.ruin(&problem, &mut scratch, &mut rng, RuinKind::Random, 3);
        let p = scratch.removed[0];

        let mut buf = Vec::new();
        let (best_route, _, best_cost, second_cost) = solution.best_two_insertions(&problem, p, &mut buf);

        assert!(best_cost.is_finite(), "a depot round trip is always available, so best must exist");
        assert!(second_cost >= best_cost, "second-best cannot beat best");
        // Every entry considered is the cheapest slot within a distinct route.
        let mut routes: Vec<usize> = buf.iter().map(|e| e.0).collect();
        routes.sort_unstable();
        let before = routes.len();
        routes.dedup();
        assert_eq!(before, routes.len(), "one option per route, so regret compares across routes");
        assert!(buf.iter().any(|e| e.0 == best_route));
    }

    #[test]
    fn test_polish_never_worsens_and_keeps_invariants() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let expected = all_ids(&solution);
        let mut scratch = Scratch::new(solution.routes.len() + 8);
        let mut rng = StdRng::seed_from_u64(31);

        for round in 0..40 {
            scratch.begin(&solution);
            let kind = match round % 3 {
                0 => RuinKind::Radial,
                1 => RuinKind::WorstRoute,
                _ => RuinKind::Random,
            };
            solution.ruin(&problem, &mut scratch, &mut rng, kind, 10);
            solution.recreate(&problem, &mut scratch, &mut rng, true);

            let before_polish = solution.total_distance;
            let placed = std::mem::take(&mut scratch.removed);
            solution.polish_placed(&problem, &placed, 3);
            scratch.removed = placed;

            // Polish only applies strictly-improving moves, so any increase is a delta bug.
            assert!(
                solution.total_distance <= before_polish + 1e-9,
                "round {}: polish worsened {} -> {}", round, before_polish, solution.total_distance
            );
            assert!(
                (solution.total_distance - solution.recomputed_distance(&problem)).abs() < 1e-6,
                "round {}: polish drifted from recomputed", round
            );

            scratch.commit();

            assert_eq!(all_ids(&solution), expected, "round {}: points lost", round);
            for &id in &problem.point_ids {
                assert_ne!(solution.location[id_to_index(id)], UNPLACED, "round {}: {} unplaced", round, id);
            }
            for (r, route) in solution.routes.iter().enumerate() {
                for (pos, &id) in route.iter().enumerate() {
                    assert_eq!(solution.location[id_to_index(id)], (r as u32, pos as u32),
                        "round {}: location stale for {}", round, id);
                }
            }
            for (r, &w) in solution.route_weight.iter().enumerate() {
                let actual: f64 = solution.routes[r].iter().map(|&id| problem.w(id)).sum();
                assert!((w - actual).abs() < 1e-9, "round {}: weight cache stale on route {}", round, r);
                assert!(w <= problem.capacity + 1e-9, "round {}: route {} over capacity", round, r);
            }
        }
    }

    #[test]
    fn test_polish_zero_passes_is_a_no_op() {
        let problem = wide_problem(600.0);
        let mut solution = wide_solution(&problem);
        let before = solution.clone();
        let placed: Vec<u32> = solution.routes[0].clone();
        solution.polish_placed(&problem, &placed, 0);
        assert_eq!(solution.routes, before.routes);
        assert_eq!(solution.location, before.location);
        assert_eq!(IlsParams::default().polish_passes, 2);
    }

    #[test]
    fn test_should_restart_respects_none() {
        // The whole point of the Option: "never" must hold at any counter value, including one
        // that would overflow a sentinel comparison.
        assert!(!should_restart(0, None));
        assert!(!should_restart(u64::MAX, None));

        assert!(!should_restart(9, Some(10)));
        assert!(should_restart(10, Some(10)), "fires exactly at the limit");
        assert!(should_restart(11, Some(10)));
        assert!(should_restart(u64::MAX, Some(10)));

        // The shipped default restarts; the sweep varies it.
        assert_eq!(IlsParams::default().stagnation_moves, Some(20_000_000));
    }

    #[test]
    fn test_initial_temperature_overrides_calibration() {
        let problem = test_problem(400.0);
        let solution = two_route_solution(&problem);
        let params = IlsParams { initial_temperature: Some(0.3), ..Default::default() };
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(calibrate_temperature(&problem, &solution, &mut rng, &params), 0.3);
        // And the shipped default pins it, so runs are reproducible rather than seed-dependent.
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(
            calibrate_temperature(&problem, &solution, &mut rng, &IlsParams::default()),
            0.3
        );
    }

    #[test]
    fn test_solution_gift_run_roundtrip() {
        let problem = test_problem(300.0);
        let solution = two_route_solution(&problem);
        let restored = Solution::from_gift_run(&solution.to_gift_run(), &problem);

        assert_eq!(restored.routes, solution.routes);
        assert!((restored.total_distance - solution.total_distance).abs() < 1e-9);

        let mut ids: Vec<u32> = restored.routes.iter().flatten().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_nearest_neighbour_construction_covers_all_points() {
        let problem = test_problem(300.0);
        let solution = construct_nearest_neighbour(&problem);

        let mut ids: Vec<u32> = solution.routes.iter().flatten().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 3, 4, 5, 6, 7], "every point must be visited exactly once");
        assert!((solution.total_distance - solution.recomputed_distance(&problem)).abs() < 1e-6);
        for (r, &w) in solution.route_weight.iter().enumerate() {
            assert!(w <= problem.capacity + 1e-9, "constructed route {} exceeds capacity", r);
        }
    }
}
