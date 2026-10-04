use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use log::{debug, info};
use ndarray::{Array, Array1, Array2};
use rand::{thread_rng, Rng, seq::SliceRandom};
use rayon::prelude::*;
use serde::Deserialize;

pub mod point;
pub mod gift_run;
pub mod ant;
pub mod ils;

use point::{Point, KORVATUNTURI};
use gift_run::GiftRun;
use ant::Ant;

const R: f64 = 6378.0; // kilometers
const MAX_LOCAL_SEARCH_ROUNDS: usize = 3; // alternation rounds between 2-opt and Or-opt per ant per iteration
// A new, distinct name -- do not reuse the existing stale, untracked 763MB `pheromone.bin` left over
// from an earlier experiment, which no current code path reads or writes.
const PHEROMONE_STATE_PATH: &str = "pheromone_state.bin";

pub fn solve() {
    let points = read_points("nicelist.txt").expect("Failed to read points");
    let all_distances = calculate_all_distances(&points);    
    let params = Hyperparameters {
        num_ants: 100,
        candidate_list_size: 50,
        alpha: 1.0, // Influence of pheromone
        beta: 2.0,  // Influence of heuristic (attractiveness)
        rho: 0.1,
        // Raised from 0.9: greedier construction hands better raw tours to 2-opt/Or-opt. This is
        // not pheromone learning -- measured convergence is flat either way -- but it lowered the
        // whole population's cost level and set a new best (8,296,883 vs 8,312,390).
        q0: 0.98,
        tau_min_divisor: 100.0,
        // MMAS-style: a single depositing elite plus a dominant global best, with quality-scaled
        // deposits. The previous num_elites=40 / elitist_weight=1 smeared 40 solutions together,
        // leaving edges shared by 2 mediocre ants better reinforced than edges unique to the
        // global best. This yields a cleanly bimodal matrix and ~22% tighter population variance.
        num_elites: 1,
        elitist_weight: 10,
        deposit_quality_exponent: 50.0,
        log_interval_secs: 30,
        pheromone_deposit_weight: 0.15,
        or_opt_candidate_list_size: 20,
        plateau_patience: Some(2000),
        kick_routes_per_event: 5,
        kick_deposit_weight: 1.0,
        max_iterations: std::env::var("ACO_ITERATIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(10000),
    };

    optimize(&points, 10000000.0, &all_distances, false, true, &params);
}

// SA/ILS entry point. Unlike `solve`, this holds one solution and mutates it with O(1)-delta moves
// rather than rebuilding a population every iteration; see `ils` for why that suits this instance.
pub fn solve_ils() {
    let points = read_points("nicelist.txt").expect("Failed to read points");
    let all_distances = calculate_all_distances(&points);
    // ILS_T0 / ILS_BUDGET let a sweep vary the schedule without a rebuild. Applied as mutations
    // rather than struct-literal fields so an absent variable leaves the default intact -- setting
    // the field directly wrote None over the tuned default whenever ILS_T0 was unset.
    let mut params = ils::IlsParams::default();
    if let Some(t0) = std::env::var("ILS_T0").ok().and_then(|v| v.parse().ok()) {
        params.initial_temperature = Some(t0);
    }
    if let Some(budget) = std::env::var("ILS_BUDGET").ok().and_then(|v| v.parse().ok()) {
        params.time_budget_secs = budget;
    }
    if let Ok(v) = std::env::var("ILS_REGRET") {
        params.use_regret_insertion = v != "0";
    }
    if let Some(n) = std::env::var("ILS_RR_INTERVAL").ok().and_then(|v| v.parse().ok()) {
        params.moves_per_ruin_recreate = n;
    }
    if let Some(n) = std::env::var("ILS_RUIN_MIN").ok().and_then(|v| v.parse().ok()) {
        params.ruin_min = n;
    }
    if let Some(n) = std::env::var("ILS_RUIN_MAX").ok().and_then(|v| v.parse().ok()) {
        params.ruin_max = n;
    }
    if let Some(n) = std::env::var("ILS_POLISH").ok().and_then(|v| v.parse().ok()) {
        params.polish_passes = n;
    }
    // "0" or "never" means never restart, leaving trajectories fully independent.
    if let Ok(v) = std::env::var("ILS_STAGNATION") {
        params.stagnation_moves = if v == "never" || v == "0" {
            None
        } else {
            v.parse().ok().or(params.stagnation_moves)
        };
    }

    ils::run(&points, 10000000.0, all_distances, &params);
}

#[derive(Deserialize)]
struct RawPoint {
    id: u32,
    latitude: f64,
    longitude: f64,
    weight: f64,
}

pub fn read_points(filename: &str) -> Result<Vec<Point>, csv::Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(false)
        .from_path(filename)?;

    let mut points = Vec::new();
    for result in rdr.deserialize() {
        let raw_point: RawPoint = result?;
        let point = Point::new(raw_point.id, raw_point.latitude, raw_point.longitude, raw_point.weight);
        points.push(point);
    }
    Ok(points)
}

pub fn calculate_all_distances(points: &[Point]) -> Array2<f64> {
    let mut vectors = vec![KORVATUNTURI.vector.clone()];
    vectors.extend(points.iter().map(|p| p.vector.clone()));

    let p1 = Array::from_shape_vec((vectors.len(), 3), vectors.into_iter().flatten().collect()).unwrap();
    let p2 = p1.t();

    let v: Array2<f64> = p1.dot(&p2) / (R * R);

    v.mapv(|val| R * val.min(1.0).acos())
}

fn id_to_index(id: u32) -> usize {
    if id == 0 { 0 } else { (id - 1) as usize }
}

fn index_to_id(index: usize) -> u32 {
    if index == 0 { 0 } else { ((index + 1) as usize).try_into().unwrap() }
}

pub fn apply_2_opt(route: &mut Vec<u32>, all_distances: &Array2<f64>) -> bool {
    debug!("2-Opt: Initial route: {:?}", route);
    let mut improvement_found = true;
    let mut any_improvement = false;
    let mut full_route = vec![KORVATUNTURI.id];
    full_route.extend(route.iter());
    full_route.push(KORVATUNTURI.id);

    while improvement_found {
        improvement_found = false;
        'outer: for i in 1..full_route.len() - 2 { // Corrected loop start
            for j in i + 2..full_route.len() - 2 {
                let node_i_start_idx = id_to_index(full_route[i]);
                let node_i_end_idx = id_to_index(full_route[i+1]);
                let node_j_start_idx = id_to_index(full_route[j]);
                let node_j_end_idx = id_to_index(full_route[j + 1]);

                let current_distance = all_distances[[node_i_start_idx, node_i_end_idx]] + all_distances[[node_j_start_idx, node_j_end_idx]];
                let new_distance = all_distances[[node_i_start_idx, node_j_start_idx]] + all_distances[[node_i_end_idx, node_j_end_idx]];
                if new_distance < current_distance {
                    debug!("New distance {} < {} current distance", new_distance, current_distance);
                    debug!("2-Opt: Found improvement. Swapping edges ({}, {}) and ({}, {}).", full_route[i], full_route[i+1], full_route[j], full_route[j+1]);
                    debug!("2-Opt: Reversing segment from index {} to {}: {:?}", i + 1, j, &full_route[i+1..=j]);
                    full_route[i+1..=j].reverse();
                    debug!("2-Opt: New route segment: {:?}", &full_route);
                    improvement_found = true;
                    any_improvement = true;
                    break 'outer;
                }
            }
        }
    }
    *route = full_route[1..full_route.len()-1].to_vec();
    debug!("2-Opt: Final route: {:?}", route);
    any_improvement
}

// Quality-scaled deposit. The previous `weight * 9e6/distance` form had only ~4.5% dynamic range
// across the entire population -- every solution to this problem lands within a few percent of
// every other, so reinforcement was effectively blind to solution quality. Scaling by relative
// excess over the reference (best-known) distance restores real contrast; `k` controls how
// sharply worse solutions are discounted. A solution matching the reference deposits the full
// `pheromone_deposit_weight`; one k-scaled-percent worse deposits exponentially less.
fn pheromone_deposit_amount(distance: f64, reference_distance: f64, pheromone_deposit_weight: f64, k: f64) -> f64 {
    if distance == 0.0 || distance == f64::MAX { return 0.0; } // Avoid division by zero or using max float
    if reference_distance <= 0.0 || reference_distance == f64::MAX { return 0.0; }
    let relative_excess = ((distance - reference_distance) / reference_distance).max(0.0);
    pheromone_deposit_weight * (-k * relative_excess).exp()
}

fn deposit_pheromone_for_run(pheromone: &mut Array2<f64>, run: &GiftRun, amount: f64, symmetric: bool) {
    if amount == 0.0 { return; }
    for route in &run.all {
        let mut last_index = KORVATUNTURI.index();
        for &next_point_id in route {
            let next_index = id_to_index(next_point_id);
            pheromone[[last_index, next_index]] += amount;
            if symmetric {
                pheromone[[next_index, last_index]] += amount; // Symmetric deposit
            }
            last_index = next_index;
        }
        pheromone[[last_index, KORVATUNTURI.index()]] += amount;
        if symmetric {
            pheromone[[KORVATUNTURI.index(), last_index]] += amount; // Symmetric deposit
        }
    }
}

pub fn update_pheromone(pheromone: &mut Array2<f64>, best_so_far: &GiftRun, iter_ants: &[GiftRun], rho: f64, num_elites: usize, elitist_weight: usize, tau_min: f64, tau_max: f64, pheromone_deposit_weight: f64, deposit_quality_exponent: f64) {
    // Global update (evaporation)
    *pheromone *= 1.0 - rho;

    // Reference point for quality scaling: the best distance known. Falls back to the iteration
    // best on the very first update, before any best-so-far exists. `iter_ants` is sorted
    // ascending by total_distance by the caller, so the first entry is the iteration best.
    let reference_distance = if best_so_far.total_distance != f64::MAX {
        best_so_far.total_distance
    } else {
        iter_ants.first().map_or(f64::MAX, |a| a.total_distance)
    };

    // Pheromone deposit for iteration's elite ants
    for ant in iter_ants.iter().take(num_elites) {
        let amount = pheromone_deposit_amount(ant.total_distance, reference_distance, pheromone_deposit_weight, deposit_quality_exponent);
        deposit_pheromone_for_run(pheromone, ant, amount, false);
    }

    // Pheromone deposit for best-so-far ant
    if best_so_far.total_distance != f64::MAX {
        let amount = (elitist_weight as f64) * pheromone_deposit_amount(best_so_far.total_distance, reference_distance, pheromone_deposit_weight, deposit_quality_exponent);
        deposit_pheromone_for_run(pheromone, best_so_far, amount, true);
    }

    // Enforce pheromone bounds (Max-Min)
    pheromone.mapv_inplace(|p| p.min(tau_max).max(tau_min));
}


fn create_candidate_list(points: &[Point], attractiveness: &Array2<f64>) -> Vec<Vec<u32>> {
    let num_total_points = points.len() + 1; // +1 for Korvatunturi
    let mut all_points = vec![KORVATUNTURI.clone()];
    all_points.extend_from_slice(points);

    // Ids only, not Point clones -- rows are intentionally NOT truncated (every other point stays
    // present) so Ant::choose_next's filter+take chain is guaranteed to always find something,
    // keeping its greedy-fallback branch permanently unreachable.
    (0..num_total_points).into_par_iter().map(|i| {
        let last_point = &all_points[i];
        let mut candidates: Vec<(u32, f64)> = all_points.iter()
            .filter(|p| p.id != last_point.id)
            .map(|p| (p.id, attractiveness[[last_point.index(), p.index()]]))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().map(|(id, _)| id).collect()
    }).collect()
}

// A separate, pure-distance nearest-neighbor list (ids only, truncated to `k` at creation time),
// used for Or-opt relocation. Unlike `create_candidate_list` (sorted by ACO "attractiveness", which
// factors in weight/capacity and is not truncated until read time), this is a plain proximity ranking
// since relocation's only goal is minimizing insertion distance.
fn create_distance_candidate_list(points: &[Point], all_distances: &Array2<f64>, k: usize) -> Vec<Vec<u32>> {
    let num_total_points = points.len() + 1; // +1 for Korvatunturi
    let mut all_points = vec![KORVATUNTURI.clone()];
    all_points.extend_from_slice(points);

    (0..num_total_points).into_par_iter().map(|i| {
        let last_point = &all_points[i];
        let mut candidates: Vec<(u32, f64)> = all_points.iter()
            .filter(|p| p.id != last_point.id)
            .map(|p| (p.id, all_distances[[last_point.index(), p.index()]]))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates.into_iter().map(|(id, _)| id).collect()
    }).collect()
}

// Double-bridge (4-opt) perturbation: splits a route into segments A-B-C-D at 3 random interior
// cut points and reconnects them as A-C-B-D. Used as an ILS-style "kick" to escape local optima
// that 2-opt cannot reach -- no single 2-opt move can undo this specific reconnection, unlike a
// plain segment reversal. Pure reordering of the same points, so route weight/capacity is unaffected.
fn double_bridge(route: &mut Vec<u32>, rng: &mut impl Rng) {
    let n = route.len();
    if n < 4 {
        return;
    }
    let mut cuts: Vec<usize> = (1..n).collect();
    cuts.shuffle(rng);
    let mut chosen: Vec<usize> = cuts.into_iter().take(3).collect();
    chosen.sort_unstable();
    let (p1, p2, p3) = (chosen[0], chosen[1], chosen[2]);

    let mut new_route = Vec::with_capacity(n);
    new_route.extend_from_slice(&route[0..p1]);
    new_route.extend_from_slice(&route[p2..p3]);
    new_route.extend_from_slice(&route[p1..p2]);
    new_route.extend_from_slice(&route[p3..n]);
    *route = new_route;
}

// Or-opt: relocate a single point to a different sleigh run when doing so strictly reduces total
// distance and fits the receiving run's remaining capacity. `apply_2_opt` only ever reorders points
// within one run; this is the complementary inter-route move. Scoped to relocation between DIFFERENT
// routes only (same-route relocation is skipped) to avoid a class of same-route index-shift edge
// cases, since 2-opt already handles intra-route reordering.
pub fn apply_or_opt(
    all: &mut Vec<Vec<u32>>,
    all_distances: &Array2<f64>,
    weights: &Array1<f64>,
    capacity: f64,
    distance_candidate_list: &[Vec<u32>],
) -> bool {
    const EPSILON: f64 = 1e-9;
    const MAX_PASSES: usize = 1000;

    let edge_dist = |a: u32, b: u32| -> f64 {
        all_distances[[id_to_index(a), id_to_index(b)]]
    };

    // location[id_to_index(point_id)] = (route_idx, pos_idx), kept current across every move.
    let mut location: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); all_distances.nrows()];
    let mut route_weight: Vec<f64> = vec![0.0; all.len()];
    let mut point_order: Vec<u32> = Vec::new();

    for (r_idx, route) in all.iter().enumerate() {
        let mut w = 0.0;
        for (p_idx, &id) in route.iter().enumerate() {
            location[id_to_index(id)] = (r_idx, p_idx);
            w += weights[id_to_index(id)];
            point_order.push(id);
        }
        route_weight[r_idx] = w;
    }
    point_order.sort_unstable();

    let mut any_move_ever = false;

    for _pass in 0..MAX_PASSES {
        let mut moved_this_pass = false;

        for &point_id in &point_order {
            let (r_old, pos_old) = location[id_to_index(point_id)];
            let (prev_id, next_id) = {
                let route_old = &all[r_old];
                let prev_id = if pos_old == 0 { KORVATUNTURI.id } else { route_old[pos_old - 1] };
                let next_id = if pos_old == route_old.len() - 1 { KORVATUNTURI.id } else { route_old[pos_old + 1] };
                (prev_id, next_id)
            };
            let removal_gain = edge_dist(prev_id, point_id) + edge_dist(point_id, next_id) - edge_dist(prev_id, next_id);
            let point_weight = weights[id_to_index(point_id)];

            let mut best: Option<(u32, bool, f64)> = None; // (candidate_id, insert_after, net_delta)

            for &cand_id in &distance_candidate_list[id_to_index(point_id)] {
                if cand_id == KORVATUNTURI.id {
                    continue;
                }
                let (r_cand, pos_cand) = location[id_to_index(cand_id)];
                if r_cand == r_old {
                    continue; // inter-route only, by design
                }
                if route_weight[r_cand] + point_weight > capacity {
                    continue;
                }
                let route_cand = &all[r_cand];
                let cand_prev_id = if pos_cand == 0 { KORVATUNTURI.id } else { route_cand[pos_cand - 1] };
                let cand_next_id = if pos_cand == route_cand.len() - 1 { KORVATUNTURI.id } else { route_cand[pos_cand + 1] };

                let cost_before = edge_dist(cand_prev_id, point_id) + edge_dist(point_id, cand_id) - edge_dist(cand_prev_id, cand_id);
                let cost_after = edge_dist(cand_id, point_id) + edge_dist(point_id, cand_next_id) - edge_dist(cand_id, cand_next_id);
                let (insertion_cost, insert_after) = if cost_before <= cost_after { (cost_before, false) } else { (cost_after, true) };

                let net_delta = insertion_cost - removal_gain;
                if best.map_or(true, |(_, _, d)| net_delta < d) {
                    best = Some((cand_id, insert_after, net_delta));
                }
            }

            if let Some((cand_id, insert_after, net_delta)) = best {
                if net_delta < -EPSILON {
                    let (r_new, cand_pos) = location[id_to_index(cand_id)];
                    let insert_pos = if insert_after { cand_pos + 1 } else { cand_pos };

                    all[r_old].remove(pos_old);
                    route_weight[r_old] -= point_weight;

                    all[r_new].insert(insert_pos, point_id);
                    route_weight[r_new] += point_weight;

                    // Positions shifted in both touched routes; refresh their location entries.
                    for (p_idx, &id) in all[r_old].iter().enumerate() {
                        location[id_to_index(id)] = (r_old, p_idx);
                    }
                    for (p_idx, &id) in all[r_new].iter().enumerate() {
                        location[id_to_index(id)] = (r_new, p_idx);
                    }

                    moved_this_pass = true;
                }
            }
        }

        if moved_this_pass {
            any_move_ever = true;
        } else {
            break;
        }
    }

    any_move_ever
}

fn run_local_search(run: &mut GiftRun, all_distances: &Array2<f64>, weights: &Array1<f64>, capacity: f64, distance_candidate_list: &[Vec<u32>]) {
    for _round in 0..MAX_LOCAL_SEARCH_ROUNDS {
        let mut changed = false;
        for tour in &mut run.all {
            changed |= apply_2_opt(tour, all_distances);
        }
        changed |= apply_or_opt(&mut run.all, all_distances, weights, capacity, distance_candidate_list);
        if !changed {
            break;
        }
    }
    run.all.retain(|route| !route.is_empty());
    run.recalculate_total_distance(all_distances);
}

pub struct Hyperparameters {
    pub num_ants: usize,
    pub candidate_list_size: usize,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub q0: f64,
    pub tau_min_divisor: f64,
    pub num_elites: usize,
    pub elitist_weight: usize,
    pub log_interval_secs: u64,
    pub pheromone_deposit_weight: f64,
    pub or_opt_candidate_list_size: usize,
    pub plateau_patience: Option<usize>,
    pub kick_routes_per_event: usize,
    pub kick_deposit_weight: f64,
    pub deposit_quality_exponent: f64,
    pub max_iterations: usize,
}

pub fn optimize(points: &[Point], capacity: f64, all_distances: &Array2<f64>, load_previous: bool, save_results: bool, params: &Hyperparameters) -> GiftRun {
    info!("--- Starting Optimization ---");
    info!("Problem parameters:");
    info!("- Number of points: {}", points.len());
    info!("- Capacity: {}", capacity);
    info!("- Load previous solution: {}", load_previous);
    info!("- Save results: {}", save_results);
    info!("Hyperparameters:");
    info!("- num_ants: {}", params.num_ants);
    info!("- candidate_list_size: {}", params.candidate_list_size);
    info!("- alpha: {}", params.alpha);
    info!("- beta: {}", params.beta);
    info!("- rho: {}", params.rho);
    info!("- q0: {}", params.q0);
    info!("- tau_min_divisor: {}", params.tau_min_divisor);
    info!("- num_elites: {}", params.num_elites);
    info!("- elitist_weight: {}", params.elitist_weight);
    info!("- pheromone_deposit_weight: {}", params.pheromone_deposit_weight);
    info!("- log_interval_secs: {}", params.log_interval_secs);
    info!("- or_opt_candidate_list_size: {}", params.or_opt_candidate_list_size);
    info!("- plateau_patience (iterations of no improvement before a double-bridge kick): {:?}", params.plateau_patience);
    info!("- kick_routes_per_event: {}", params.kick_routes_per_event);
    info!("- kick_deposit_weight: {}", params.kick_deposit_weight);
    info!("- deposit_quality_exponent: {}", params.deposit_quality_exponent);
    info!("- max_iterations: {}", params.max_iterations);
    info!("---------------------------");

    // Flush the log buffer to ensure the header is written to the file immediately.
    log::logger().flush();

    let mut best_so_far_run = if load_previous {
        load_best().unwrap_or_else(|| GiftRun::new_with_max_dist())
    } else {
        GiftRun::new_with_max_dist()
    };

    let mut weights_vec = vec![0.0]; // Korvatunturi has 0 weight
    weights_vec.extend(points.iter().map(|p| p.weight));
    let weights: Array1<f64> = Array1::from(weights_vec);
    let attractiveness = Arc::new(
        (&weights / capacity) / (&(all_distances + 1e-12) / R)
    );

    info!("Pre-calculating candidate lists...");
    let candidate_list = Arc::new(create_candidate_list(points, &attractiveness));

    info!("Pre-calculating distance-based candidate lists for local search...");
    let distance_candidate_list = create_distance_candidate_list(points, all_distances, params.or_opt_candidate_list_size);

    let tau_max = 1.0 / params.rho;
    let tau_min = tau_max / params.tau_min_divisor;

    let (initial_pheromone, pheromone_loaded) = if load_previous {
        match load_pheromone_state(PHEROMONE_STATE_PATH, all_distances.shape()) {
            Some(p) => (p, true),
            None => (Array2::from_elem(all_distances.raw_dim(), 1.0), false),
        }
    } else {
        (Array2::from_elem(all_distances.raw_dim(), 1.0), false)
    };
    info!("Pheromone state: {}", if pheromone_loaded { "resumed from disk" } else { "fresh (uniform) start" });
    let pheromone = Arc::new(Mutex::new(initial_pheromone));

    info!("Using {} ants per iteration on {} CPU cores.", params.num_ants, num_cpus::get());

    let mut last_log_time = Instant::now();
    let mut last_log_iter = 0;
    let mut improvements_since_last_log = 0;
    let mut iters_since_improvement: usize = 0;
    let mut kick_count: usize = 0;
    let total_iterations = params.max_iterations;

    let points_arc = Arc::new(points.to_vec());
    let distances_arc = Arc::new(all_distances.clone());

    for i in 0..total_iterations {
        let pheromone_snapshot = Arc::new(pheromone.lock().unwrap().clone());

        let results: Vec<(GiftRun, u32, u32)> = (0..params.num_ants)
            .into_par_iter()
            .map(|_| {
                let mut ant = Ant::new(
                    points_arc.clone(),
                    capacity,
                    distances_arc.clone(),
                    pheromone_snapshot.clone(),
                    candidate_list.clone(),
                    params.candidate_list_size,
                    params.alpha,
                    params.beta,
                    params.q0,
                );
                let gift_run = ant.find_solution(attractiveness.clone());
                (gift_run, ant.greedy_fallback_count, ant.total_choices_count)
            })
            .collect();

        let mut iter_gift_runs: Vec<GiftRun> = results.iter().map(|r| r.0.clone()).collect();
        let total_greedy_fallbacks: u32 = results.iter().map(|r| r.1).sum();
        let total_choices: u32 = results.iter().map(|r| r.2).sum();

        iter_gift_runs.par_iter_mut().for_each(|run| {
            run_local_search(run, &distances_arc, &weights, capacity, &distance_candidate_list);
        });

        iter_gift_runs.sort_by(|a, b| a.total_distance.partial_cmp(&b.total_distance).unwrap());

        if iter_gift_runs[0].total_distance < best_so_far_run.total_distance {
            info!("New best found: {}", iter_gift_runs[0].total_distance);
            best_so_far_run = iter_gift_runs[0].clone();
            improvements_since_last_log += 1;
            iters_since_improvement = 0;

            if save_results {
                if let Err(e) = save_best(&best_so_far_run) {
                    log::error!("Failed to save best solution: {}", e);
                }
            }
        } else {
            iters_since_improvement += 1;
        }

        let mut pending_kick: Option<GiftRun> = None;

        if let Some(patience) = params.plateau_patience {
            if iters_since_improvement >= patience {
                let mut kicked_run = best_so_far_run.clone();
                let eligible: Vec<usize> = kicked_run.all.iter().enumerate()
                    .filter(|(_, r)| r.len() >= 4)
                    .map(|(idx, _)| idx)
                    .collect();

                if !eligible.is_empty() {
                    let mut rng = thread_rng();
                    let n_kick = params.kick_routes_per_event.min(eligible.len());
                    let chosen: Vec<usize> = eligible.choose_multiple(&mut rng, n_kick).cloned().collect();
                    for &idx in &chosen {
                        double_bridge(&mut kicked_run.all[idx], &mut rng);
                    }
                    run_local_search(&mut kicked_run, &distances_arc, &weights, capacity, &distance_candidate_list);
                    kick_count += 1;

                    if kicked_run.total_distance < best_so_far_run.total_distance {
                        info!(
                            "Kick #{}: improved best {:.2} -> {:.2} (perturbed {} routes)",
                            kick_count, best_so_far_run.total_distance, kicked_run.total_distance, chosen.len()
                        );
                        best_so_far_run = kicked_run.clone();
                        if save_results {
                            if let Err(e) = save_best(&best_so_far_run) {
                                log::error!("Failed to save best solution: {}", e);
                            }
                        }
                    } else {
                        info!(
                            "Kick #{}: perturbed {} routes, result {:.2} (best remains {:.2})",
                            kick_count, chosen.len(), kicked_run.total_distance, best_so_far_run.total_distance
                        );
                    }
                    pending_kick = Some(kicked_run);
                }
                iters_since_improvement = 0;
            }
        }

        let mut pheromone_guard = pheromone.lock().unwrap();
        update_pheromone(&mut pheromone_guard, &best_so_far_run, &iter_gift_runs, params.rho, params.num_elites, params.elitist_weight, tau_min, tau_max, params.pheromone_deposit_weight, params.deposit_quality_exponent);
        if let Some(kicked_run) = pending_kick {
            let amount = params.kick_deposit_weight * pheromone_deposit_amount(kicked_run.total_distance, best_so_far_run.total_distance, params.pheromone_deposit_weight, params.deposit_quality_exponent);
            deposit_pheromone_for_run(&mut pheromone_guard, &kicked_run, amount, true);
        }

        if last_log_time.elapsed().as_secs() >= params.log_interval_secs {
            let progress = (i + 1) as f64 / total_iterations as f64 * 100.0;
            let elapsed_secs = last_log_time.elapsed().as_secs_f64();
            let iters_since_last_log = i + 1 - last_log_iter;
            let secs_per_iter = elapsed_secs / iters_since_last_log as f64;

            let distances: Vec<f64> = iter_gift_runs.iter().map(|r| r.total_distance).collect();
            let iter_mean = distances.iter().sum::<f64>() / distances.len() as f64;
            let iter_std_dev = (distances.iter().map(|d| (d - iter_mean).powi(2)).sum::<f64>() / distances.len() as f64).sqrt();

            let p_min = *pheromone_guard.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let p_max = *pheromone_guard.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            let (q95, q99, q995, q999, q9995, q9999) = {
                let mut p_values: Vec<f64> = pheromone_guard.iter().cloned().collect();
                if p_values.is_empty() {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                } else {
                    p_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = p_values.len();
                    let q95 = p_values[((n - 1) as f64 * 0.95) as usize];
                    let q99 = p_values[((n - 1) as f64 * 0.99) as usize];
                    let q995 = p_values[((n - 1) as f64 * 0.995) as usize];
                    let q999 = p_values[((n - 1) as f64 * 0.999) as usize];
                    let q9995 = p_values[((n - 1) as f64 * 0.9995) as usize];
                    let q9999 = p_values[((n - 1) as f64 * 0.9999) as usize];
                    (q95, q99, q995, q999, q9995, q9999)
                }
            };

            let greedy_fallback_pct = if total_choices == 0 { 0.0 } else { (total_greedy_fallbacks as f64 / total_choices as f64) * 100.0 };

            info!(
                "[{:.2}%] Iter: {}/{}, {:.2}s/it, Best: {:.2}, Iter Best: {:.2}, Iter Avg: {:.2}, StdDev: {:.2e}, Impr: {}, NoImprove: {}, Greedy Fallbacks: {} ({:.2}%), Pheromone (Min/Q95/Q99/Q99.5/Q99.9/Q99.95/Q99.99/Max): {:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}",
                progress,
                i + 1,
                total_iterations,
                secs_per_iter,
                best_so_far_run.total_distance,
                iter_gift_runs[0].total_distance,
                iter_mean,
                iter_std_dev,
                improvements_since_last_log,
                iters_since_improvement,
                total_greedy_fallbacks,
                greedy_fallback_pct,
                p_min,
                q95,
                q99,
                q995,
                q999,
                q9995,
                q9999,
                p_max
            );
            log_pheromone_heatmap(&pheromone_guard, 20);

            // Clone-then-drop before saving so the ~800MB write doesn't hold the lock and delay
            // the next iteration's ant work -- mirrors the pheromone_snapshot pattern used above.
            let snapshot_to_save = pheromone_guard.clone();
            drop(pheromone_guard);
            if save_results {
                if let Err(e) = save_pheromone_state(&snapshot_to_save, PHEROMONE_STATE_PATH) {
                    log::error!("Failed to save pheromone state: {}", e);
                }
            }

            last_log_time = Instant::now();
            last_log_iter = i + 1;
            improvements_since_last_log = 0;
        }
    }

    if save_results {
        let final_pheromone = pheromone.lock().unwrap();
        if let Err(e) = save_pheromone_state(&final_pheromone, PHEROMONE_STATE_PATH) {
            log::error!("Failed to save final pheromone state: {}", e);
        }
    }

    info!("Total plateau kicks applied: {}", kick_count);

    best_so_far_run
}

fn log_pheromone_heatmap(pheromone: &Array2<f64>, top_n: usize) {
    let mut trails: Vec<(usize, usize, f64)> = Vec::new();
    for ((i, j), &p) in pheromone.indexed_iter() {
        if i < j { // Avoid duplicates and self-loops
            trails.push((i, j, p));
        }
    }

    // Sort by pheromone level in descending order
    trails.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    info!("--- Top {} Pheromone Trails ---", top_n);
    for (i, j, p) in trails.iter().take(top_n) {
        info!("Trail ({: >4} - {: >4}): {:.2e}", index_to_id(*i), index_to_id(*j), p);
    }
    info!("---------------------------------");
}

fn save_pheromone_state(pheromone: &Array2<f64>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    bincode::serialize_into(file, pheromone)?;
    Ok(())
}

fn load_pheromone_state(path: &str, expected_shape: &[usize]) -> Option<Array2<f64>> {
    let file = BufReader::new(File::open(path).ok()?);
    let pheromone: Array2<f64> = bincode::deserialize_from(file).ok()?;
    if pheromone.shape() == expected_shape {
        Some(pheromone)
    } else {
        info!("Saved pheromone state shape {:?} doesn't match current problem shape {:?}; starting fresh.", pheromone.shape(), expected_shape);
        None
    }
}

fn save_best(gift_run: &GiftRun) -> Result<(), std::io::Error> {
    let path = format!("results/{}.txt", gift_run.total_distance);
    fs::create_dir_all("results")?;
    let mut file = File::create(path)?;
    file.write_all(gift_run.to_string().as_bytes())?;
    info!("Saving new best solution...");
    Ok(())
}

fn load_best() -> Option<GiftRun> {
    let path = Path::new("results");
    if !path.exists() {
        info!("no results/ directory found.");
        return None;
    }

    let mut best_distance = f64::MAX;
    let mut best_file = None;

    for entry in fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if let Some(stem) = path.file_stem() {
            if let Some(stem_str) = stem.to_str() {
                if let Ok(distance) = stem_str.parse::<f64>() {
                    if distance < best_distance {
                        best_distance = distance;
                        best_file = Some(path.to_path_buf());
                    }
                }
            }
        }
    }

    if let Some(best_file) = best_file {
        info!("load solution from: {}", best_file.display());
        let lines = fs::read_to_string(best_file).ok()?;
        Some(GiftRun::parse(&lines, best_distance))
    } else {
        info!("no solutions found from results/");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn float_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_double_bridge_reorders_without_losing_points() {
        let original: Vec<u32> = vec![2, 3, 4, 5, 6, 7, 8, 9];
        let mut route = original.clone();
        let mut rng = rand::thread_rng();

        double_bridge(&mut route, &mut rng);

        let mut sorted_route = route.clone();
        sorted_route.sort_unstable();
        let mut sorted_original = original.clone();
        sorted_original.sort_unstable();
        assert_eq!(sorted_route, sorted_original, "double_bridge must not lose or duplicate points");
        assert_ne!(route, original, "double_bridge must change the route's order");
    }

    #[test]
    fn test_double_bridge_leaves_short_routes_unchanged() {
        let mut route: Vec<u32> = vec![2, 3, 4];
        let original = route.clone();
        let mut rng = rand::thread_rng();

        double_bridge(&mut route, &mut rng);

        assert_eq!(route, original, "routes shorter than 4 points have no valid cut points");
    }

    #[test]
    fn test_all_distances() {
        let helsinki = Point::new(2, 60.1100895, 24.8782586, 10.0);
        let newyork = Point::new(3, 40.697403, -74.1201045, 10.0);
        let all = calculate_all_distances(&[helsinki, newyork]);

        assert_eq!(all.shape(), &[3, 3]);
        assert!(float_eq(all[[0, 0]], 0.0, 0.001));
        assert!(float_eq(all[[1, 1]], 0.0, 0.001));
        assert!(float_eq(all[[2, 2]], 0.0, 0.001));

        assert!(float_eq(all[[0, 1]], 892.0, 20.0)); // Korvatunturi -> Helsinki
        assert!(float_eq(all[[1, 0]], 892.0, 20.0)); // Helsinki -> Korvatunturi

        assert!(float_eq(all[[0, 2]], 6372.0, 20.0)); // Korvatunturi -> New York
        assert!(float_eq(all[[2, 0]], 6372.0, 20.0)); // New York -> Korvatunturi

        assert!(float_eq(all[[1, 2]], 6625.0, 20.0)); // Helsinki -> New York
        assert!(float_eq(all[[2, 1]], 6625.0, 20.0)); // New York -> Helsinki
    }

    #[test]
    fn test_attractiveness_calculation() {
        let points = vec![
            Point::new(1, 60.0, 25.0, 10.0),
            Point::new(2, 61.0, 25.0, 20.0),
        ];
        let all_distances = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 111.32],
            [0.0, 111.32, 0.0],
        ]);
        let capacity = 1000.0;

        let mut weights_vec = vec![0.0]; // Korvatunturi has 0 weight
        weights_vec.extend(points.iter().map(|p| p.weight));
        let weights: Array1<f64> = Array1::from(weights_vec);
        let attractiveness = (&weights / capacity) / (&(&all_distances + 1e-12) / R);

        // Attractiveness from point 1 (index 1) to point 2 (index 2)
        let expected_attractiveness_1_2 = (20.0 / 1000.0) / (111.32 / R);
        assert!(float_eq(attractiveness[[1, 2]], expected_attractiveness_1_2, 1e-9));

        // Attractiveness from point 2 (index 2) to point 1 (index 1)
        let expected_attractiveness_2_1 = (10.0 / 1000.0) / (111.32 / R);
        assert!(float_eq(attractiveness[[2, 1]], expected_attractiveness_2_1, 1e-9));
    }

    #[test]
    fn test_pheromone_state_roundtrip() {
        let path = "test_pheromone_roundtrip.bin";
        let original = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        save_pheromone_state(&original, path).unwrap();
        let loaded = load_pheromone_state(path, &[3, 3]).unwrap();
        assert_eq!(loaded, original);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_pheromone_state_shape_mismatch_returns_none() {
        let path = "test_pheromone_shape_mismatch.bin";
        let saved = Array2::from_elem((3, 3), 1.0);
        save_pheromone_state(&saved, path).unwrap();
        assert!(load_pheromone_state(path, &[4, 4]).is_none());
        let _ = fs::remove_file(path);
    }
}
