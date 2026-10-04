use traveling_santa::point::Point;
use traveling_santa::{calculate_all_distances, update_pheromone, apply_2_opt, apply_or_opt, Hyperparameters};
use traveling_santa::ant::Ant;
use traveling_santa::gift_run::GiftRun;
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;

#[test]
fn test_full_optimization_run_small() {
    // This is a full integration test. It's slow, so it's in the tests/ directory.
    // It will only be run with `cargo test`.

    // 1. Setup a small, manageable dataset
    let points = vec![
        Point::new(2, 60.1, 24.8, 10.0), // Helsinki
        Point::new(3, 52.5, 13.4, 20.0), // Berlin
        Point::new(4, 48.8, 2.3, 30.0),  // Paris
    ];
    let capacity = 1000.0;
    let all_distances = calculate_all_distances(&points);
    let params = Hyperparameters {
        num_ants: 10,
        candidate_list_size: 2,
        alpha: 1.0,
        beta: 2.0,
        rho: 0.5,
        q0: 0.85,
        tau_min_divisor: 5.0,
        num_elites: 10,
        elitist_weight: 1,
        log_interval_secs: 60,
        pheromone_deposit_weight: 1.0,
        or_opt_candidate_list_size: 2,
        plateau_patience: None,
        kick_routes_per_event: 5,
        kick_deposit_weight: 1.0,
        deposit_quality_exponent: 50.0,
        max_iterations: 10000,
    };

    // 2. Run the optimization function in test mode (no loading/saving)
    let gift_run = traveling_santa::optimize(&points, capacity, &all_distances, false, false, &params);

    // 3. Verify the result
    assert!(gift_run.total_distance > 0.0);
    assert!(gift_run.total_distance < f64::MAX);

    let visited_points: std::collections::HashSet<u32> = gift_run.all.into_iter().flatten().collect();
    let expected_points: std::collections::HashSet<u32> = [2, 3, 4].iter().cloned().collect();

    assert_eq!(visited_points, expected_points);
}

#[test]
fn test_2_opt_improves_crossed_path() {
    // Initialize the logger to see debug output for this test
    let _ = env_logger::builder().is_test(true).try_init();

    // This test verifies that the 2-opt algorithm correctly un-crosses a simple path.
    // Points form a square: A(0,0), B(0,1), C(1,1), D(1,0)
    let points = vec![
        Point::new(2, 0.0, 0.0, 10.0), // A
        Point::new(3, 0.0, 1.0, 10.0), // B
        Point::new(4, 1.0, 1.0, 10.0), // C
        Point::new(5, 1.0, 0.0, 10.0), // D
    ];
    let all_distances = calculate_all_distances(&points);

    // A crossed tour: A -> C -> B -> D
    let mut crossed_tour = vec![2, 4, 3, 5];

    // The expected optimal tour is A -> B -> C -> D
    let expected_optimal_tour = vec![2, 3, 4, 5];

    apply_2_opt(&mut crossed_tour, &all_distances);

    assert_eq!(crossed_tour, expected_optimal_tour);
}

// Shared geometry for the Or-opt tests below: X sits almost exactly on top of Y (a point in a
// different route), while Korvatunturi is far away from the X/Y/Z cluster. So a lone round trip to
// deliver just X is hugely expensive, but inserting X next to Y in Y's route is nearly free.
fn or_opt_test_geometry() -> (Vec<Point>, Array2<f64>) {
    let x = Point::new(2, 60.0, 25.0001, 0.0); // weight filled in per-test
    let y = Point::new(3, 60.0, 25.0, 0.0);
    let z = Point::new(4, 60.0, 25.2, 0.0);
    let points = vec![x, y, z];
    let all_distances = calculate_all_distances(&points);
    (points, all_distances)
}

#[test]
fn test_or_opt_relocates_point_to_cheaper_route() {
    let (_points, all_distances) = or_opt_test_geometry();

    let weights = Array1::from(vec![0.0, 5.0, 5.0, 5.0]); // Korvatunturi, X, Y, Z
    let capacity = 1000.0; // ample spare capacity in route B

    let mut gift_run = GiftRun::new();
    gift_run.all = vec![
        vec![2u32],       // route A: X alone
        vec![3u32, 4u32], // route B: Y, Z
    ];
    gift_run.recalculate_total_distance(&all_distances);
    let distance_before = gift_run.total_distance;

    // Hand-built pure-distance candidate list (X's nearest neighbor is Y, and vice versa).
    let distance_candidate_list: Vec<Vec<u32>> = vec![
        vec![],     // Korvatunturi - never queried by apply_or_opt
        vec![3, 4], // X's neighbors: Y, then Z
        vec![2, 4], // Y's neighbors: X, then Z
        vec![3, 2], // Z's neighbors: Y, then X
    ];

    let moved = apply_or_opt(&mut gift_run.all, &all_distances, &weights, capacity, &distance_candidate_list);
    gift_run.recalculate_total_distance(&all_distances);

    assert!(moved, "Or-opt should relocate the isolated point into the cheaper route");
    assert!(gift_run.total_distance < distance_before, "total distance should strictly decrease");
    assert!(gift_run.all[0].is_empty(), "route A should be emptied once its only point is relocated");

    let remaining: std::collections::HashSet<u32> = gift_run.all.iter().flatten().cloned().collect();
    assert_eq!(remaining, [2u32, 3, 4].iter().cloned().collect(), "no points should be lost or duplicated");
}

#[test]
fn test_or_opt_respects_capacity() {
    let (_points, all_distances) = or_opt_test_geometry();

    // Weights chosen so that NO single-point relocation fits either direction:
    // X (600) can't join route B (450+450=900, +600=1500 > 1000), and neither Y nor Z (450 each)
    // can join route A (600+450=1050 > 1000) -- isolating the capacity guard from any other move.
    let weights = Array1::from(vec![0.0, 600.0, 450.0, 450.0]);
    let capacity = 1000.0;

    let mut gift_run = GiftRun::new();
    gift_run.all = vec![
        vec![2u32],
        vec![3u32, 4u32],
    ];
    gift_run.recalculate_total_distance(&all_distances);
    let distance_before = gift_run.total_distance;
    let all_before = gift_run.all.clone();

    let distance_candidate_list: Vec<Vec<u32>> = vec![
        vec![],
        vec![3, 4],
        vec![2, 4],
        vec![3, 2],
    ];

    let moved = apply_or_opt(&mut gift_run.all, &all_distances, &weights, capacity, &distance_candidate_list);
    gift_run.recalculate_total_distance(&all_distances);

    assert!(!moved, "Or-opt must not relocate a point when it would exceed the receiving route's capacity");
    assert_eq!(gift_run.all, all_before);
    assert_eq!(gift_run.total_distance, distance_before);
}

#[test]
fn test_pheromone_convergence_mechanism() {
    // This test creates a deterministic scenario to prove that better paths
    // receive more pheromone, which is the core of convergence.

    // 1. Create a simple, obvious problem: 4 cities in a line.
    // Optimal path is Korvatunturi -> A -> B -> C -> D -> Korvatunturi
    let points = vec![
        Point::new(2, 60.0, 10.0, 10.0), // A (index 1)
        Point::new(3, 60.0, 11.0, 10.0), // B (index 2)
        Point::new(4, 60.0, 12.0, 10.0), // C (index 3)
        Point::new(5, 60.0, 13.0, 10.0), // D (index 4)
    ];
    let all_distances = calculate_all_distances(&points);

    // 2. Use hyperparameters that force deterministic "greedy" behavior
    let alpha: f64 = 1.0;
    let beta: f64 = 5.0; // High beta to emphasize distance
    let rho: f64 = 0.1;
    let q0: f64 = 1.0; // q0 = 1.0 removes all random exploration
    let num_ants: usize = 10;
    let num_elites: usize = 1;
    let elitist_weight: usize = 1;
    let tau_max = 1.0 / rho;
    let tau_min = tau_max / 5.0;
    let pheromone_deposit_weight = 1.0;

    let attractiveness = 1.0 / (&all_distances + 1e-12);
    let pheromone = Arc::new(Mutex::new(Array2::ones(all_distances.raw_dim())));

    let points_arc = Arc::new(points);
    let distances_arc = Arc::new(all_distances);

    // Create a dummy candidate list for the test
    let candidate_list = Arc::new(
        (0..points_arc.len() + 1)
            .map(|_| points_arc.iter().map(|p| p.id).take(4).collect())
            .collect::<Vec<Vec<u32>>>()
    );

    // 3. Run a few iterations of the optimization loop
    for _i in 0..10 {
        let attractiveness_arc = Arc::new(attractiveness.clone());
        let pheromone_snapshot = pheromone.lock().unwrap().clone();

        let mut iter_gift_runs: Vec<_> = (0..num_ants)
            .into_iter() // Use a sequential iterator to make the test deterministic
            .map(|_| {
                let mut ant = Ant::new(
                    points_arc.clone(),
                    1000.0,
                    distances_arc.clone(),
                    Arc::new(pheromone_snapshot.clone()),
                    candidate_list.clone(),
                    candidate_list.len(),
                    alpha,
                    beta,
                    q0,
                );
                ant.find_solution(attractiveness_arc.clone())
            }).map(|run| (run, Array2::<f64>::zeros((1,1)))) // Wrap in tuple for update_pheromone
            .collect();

        // Sort by total distance, with a deterministic tie-breaker on the path itself.
        iter_gift_runs.sort_by(|a, b| {
            a.0.total_distance.partial_cmp(&b.0.total_distance).unwrap_or(Ordering::Equal)
                .then_with(|| a.0.all.cmp(&b.0.all))
        });

        let best_run = &iter_gift_runs[0].0;
        let mut pheromone_guard = pheromone.lock().unwrap();
        let just_gift_runs: Vec<GiftRun> = iter_gift_runs.iter().map(|(run, _)| run.clone()).collect();

        update_pheromone(&mut pheromone_guard, best_run, &just_gift_runs, rho, num_elites, elitist_weight, tau_min, tau_max, pheromone_deposit_weight, 50.0);
    }

    let pheromone_guard = pheromone.lock().unwrap();
    // 4. Assert that the optimal path has stronger pheromone trails
    // Optimal path is 0->1->2->3->4->0 (by index)
    let optimal_path_pheromone = pheromone_guard[[0, 1]] + pheromone_guard[[1, 2]] + pheromone_guard[[2, 3]] + pheromone_guard[[3, 4]] + pheromone_guard[[4, 0]];

    // A clearly suboptimal path (e.g., zig-zagging) 0 -> 1(A) -> 3(C) -> 2(B) -> 4(D) -> 0
    let suboptimal_path_pheromone = pheromone_guard[[0, 1]] + pheromone_guard[[1, 3]] + pheromone_guard[[3, 2]] + pheromone_guard[[2, 4]] + pheromone_guard[[4, 0]];

    println!("Optimal Path Pheromone Sum: {}", optimal_path_pheromone);
    println!("Suboptimal Path Pheromone Sum: {}", suboptimal_path_pheromone);

    assert!(
        optimal_path_pheromone > suboptimal_path_pheromone,
        "Pheromone on the optimal path ({}) should be higher than on a suboptimal path ({}).",
        optimal_path_pheromone,
        suboptimal_path_pheromone
    );
}
