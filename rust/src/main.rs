use std::time::Instant;
use log::info;
use traveling_santa::{solve, solve_ils};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let start_time = Instant::now();

    // SOLVER=aco selects the original ant colony optimisation; anything else runs the SA/ILS
    // local search. Both are kept so they can be compared on equal wall-clock.
    let solver = std::env::var("SOLVER").unwrap_or_else(|_| "ils".to_string());

    info!("Starting Traveling Santa optimization (solver: {})...", solver);
    match solver.as_str() {
        "aco" => solve(),
        _ => solve_ils(),
    }
    info!("Optimization finished in {:?}", start_time.elapsed());
}
