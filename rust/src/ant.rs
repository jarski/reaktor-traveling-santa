use std::sync::Arc;
use ndarray::Array2;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use std::cmp::Ordering;

use super::point::{Point, KORVATUNTURI};
use super::gift_run::GiftRun;

pub struct Ant {
    points: Arc<Vec<Point>>,
    visited: Vec<bool>,
    remaining_count: usize,
    capacity: f64,
    capacity_left: f64,
    all_distances: Arc<Array2<f64>>,
    pheromone: Arc<Array2<f64>>, // Pheromone levels for decision making
    candidate_list: Arc<Vec<Vec<u32>>>, // Pre-computed lists of promising next point ids
    candidate_list_size: usize,
    alpha: f64,
    beta: f64,
    q0: f64,
    pub greedy_fallback_count: u32,
    pub total_choices_count: u32,
}

impl Ant {
    pub fn new(
        points: Arc<Vec<Point>>,
        capacity: f64,
        all_distances: Arc<Array2<f64>>,
        pheromone: Arc<Array2<f64>>,
        candidate_list: Arc<Vec<Vec<u32>>>,
        candidate_list_size: usize,
        alpha: f64,
        beta: f64,
        q0: f64,
    ) -> Self {
        let remaining_count = points.len();
        let visited = vec![false; all_distances.nrows()];
        Self {
            points,
            visited,
            remaining_count,
            capacity,
            capacity_left: capacity,
            all_distances,
            pheromone,
            candidate_list,
            candidate_list_size,
            alpha,
            beta,
            q0,
            greedy_fallback_count: 0,
            total_choices_count: 0,
        }
    }

    fn is_available(&self, p: &Point) -> bool {
        !self.visited[p.index()] && p.weight <= self.capacity_left
    }

    // Resolve a candidate-list id to its Point in O(1), no allocation. Relies on the same
    // id<->position invariant lib.rs's id_to_index/index_to_id already assume crate-wide:
    // nicelist.txt ids are contiguous starting at 2 (verified), so self.points[i].id == i + 2
    // for every i.
    fn resolve(&self, id: u32) -> &Point {
        if id == 0 {
            &KORVATUNTURI
        } else {
            &self.points[(id - 2) as usize]
        }
    }

    pub fn find_solution(&mut self, attractiveness: Arc<Array2<f64>>) -> GiftRun {
        let mut gift_run = GiftRun::new();
        let mut last_point = KORVATUNTURI.clone();

        while self.remaining_count > 0 {
            // At each step, check whether anything remaining still fits in the sleigh.
            let any_possible = self.points.iter().any(|p| self.is_available(p));

            if !any_possible {
                gift_run.run_finished(self.all_distances[[last_point.index(), KORVATUNTURI.index()]]);
                self.capacity_left = self.capacity;
                last_point = KORVATUNTURI.clone();
                continue;
            }

            let next_point = self.choose_next(&last_point, &attractiveness);

            if next_point.id != KORVATUNTURI.id {
                gift_run.append(next_point.id, self.all_distances[[last_point.index(), next_point.index()]]);

                self.visited[next_point.index()] = true;
                self.remaining_count -= 1;

                self.capacity_left -= next_point.weight;

                last_point = next_point;

            } else { // Chosen to return to Korvatunturi
                gift_run.run_finished(self.all_distances[[last_point.index(), KORVATUNTURI.index()]]);
                self.capacity_left = self.capacity;
                last_point = KORVATUNTURI.clone();
            }
        }
        gift_run.run_finished(self.all_distances[[last_point.index(), KORVATUNTURI.index()]]);
        gift_run
    }

    fn choose_next(&mut self, last_point: &Point, attractiveness: &Array2<f64>) -> Point {
        self.total_choices_count += 1;
        let mut rng = thread_rng();

        // Use the pre-computed candidate list, filtering for points that are still possible to visit.
        // Ids only -- cheap u32 copies, no Point clones/heap allocations here at all.
        let candidates: Vec<u32> = self.candidate_list[last_point.index()].iter()
            .copied()
            .filter(|&id| self.is_available(self.resolve(id)))
            .take(self.candidate_list_size)
            .collect();

        if candidates.is_empty() {
            self.greedy_fallback_count += 1;
            let mut possible_points: Vec<Point> = self.points.iter().filter(|p| self.is_available(p)).cloned().collect();
            if possible_points.is_empty() {
                return KORVATUNTURI.clone();
            }
            possible_points.sort_by(|a, b| {
                let attractive_a = attractiveness[[last_point.index(), a.index()]];
                let attractive_b = attractiveness[[last_point.index(), b.index()]];
                // Sort descending by attractiveness, then by ID for determinism
                attractive_b.partial_cmp(&attractive_a).unwrap_or(Ordering::Equal).then_with(|| a.id.cmp(&b.id))
            });
            return possible_points[0].clone();
        }

        // Scores keyed by id, not by owned Point -- only the finally-chosen id gets resolved+cloned.
        let mut scores: Vec<(u32, f64)> = candidates.into_iter().map(|id| {
            let p = self.resolve(id);
            let pheromone_val = self.pheromone[[last_point.index(), p.index()]].powf(self.alpha);
            let attractiveness_val = attractiveness[[last_point.index(), p.index()]].powf(self.beta);
            (id, (pheromone_val * attractiveness_val).max(1e-10))
        }).collect();

        let q: f64 = rng.gen();
        let chosen_id = if q <= self.q0 {
            scores.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0))
            });
            scores[0].0
        } else {
            let weights: Vec<f64> = scores.iter().map(|s| s.1).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            scores[dist.sample(&mut rng)].0
        };
        self.resolve(chosen_id).clone()
    }
}
