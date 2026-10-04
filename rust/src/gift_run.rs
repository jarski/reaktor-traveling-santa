use std::fmt;
use ndarray::Array2;
use crate::point::KORVATUNTURI;

#[derive(Debug, Clone)]
pub struct GiftRun {
    pub current: Vec<u32>,
    pub all: Vec<Vec<u32>>,
    pub total_distance: f64,
}

impl GiftRun {
    pub fn new() -> Self {
        Self {
            current: Vec::new(),
            all: Vec::new(),
            total_distance: 0.0,
        }
    }

    pub fn new_with_max_dist() -> Self {
        Self {
            current: Vec::new(),
            all: Vec::new(),
            total_distance: f64::MAX,
        }
    }

    pub fn append(&mut self, point_id: u32, distance: f64) {
        self.current.push(point_id);
        self.total_distance += distance;
    }

    pub fn run_finished(&mut self, distance: f64) {
        if !self.current.is_empty() {
            self.all.push(self.current.clone());
            self.current.clear();
        }
        self.total_distance += distance;
    }

    pub fn recalculate_total_distance(&mut self, all_distances: &Array2<f64>) {
        let mut new_total_distance = 0.0;
        for run in &self.all {
            let mut last_index = KORVATUNTURI.index();
            for &point_id in run {
                let current_index = (point_id - 1) as usize;
                new_total_distance += all_distances[[last_index, current_index]];
                last_index = current_index;
            }
            new_total_distance += all_distances[[last_index, KORVATUNTURI.index()]];
        }
        self.total_distance = new_total_distance;
    }

    pub fn parse(lines: &str, total_distance: f64) -> Self {
        let all = lines
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split(';')
                    .map(|id_str| id_str.trim().parse::<u32>().unwrap())
                    .collect()
            })
            .collect();

        Self {
            current: Vec::new(),
            all,
            total_distance,
        }
    }
}

impl fmt::Display for GiftRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for run in &self.all {
            let line = run
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join("; ");
            writeln!(f, "{}", line)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gift_run() {
        let input = "2; 3; 4\n5; 6\n\n7\n";
        let gift_run = GiftRun::parse(input, 12345.0);

        assert_eq!(gift_run.total_distance, 12345.0);
        assert_eq!(gift_run.all.len(), 3);
        assert_eq!(gift_run.all[0], vec![2, 3, 4]);
        assert_eq!(gift_run.all[1], vec![5, 6]);
        assert_eq!(gift_run.all[2], vec![7]);
    }

    #[test]
    fn test_gift_run_to_string() {
        let mut gift_run = GiftRun::new();
        gift_run.all.push(vec![2, 3, 4]);
        gift_run.all.push(vec![5, 6]);
        gift_run.all.push(vec![7]);

        let expected_output = "2; 3; 4\n5; 6\n7\n";
        assert_eq!(gift_run.to_string(), expected_output);
    }
}
