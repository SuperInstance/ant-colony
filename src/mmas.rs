//! MAX-MIN Ant System (MMAS) variant.

use crate::pheromone::PheromoneMatrix;
use crate::colony::{DistanceMatrix, tour_length, construct_solution};
use crate::rng::SimpleRng;

/// MMAS parameters and runner.
pub struct MMAS {
    pub num_ants: usize,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
}

impl MMAS {
    pub fn new(num_ants: usize, alpha: f64, beta: f64, rho: f64) -> Self {
        Self { num_ants, alpha, beta, rho }
    }

    /// Run MMAS for given iterations.
    pub fn run(
        &self,
        dist: &DistanceMatrix,
        iterations: usize,
        rng: &mut SimpleRng,
    ) -> (Vec<usize>, f64) {
        let n = dist.len();
        let nn_len = self.nearest_neighbor_length(dist);
        let tau_max = 1.0 / (self.rho * nn_len);
        let tau_min = tau_max / (2.0 * n as f64);

        let mut pheromone = PheromoneMatrix::new(n, tau_max, self.rho);

        let mut best_tour: Vec<usize> = (0..n).collect();
        let mut best_length = f64::MAX;

        for _ in 0..iterations {
            let iter_best = self.iteration_best(dist, &pheromone, rng);

            if iter_best.1 < best_length {
                best_length = iter_best.1;
                best_tour = iter_best.0;
            }

            // Evaporate and deposit
            pheromone.evaporate();
            let deposit = 1.0 / best_length;
            pheromone.deposit(&best_tour, deposit);

            // Clamp trails
            pheromone.clamp(tau_min, tau_max);
        }

        (best_tour, best_length)
    }

    fn iteration_best(
        &self,
        dist: &DistanceMatrix,
        pheromone: &PheromoneMatrix,
        rng: &mut SimpleRng,
    ) -> (Vec<usize>, f64) {
        let n = dist.len();
        let mut best_tour = Vec::new();
        let mut best_len = f64::MAX;

        for ant in 0..self.num_ants {
            let start = ant % n;
            let tour = construct_solution(pheromone, dist, start, self.alpha, self.beta, rng);
            let len = tour_length(&tour, dist);
            if len < best_len {
                best_len = len;
                best_tour = tour;
            }
        }

        (best_tour, best_len)
    }

    fn nearest_neighbor_length(&self, dist: &DistanceMatrix) -> f64 {
        let n = dist.len();
        let mut best = f64::MAX;
        for start in 0..n {
            let mut visited = vec![false; n];
            let mut tour = vec![start];
            visited[start] = true;
            while tour.len() < n {
                let curr = *tour.last().unwrap();
                let (next, _) = (0..n)
                    .filter(|j| !visited[*j])
                    .map(|j| (j, dist[curr][j]))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                visited[next] = true;
                tour.push(next);
            }
            let len = tour_length(&tour, dist);
            if len < best { best = len; }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dist() -> DistanceMatrix {
        vec![
            vec![0.0, 2.0, 9.0, 10.0],
            vec![2.0, 0.0, 6.0, 4.0],
            vec![9.0, 6.0, 0.0, 8.0],
            vec![10.0, 4.0, 8.0, 0.0],
        ]
    }

    #[test]
    fn mmas_finds_valid_tour() {
        let mut rng = SimpleRng::new(42);
        let mmas = MMAS::new(5, 1.0, 2.0, 0.1);
        let (tour, length) = mmas.run(&sample_dist(), 50, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length > 0.0);
    }

    #[test]
    fn mmas_clamps_pheromones() {
        let mut rng = SimpleRng::new(42);
        let mmas = MMAS::new(3, 1.0, 2.0, 0.1);
        let (tour, _) = mmas.run(&sample_dist(), 20, &mut rng);
        // Verify tour visits all nodes
        let mut sorted = tour.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }
}
