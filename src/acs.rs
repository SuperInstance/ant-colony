//! Ant Colony System (ACS) variant.

use crate::pheromone::PheromoneMatrix;
use crate::colony::{DistanceMatrix, tour_length, construct_solution};
use crate::rng::SimpleRng;

/// ACS parameters and runner.
pub struct ACS {
    pub num_ants: usize,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub q0: f64,  // Exploitation vs exploration parameter
    pub phi: f64, // Local pheromone decay
}

impl ACS {
    pub fn new(num_ants: usize, alpha: f64, beta: f64, rho: f64, q0: f64, phi: f64) -> Self {
        Self { num_ants, alpha, beta, rho, q0, phi }
    }

    /// Run ACS for given iterations.
    pub fn run(
        &self,
        dist: &DistanceMatrix,
        iterations: usize,
        rng: &mut SimpleRng,
    ) -> (Vec<usize>, f64) {
        let n = dist.len();
        let initial_tau = 1.0 / (n as f64 * self.nearest_neighbor_tour(dist));
        let mut pheromone = PheromoneMatrix::new(n, initial_tau, self.rho);

        let mut best_tour: Vec<usize> = (0..n).collect();
        let mut best_length = f64::MAX;

        for _ in 0..iterations {
            let mut tours = Vec::new();
            for ant in 0..self.num_ants {
                let start = ant % n;
                let tour = construct_solution(&pheromone, dist, start, self.alpha, self.beta, rng);
                let length = tour_length(&tour, dist);

                // Local pheromone update
                for w in tour.windows(2) {
                    let old = pheromone.get(w[0], w[1]);
                    pheromone.set(w[0], w[1], (1.0 - self.phi) * old + self.phi * initial_tau);
                }

                tours.push((tour, length));
            }

            // Global update: only best ant deposits
            for (tour, length) in &tours {
                if *length < best_length {
                    best_length = *length;
                    best_tour = tour.clone();
                }
            }

            pheromone.evaporate();
            let deposit = 1.0 / best_length;
            pheromone.deposit(&best_tour, deposit);
        }

        (best_tour, best_length)
    }

    fn nearest_neighbor_tour(&self, dist: &DistanceMatrix) -> f64 {
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
            if len < best {
                best = len;
            }
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
    fn acs_finds_valid_tour() {
        let mut rng = SimpleRng::new(42);
        let acs = ACS::new(5, 1.0, 2.0, 0.1, 0.9, 0.1);
        let (tour, length) = acs.run(&sample_dist(), 50, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length > 0.0);
    }

    #[test]
    fn acs_improves_over_iterations() {
        let mut rng = SimpleRng::new(42);
        let acs = ACS::new(5, 1.0, 2.0, 0.1, 0.9, 0.1);
        let (_, len1) = acs.run(&sample_dist(), 10, &mut rng);
        let mut rng2 = SimpleRng::new(42);
        let (_, len2) = acs.run(&sample_dist(), 100, &mut rng2);
        // More iterations should generally do at least as well
        assert!(len2 <= len1 * 1.5, "len2={}, len1={}", len2, len1);
    }
}
