//! TSP solver utilities and problem construction.

use crate::colony::{DistanceMatrix, tour_length};
use crate::pheromone::PheromoneMatrix;
use crate::rng::SimpleRng;

/// Compute Euclidean distance between two 2D points.
pub fn euclidean_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

/// Build a distance matrix from 2D coordinates.
pub fn build_distance_matrix(coords: &[(f64, f64)]) -> DistanceMatrix {
    let n = coords.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            dist[i][j] = euclidean_distance(coords[i], coords[j]);
        }
    }
    dist
}

/// A simple ACO TSP solver (basic variant).
pub struct TSPSolver {
    pub num_ants: usize,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub iterations: usize,
    pub elitist_ants: usize,
}

impl TSPSolver {
    pub fn new(num_ants: usize, alpha: f64, beta: f64, rho: f64, iterations: usize) -> Self {
        Self { num_ants, alpha, beta, rho, iterations, elitist_ants: 0 }
    }

    pub fn with_elitism(mut self, n: usize) -> Self {
        self.elitist_ants = n;
        self
    }

    /// Solve TSP given a distance matrix.
    pub fn solve(&self, dist: &DistanceMatrix, rng: &mut SimpleRng) -> (Vec<usize>, f64) {
        let n = dist.len();
        let initial = 1.0 / n as f64;
        let mut pheromone = PheromoneMatrix::new(n, initial, self.rho);

        let mut best_tour: Vec<usize> = (0..n).collect();
        let mut best_length = tour_length(&best_tour, dist);

        for _ in 0..self.iterations {
            pheromone.evaporate();

            for ant in 0..self.num_ants {
                let start = ant % n;
                let tour = crate::colony::construct_solution(
                    &pheromone, dist, start, self.alpha, self.beta, rng,
                );
                let length = tour_length(&tour, dist);

                if length < best_length {
                    best_length = length;
                    best_tour = tour.clone();
                }

                let deposit = 1.0 / length;
                pheromone.deposit(&tour, deposit);
            }

            // Elitist ants deposit extra
            if self.elitist_ants > 0 {
                let deposit = self.elitist_ants as f64 / best_length;
                pheromone.deposit(&best_tour, deposit);
            }
        }

        (best_tour, best_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_distance_unit() {
        let d = euclidean_distance((0.0, 0.0), (3.0, 4.0));
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn euclidean_distance_zero() {
        let d = euclidean_distance((1.0, 1.0), (1.0, 1.0));
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn build_distance_matrix_symmetric() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let dm = build_distance_matrix(&coords);
        assert!((dm[0][1] - 1.0).abs() < 1e-10);
        assert!((dm[1][0] - 1.0).abs() < 1e-10);
        assert!((dm[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tsp_solver_basic() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dm = build_distance_matrix(&coords);
        let mut rng = SimpleRng::new(42);
        let solver = TSPSolver::new(5, 1.0, 2.0, 0.1, 50);
        let (tour, length) = solver.solve(&dm, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length > 0.0);
    }

    #[test]
    fn tsp_solver_with_elitism() {
        let coords = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let dm = build_distance_matrix(&coords);
        let mut rng = SimpleRng::new(42);
        let solver = TSPSolver::new(5, 1.0, 2.0, 0.1, 50).with_elitism(3);
        let (tour, length) = solver.solve(&dm, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length > 0.0);
    }
}
