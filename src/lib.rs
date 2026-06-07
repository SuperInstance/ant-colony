//! # Ant Colony Optimization
//!
//! A comprehensive ACO library with pheromone management, Ant Colony System (ACS),
//! MAX-MIN Ant System (MMAS), and TSP solving. Zero external dependencies.
//!
//! # Example
//! ```
//! use ant_colony::{TSPSolver, SimpleRng};
//!
//! let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
//! let dm = ant_colony::tsp::build_distance_matrix(&coords);
//! let mut rng = SimpleRng::new(42);
//! let solver = TSPSolver::new(5, 1.0, 2.0, 0.1, 50);
//! let (tour, length) = solver.solve(&dm, &mut rng);
//! println!("Tour: {:?}, Length: {}", tour, length);
//! ```

pub mod pheromone;
pub mod colony;
pub mod acs;
pub mod mmas;
pub mod tsp;
mod rng;

pub use rng::SimpleRng;
pub use pheromone::PheromoneMatrix;
pub use colony::{DistanceMatrix, construct_solution, tour_length};
pub use acs::ACS;
pub use mmas::MMAS;
pub use tsp::TSPSolver;

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
    fn integration_acs_solves_tsp() {
        let mut rng = SimpleRng::new(42);
        let acs = ACS::new(5, 1.0, 2.0, 0.1, 0.9, 0.1);
        let (tour, length) = acs.run(&sample_dist(), 50, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length < 50.0);
    }

    #[test]
    fn integration_mmas_solves_tsp() {
        let mut rng = SimpleRng::new(42);
        let mmas = MMAS::new(5, 1.0, 2.0, 0.1);
        let (tour, length) = mmas.run(&sample_dist(), 50, &mut rng);
        assert_eq!(tour.len(), 4);
        assert!(length < 50.0);
    }
}
