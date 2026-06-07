//! Colony management: ant construction of solutions.

use crate::pheromone::PheromoneMatrix;
use crate::rng::SimpleRng;

/// A graph represented as a distance matrix.
pub type DistanceMatrix = Vec<Vec<f64>>;

/// Build a solution (tour) for a single ant using probabilistic selection.
pub fn construct_solution(
    pheromone: &PheromoneMatrix,
    dist: &DistanceMatrix,
    start: usize,
    alpha: f64,
    beta: f64,
    rng: &mut SimpleRng,
) -> Vec<usize> {
    let n = pheromone.n;
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = start;
    visited[current] = true;
    tour.push(current);

    while tour.len() < n {
        let mut probs = Vec::new();
        let mut total = 0.0;
        for j in 0..n {
            if !visited[j] && j != current {
                let d = dist[current][j];
                let tau = pheromone.get(current, j);
                let eta = if d > 0.0 { 1.0 / d } else { 1e10 };
                let p = tau.powf(alpha) * eta.powf(beta);
                probs.push((j, p));
                total += p;
            }
        }

        if probs.is_empty() {
            break;
        }

        // Roulette selection
        let r = rng.gen_f64() * total;
        let mut cumsum = 0.0;
        let mut next = probs[0].0;
        for (j, p) in &probs {
            cumsum += p;
            if cumsum >= r {
                next = *j;
                break;
            }
        }

        visited[next] = true;
        tour.push(next);
        current = next;
    }

    tour
}

/// Compute tour length.
pub fn tour_length(tour: &[usize], dist: &DistanceMatrix) -> f64 {
    let mut len = 0.0;
    for w in tour.windows(2) {
        len += dist[w[0]][w[1]];
    }
    // Close the tour
    if tour.len() > 1 {
        len += dist[*tour.last().unwrap()][tour[0]];
    }
    len
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
    fn construct_visits_all_nodes() {
        let dist = sample_dist();
        let pm = PheromoneMatrix::new(4, 1.0, 0.1);
        let mut rng = SimpleRng::new(42);
        let tour = construct_solution(&pm, &dist, 0, 1.0, 2.0, &mut rng);
        assert_eq!(tour.len(), 4);
        let mut sorted = tour.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn tour_length_computes_correctly() {
        let dist = sample_dist();
        let tour = vec![0, 1, 2, 3];
        let len = tour_length(&tour, &dist);
        // 0->1: 2, 1->2: 6, 2->3: 8, 3->0: 10 = 26
        assert!((len - 26.0).abs() < 1e-10);
    }

    #[test]
    fn tour_length_single_node() {
        let dist = vec![vec![0.0]];
        let tour = vec![0];
        let len = tour_length(&tour, &dist);
        assert!((len - 0.0).abs() < 1e-10);
    }
}
