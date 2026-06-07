//! Pheromone trail management: matrix, evaporation, deposit.

/// Pheromone matrix for a graph of `n` nodes.
#[derive(Clone, Debug)]
pub struct PheromoneMatrix {
    /// n x n matrix of pheromone values.
    pub trails: Vec<Vec<f64>>,
    pub n: usize,
    /// Evaporation rate (0..1).
    pub rho: f64,
}

impl PheromoneMatrix {
    pub fn new(n: usize, initial: f64, rho: f64) -> Self {
        let trails = vec![vec![initial; n]; n];
        Self { trails, n, rho }
    }

    /// Evaporate all pheromone trails.
    pub fn evaporate(&mut self) {
        for i in 0..self.n {
            for j in 0..self.n {
                self.trails[i][j] *= 1.0 - self.rho;
            }
        }
    }

    /// Deposit pheromone on path.
    pub fn deposit(&mut self, path: &[usize], amount: f64) {
        for w in path.windows(2) {
            let i = w[0];
            let j = w[1];
            self.trails[i][j] += amount;
            self.trails[j][i] += amount; // symmetric
        }
    }

    /// Get pheromone level between nodes i and j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.trails[i][j]
    }

    /// Set pheromone level.
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.trails[i][j] = val;
        self.trails[j][i] = val;
    }

    /// Clamp all trails to [tau_min, tau_max].
    pub fn clamp(&mut self, tau_min: f64, tau_max: f64) {
        for row in &mut self.trails {
            for val in row.iter_mut() {
                *val = val.clamp(tau_min, tau_max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrix_has_initial_value() {
        let pm = PheromoneMatrix::new(5, 1.0, 0.1);
        assert_eq!(pm.get(0, 1), 1.0);
    }

    #[test]
    fn evaporation_reduces_values() {
        let mut pm = PheromoneMatrix::new(3, 1.0, 0.5);
        pm.evaporate();
        assert!((pm.get(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn deposit_adds_symmetrically() {
        let mut pm = PheromoneMatrix::new(4, 0.0, 0.1);
        pm.deposit(&[0, 1, 2], 5.0);
        assert!((pm.get(0, 1) - 5.0).abs() < 1e-10);
        assert!((pm.get(1, 0) - 5.0).abs() < 1e-10);
        assert!((pm.get(1, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn clamp_limits_values() {
        let mut pm = PheromoneMatrix::new(3, 0.0, 0.1);
        pm.set(0, 1, 10.0);
        pm.clamp(0.1, 5.0);
        assert!((pm.get(0, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn set_is_symmetric() {
        let mut pm = PheromoneMatrix::new(3, 0.0, 0.1);
        pm.set(0, 2, 3.0);
        assert!((pm.get(2, 0) - 3.0).abs() < 1e-10);
    }
}
