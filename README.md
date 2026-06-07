# ant-colony

Ant colony optimization library in Rust with zero external dependencies.

## Features

- **Pheromone Management**: Matrix operations, evaporation, deposit, clamping
- **Ant Colony System (ACS)**: Local/global pheromone updates with exploitation parameter
- **MAX-MIN Ant System (MMAS)**: Bounded pheromone trails for exploration control
- **TSP Solver**: Full traveling salesman problem solver with elitist ants
- **Utilities**: Euclidean distance, distance matrix construction from coordinates

## Usage

```rust
use ant_colony::{TSPSolver, SimpleRng};

let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
let dm = ant_colony::tsp::build_distance_matrix(&coords);
let mut rng = SimpleRng::new(42);

let solver = TSPSolver::new(5, 1.0, 2.0, 0.1, 50);
let (tour, length) = solver.solve(&dm, &mut rng);
println!("Tour: {:?}, Length: {}", tour, length);
```

## License

MIT
