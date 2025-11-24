# 🏆 TSP Solver Benchmark: berlin52.tsp

**Test Run:** 2025-11-16 15:17:09
**Problem:** `berlin52.tsp` (52 cities)
**Known Optimal Distance:** `7542.0`
**Runs per Solver:** `10`

## 🥇 Best Performing Solver (by Avg. Distance)

**Ant Colony (1000 iter)**
* **Average Distance:** 7544.98 (+0.04% from optimal)
* **Best Distance:** 7544.37 (+0.03% from optimal)
* **Average Time:** 57.264s

## 📊 Full Comparison Table
*(Sorted by `Avg. Distance`)*

| Solver                       | Avg. Distance   | Best Distance   |   Avg. Time (s) | Avg. Error (%)   | Best Error (%)   |   Std. Dev. |
|:-----------------------------|:----------------|:----------------|----------------:|:-----------------|:-----------------|------------:|
| Ant Colony (1000 iter)       | 7,544.98        | 7,544.37        |          57.264 | +0.04%           | +0.03%           |        1.44 |
| Hybrid (DRL + ACO 500 iter)  | 7,560.67        | 7,544.66        |          24.754 | +0.25%           | +0.04%           |       34.13 |
| DRL (Model + 2-Opt)          | 8,319.74        | 8,319.74        |           0.789 | +10.31%          | +10.31%          |        0    |
| Genetic Algorithm (1000 gen) | 9,383.14        | 8,739.54        |          16.232 | +24.41%          | +15.88%          |      507.36 |
