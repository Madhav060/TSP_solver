# 🏆 TSP Solver Benchmark: berlin52.tsp

**Test Run:** 2025-11-04 08:54:18
**Problem:** `berlin52.tsp` (52 cities)
**Known Optimal Distance:** `7542.0`
**Runs per Solver:** `10`

## 🥇 Best Performing Solver (by Avg. Distance)

**Ant Colony (1000 iter)**
* **Average Distance:** 7544.89 (+0.04% from optimal)
* **Best Distance:** 7544.37 (+0.03% from optimal)
* **Average Time:** 58.876s

## 📊 Full Comparison Table
*(Sorted by `Avg. Distance`)*

| Solver                       | Avg. Distance   | Best Distance   |   Avg. Time (s) | Avg. Error (%)   | Best Error (%)   |   Std. Dev. |
|:-----------------------------|:----------------|:----------------|----------------:|:-----------------|:-----------------|------------:|
| Ant Colony (1000 iter)       | 7,544.89        | 7,544.37        |          58.876 | +0.04%           | +0.03%           |        1.37 |
| Hybrid (DRL + ACO 500 iter)  | 7,549.09        | 7,544.37        |          26.168 | +0.09%           | +0.03%           |        9.36 |
| DRL (Model + 2-Opt)          | 8,319.74        | 8,319.74        |           0.793 | +10.31%          | +10.31%          |        0    |
| Genetic Algorithm (1000 gen) | 9,588.65        | 8,528.00        |          18.691 | +27.14%          | +13.07%          |      618.31 |
