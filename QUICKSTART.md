# 🚀 Quick Start Guide

## Installation

1. **Download all files** to a folder on your computer

2. **Install dependencies:**
```bash
pip install numpy matplotlib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 5-Minute Quick Start

### Option 1: Run the Benchmark (Recommended)
Compare all algorithms on the same problem:

```bash
python main.py --benchmark --cities 30
```

This will:
- ✅ Generate 30 random cities
- ✅ Solve using 4 different methods
- ✅ Show detailed comparison
- ✅ Generate beautiful visualizations

**Expected output:**
```
Method                    Distance        Time (s)        Quality   
----------------------------------------------------------------------
DRL Fast                  ~850            0.05           ~94%     
Genetic Algorithm         ~800            12.0           ~100%     
Ant Colony               ~810            15.0           ~99%     
Hybrid (DRL+GA)          ~800            3.5            ~100%    
```

### Option 2: Run Simple Example
For a guided walkthrough:

```bash
python example.py
```

Choose option 1 and watch it work!

### Option 3: Run Individual Solver
Try a specific algorithm:

```bash
# Genetic Algorithm (best quality)
python main.py --solver ga --cities 50

# Hybrid (best overall)
python main.py --solver hybrid --cities 50

# DRL Fast (instant solution)
python main.py --solver drl --cities 100
```

## Understanding the Output

### Console Output
```
GENETIC ALGORITHM SOLVER
============================================================
Population Size: 100
Mutation Rate: 0.015
...
Generation  100 | Best:   850.23 | Avg:   920.45
Generation  200 | Best:   820.15 | Avg:   875.30
...
Final Best Distance: 802.15
Improvement: 12.50%
```

### Visualizations
You'll see three types of plots:

1. **Tour Visualization** - The actual route with cities and paths
2. **Convergence Plot** - How the solution improves over time
3. **Comparison Plot** - Side-by-side comparison of different methods

## Common Use Cases

### Small Problem (Testing)
```bash
python main.py --benchmark --cities 10
# Fast, good for testing
```

### Medium Problem (Standard)
```bash
python main.py --benchmark --cities 50
# Most realistic, takes a few minutes
```

### Large Problem (Speed Test)
```bash
python main.py --solver drl --cities 200 --no-viz
# Instant solution, no visualization overhead
```

### Circle Pattern (Visual Verification)
```bash
python main.py --solver ga --cities 20 --pattern circle
# Should produce a perfect circle route
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Modify parameters** in the solvers to see how they affect results
3. **Try custom city layouts** by editing example.py
4. **Experiment with different problem sizes**

## Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy matplotlib
```

### Plots don't show up
Make sure you're not using `--no-viz` flag

### "Command not found: python"
Try `python3` instead:
```bash
python3 main.py --benchmark --cities 30
```

### Code runs but no output
Check that you're in the correct directory with all the .py files

## File Overview

| File | Purpose |
|------|---------|
| `main.py` | Main application with CLI |
| `example.py` | Simple guided examples |
| `tsp_core.py` | Core data structures |
| `genetic_algorithm.py` | GA solver |
| `ant_colony.py` | ACO solver |
| `drl_solver.py` | DRL + Hybrid solver |
| `visualization.py` | Plotting tools |
| `README.md` | Full documentation |

## Need Help?

1. Check the **README.md** for detailed explanations
2. Look at **example.py** for code examples
3. Run with `-h` flag: `python main.py -h`

---

**You're ready to solve TSP! 🎉**

Start with:
```bash
python main.py --benchmark --cities 30
```