# Simple Reinforcement Learning Configuration
This project demonstrates how Reinforcement Learning (PPO) can be used for algorithm configuration — automatically tuning solver parameters to improve performance.

I applied a simple Actor–Critic PPO framework to a small Capacitated Vehicle Routing Problem (CVRP) solved with IBM CPLEX (via Docplex).
The RL agent observes basic instance features (like demand utilization and distances), then proposes different solver parameter settings such as emphasis, heuristic frequency, threads, and MIP gap.
Each configuration is tested in CPLEX, and the solver’s objective and runtime are turned into a reward signal for learning.

Although the instance is small and deterministic (so results converge quickly), the project shows the full workflow of combining deep reinforcement learning with mathematical optimization solvers — a practical example of intelligent algorithm tuning.

## How to run:
Just open a terminal and run.
