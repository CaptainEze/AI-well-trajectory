# AI-Based Gas Well Placement and Trajectory Optimization

This system uses Deep Reinforcement Learning (PPO) to optimize gas well trajectories based on reservoir properties, drilling constraints, and production objectives.

## Quick Start

### First Time (Training Mode)
Train a new model from scratch:
```bash
python src/main.py
```
This will take ~1-2 hours depending on your system.

### Subsequent Runs (Load Pre-trained Model)
Skip training and use the trained model:
```bash
python src/main.py --skip-training
```
or
```bash
python src/main.py --load-model
```
This completes in ~1-2 minutes.

## Project Structure

```
src/
  main.py                    # Main orchestrator (aligned with study objectives)
  modules/
    physics.py               # Trajectory calculations, torque/drag, wellbore stability
    reservoir.py             # Reservoir model and synthetic data generation
    ppo_agent.py            # PPO neural networks and training
    environment.py          # Reinforcement learning environment
    visualization.py        # All plotting functions

models/                     # Trained models saved here
results/                    # CSV output files
plots/                      # Generated visualizations
```

## Outputs

### CSV Files (results/)
- `objective1_model_specs.csv` - Model architecture specifications
- `objective2_training_history.csv` - Episode rewards during training
- `objective2_optimized_trajectory.csv` - Optimized well path coordinates
- `objective3_performance_comparison.csv` - AI vs Conventional comparison

### Plots (plots/)
1. `01_conventional_trajectory_3d.png` - Baseline trajectory
2. `02_conventional_inclination_azimuth.png` - Baseline angles
3. `03_dogleg_severity.png` - DLS along wellbore
4. `04_training_rewards.png` - Training progress
5. `05_reservoir_properties.png` - Porosity and permeability maps
6. `06_torque_drag.png` - Mechanical loads
7. `07_trajectory_comparison.png` - AI vs Conventional (4-panel)
8. `08_optimized_trajectory_3d.png` - AI-optimized path
9. `09_optimized_inclination_azimuth.png` - Optimized angles
10. `10_3d_reservoir_trajectory.png` - 3D reservoir with well path overlay

## Study Objectives

The system addresses three objectives:

1. **Objective 1**: Mathematical model integrating AI for well optimization
   - Physics-based trajectory calculations
   - PPO neural networks (Actor-Critic)
   - Reservoir model integration

2. **Objective 2**: AI-driven algorithm implementation
   - PPO training with 200 episodes
   - Multi-objective reward function
   - Constraint handling (DLS, torque, stability)

3. **Objective 3**: Validation and comparison
   - Performance metrics comparison
   - Improvement percentages
   - Visualization of results

## Key Features

- **Physics-based**: Minimum curvature method, torque/drag, wellbore stability
- **AI-optimized**: PPO reinforcement learning with improved reward shaping
- **Multi-objective**: Balances target accuracy, efficiency, safety, and production
- **Constraint-aware**: DLS limits, mud weight window, mechanical limits
- **Comprehensive visualization**: 10 different plots showing all aspects

## Requirements

See `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- torch
- tqdm
