# AI-Based Gas Well Trajectory Optimization

Advanced well trajectory optimization using multiple AI algorithms including Hybrid RL+Planning, SAC, TD3, and PPO.

## Project Structure

```
project/
├── src/
│   ├── main.py                    # Objectives-based workflow
│   ├── train.py                   # Standalone training script
│   ├── compare_algorithms.py      # Algorithm comparison tool
│   │
│   └── modules/
│       ├── environment.py         # Well planning environment
│       ├── physics.py             # Trajectory physics and constraints
│       ├── reservoir.py           # Reservoir model and properties
│       ├── visualization.py       # Plotting and visualization
│       │
│       ├── hybrid_agent.py        # Hybrid RL + Planning (RECOMMENDED)
│       ├── sac_agent.py           # Soft Actor-Critic
│       ├── td3_agent.py           # Twin Delayed DDPG
│       └── ppo_agent.py           # Proximal Policy Optimization
│
├── models/                        # Trained model weights
│   ├── hybrid/
│   ├── sac/
│   ├── td3/
│   └── ppo/
│
├── plots/                         # Visualization outputs
│   ├── hybrid/
│   ├── sac/
│   ├── td3/
│   └── ppo/
│
├── results/                       # CSV results and metrics
│   ├── hybrid/
│   ├── sac/
│   ├── td3/
│   └── ppo/
│
└── training_outputs/              # Detailed training logs
    └── [algorithm]_[timestamp]/
```

## Quick Start

### Installation

Install required dependencies: numpy, pandas, torch, matplotlib, seaborn, scipy.

### Basic Usage

Run with default algorithm (Hybrid - recommended) from the src directory. 

Run with specific algorithm by specifying: hybrid (20-30 min), sac (2-3 hours), td3 (2-3 hours), or ppo (not recommended).

Load pre-trained model to skip training and use existing weights.

## Workflow

The main.py script implements a three-objective structured approach:

### Objective 1: Mathematical Model Development

- Creates 3D reservoir with realistic properties
- Defines target location and drilling constraints
- Initializes AI agent
- Generates conventional trajectory baseline

Outputs: reservoir_formation.csv, objective1_model_specs.csv, visualizations

### Objective 2: AI Algorithm Implementation

- Trains selected algorithm on well optimization task
- Generates optimized trajectory
- Creates comprehensive visualizations

Outputs: trained model, training_history.csv, optimized_trajectory.csv, plots

### Objective 3: Validation and Comparison

- Compares AI vs conventional drilling
- Computes performance metrics
- Generates comparison visualizations

Outputs: performance_comparison.csv, comparison plots

## Standalone Training

Use train.py for focused training with detailed monitoring. Generates organized training_outputs/[algorithm]_[timestamp]/ directory containing model checkpoints, visualizations, and CSV training logs.

## Algorithm Comparison

Use compare_algorithms.py to compare results from multiple algorithm runs. Generates performance summaries and comparison visualizations.

## Algorithm Guide

### Hybrid RL + Planning (RECOMMENDED)

Best for production use and most applications.

**Training time:** 20-30 minutes  
**Episodes:** 150  
**Success rate:** >95%  
**Accuracy:** <200 ft to target

Combines classical trajectory planning with RL optimization:
- Classical planner generates reference trajectory
- RL agent learns optimal corrections
- Guaranteed feasibility with faster convergence

### SAC (Soft Actor-Critic)

Best performance when training time is available.

**Training time:** 2-3 hours  
**Episodes:** 500  
**Success rate:** >80%  
**Accuracy:** <200 ft to target

Advantages:
- State-of-the-art exploration strategy
- Automatic entropy tuning
- Globally optimal solutions
- Excellent sample efficiency

### TD3 (Twin Delayed DDPG)

Good balance of performance and stability.

**Training time:** 2-3 hours  
**Episodes:** 500  
**Success rate:** >75%  
**Accuracy:** <250 ft to target

Advantages:
- Simpler than SAC
- Very stable training
- Proven in robotics applications
- Reliable convergence

```bash
python main.py --algorithm td3
``` recommended for well planning task.

**Training time:** 2+ hours  
**Episodes:** 300  
**Success rate:** <20%

Issues:
- Inefficient with long-horizon tasks
- Poor sample efficiency
- Frequently gets stuck in local minima
- Often fails to reach target

## Output Files

### Models Directory

```
models/[algorithm]/best_[algorithm]_agent.pth
```

### Results Directory

```
results/[algorithm]/
├── reservoir_formation.csv
├── objective1_model_specs.csv
├── objective2_training_history.csv
├── objective2_optimized_trajectory.csv
└── objective3_performance_comparison.csv
```

### Plots Directory

```
plots/[algorithm]/
├── 01_conventional_trajectory_3d.png
├── 02_conventional_inclination_azimuth.png
├── 02_optimized_inclination_azimuth.png
├── 03_dogleg_severity.png
├── 04_training_rewards.png
├── 05_reservoir_properties.png
├── 06_torque_drag.png
├── 07_trajectory_comparison.png
├── 08_optimized_trajectory_3d.png
├── 08_reservoir_properties_4panel.png
├── 09_reservoir_combined_well.png
├── 10_3d_reservoir_trajectory.png
└── 11_reservoir_contour_slices.png
```

## Environment Configuration

Target location and drilling constraints defined in main.py:

```python
target = {
    'KOP': 3000,           # Kick-off poiare defined in main.py. Configuration includes kick-off point, target true vertical depth, north and east coordinates, initial azimuth, maximum dogleg severity, friction coefficient, minimum well separation, and maximum inclination.formance comparison across algorithms:

| Metric | Hybrid | SAC | TD3 | Conventional |
|--------|--------|-----|-----|--------------|
| Distance to Target | 150 ft | 180 ft | 220 ft | 350 ft |
| Training Episodes | 150 | 500 | 500 | N/A |
| Training Time | 30 min | 3 hr | 3 hr | N/A |
| Max DLS | 7.5 deg/100ft | 7.8 deg/100ft | 8.2 deg/100ft | 3.0 deg/100ft |
| Productivity | 760k | 780k | 750k | 590k |

## Hyperparameters

### Hybrid Controller

```python
HybridController(
    state_dim=26,
    action_dim=5,
    learning_rate=1e-3
)
```

Training: 150 episodes, 500 steps/episode

### SAC Agent

```python
SACAgent(
    state_dim=26,
    action_dim=5,
    learning_rate=3e-4,
State dimension: 26, Action dimension: 5, Learning rate: 1e-3. Training: 150 episodes, 500 steps/episode.

### SAC Agent

State dimension: 26, Action dimension: 5, Learning rate: 3e-4, Discount factor: 0.99, Target network update: 0.005. Training: 500 episodes, 500 steps/episode, 10k random exploration steps.

### TD3 Agent

State dimension: 26, Action dimension: 5, Learning rate: 3e-4, Discount factor: 0.99, Target network update: 0.005, Policy noise: 0.2, Noise clip: 0.5. Training: 500 episodes, 500 steps/episode, 10k random exploration steps.
- Wellbore stability assessment

WellboreMechanics class:
- Critical slide angle calculation
- Mud weight window determination
- Build-up rate from radius of curvature

### reservoir.py

ReservoirModel class:
- 3D synthetic reservoir generation
- Property fields: porosity, permeability, pore pressure, fracture gradient
- Spatial interpolation for property queries
- CSV export functionality

### visualization.py

Visualizer class:
- 3D trajectory plots
- Inclination/azimuth vs depth profiles
- Dogleg severity visualization
- Reservoir property maps
- Training progress curves
- Comparison plots

## Command Reference

```bash
# Basic runs
python main.py
python main.py --algorithm sac
python main.py --algorithm hybrid --skip-training

# Training
python train.py --algorithm hybrid --episodes 150
python train.py --algorithm sac --episodes 500 --verbose

# Comparison
python compare_algorithms.py
python compare_algorithms.py --algorithms hybrid sac td3

# Directory structure
ls models/          # Trained models
ls plots/           # All visualizations
ls results/         # CSV outputs
ls training_outputs/  # Training logs
```

## Contributing

To add a new algorithm, create a new agent module implementing select_action(), update(), save(), and load() methods. Register the new algorithm in main.py's _create_agent() method and add to argument parser choices.

Models directory contains trained model weights. Plots directory contains all visualizations. Results directory contains CSV outputs. Training outputs directory contains detailed training logs.