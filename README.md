# 🚁 Drone Pursuit Environment - Deep Reinforcement Learning

A sophisticated drone pursuit environment built with Gymnasium and Pygame, featuring beautiful graphics, adaptive curriculum learning, and state-of-the-art RL algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Speed Analysis](#speed-analysis)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## ✨ Features

### Environment Features
- 🎨 **Beautiful Graphics**: Gradient backgrounds, particle effects, glow effects, glassmorphism UI
- 🎯 **Smart Evader**: Intelligent opponent with waypoint navigation and panic behavior
- 🚧 **Dynamic Obstacles**: Procedurally generated obstacles with collision detection
- ⚡ **Jam Zones**: No-fly zones that end the episode if entered
- 📊 **Real-time Metrics**: Live performance graphs, mini-map, and HUD

### Training Features
- 🎓 **Adaptive Curriculum**: Automatically adjusts difficulty based on performance
- 🧠 **Multiple Algorithms**: PPO, SAC, TD3 support
- 📈 **Enhanced Metrics**: Speed-focused metrics and efficiency tracking
- 💾 **Auto-checkpointing**: Saves best models automatically
- 🔄 **Parallel Training**: Multi-process environment support

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/drone-pursuit-rl.git
cd drone-pursuit-rl
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python Enviroment.py
```
You should see a Pygame window with the drone pursuit simulation running.

---

## 🚀 Quick Start

### 1. Test the Environment
Run a demo with random actions to see the environment in action:
```bash
python Enviroment.py
```
**Controls**: Press `ESC` to exit

### 2. Train Your First Agent (Quick)
Start training with default settings (PPO, 500K steps):
```bash
python Training.py --total-timesteps 500000
```

### 3. Watch Your Trained Agent
```bash
python Eval.py --model-path models/ppo_optimized_YYYYMMDD_HHMMSS/best/best_model.zip --render
```

---

## 🎓 Training

### Basic Training
Train with default PPO algorithm:
```bash
python Training.py --total-timesteps 2000000
```

### Advanced Training Options

#### Choose Algorithm
```bash
# PPO (Best for beginners, stable)
python Training.py --algo ppo --total-timesteps 2000000

# SAC (Good for continuous control)
python Training.py --algo sac --total-timesteps 2000000

# TD3 (Fast convergence, less stable)
python Training.py --algo td3 --total-timesteps 2000000
```

#### Parallel Training (Faster)
```bash
python Training.py --num-envs 16 --total-timesteps 2000000
```
**Tip**: Use 4-16 environments for faster training. More isn't always better!

#### Custom Difficulty
```bash
# Start easy
python Training.py --start-obstacles 3 --start-jam-radius 120

# Start hard
python Training.py --start-obstacles 15 --start-jam-radius 60
```

#### Adjust Learning Rate
```bash
# Faster learning (less stable)
python Training.py --lr 0.001

# Slower learning (more stable)
python Training.py --lr 0.0001
```

#### GPU Training
```bash
python Training.py --device cuda
```

#### Custom Experiment Name
```bash
python Training.py --experiment-name my_awesome_agent
```

### Training Output
Models are saved to:
```
models/
  └── [experiment_name]/
      ├── best/              # Best performing model
      │   └── best_model.zip
      ├── final_model.zip    # Final model after training
      ├── vec_normalize.pkl  # Normalization stats
      └── [algo]_[step]_steps.zip  # Checkpoints
```

### Monitor Training Progress
Open TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```
Then open http://localhost:6006 in your browser

**Key Metrics to Watch**:
- `metrics/capture_rate`: Should increase to 70%+
- `metrics/avg_capture_time`: Should decrease (faster captures)
- `metrics/efficiency_score`: Higher is better

---

## 📊 Evaluation

### Basic Evaluation
```bash
python Eval.py \
    --model-path models/your_experiment/best/best_model.zip \
    --n-episodes 10 \
    --render
```

### Evaluation Options

#### Silent Evaluation (No Rendering)
```bash
python Eval.py \
    --model-path models/your_experiment/best/best_model.zip \
    --n-episodes 50
```

#### Compare with Random Agent
```bash
python Eval.py \
    --model-path models/your_experiment/best/best_model.zip \
    --compare \
    --render
```

#### Stress Test (Multiple Difficulties)
```bash
python Eval.py \
    --model-path models/your_experiment/best/best_model.zip \
    --stress-test \
    --n-episodes 20
```

#### With Normalization Stats
```bash
python Eval.py \
    --model-path models/your_experiment/best/best_model.zip \
    --vec-normalize-path models/your_experiment/vec_normalize.pkl \
    --render
```

### Expected Performance
**Good Agent**:
- Capture rate: 70-90%
- Average capture time: 300-600 steps
- Jam rate: <10%

**Excellent Agent**:
- Capture rate: 90%+
- Average capture time: <300 steps
- Jam rate: <5%

---

## ⚡ Speed Analysis

Analyze your agent's capture speed performance:

### Basic Speed Analysis
```bash
python Speed_Analysis.py \
    --model-path models/your_experiment/best/best_model.zip \
    --n-episodes 50
```

### Test All Difficulties
```bash
python Speed_Analysis.py \
    --model-path models/your_experiment/best/best_model.zip \
    --test-all-difficulties \
    --n-episodes 50
```

### Analyze Path Efficiency
```bash
python Speed_Analysis.py \
    --model-path models/your_experiment/best/best_model.zip \
    --analyze-trajectories \
    --test-all-difficulties
```

### Output
- **Console**: Detailed statistics and performance ratings
- **Plot**: `speed_analysis.png` with 4 visualization charts
  - Capture time distribution
  - Success rate by difficulty
  - Average capture time
  - Speed score comparison

---

## 📁 Project Structure

```
drone-pursuit-rl/
│
├── Enviroment.py           # Main environment (Gymnasium)
├── Training.py             # Training script with curriculum
├── Eval.py                 # Evaluation and testing
├── Speed_Analysis.py       # Speed benchmarking tool
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── models/                # Saved models (created during training)
│   └── [experiments]/
│
└── logs/                  # Training logs (created during training)
    └── tensorboard/
```

---

## ⚙️ Configuration

### Environment Config
Modify `config` dictionary in training scripts:

```python
config = {
    'num_obstacles': 8,        # Number of obstacles (3-20)
    'jam_zone_radius': 80,     # Jam zone size (50-120)
    'random_start': True,      # Random starting positions
}
```

### Observation Space (27 dimensions)
1. Relative position to evader (2D)
2. Relative velocity (2D)
3. Distance to evader (scalar)
4. Distance to jam zone (scalar)
5. Chaser heading (scalar)
6. Chaser velocity (2D)
7. Minimum obstacle distance (scalar)
8. Ray distances (8D) - obstacle detection
9. Angle to target (scalar)
10. Angle to predicted position (scalar)
11. Relative heading (scalar)
12. Closing speed (scalar)
13. Time to intercept (scalar)
14. Episode progress (scalar)

### Action Space (2D continuous)
- `throttle`: [-1, 1] - Forward/backward acceleration
- `turn_rate`: [-1, 1] - Left/right turning

### Reward Function
```python
# Distance-based reward
reward += (prev_distance - curr_distance) * 2.0

# Closing speed bonus
if closing_speed > 0:
    reward += closing_speed * 0.3

# Proximity bonus (when close)
if distance < 200:
    reward += (200 - distance) / 200 * 0.5

# Heading alignment
reward += heading_alignment * 0.2

# Time penalty
reward -= 0.02

# Success bonus
if captured:
    reward += 200.0 + time_bonus

# Failure penalty
if jammed or collision:
    reward -= 100.0
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'pygame'
```bash
pip install pygame
```

#### 2. CUDA out of memory
```bash
# Use CPU instead
python Training.py --device cpu

# Or reduce number of parallel environments
python Training.py --num-envs 4
```

#### 3. Pygame window not appearing
- Make sure you have a display available
- On Linux, you may need: `export DISPLAY=:0`
- For headless servers, remove `--render` flag

#### 4. Training is too slow
```bash
# Increase parallel environments
python Training.py --num-envs 16

# Use GPU
python Training.py --device cuda

# Reduce total timesteps for testing
python Training.py --total-timesteps 500000
```

#### 5. Model not learning (capture rate stays low)
- Try lower learning rate: `--lr 0.0001`
- Start with easier difficulty: `--start-obstacles 3`
- Train longer: `--total-timesteps 5000000`
- Check tensorboard for diverging losses

#### 6. "VecNormalize" file not found during evaluation
```bash
# Don't use vec-normalize if file doesn't exist
python Eval.py --model-path models/.../best_model.zip
# (Remove --vec-normalize-path argument)
```

---

## 🎯 Advanced Usage

### 1. Custom Reward Function
Edit `_calculate_reward()` in `Enviroment.py`:
```python
def _calculate_reward(self, prev_dist, curr_dist, terminated, info):
    reward = 0.0
    
    # Your custom reward logic here
    distance_change = prev_dist - curr_dist
    reward += distance_change * 3.0  # Increase weight
    
    # Add more custom rewards...
    
    return reward
```

### 2. Modify Drone Physics
Edit `DronePhysics` class in `Enviroment.py`:
```python
self.chaser = DronePhysics(
    max_speed=6.0,      # Increase speed
    max_accel=0.5,      # Increase acceleration
    max_turn_rate=0.25  # Increase turn rate
)
```

### 3. Add More Obstacles
```python
python Training.py --start-obstacles 20
```

### 4. Disable Adaptive Curriculum
```python
python Training.py --use-adaptive-curriculum False
```

### 5. Custom Neural Network Architecture
Edit `create_optimized_model()` in `Training.py`:
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[1024, 1024, 512],  # Larger actor network
        vf=[1024, 1024, 512]   # Larger critic network
    ),
    activation_fn=torch.nn.ReLU,
)
```

### 6. Resume Training from Checkpoint
```python
# Load existing model
model = PPO.load("models/experiment/ppo_1000000_steps.zip")

# Continue training
model.learn(total_timesteps=1000000)
```

### 7. Hyperparameter Tuning with Optuna
```python
import optuna
from stable_baselines3 import PPO

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    
    model = PPO('MlpPolicy', env, learning_rate=lr, gamma=gamma)
    model.learn(total_timesteps=100000)
    
    # Evaluate and return performance
    return eval_performance(model)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 📈 Performance Benchmarks

### Hardware Requirements

| Configuration | CPU | RAM | GPU | Training Time (2M steps) |
|--------------|-----|-----|-----|-------------------------|
| Minimum | 4 cores | 8 GB | - | ~8 hours |
| Recommended | 8 cores | 16 GB | GTX 1060 | ~2 hours |
| High-end | 16 cores | 32 GB | RTX 3080 | ~45 min |

### Expected Results

| Difficulty | Success Rate | Avg Capture Time | Training Steps |
|-----------|--------------|------------------|----------------|
| Easy | 95%+ | 200-300 | 500K |
| Medium | 80-90% | 400-600 | 1M |
| Hard | 60-80% | 600-900 | 2M |
| Expert | 40-60% | 800-1200 | 5M+ |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- RL algorithms from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Graphics with [Pygame](https://www.pygame.org/)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🎓 Citation

If you use this environment in your research, please cite:
```bibtex
@software{drone_pursuit_rl,
  author = {Your Name},
  title = {Drone Pursuit Reinforcement Learning Environment},
  year = {2024},
  url = {https://github.com/yourusername/drone-pursuit-rl}
}
```

---

**Happy Training! 🚁🎯**