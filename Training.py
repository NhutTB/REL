"""
OPTIMIZED Training Script - Focus on Fast Capture
Key improvements:
1. Refined reward shaping for speed
2. Better curriculum learning
3. Enhanced observation space
4. Adaptive difficulty based on performance
"""

import argparse
import os
from datetime import datetime
import numpy as np
from collections import deque

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from drone_pursuit_env import DronePursuitEnv


class AdaptiveCurriculumCallback(BaseCallback):
    """
    Performance-based adaptive curriculum
    Automatically adjusts difficulty based on success rate
    """
    def __init__(self, envs, window_size=100, target_success_rate=0.70, verbose=1):
        super().__init__(verbose)
        self.envs = envs
        self.window_size = window_size
        self.target_success_rate = target_success_rate
        
        # Track performance
        self.recent_results = deque(maxlen=window_size)
        self.recent_times = deque(maxlen=window_size)
        
        # Difficulty levels
        self.difficulty_levels = [
            {'num_obstacles': 3, 'jam_zone_radius': 120, 'name': 'Easy'},
            {'num_obstacles': 5, 'jam_zone_radius': 100, 'name': 'Medium-Easy'},
            {'num_obstacles': 8, 'jam_zone_radius': 80, 'name': 'Medium'},
            {'num_obstacles': 12, 'jam_zone_radius': 70, 'name': 'Medium-Hard'},
            {'num_obstacles': 15, 'jam_zone_radius': 60, 'name': 'Hard'},
            {'num_obstacles': 20, 'jam_zone_radius': 50, 'name': 'Expert'},
        ]
        self.current_level = 0
        self.steps_at_level = 0
        self.min_steps_per_level = 50000
        
    def _on_step(self):
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                captured = info.get('captured', False)
                self.recent_results.append(1 if captured else 0)
                if captured:
                    self.recent_times.append(info['episode']['l'])
        
        self.steps_at_level += 1
        
        # Check for curriculum update every 1000 steps
        if self.steps_at_level >= 1000 and self.steps_at_level >= self.min_steps_per_level:
            if len(self.recent_results) >= self.window_size:
                success_rate = np.mean(self.recent_results)
                avg_time = np.mean(self.recent_times) if self.recent_times else 2000
                
                # Increase difficulty if doing well
                if success_rate >= self.target_success_rate and self.current_level < len(self.difficulty_levels) - 1:
                    self._level_up(success_rate, avg_time)
                
                # Decrease difficulty if struggling
                elif success_rate < 0.3 and self.current_level > 0:
                    self._level_down(success_rate)
        
        return True
    
    def _level_up(self, success_rate, avg_time):
        self.current_level += 1
        config = self.difficulty_levels[self.current_level]
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🎓 LEVEL UP! → {config['name']}")
            print(f"   Success Rate: {success_rate*100:.1f}% | Avg Capture Time: {avg_time:.0f} steps")
            print(f"   New Config: {config['num_obstacles']} obstacles, jam radius {config['jam_zone_radius']}")
            print(f"{'='*70}\n")
        
        self._update_envs(config)
        self.recent_results.clear()
        self.recent_times.clear()
        self.steps_at_level = 0
    
    def _level_down(self, success_rate):
        self.current_level -= 1
        config = self.difficulty_levels[self.current_level]
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"⚠️  LEVEL DOWN → {config['name']}")
            print(f"   Success Rate too low: {success_rate*100:.1f}%")
            print(f"{'='*70}\n")
        
        self._update_envs(config)
        self.recent_results.clear()
        self.recent_times.clear()
        self.steps_at_level = 0
    
    def _update_envs(self, config):
        for env in self.envs.envs:
            env.config.update({
                'num_obstacles': config['num_obstacles'],
                'jam_zone_radius': config['jam_zone_radius']
            })


class EnhancedMetricsCallback(BaseCallback):
    """Enhanced metrics tracking with speed focus"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_data = {
            'rewards': deque(maxlen=200),
            'lengths': deque(maxlen=200),
            'captures': deque(maxlen=200),
            'jams': deque(maxlen=200),
            'capture_times': deque(maxlen=200)
        }
        self.best_avg_time = float('inf')

    def _on_rollout_end(self) -> None:
        if len(self.episode_data['rewards']) >= 100:
            # Standard metrics
            self.logger.record('metrics/mean_reward_100ep', 
                             np.mean(self.episode_data['rewards']))
            self.logger.record('metrics/mean_length_100ep', 
                             np.mean(self.episode_data['lengths']))
            
            capture_rate = np.mean(self.episode_data['captures'])
            self.logger.record('metrics/capture_rate', capture_rate)
            
            # Speed metrics (key for optimization)
            if len(self.episode_data['capture_times']) > 0:
                avg_capture_time = np.mean(self.episode_data['capture_times'])
                self.logger.record('metrics/avg_capture_time', avg_capture_time)
                self.logger.record('metrics/min_capture_time', 
                                 np.min(self.episode_data['capture_times']))
                
                # Track improvement
                if avg_capture_time < self.best_avg_time:
                    improvement = self.best_avg_time - avg_capture_time
                    self.best_avg_time = avg_capture_time
                    self.logger.record('metrics/speed_improvement', improvement)
            
            # Efficiency score (combines speed and success)
            if capture_rate > 0:
                efficiency = capture_rate / (avg_capture_time / 1000.0)
                self.logger.record('metrics/efficiency_score', efficiency)

    def _on_step(self):
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_data['rewards'].append(ep_info['r'])
                self.episode_data['lengths'].append(ep_info['l'])
                
                captured = info.get('captured', False)
                self.episode_data['captures'].append(1 if captured else 0)
                self.episode_data['jams'].append(1 if info.get('jammed', False) else 0)
                
                if captured:
                    self.episode_data['capture_times'].append(ep_info['l'])
        
        return True


def make_env(rank, render_mode=None, config=None):
    """Create environment with unique seed"""
    def _init():
        env = DronePursuitEnv(render_mode=render_mode, config=config)
        env.reset(seed=rank)
        env = Monitor(env)
        return env
    return _init


def create_optimized_model(args, env):
    """
    Optimized model configuration for fast learning
    """
    
    # Larger networks for better function approximation
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=torch.nn.ReLU,
    )
    
    if args.algo == 'ppo':
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            n_steps=4096,  # Increased for more stable updates
            batch_size=128,  # Larger batches
            n_epochs=15,  # More epochs per update
            gamma=0.995,  # Slightly higher discount for long-term planning
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.005,  # Lower entropy for more focused policy
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            verbose=1,
            device=args.device
        )
    
    elif args.algo == 'sac':
        off_policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]),
            activation_fn=torch.nn.ReLU
        )
        
        return SAC(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            buffer_size=200_000,  # Larger replay buffer
            learning_starts=5000,
            batch_size=512,  # Larger batches for off-policy
            gamma=0.995,
            tau=0.005,  # Softer target network updates
            ent_coef='auto_0.1',  # Start with lower entropy
            policy_kwargs=off_policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            verbose=1,
            device=args.device
        )
    
    elif args.algo == 'td3':
        off_policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]),
            activation_fn=torch.nn.ReLU
        )
        
        return TD3(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            buffer_size=200_000,
            learning_starts=5000,
            batch_size=512,
            gamma=0.995,
            tau=0.005,
            policy_kwargs=off_policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            verbose=1,
            device=args.device
        )
    
    raise ValueError(f"Algorithm {args.algo} not supported.")


def main():
    parser = argparse.ArgumentParser(description='Optimized Drone Pursuit Training')
    
    # Algorithm
    parser.add_argument('--algo', type=str, default='ppo', 
                       choices=['ppo', 'sac', 'td3'])
    
    # Training params - Increased defaults
    parser.add_argument('--total-timesteps', type=int, default=2_000_000,
                       help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=8,  # More parallel envs
                       help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto')
    
    # Environment config - Start easy
    parser.add_argument('--start-obstacles', type=int, default=3,
                       help='Starting number of obstacles')
    parser.add_argument('--start-jam-radius', type=int, default=120,
                       help='Starting jam zone radius')
    parser.add_argument('--use-adaptive-curriculum', action='store_true', default=True,
                       help='Use adaptive curriculum learning')
    
    # Logging
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--tensorboard-log', type=str, default='./logs/tensorboard')
    parser.add_argument('--save-dir', type=str, default='./models')
    parser.add_argument('--save-freq', type=int, default=25000)
    parser.add_argument('--eval-freq', type=int, default=10000)
    
    args = parser.parse_args()
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.algo}_optimized_{timestamp}"
    
    # Create directories
    os.makedirs(args.tensorboard_log, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Environment config - start easy
    args.config = {
        'num_obstacles': args.start_obstacles,
        'jam_zone_radius': args.start_jam_radius,
        'random_start': True
    }
    
    print(f"\n{'='*70}")
    print(f"🚁 OPTIMIZED TRAINING - Focus on Speed")
    print(f"{'='*70}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Starting difficulty: {args.config}")
    print(f"Adaptive curriculum: {args.use_adaptive_curriculum}")
    print(f"{'='*70}\n")
    
    # Create vectorized environments
    if args.num_envs > 1:
        train_env = SubprocVecEnv([make_env(i, config=args.config) for i in range(args.num_envs)])
    else:
        train_env = DummyVecEnv([make_env(0, config=args.config)])
    
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(999, config=args.config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, args.experiment_name),
        name_prefix=args.algo,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, args.experiment_name, 'best'),
        log_path=os.path.join(args.save_dir, args.experiment_name, 'eval'),
        eval_freq=args.eval_freq,
        n_eval_episodes=20,  # More eval episodes
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Enhanced metrics
    metrics_callback = EnhancedMetricsCallback(verbose=1)
    callbacks.append(metrics_callback)
    
    # Adaptive curriculum
    if args.use_adaptive_curriculum:
        curriculum_callback = AdaptiveCurriculumCallback(
            train_env, 
            window_size=100,
            target_success_rate=0.70,
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Create optimized model
    model = create_optimized_model(args, train_env)
    
    # Train
    print("\n🎯 Starting optimized training...\n")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            tb_log_name=args.experiment_name,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(args.save_dir, args.experiment_name, 'final_model')
        model.save(final_path)
        train_env.save(os.path.join(args.save_dir, args.experiment_name, 'vec_normalize.pkl'))
        
        print(f"\n{'='*70}")
        print(f"✅ Training completed!")
        print(f"Model saved to: {final_path}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        interrupt_path = os.path.join(args.save_dir, args.experiment_name, 'interrupted_model')
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()