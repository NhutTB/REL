"""
Advanced Speed Analysis Tool
Measures and visualizes capture speed performance
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_pursuit_env import DronePursuitEnv


class SpeedBenchmark:
    """Comprehensive speed benchmarking"""
    
    def __init__(self, model, env, n_episodes=50):
        self.model = model
        self.env = env
        self.n_episodes = n_episodes
        self.results = defaultdict(list)
        
    def run_benchmark(self, difficulty='medium'):
        """Run comprehensive speed benchmark"""
        print(f"\n{'='*70}")
        print(f"🏁 SPEED BENCHMARK - {difficulty.upper()} Difficulty")
        print(f"{'='*70}\n")
        
        difficulty_configs = {
            'easy': {'num_obstacles': 3, 'jam_zone_radius': 120},
            'medium': {'num_obstacles': 8, 'jam_zone_radius': 80},
            'hard': {'num_obstacles': 15, 'jam_zone_radius': 60},
        }
        
        config = difficulty_configs.get(difficulty, difficulty_configs['medium'])
        
        # Update environment
        for e in self.env.envs:
            e.config.update(config)
        
        capture_times = []
        episode_rewards = []
        total_steps = []
        success_count = 0
        
        for episode in range(self.n_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0
            
            start_time = time.time()
            
            while not done and steps < 2000:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                steps += 1
                episode_reward += reward[0]
                
                if info[0].get('captured', False):
                    capture_times.append(steps)
                    success_count += 1
                    break
            
            elapsed = time.time() - start_time
            total_steps.append(steps)
            episode_rewards.append(episode_reward)
            
            status = "✓ CAPTURED" if info[0].get('captured', False) else "✗ FAILED"
            print(f"Episode {episode+1:3d}: {status:12s} | Steps: {steps:4d} | Reward: {episode_reward:7.2f} | Time: {elapsed:.2f}s")
        
        # Calculate statistics
        success_rate = success_count / self.n_episodes
        
        if capture_times:
            avg_capture_time = np.mean(capture_times)
            median_capture_time = np.median(capture_times)
            min_capture_time = np.min(capture_times)
            std_capture_time = np.std(capture_times)
            
            # Speed score (lower is better)
            speed_score = avg_capture_time / success_rate if success_rate > 0 else float('inf')
        else:
            avg_capture_time = median_capture_time = min_capture_time = std_capture_time = None
            speed_score = float('inf')
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK RESULTS - {difficulty.upper()}")
        print(f"{'='*70}")
        print(f"Success Rate:       {success_rate*100:6.2f}% ({success_count}/{self.n_episodes})")
        
        if capture_times:
            print(f"\n🏃 SPEED METRICS:")
            print(f"  Average Time:     {avg_capture_time:6.1f} steps")
            print(f"  Median Time:      {median_capture_time:6.1f} steps")
            print(f"  Best Time:        {min_capture_time:6.0f} steps ⭐")
            print(f"  Std Dev:          {std_capture_time:6.1f} steps")
            print(f"  Speed Score:      {speed_score:6.1f} (lower = better)")
            
            # Performance rating
            if avg_capture_time < 300:
                rating = "🏆 EXCELLENT"
            elif avg_capture_time < 500:
                rating = "🥇 VERY GOOD"
            elif avg_capture_time < 800:
                rating = "🥈 GOOD"
            elif avg_capture_time < 1200:
                rating = "🥉 FAIR"
            else:
                rating = "📊 NEEDS IMPROVEMENT"
            
            print(f"\n  Performance:      {rating}")
        else:
            print(f"\n❌ No successful captures!")
        
        print(f"{'='*70}\n")
        
        # Store results
        self.results[difficulty] = {
            'capture_times': capture_times,
            'success_rate': success_rate,
            'avg_time': avg_capture_time if capture_times else None,
            'min_time': min_capture_time if capture_times else None,
            'speed_score': speed_score
        }
        
        return self.results[difficulty]
    
    def compare_difficulties(self):
        """Compare performance across difficulties"""
        print(f"\n{'='*70}")
        print(f"📈 DIFFICULTY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Difficulty':<15} {'Success %':>10} {'Avg Time':>10} {'Best Time':>10} {'Score':>10}")
        print(f"{'-'*70}")
        
        for diff in ['easy', 'medium', 'hard']:
            if diff in self.results:
                r = self.results[diff]
                print(f"{diff.capitalize():<15} {r['success_rate']*100:>9.1f}% "
                      f"{r['avg_time'] if r['avg_time'] else 'N/A':>10} "
                      f"{r['min_time'] if r['min_time'] else 'N/A':>10} "
                      f"{r['speed_score']:>10.1f}")
        
        print(f"{'='*70}\n")
    
    def plot_results(self, save_path='speed_analysis.png'):
        """Visualize speed performance"""
        if not self.results:
            print("No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Drone Pursuit Speed Analysis', fontsize=16, fontweight='bold')
        
        difficulties = [d for d in ['easy', 'medium', 'hard'] if d in self.results]
        
        # Plot 1: Capture Time Distribution
        ax = axes[0, 0]
        for diff in difficulties:
            times = self.results[diff]['capture_times']
            if times:
                ax.hist(times, bins=20, alpha=0.6, label=diff.capitalize())
        ax.set_xlabel('Capture Time (steps)')
        ax.set_ylabel('Frequency')
        ax.set_title('Capture Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success Rate by Difficulty
        ax = axes[0, 1]
        success_rates = [self.results[d]['success_rate']*100 for d in difficulties]
        bars = ax.bar(difficulties, success_rates, color=['green', 'orange', 'red'])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Difficulty')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Average Capture Time
        ax = axes[1, 0]
        avg_times = [self.results[d]['avg_time'] for d in difficulties if self.results[d]['avg_time']]
        diff_labels = [d for d in difficulties if self.results[d]['avg_time']]
        if avg_times:
            bars = ax.bar(diff_labels, avg_times, color=['lightgreen', 'gold', 'salmon'])
            ax.set_ylabel('Average Capture Time (steps)')
            ax.set_title('Average Capture Time by Difficulty')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, time in zip(bars, avg_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.0f}', ha='center', va='bottom')
        
        # Plot 4: Speed Score (Combined Metric)
        ax = axes[1, 1]
        speed_scores = [self.results[d]['speed_score'] for d in difficulties 
                       if self.results[d]['speed_score'] != float('inf')]
        score_labels = [d for d in difficulties 
                       if self.results[d]['speed_score'] != float('inf')]
        if speed_scores:
            bars = ax.bar(score_labels, speed_scores, color=['lightblue', 'steelblue', 'navy'])
            ax.set_ylabel('Speed Score (lower = better)')
            ax.set_title('Overall Speed Score')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, score in zip(bars, speed_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
        plt.show()
    
    def analyze_trajectory_efficiency(self, n_test_episodes=5):
        """Analyze path efficiency"""
        print(f"\n{'='*70}")
        print(f"🛤️  TRAJECTORY EFFICIENCY ANALYSIS")
        print(f"{'='*70}\n")
        
        path_lengths = []
        direct_distances = []
        
        for episode in range(n_test_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            
            # Record start positions
            chaser_start = self.env.envs[0].chaser.pos.copy()
            evader_start = self.env.envs[0].evader.pos.copy()
            direct_distance = np.linalg.norm(evader_start - chaser_start)
            
            total_path_length = 0
            prev_pos = chaser_start.copy()
            
            while not done and steps < 2000:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                curr_pos = self.env.envs[0].chaser.pos.copy()
                total_path_length += np.linalg.norm(curr_pos - prev_pos)
                prev_pos = curr_pos
                
                steps += 1
                
                if info[0].get('captured', False):
                    break
            
            if info[0].get('captured', False):
                path_lengths.append(total_path_length)
                direct_distances.append(direct_distance)
                efficiency = (direct_distance / total_path_length) * 100
                
                print(f"Episode {episode+1}: "
                      f"Path: {total_path_length:.1f} | "
                      f"Direct: {direct_distance:.1f} | "
                      f"Efficiency: {efficiency:.1f}%")
        
        if path_lengths:
            avg_efficiency = np.mean([d/p for d, p in zip(direct_distances, path_lengths)]) * 100
            print(f"\n📊 Average Path Efficiency: {avg_efficiency:.1f}%")
            
            if avg_efficiency > 80:
                rating = "🏆 Excellent (Nearly optimal paths)"
            elif avg_efficiency > 60:
                rating = "🥇 Very Good (Efficient pursuit)"
            elif avg_efficiency > 40:
                rating = "🥈 Good (Some detours)"
            else:
                rating = "📊 Needs improvement (Inefficient paths)"
            
            print(f"Rating: {rating}")
        
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Speed Analysis for Drone Pursuit')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3'])
    parser.add_argument('--n-episodes', type=int, default=50,
                       help='Episodes per difficulty')
    parser.add_argument('--vec-normalize-path', type=str, default=None,
                       help='Path to VecNormalize stats')
    parser.add_argument('--test-all-difficulties', action='store_true',
                       help='Test on all difficulty levels')
    parser.add_argument('--analyze-trajectories', action='store_true',
                       help='Analyze path efficiency')
    parser.add_argument('--save-plot', type=str, default='speed_analysis.png',
                       help='Save plot to file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚁 DRONE PURSUIT SPEED ANALYZER")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"{'='*70}\n")
    
    # Load model
    if args.algo == 'ppo':
        model = PPO.load(args.model_path)
    elif args.algo == 'sac':
        model = SAC.load(args.model_path)
    elif args.algo == 'td3':
        model = TD3.load(args.model_path)
    
    # Create environment
    env = DummyVecEnv([lambda: DronePursuitEnv(render_mode=None)])
    
    if args.vec_normalize_path:
        env = VecNormalize.load(args.vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Create benchmark
    benchmark = SpeedBenchmark(model, env, n_episodes=args.n_episodes)
    
    try:
        # Run benchmarks
        if args.test_all_difficulties:
            for difficulty in ['easy', 'medium', 'hard']:
                benchmark.run_benchmark(difficulty)
            benchmark.compare_difficulties()
        else:
            benchmark.run_benchmark('medium')
        
        # Trajectory analysis
        if args.analyze_trajectories:
            benchmark.analyze_trajectory_efficiency()
        
        # Plot results
        if args.test_all_difficulties:
            benchmark.plot_results(args.save_plot)
        
    finally:
        env.close()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()