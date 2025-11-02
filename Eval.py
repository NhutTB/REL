"""
Evaluation and visualization script for trained agents
"""

import argparse
import numpy as np
import time
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_pursuit_env import DronePursuitEnv


def evaluate_agent(model, env, n_episodes=10, render=True, deterministic=True):
    """
    Evaluate trained agent
    """
    episode_rewards = []
    episode_lengths = []
    captures = 0
    jams = 0
    collisions = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.01)  # Slow down for visualization
            
            if terminated or truncated:
                if info[0].get('captured', False):
                    captures += 1
                    print(f"[SUCCESS] Episode {episode + 1}: CAPTURED in {episode_length} steps!")
                elif info[0].get('jammed', False):
                    jams += 1
                    print(f"[FAIL] Episode {episode + 1}: JAMMED at step {episode_length}")
                elif info[0].get('collision', False):
                    collisions += 1
                    print(f"[FAIL] Episode {episode + 1}: COLLISION at step {episode_length}")
                else:
                    print(f"[TIMEOUT] Episode {episode + 1}: Timeout after {episode_length} steps")
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Capture rate: {captures/n_episodes*100:.1f}% ({captures}/{n_episodes})")
    print(f"Jam rate: {jams/n_episodes*100:.1f}% ({jams}/{n_episodes})")
    print(f"Collision rate: {collisions/n_episodes*100:.1f}% ({collisions}/{n_episodes})")
    print(f"{'='*60}\n")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'capture_rate': captures/n_episodes,
        'jam_rate': jams/n_episodes,
        'collision_rate': collisions/n_episodes
    }


def compare_random_vs_trained(trained_model, env, n_episodes=5):
    """
    Compare trained agent vs random agent
    """
    print("\n" + "="*60)
    print("Comparison: Random Agent vs Trained Agent")
    print("="*60 + "\n")
    
    # Evaluate random agent
    print("Testing RANDOM agent...")
    random_stats = []
    for episode in range(n_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_length = 0
        captured = False
        
        while not (terminated or truncated) and episode_length < 2000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if info[0].get('captured', False):
                captured = True
        
        random_stats.append({'length': episode_length, 'captured': captured})
        status = "[SUCCESS] CAPTURED" if captured else "[FAIL] FAILED"
        print(f"  Episode {episode+1}: {status} (steps: {episode_length})")
    
    # Evaluate trained agent
    print("\nTesting TRAINED agent...")
    trained_stats = []
    for episode in range(n_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_length = 0
        captured = False
        
        while not (terminated or truncated) and episode_length < 2000:
            action, _states = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if info[0].get('captured', False):
                captured = True
        
        trained_stats.append({'length': episode_length, 'captured': captured})
        status = "[SUCCESS] CAPTURED" if captured else "[FAIL] FAILED"
        print(f"  Episode {episode+1}: {status} (steps: {episode_length})")
    
    # Compare
    random_captures = sum(1 for s in random_stats if s['captured'])
    trained_captures = sum(1 for s in trained_stats if s['captured'])
    
    random_avg_length = np.mean([s['length'] for s in random_stats if s['captured']]) if random_captures > 0 else 2000
    trained_avg_length = np.mean([s['length'] for s in trained_stats if s['captured']]) if trained_captures > 0 else 2000
    
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"Random Agent:")
    print(f"  Capture rate: {random_captures}/{n_episodes} ({random_captures/n_episodes*100:.1f}%)")
    print(f"  Avg capture time: {random_avg_length:.1f} steps")
    print(f"\nTrained Agent:")
    print(f"  Capture rate: {trained_captures}/{n_episodes} ({trained_captures/n_episodes*100:.1f}%)")
    print(f"  Avg capture time: {trained_avg_length:.1f} steps")
    print(f"\nImprovement:")
    print(f"  Capture rate: +{(trained_captures-random_captures)/n_episodes*100:.1f}%")
    if random_captures > 0 and trained_captures > 0:
        print(f"  Speed improvement: {(random_avg_length-trained_avg_length)/random_avg_length*100:.1f}% faster")
    print(f"{'='*60}\n")


def stress_test(model, env, n_episodes=20):
    """
    Stress test with various difficulty levels
    """
    print("\n" + "="*60)
    print("Stress Test: Multiple Difficulty Levels")
    print("="*60 + "\n")
    
    configs = [
        {'name': 'Easy', 'num_obstacles': 3, 'jam_zone_radius': 100},
        {'name': 'Medium', 'num_obstacles': 8, 'jam_zone_radius': 80},
        {'name': 'Hard', 'num_obstacles': 15, 'jam_zone_radius': 60},
        {'name': 'Expert', 'num_obstacles': 20, 'jam_zone_radius': 50},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} difficulty...")
        print(f"  Obstacles: {config['num_obstacles']}, Jam radius: {config['jam_zone_radius']}")
        
        # Update environment config
        for e in env.envs:
            e.config.update({'num_obstacles': config['num_obstacles'], 
                           'jam_zone_radius': config['jam_zone_radius']})
        
        captures = 0
        jams = 0
        avg_length = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            terminated = False
            truncated = False
            episode_length = 0
            
            while not (terminated or truncated) and episode_length < 2000:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_length += 1
                
                if terminated or truncated:
                    if info[0].get('captured', False):
                        captures += 1
                        avg_length.append(episode_length)
                    elif info[0].get('jammed', False):
                        jams += 1
        
        avg = np.mean(avg_length) if avg_length else 2000
        results.append({
            'difficulty': config['name'],
            'capture_rate': captures/n_episodes,
            'jam_rate': jams/n_episodes,
            'avg_capture_time': avg
        })
        
        print(f"  Capture: {captures}/{n_episodes} ({captures/n_episodes*100:.1f}%)")
        print(f"  Jammed: {jams}/{n_episodes} ({jams/n_episodes*100:.1f}%)")
        print(f"  Avg time: {avg:.1f} steps")
    
    # Summary
    print(f"\n{'='*60}")
    print("Stress Test Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['difficulty']:10s} | Capture: {r['capture_rate']*100:5.1f}% | "
              f"Jammed: {r['jam_rate']*100:5.1f}% | Avg time: {r['avg_capture_time']:6.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Drone Pursuit Agent')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3'],
                       help='Algorithm used for training')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with random agent')
    parser.add_argument('--stress-test', action='store_true',
                       help='Run stress test with multiple difficulties')
    parser.add_argument('--vec-normalize-path', type=str, default=None,
                       help='Path to VecNormalize stats')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    print(f"Model path: {args.model_path}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    if args.algo == 'ppo':
        model = PPO.load(args.model_path)
    elif args.algo == 'sac':
        model = SAC.load(args.model_path)
    elif args.algo == 'td3':
        model = TD3.load(args.model_path)
    
    # Create environment
    render_mode = "human" if args.render else None
    env = DummyVecEnv([lambda: DronePursuitEnv(render_mode=render_mode)])
    
    # Load VecNormalize stats if available
    if args.vec_normalize_path:
        env = VecNormalize.load(args.vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    try:
        # Standard evaluation
        print("Running standard evaluation...")
        evaluate_agent(model, env, 
                      n_episodes=args.n_episodes, 
                      render=args.render,
                      deterministic=args.deterministic)
        
        # Comparison with random
        if args.compare:
            compare_random_vs_trained(model, env, n_episodes=5)
        
        # Stress test
        if args.stress_test:
            stress_test(model, env, n_episodes=20)
        
    finally:
        env.close()
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()