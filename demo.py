import argparse
import os
from stable_baselines3 import PPO, SAC, TD3
import pygame
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from Enviroment import DronePursuitEnv

def main():

    MODEL_PATH = r"D:\REL\model_best\medium_SAC.zip"

    parser = argparse.ArgumentParser(description="Evaluate a trained Drone Pursuit agent.")
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac", "td3"], help="RL algorithm used for training.")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"], help="Difficulty level for evaluation.")
    parser.add_argument("--n-obstacles-hard", type=int, default=6, help="Max number of obstacles for observation space consistency.")
    args = parser.parse_args()

    print(f"---Starting Fast Agent Evaluation (No Rendering) ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Difficulty: {args.difficulty.upper()}")
    print(f"Episodes: {args.n_episodes}")

    model_dir = os.path.dirname(MODEL_PATH)
    vecnormalize_path = os.path.join(model_dir, "medium_SAC.pkl")
    if not os.path.exists(vecnormalize_path):
        vecnormalize_path = os.path.join(os.path.dirname(model_dir), "medium_SAC.pkl")

    base_env = DummyVecEnv([lambda: DronePursuitEnv(render_mode='human', 
                                                    difficulty=args.difficulty,
                                                    n_obstacles_hard=args.n_obstacles_hard)])
    
    if os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = False
        env.norm_reward = False
    else:
        print("VecNormalize file not found. Running without normalization.")
        env = base_env
    
    algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
    model_class = algo_map.get(args.algo.lower())
    
    if not model_class:
        raise ValueError(f"Algorithm {args.algo} is not supported.")
        
    print("Loading model...")
    model = model_class.load(MODEL_PATH, env=env)
    print("Model loaded successfully. Starting evaluation...")
        
    total_captures = 0
    total_jams = 0
    total_collisions = 0
    total_timeouts = 0
    
    for i in range(args.n_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info_list = env.step(action)
            
            terminated = done[0]
            truncated = info_list[0].get("TimeLimit.truncated", False)
            info = info_list[0]
        if info.get("captured"):
            print("Result: TARGET CAPTURED")
            total_captures += 1
        elif info.get("jammed"):
            print("Result: JAMMED")
            total_jams += 1
        elif info.get("collision"):
            print("Result: COLLISION")
            total_collisions += 1
        elif truncated:
            print("Result: TIMEOUT")
            total_timeouts += 1
            
    env.close()
    
    print("\n--- OVERALL EVALUATION RESULTS ---")
    print(f"Total episodes: {args.n_episodes}")
    if args.n_episodes > 0:
        print(f"Capture Rate: {total_captures / args.n_episodes * 100:.1f}%")
        print(f"Collision Rate: {total_collisions / args.n_episodes * 100:.1f}%")
        print(f"Jammed Rate: {total_jams / args.n_episodes * 100:.1f}%")
        print(f"Timeout Rate: {total_timeouts / args.n_episodes * 100:.1f}%")

if __name__ == "__main__":
    main()