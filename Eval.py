import argparse
import os
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from tqdm import tqdm
from Enviroment import DronePursuitEnv

def safe_extract_info(done, info):
    if isinstance(done, (list, tuple)):
        terminated = bool(done[0])
    else:
        try:
            terminated = bool(done)
        except Exception:
            terminated = False

    info_dict = {}
    truncated = False
    if isinstance(info, (list, tuple)) and len(info) > 0:
        info_dict = info[0] or {}
    elif isinstance(info, dict):
        info_dict = info
    else:
        info_dict = {}

    truncated = bool(info_dict.get("TimeLimit.truncated", False))

    return terminated, truncated, info_dict

def main():

    MODEL_PATH = r"D:\REL\model_best\medium_SAC.zip"

    parser = argparse.ArgumentParser(description="Evaluate a trained Drone Pursuit agent.")
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac", "td3"],
                        help="RL algorithm used for training.")
    parser.add_argument("--n-episodes", type=int, default=1000,
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--difficulty", type=str, default="hard",
                        choices=["easy", "medium", "hard"], help="Difficulty level for evaluation.")
    parser.add_argument("--n-obstacles-hard", type=int, default=6,
                        help="Max number of obstacles for observation space consistency.")
    args = parser.parse_args()

    print(f"--- Starting Fast Agent Evaluation (No Rendering) ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Difficulty: {args.difficulty.upper()}")
    print(f"Episodes: {args.n_episodes}")

    model_dir = os.path.dirname(MODEL_PATH)
    vecnormalize_path = os.path.join(model_dir, "medium_SAC.pkl")
    if not os.path.exists(vecnormalize_path):
        vecnormalize_path = os.path.join(os.path.dirname(model_dir), "medium_SAC.pkl")

    base_env = DummyVecEnv([lambda: DronePursuitEnv(render_mode=None,
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
    total_other = 0

    for i in tqdm(range(args.n_episodes), desc="Evaluating Episodes"):
        obs = env.reset()
        terminated = False
        truncated = False
        info = {}

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            terminated, truncated, info_dict = safe_extract_info(done, info)

            if terminated or truncated:
                info = info_dict
                break

        if info.get("TimeLimit.truncated", False):
            total_timeouts += 1

        elif info.get("captured", False):
            total_captures += 1

        elif info.get("jammed", False):
            total_jams += 1

        elif info.get("collision", False):
            total_collisions += 1

        else:
            expected_keys = {
                'distance', 'captured', 'jammed', 'collision',
                'evader_in_sensor', 'TimeLimit.truncated', 'terminal_observation'
            }
            if expected_keys.issubset(set(info.keys())):
                total_collisions += 1
            else:
                total_other += 1

    env.close()

    print("\n--- OVERALL EVALUATION RESULTS ---")
    print(f"Total episodes: {args.n_episodes}")
    if args.n_episodes > 0:
        def pct(x):
            return x / args.n_episodes * 100.0

        print(f"Capture Rate : {pct(total_captures):5.1f}% ({total_captures}/{args.n_episodes})")
        print(f"Collision Rate: {pct(total_collisions):5.1f}% ({total_collisions}/{args.n_episodes})")
        print(f"Jammed Rate  : {pct(total_jams):5.1f}% ({total_jams}/{args.n_episodes})")
        print(f"Timeout Rate : {pct(total_timeouts):5.1f}% ({total_timeouts}/{args.n_episodes})")
        print(f"Other Rate   : {pct(total_other):5.1f}% ({total_other}/{args.n_episodes})")
        print(f"SUM          : {pct(total_captures + total_collisions + total_jams + total_timeouts + total_other):5.1f}%")

if __name__ == "__main__":
    main()
