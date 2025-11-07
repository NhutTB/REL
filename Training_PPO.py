import os
import time
import wandb
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecNormalize
import torch
import torch.nn as nn
from Environment import DronePursuitEnv

class RolloutCallback(BaseCallback):
    def __init__(self, eval_freq: int, difficulty: str = "hard", n_obstacles_hard: int = 6, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = DronePursuitEnv(render_mode="rgb_array", 
                                      difficulty=difficulty, 
                                      n_obstacles_hard=n_obstacles_hard)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n[RolloutCallback] Running rollout at {self.num_timesteps} timesteps...")

            frames = []
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            vec_env = self.model.get_vec_normalize_env()
            if vec_env is None:
                eval_obs = obs
            else:
                eval_obs = vec_env.normalize_obs(obs)


            while not (done or truncated):
                action, _ = self.model.predict(eval_obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                if vec_env is not None:
                    eval_obs = vec_env.normalize_obs(obs)
                else:
                    eval_obs = obs

                episode_reward += reward
                episode_length += 1
                frame = self.eval_env.render()
                frames.append(frame)

            if self.verbose > 0:
                print(f"[RolloutCallback] Rollout finished. Reward: {episode_reward:.2f}, Length: {episode_length}")

            video_frames = np.array(frames).transpose(0, 3, 1, 2)
            wandb.log({
                "rollout/video": wandb.Video(video_frames, fps=60, format="mp4"),
                "rollout/episode_reward": episode_reward,
                "rollout/episode_length": episode_length,
                "rollout/captured": 1 if info.get("captured") else 0,
                "rollout/jammed": 1 if info.get("jammed") else 0,
                "rollout/collision": 1 if info.get("collision") else 0,
            })
        return True


def make_env(difficulty="hard", n_obstacles_hard=6, seed=0, rank=0):
    def _init():
        env = DronePursuitEnv(render_mode=None, 
                              difficulty=difficulty, 
                              n_obstacles_hard=n_obstacles_hard)
        env.reset(seed=seed + rank) 
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    ALGO = "PPO"
    TOTAL_TIMESTEPS = 5_000_000
    
    DIFFICULTY_TO_TRAIN = "medium"
    
    N_OBSTACLES_HARD_MODE = 6
    NUM_ENVS = 16

    RUN_NAME = f"{ALGO}_Fast_{DIFFICULTY_TO_TRAIN}_{int(time.time())}"

    run = wandb.init(
        project="DronePursuit_Project",
        name=RUN_NAME,
        config={
            "algorithm": ALGO,
            "timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "policy": "MlpPolicy",
            "difficulty": DIFFICULTY_TO_TRAIN,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    print(f"Creating {NUM_ENVS} environments (Difficulty: {DIFFICULTY_TO_TRAIN})...")
    env = VecNormalize(
        SubprocVecEnv([make_env(difficulty=DIFFICULTY_TO_TRAIN, 
                                n_obstacles_hard=N_OBSTACLES_HARD_MODE, 
                                rank=i) for i in range(NUM_ENVS)]),
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path=f"./models/{RUN_NAME}/",
        name_prefix=f"{ALGO}_drone"
    )
    wandb_callback = WandbCallback(gradient_save_freq=10000, model_save_path=f"models/wandb/{run.id}", verbose=2)
    rollout_callback = RolloutCallback(eval_freq=max(50_000 // NUM_ENVS, 1), 
                                     difficulty=DIFFICULTY_TO_TRAIN,
                                     n_obstacles_hard=N_OBSTACLES_HARD_MODE)

    print(f"Initializing {ALGO} model...")
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print(f"🚁 Starting training for {ALGO} (Difficulty: {DIFFICULTY_TO_TRAIN})")
    print(f"Track progress at: {run.url}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, wandb_callback, rollout_callback]
    )

    os.makedirs(f"models/{RUN_NAME}", exist_ok=True)
    model.save(f"models/{RUN_NAME}/final_model.zip")
    env.save(f"models/{RUN_NAME}/vecnormalize.pkl")
    print(f"Final model saved at: models/{RUN_NAME}/final_model.zip")
    print(f"VecNormalize stats saved at: models/{RUN_NAME}/vecnormalize.pkl")

    env.close()

    run.finish()
