# Training.py
import os
import time
import wandb
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecNormalize

import torch
# Import môi trường
from Enviroment import DronePursuitEnv


# ========== CALLBACK ROLLOUT & VIDEO ========== #
class RolloutCallback(BaseCallback):
    """
    Callback để chạy một lượt rollout (đánh giá) định kỳ,
    quay video và log lên Wandb.
    """
    def __init__(self, eval_freq: int, n_obstacles: int = 4, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        # Tạo một môi trường riêng chỉ để đánh giá và render
        self.eval_env = DronePursuitEnv(render_mode="rgb_array", n_obstacles=n_obstacles)

    def _on_step(self) -> bool:
        # Chạy rollout sau mỗi `eval_freq` bước
        if self.num_timesteps % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n[RolloutCallback] Running rollout at {self.num_timesteps} timesteps...")

            frames = []
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                # Lấy frame từ môi trường và thêm vào list
                frame = self.eval_env.render()
                frames.append(frame)

            if self.verbose > 0:
                print(f"[RolloutCallback] Rollout finished. Reward: {episode_reward:.2f}, Length: {episode_length}")

            # Log video lên Wandb
            # Chuyển đổi list các frame (T, H, W, C) thành (T, C, H, W) mà wandb yêu cầu
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

# ========== CALLBACK TÙY CHỈNH ========== #
class EpisodeStatsCallback(BaseCallback):
    """
    Callback tùy chỉnh để log thêm số liệu thật:
    - Tỷ lệ bắt được (capture)
    - Tỷ lệ va chạm (collision)
    - Tỷ lệ bị jammed
    - Tỷ lệ hết giờ (timeout)
    """
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.captures = 0
        self.collisions = 0
        self.jammeds = 0
        self.timeouts = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "captured" in info:
                self.episode_count += 1
                if info["captured"]:
                    self.captures += 1
                elif info["collision"]:
                    self.collisions += 1
                elif info["jammed"]:
                    self.jammeds += 1
                else:
                    self.timeouts += 1

        if self.episode_count > 0 and self.episode_count % 20 == 0:
            wandb.log({
                "stats/capture_rate": self.captures / self.episode_count,
                "stats/collision_rate": self.collisions / self.episode_count,
                "stats/jammed_rate": self.jammeds / self.episode_count,
                "stats/timeout_rate": self.timeouts / self.episode_count,
            })
        return True


# ========== TẠO MÔI TRƯỜNG ========== #
def make_env(rank, seed=0):
    """
    Hàm tiện ích tạo môi trường song song.
    """
    def _init():
        env = DronePursuitEnv(render_mode=None, n_obstacles=4)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ========== HUẤN LUYỆN ========== #
if __name__ == "__main__":
    ALGO = "PPO"  # Chọn: PPO / SAC / TD3
    TOTAL_TIMESTEPS = 1_000_000
    NUM_ENVS = 16  # Tăng số môi trường song song để thu thập dữ liệu nhanh hơn
    RUN_NAME = f"{ALGO}_{int(time.time())}"

    # 1️⃣ Khởi tạo Weights & Biases
    run = wandb.init(
        project="DronePursuit_Project",
        name=RUN_NAME,
        config={
            "algorithm": ALGO,
            "timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "policy": "MlpPolicy",
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # 2️⃣ Tạo môi trường song song
    print(f"🧠 Đang tạo {NUM_ENVS} môi trường song song...")
    env = VecNormalize(
        SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]),
        norm_obs=True,      # Chuẩn hóa observation
        norm_reward=True,   # Chuẩn hóa reward
        clip_reward=10.0    # Cắt bớt reward quá lớn (ví dụ: +500)
    )

    # 3️⃣ Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=f"./models/{RUN_NAME}/",
        name_prefix=f"{ALGO}_drone"
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=10000,
        model_save_path=f"models/wandb/{run.id}",
        verbose=2,
    )

    stats_callback = EpisodeStatsCallback()

    # Callback mới để chạy rollout và quay video
    # Sẽ chạy mỗi 50,000 timesteps
    rollout_callback = RolloutCallback(eval_freq=max(50_000 // NUM_ENVS, 1))

    # 4️⃣ Khởi tạo Model
    print(f"🚀 Đang khởi tạo mô hình {ALGO} ...")
    if ALGO == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_steps=2048,
            batch_size=512,  # Tăng batch size để tận dụng GPU tốt hơn
            n_epochs=10,     # Bắt GPU học nhiều hơn trên mỗi lô dữ liệu
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=1.0,
        )
    elif ALGO == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device="cuda",
            buffer_size=300_000,
            learning_rate=7e-4,
            batch_size=256,
            gamma=0.99,
        )
    elif ALGO == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device="cuda",
            buffer_size=300_000,
            learning_rate=1e-3,
            batch_size=128,
            gamma=0.99,
        )
    else:
        raise ValueError(f"Thuật toán {ALGO} không được hỗ trợ.")

    # 5️⃣ Huấn luyện
    print(f"🏁 Bắt đầu huấn luyện {ALGO}")
    print(f"Theo dõi tại: {run.get_url()}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, wandb_callback, stats_callback, rollout_callback]
    )

    # 6️⃣ Lưu mô hình cuối
    os.makedirs(f"models/{RUN_NAME}", exist_ok=True)
    model.save(f"models/{RUN_NAME}/final_model.zip")
    print(f"✅ Đã lưu model cuối cùng tại: models/{RUN_NAME}/final_model.zip")

    env.close()
    run.finish()
