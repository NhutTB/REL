# Training.py
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import time
import os

# Import môi trường
from Enviroment import DronePursuitEnv 

def make_env(rank, seed=0):
    """
    Hàm tiện ích để tạo môi trường cho VecEnv.
    """
    def _init():
        # Quan trọng: KHÔNG render khi huấn luyện
        env = DronePursuitEnv(render_mode=None) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # --- CẤU HÌNH HUẤN LUYỆN ---
    ALGO = "PPO"  # Chọn 1 trong 3: "PPO", "SAC", "TD3"
    TOTAL_TIMESTEPS = 2_000_000  # 2 triệu bước
    NUM_ENVS = 8  # Số môi trường chạy song song
    RUN_NAME = f"{ALGO}_{int(time.time())}"
    # -----------------------------

    # 1. Khởi tạo Weights & Biases (wandb)
    run = wandb.init(
        project="DronePursuit_Project",  # Tên dự án trên wandb
        name=RUN_NAME,
        config={
            "algorithm": ALGO,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "policy": "MlpPolicy",
        },
        sync_tensorboard=True,  # Tự động đồng bộ log TensorBoard
        monitor_gym=True,       # Tự động theo dõi video (nếu có)
        save_code=True,         # Lưu code của file này lên wandb
    )

    # 2. Tạo môi trường VecEnv (chạy song song)
    print(f"Đang tạo {NUM_ENVS} môi trường song song...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # 3. Tạo Callbacks
    # Callback để lưu model checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=f"./models/{RUN_NAME}/",
        name_prefix=f"{ALGO}_drone"
    )
    
    # Callback để log lên wandb
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"models/wandb/{run.id}",
        verbose=2,
    )

    # 4. Chọn và khởi tạo Model
    print(f"Đang khởi tạo model {ALGO}...")
    if ALGO == "PPO":
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=f"runs/{run.id}",
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )
    elif ALGO == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            buffer_size=300_000,
            learning_rate=7e-4,
            batch_size=256,
            gamma=0.99
        )
    elif ALGO == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            buffer_size=300_000,
            learning_rate=1e-3,
            batch_size=128,
            gamma=0.99
        )
    else:
        raise ValueError(f"Thuật toán {ALGO} không được hỗ trợ.")

    # 5. Bắt đầu Huấn luyện
    print(f"--- BẮT ĐẦU HUẤN LUYỆN {ALGO} ---")
    print(f"Theo dõi tiến trình tại: {run.get_url()}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, wandb_callback]
    )
    
    # 6. Kết thúc
    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    
    # Lưu model cuối cùng
    final_model_path = f"models/{RUN_NAME}/final_model.zip"
    model.save(final_model_path)
    print(f"Đã lưu model cuối cùng tại: {final_model_path}")
    
    env.close()
    run.finish()