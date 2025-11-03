# Eval.py
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import time
import pygame

# Import môi trường
from Enviroment import DronePursuitEnv

# --- CẤU HÌNH ĐÁNH GIÁ ---
# THAY ĐỔI ĐƯỜNG DẪN NÀY
MODEL_PATH = "models/PPO_1730598835/final_model.zip"
ALGO = "PPO"  # Phải khớp với model bạn load: PPO, SAC, TD3
N_EPISODES = 10
# -----------------------------

if __name__ == "__main__":
    print(f"Đang tải model {ALGO} từ: {MODEL_PATH}")

    # 1. Tạo môi trường (lần này CÓ RENDER)
    env = DronePursuitEnv(render_mode="human")
    
    # 2. Load model
    if ALGO == "PPO":
        model = PPO.load(MODEL_PATH, env=env)
    elif ALGO == "SAC":
        model = SAC.load(MODEL_PATH, env=env)
    elif ALGO == "TD3":
        model = TD3.load(MODEL_PATH, env=env)
    else:
        raise ValueError(f"Thuật toán {ALGO} không được hỗ trợ.")
        
    print("Model đã tải xong. Bắt đầu đánh giá...")
    
    # 3. Chạy vòng lặp đánh giá
    total_captures = 0
    total_jams = 0
    total_collisions = 0
    total_timeouts = 0
    
    for i in range(N_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        print(f"--- Bắt đầu Lượt #{i+1} ---")
        
        while not (terminated or truncated):
            # Lấy hành động từ model
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Xử lý sự kiện thoát (nhấn nút X)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()
        
        # Ghi lại kết quả của lượt
        if info.get("captured"):
            print("Kết quả: BẮT ĐƯỢC MỤC TIÊU (CAPTURED) ✅")
            total_captures += 1
        elif info.get("jammed"):
            print("Kết quả: BỊ PHÁ SÓNG (JAMMED) ❌")
            total_jams += 1
        elif info.get("collision"):
            print("Kết quả: VA CHẠM (COLLISION) 💥")
            total_collisions += 1
        elif truncated:
            print("Kết quả: HẾT GIỜ (TIMEOUT) ⏱️")
            total_timeouts += 1
            
    env.close()
    
    # 4. In thống kê
    print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
    print(f"Tổng số lượt: {N_EPISODES}")
    print(f"Tỉ lệ bắt được: {total_captures / N_EPISODES * 100:.1f}%")
    print(f"Tỉ lệ va chạm: {total_collisions / N_EPISODES * 100:.1f}%")
    print(f"Tỉ lệ bị phá sóng: {total_jams / N_EPISODES * 100:.1f}%")
    print(f"Tỉ lệ hết giờ: {total_timeouts / N_EPISODES * 100:.1f}%")