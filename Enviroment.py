# Enviroment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class DronePursuitEnv(gym.Env):
    """
    Môi trường Drone Pursuit 2D.
    Logic:
    - Chaser (Agent): Bị 'terminated' nếu chạm vật cản hoặc jam zone.
    - Evader (Target): Bị chặn bởi vật cản, nhưng đi xuyên được jam zone.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, n_obstacles=4, world_size=(800, 600)):
        super().__init__()
        
        self.width, self.height = world_size
        self.n_obstacles = n_obstacles
        self.max_steps = 750  # Giới hạn thời gian mỗi lượt
        
        # Các hằng số vật lý
        self.CHASER_SPEED = 4.0
        self.EVADER_SPEED = 3.0
        self.TURN_RATE = 0.1  # Radian mỗi bước
        self.CAPTURE_RADIUS = 25.0
        self.DRONE_RADIUS = 10
        self.OBSTACLE_RADIUS = 30
        self.JAM_ZONE_RADIUS = 60

        # Không gian hành động: [throttle, turn]
        # throttle: [0, 1] (kiểm soát tốc độ)
        # turn: [-1, 1] (trái, phải)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Không gian quan sát (18 giá trị):
        # - Vận tốc chaser (2)
        # - Hướng chaser (cos, sin) (2)
        # - Vị trí tương đối của evader (2)
        # - Vận tốc tương đối của evader (2)
        # - Vị trí tương đối của jam zone (2)
        # - Vị trí tương đối của N vật cản (N * 2)
        obs_shape = 2 + 2 + 2 + 2 + 2 + (n_obstacles * 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # Trạng thái môi trường
        self.chaser_pos = np.zeros(2)
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        
        self.evader_pos = np.zeros(2)
        self.evader_vel = np.zeros(2)
        self.evader_turn_timer = 0
        
        self.obstacles = [np.zeros(2) for _ in range(self.n_obstacles)]
        self.jam_zone_pos = np.zeros(2)
        
        self.current_step = 0
        self.prev_dist_to_evader = np.inf

        # Pygame (cho render)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        obs = []
        
        # 1. Chaser info (4)
        obs.extend(self.chaser_vel)
        obs.extend([math.cos(self.chaser_heading), math.sin(self.chaser_heading)])
        
        # 2. Evader info (4)
        rel_pos = self.evader_pos - self.chaser_pos
        rel_vel = self.evader_vel - self.chaser_vel
        obs.extend(rel_pos)
        obs.extend(rel_vel)
        
        # 3. Jam zone info (2)
        obs.extend(self.jam_zone_pos - self.chaser_pos)
        
        # 4. Obstacles info (N * 2)
        for obs_pos in self.obstacles:
            obs.extend(obs_pos - self.chaser_pos)
            
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        return {
            "distance": dist,
            "captured": dist < self.CAPTURE_RADIUS,
            "jammed": np.linalg.norm(self.chaser_pos - self.jam_zone_pos) < self.JAM_ZONE_RADIUS + self.DRONE_RADIUS,
            "collision": self._check_chaser_obstacle_collision()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Đặt vị trí chaser và evader ở hai bên
        self.chaser_pos = np.array([50.0, self.height / 2.0])
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        
        self.evader_pos = np.array([self.width - 50.0, self.height / 2.0])
        self.evader_vel = np.array([-self.EVADER_SPEED, 0.0])
        self.evader_turn_timer = 0

        # Đặt ngẫu nhiên vật cản và jam zone ở giữa
        min_x, max_x = 150, self.width - 150
        min_y, max_y = 100, self.height - 100
        
        self.jam_zone_pos = self.np_random.uniform(
            low=[min_x, min_y], high=[max_x, max_y], size=(2,)
        )
        
        self.obstacles = []
        for _ in range(self.n_obstacles):
            pos = self.np_random.uniform(
                low=[min_x, min_y], high=[max_x, max_y], size=(2,)
            )
            # Đảm bảo vật cản không đè lên jam zone
            while np.linalg.norm(pos - self.jam_zone_pos) < self.JAM_ZONE_RADIUS + self.OBSTACLE_RADIUS:
                pos = self.np_random.uniform(
                    low=[min_x, min_y], high=[max_x, max_y], size=(2,)
                )
            self.obstacles.append(pos)
            
        self.prev_dist_to_evader = np.linalg.norm(self.chaser_pos - self.evader_pos)

        if self.render_mode == "human":
            self._init_render()
            
        return self._get_obs(), self._get_info()

    def _check_chaser_obstacle_collision(self):
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.chaser_pos - obs_pos) < self.OBSTACLE_RADIUS + self.DRONE_RADIUS:
                return True
        return False

    def _update_evader(self):
        # AI cho evader: đi ngẫu nhiên và tránh vật cản/biên
        
        # 1. Thay đổi hướng ngẫu nhiên
        if self.evader_turn_timer <= 0:
            turn = self.np_random.uniform(-0.5, 0.5)  # Đổi hướng
            self.evader_vel = self._rotate(self.evader_vel, turn)
            self.evader_turn_timer = self.np_random.integers(30, 90) # Hẹn giờ đổi hướng tiếp
        self.evader_turn_timer -= 1
        
        # 2. Tính vị trí mới
        new_pos = self.evader_pos + self.evader_vel

        # 3. Kiểm tra va chạm vật cản (LOGIC: bị chặn lại)
        collided = False
        for obs_pos in self.obstacles:
            if np.linalg.norm(new_pos - obs_pos) < self.OBSTACLE_RADIUS + self.DRONE_RADIUS:
                collided = True
                break
        
        # 4. Kiểm tra va chạm biên
        if not (self.DRONE_RADIUS < new_pos[0] < self.width - self.DRONE_RADIUS):
            collided = True
        if not (self.DRONE_RADIUS < new_pos[1] < self.height - self.DRONE_RADIUS):
            collided = True
        
        # 5. Cập nhật vị trí
        if collided:
            # Nếu va chạm, đổi hướng ngẫu nhiên và dừng lại
            self.evader_vel = self._rotate(self.evader_vel, self.np_random.uniform(-math.pi, math.pi))
        else:
            self.evader_pos = new_pos

    def _rotate(self, vec, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        x = vec[0] * cos - vec[1] * sin
        y = vec[0] * sin + vec[1] * cos
        return np.array([x, y])


    def step(self, action):
        self.current_step += 1
        
        # 1. Cập nhật Evader
        self._update_evader()
        
        # 2. Cập nhật Chaser (Agent)
        # SỬA LỖI ACTION: Dòng này đúng nếu action space của bạn là [-1, 1]
        # (Xem mục 2 để sửa action space cho đúng)
        throttle, turn = action
        throttle = (throttle + 1) / 2  # Chuyển [-1, 1] thành [0, 1]
        
        self.chaser_heading += turn * self.TURN_RATE
        self.chaser_vel[0] = math.cos(self.chaser_heading) * throttle * self.CHASER_SPEED
        self.chaser_vel[1] = math.sin(self.chaser_heading) * throttle * self.CHASER_SPEED
        self.chaser_pos += self.chaser_vel
        
        # 3. Kiểm tra điều kiện kết thúc và tính Reward
        terminated = False
        reward = 0.0
        
        info = self._get_info()
        curr_dist = info["distance"]
        
        # === CÂN BẰNG LẠI REWARD ===
        # Tăng phần thưởng và hình phạt lên để tạo tín hiệu rõ ràng
        CAPTURE_REWARD = 500.0
        DEATH_PENALTY = -500.0
        TIME_PENALTY_PER_STEP = -0.5  # Quan trọng: Phạt nặng hơn
        DISTANCE_REWARD_SCALE = 0.1   # Quan trọng: Thưởng nhiều hơn

        # (A) Thắng: Bắt được evader
        if info["captured"]:
            terminated = True
            reward = CAPTURE_REWARD
        
        # (B) Thua: Chạm vùng phá sóng
        elif info["jammed"]:
            terminated = True
            reward = DEATH_PENALTY
        
        # (C) Thua: Chạm vật cản
        elif info["collision"]:
            terminated = True
            reward = DEATH_PENALTY
            
        # (D) Thua: Chạm biên
        elif not (0 < self.chaser_pos[0] < self.width and 0 < self.chaser_pos[1] < self.height):
            terminated = True
            reward = DEATH_PENALTY
            
        # 4. Kiểm tra hết giờ
        truncated = self.current_step >= self.max_steps
        
        # 5. Tính reward shaping
        if not terminated:
            # Thưởng khi tiến lại gần (hoặc phạt khi đi xa)
            # Tăng trọng số từ 0.5 lên 1.0
            reward += (self.prev_dist_to_evader - curr_dist) * DISTANCE_REWARD_SCALE
            
            # Phạt nặng mỗi bước để buộc agent phải nhanh
            # Tăng hình phạt từ -0.1 lên -0.5
            reward += TIME_PENALTY_PER_STEP
            
        self.prev_dist_to_evader = curr_dist
        
        # 6. Render
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # --- Phần Render (Pygame) ---
    
    def _init_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Drone Pursuit Demo")
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
            
        self._init_render()
        
        self.screen.fill((20, 20, 30))  # Nền tối

        # Vẽ Jam Zone (Màu đỏ nguy hiểm)
        pygame.draw.circle(self.screen, (100, 0, 0), self.jam_zone_pos, self.JAM_ZONE_RADIUS)
        pygame.draw.circle(self.screen, (255, 0, 0), self.jam_zone_pos, self.JAM_ZONE_RADIUS, 2)
        
        # Vẽ Obstacles (Màu xám)
        for obs_pos in self.obstacles:
            pygame.draw.circle(self.screen, (80, 80, 90), obs_pos, self.OBSTACLE_RADIUS)
            pygame.draw.circle(self.screen, (150, 150, 150), obs_pos, self.OBSTACLE_RADIUS, 2)
            
        # Vẽ Evader (Màu xanh)
        self._draw_drone(self.evader_pos, math.atan2(self.evader_vel[1], self.evader_vel[0]), (0, 150, 255))
        
        # Vẽ Chaser (Màu cam)
        self._draw_drone(self.chaser_pos, self.chaser_heading, (255, 150, 0))
        
        # Vẽ vòng capture
        pygame.draw.circle(self.screen, (0, 255, 0), self.evader_pos, self.CAPTURE_RADIUS, 1)

        # Hiển thị text
        text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_drone(self, pos, angle, color):
        points = []
        # Mũi
        points.append(
            (pos[0] + self.DRONE_RADIUS * math.cos(angle), 
             pos[1] + self.DRONE_RADIUS * math.sin(angle))
        )
        # Đuôi trái
        points.append(
            (pos[0] + self.DRONE_RADIUS * math.cos(angle + 2.5), 
             pos[1] + self.DRONE_RADIUS * math.sin(angle + 2.5))
        )
        # Đuôi phải
        points.append(
            (pos[0] + self.DRONE_RADIUS * math.cos(angle - 2.5), 
             pos[1] + self.DRONE_RADIUS * math.sin(angle - 2.5))
        )
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.aalines(self.screen, (255, 255, 255), True, points)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


# --- KHỐI DEMO MÔI TRƯỜNG ---
if __name__ == "__main__":
    print("Đang chạy demo môi trường với Agent NGẪU NHIÊN...")
    print("Mũi tên CAM (Agent) sẽ tự đâm vào tường (XÁM) hoặc vùng phá sóng (ĐỎ) và bị reset.")
    print("Mũi tên XANH (Target) sẽ bị chặn bởi tường nhưng đi xuyên được vùng phá sóng.")
    print("Đây là cách kiểm tra logic môi trường.")
    
    # Bật render_mode="human" để xem
    env = DronePursuitEnv(render_mode="human")
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    
    while True:
        # Agent thực hiện hành động ngẫu nhiên
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Kết thúc lượt. Lý do: {info}")
            obs, info = env.reset()
        
        # Xử lý sự kiện thoát (nhấn nút X)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()