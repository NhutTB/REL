# Enviroment.py (Phiên bản CẢI TIẾN 2 CẢM BIẾN)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class DronePursuitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, n_obstacles=4, world_size=(800, 600)):
        super().__init__()
        
        self.width, self.height = world_size
        self.n_obstacles = n_obstacles
        self.max_steps = 750
        
        # Các hằng số vật lý
        self.CHASER_SPEED = 4.0
        self.EVADER_SPEED = 3.0
        self.TURN_RATE = 0.2  # Giữ tốc độ quay đầu nhanh
        self.CAPTURE_RADIUS = 25.0
        self.DRONE_RADIUS = 10
        self.OBSTACLE_RADIUS = 30
        self.JAM_ZONE_RADIUS = 60
        
        # === MỚI: Chiến lược 2 Cảm biến ===
        # Cảm biến phát hiện Evader (Tầm xa)
        self.EVADER_SENSOR_RADIUS = 200.0
        # Cảm biến va chạm (Tầm ngắn, "nhỏ hơn nhiều" theo yêu cầu)
        self.COLLISION_SENSOR_RADIUS = 80.0

        # Không gian hành động
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Không gian quan sát (Observation Space) - 19 giá trị
        # 4: Thông tin bản thân (vận tốc, hướng)
        # 2: Vị trí tương đối Evader (nếu thấy)
        # 2: Vận tốc tương đối Evader (nếu thấy)
        # 2: Vị trí tương đối Jam Zone (nếu thấy)
        # 8: Vị trí tương đối 4 vật cản (nếu thấy)
        # 1: Cờ báo "Evader trong cảm biến" (quan trọng)
        obs_shape = 4 + 2 + 2 + 2 + (n_obstacles * 2) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # Trạng thái môi trường (giữ nguyên)
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

        # Pygame (giữ nguyên)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    # --- HÀM _get_obs ĐÃ THAY ĐỔI HOÀN TOÀN ---
    def _get_obs(self):
        obs = []
        
        # 1. Chaser info (4)
        obs.extend(self.chaser_vel)
        obs.extend([math.cos(self.chaser_heading), math.sin(self.chaser_heading)])
        
        # --- LOGIC CẢM BIẾN EVADER (TẦM XA) ---
        _real_rel_pos_evader = self.evader_pos - self.chaser_pos
        _real_rel_vel_evader = self.evader_vel - self.chaser_vel
        dist_to_evader = np.linalg.norm(_real_rel_pos_evader)
        
        evader_in_sensor = 1.0 if dist_to_evader < self.EVADER_SENSOR_RADIUS else 0.0
        
        if evader_in_sensor:
            obs.extend(_real_rel_pos_evader) # Vị trí (2)
            obs.extend(_real_rel_vel_evader) # Vận tốc (2)
        else:
            obs.extend([0.0, 0.0]) # Mất dấu (2)
            obs.extend([0.0, 0.0]) # Mất dấu (2)

        # --- LOGIC CẢM BIẾN VA CHẠM (TẦM NGẮN) ---
        
        # 3. Jam zone info (2)
        _real_rel_pos_jam = self.jam_zone_pos - self.chaser_pos
        if np.linalg.norm(_real_rel_pos_jam) < self.COLLISION_SENSOR_RADIUS:
            obs.extend(_real_rel_pos_jam)
        else:
            obs.extend([0.0, 0.0]) # Không thấy
            
        # 4. Obstacles info (N * 2)
        for obs_pos in self.obstacles:
            _real_rel_pos_obs = obs_pos - self.chaser_pos
            if np.linalg.norm(_real_rel_pos_obs) < self.COLLISION_SENSOR_RADIUS:
                obs.extend(_real_rel_pos_obs)
            else:
                obs.extend([0.0, 0.0]) # Không thấy

        # 5. Cờ báo Evader (1) - Rất quan trọng
        obs.append(evader_in_sensor)
            
        return np.array(obs, dtype=np.float32)

    # --- HÀM _get_info (Giữ nguyên) ---
    def _get_info(self):
        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        return {
            "distance": dist,
            "captured": dist < self.CAPTURE_RADIUS,
            "jammed": np.linalg.norm(self.chaser_pos - self.jam_zone_pos) < self.JAM_ZONE_RADIUS + self.DRONE_RADIUS,
            "collision": self._check_chaser_obstacle_collision(),
            "evader_in_sensor": dist < self.EVADER_SENSOR_RADIUS
        }
        
    # --- HÀM reset (Giữ nguyên) ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.chaser_pos = np.array([50.0, self.height / 2.0])
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        self.evader_pos = np.array([self.width - 50.0, self.height / 2.0])
        self.evader_vel = np.array([-self.EVADER_SPEED, 0.0])
        self.evader_turn_timer = 0
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
            while np.linalg.norm(pos - self.jam_zone_pos) < self.JAM_ZONE_RADIUS + self.OBSTACLE_RADIUS:
                pos = self.np_random.uniform(
                    low=[min_x, min_y], high=[max_x, max_y], size=(2,)
                )
            self.obstacles.append(pos)
        self.prev_dist_to_evader = np.linalg.norm(self.chaser_pos - self.evader_pos)
        if self.render_mode == "human":
            self._init_render()
        return self._get_obs(), self._get_info()

    # --- HÀM _check_chaser_obstacle_collision (Giữ nguyên) ---
    def _check_chaser_obstacle_collision(self):
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.chaser_pos - obs_pos) < self.OBSTACLE_RADIUS + self.DRONE_RADIUS:
                return True
        return False
        
    # --- HÀM _update_evader và _rotate (Giữ nguyên) ---
    def _update_evader(self):
        if self.evader_turn_timer <= 0:
            turn = self.np_random.uniform(-0.5, 0.5)
            self.evader_vel = self._rotate(self.evader_vel, turn)
            self.evader_turn_timer = self.np_random.integers(30, 90)
        self.evader_turn_timer -= 1
        new_pos = self.evader_pos + self.evader_vel
        collided = False
        for obs_pos in self.obstacles:
            if np.linalg.norm(new_pos - obs_pos) < self.OBSTACLE_RADIUS + self.DRONE_RADIUS:
                collided = True
                break
        if not (self.DRONE_RADIUS < new_pos[0] < self.width - self.DRONE_RADIUS):
            collided = True
        if not (self.DRONE_RADIUS < new_pos[1] < self.height - self.DRONE_RADIUS):
            collided = True
        if collided:
            self.evader_vel = self._rotate(self.evader_vel, self.np_random.uniform(-math.pi, math.pi))
        else:
            self.evader_pos = new_pos

    def _rotate(self, vec, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        x = vec[0] * cos - vec[1] * sin
        y = vec[0] * sin + vec[1] * cos
        return np.array([x, y])

    # --- HÀM step ĐÃ THAY ĐỔI (REWARD) ---
    def step(self, action):
        self.current_step += 1
        self._update_evader()
        
        throttle, turn = action
        throttle_normalized = (throttle + 1) / 2 # Chuyển [-1, 1] thành [0, 1]
        
        self.chaser_heading += turn * self.TURN_RATE
        self.chaser_vel[0] = math.cos(self.chaser_heading) * throttle_normalized * self.CHASER_SPEED
        self.chaser_vel[1] = math.sin(self.chaser_heading) * throttle_normalized * self.CHASER_SPEED
        self.chaser_pos += self.chaser_vel
        
        terminated = False
        reward = 0.0
        info = self._get_info()
        curr_dist = info["distance"]
        
        # === CHIẾN LƯỢC REWARD MỚI: CHỐNG "ĐỨNG YÊN" ===
        CAPTURE_REWARD = 1000.0
        DEATH_PENALTY = -1000.0
        
        # 1. Hình phạt khi không làm gì (Throttle thấp)
        # Nếu agent "đứng yên" (ga < 20%), phạt NẶNG
        if throttle_normalized < 0.2:
            INACTIVITY_PENALTY = -5.0
            reward += INACTIVITY_PENALTY
        else:
            # Nếu agent di chuyển, chỉ phạt nhẹ
            MOVEMENT_PENALTY = -0.1
            reward += MOVEMENT_PENALTY

        # 2. Thưởng khi "Khóa mục tiêu" (Bám theo)
        # Tăng mạnh thưởng bám theo
        SENSOR_LOCK_BONUS = 2.0
        
        # 3. Thưởng khi "Tiến lại gần"
        # Tăng mạnh thưởng khi tiến lại gần (quan trọng cho việc khám phá)
        DISTANCE_REWARD_SCALE = 0.5 

        # (A) Thắng: Bắt được evader
        if info["captured"]:
            terminated = True
            reward = CAPTURE_REWARD
        
        # (B) Thua: Chạm vùng phá sóng, vật cản, hoặc biên
        elif info["jammed"] or info["collision"] or \
             not (0 < self.chaser_pos[0] < self.width and 0 < self.chaser_pos[1] < self.height):
            terminated = True
            reward = DEATH_PENALTY
            
        # 4. Kiểm tra hết giờ
        truncated = self.current_step >= self.max_steps
        
        # 5. Tính reward shaping
        if not terminated:
            # Thưởng/phạt khi thay đổi khoảng cách
            reward += (self.prev_dist_to_evader - curr_dist) * DISTANCE_REWARD_SCALE
            
            # Thưởng khi khóa mục tiêu
            if info["evader_in_sensor"]:
                reward += SENSOR_LOCK_BONUS
                
        self.prev_dist_to_evader = curr_dist
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info
    # --- HÀM _init_render (Giữ nguyên) ---
    def _init_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Drone Pursuit Demo")
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

    # --- HÀM render ĐÃ THAY ĐỔI (Vẽ 2 cảm biến) ---
    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
        self._init_render()
        self.screen.fill((20, 20, 30))
        pygame.draw.circle(self.screen, (100, 0, 0), self.jam_zone_pos, self.JAM_ZONE_RADIUS)
        pygame.draw.circle(self.screen, (255, 0, 0), self.jam_zone_pos, self.JAM_ZONE_RADIUS, 2)
        for obs_pos in self.obstacles:
            pygame.draw.circle(self.screen, (80, 80, 90), obs_pos, self.OBSTACLE_RADIUS)
            pygame.draw.circle(self.screen, (150, 150, 150), obs_pos, self.OBSTACLE_RADIUS, 2)
        self._draw_drone(self.evader_pos, math.atan2(self.evader_vel[1], self.evader_vel[0]), (0, 150, 255))
        self._draw_drone(self.chaser_pos, self.chaser_heading, (255, 150, 0))
        
        # === MỚI: Vẽ 2 Vòng Cảm Biến ===
        # Vòng 1: Cảm biến Evader (Tầm xa - Màu vàng)
        pygame.draw.circle(self.screen, (200, 200, 0), self.chaser_pos, self.EVADER_SENSOR_RADIUS, 1)
        # Vòng 2: Cảm biến Va chạm (Tầm ngắn - Màu xanh lam)
        pygame.draw.circle(self.screen, (0, 100, 200), self.chaser_pos, self.COLLISION_SENSOR_RADIUS, 1)
        
        pygame.draw.circle(self.screen, (0, 255, 0), self.evader_pos, self.CAPTURE_RADIUS, 1)
        text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    # --- HÀM _draw_drone và close (Giữ nguyên) ---
    def _draw_drone(self, pos, angle, color):
        points = []
        points.append(
            (pos[0] + self.DRONE_RADIUS * math.cos(angle), 
             pos[1] + self.DRONE_RADIUS * math.sin(angle))
        )
        points.append(
            (pos[0] + self.DRONE_RADIUS * math.cos(angle + 2.5), 
             pos[1] + self.DRONE_RADIUS * math.sin(angle + 2.5))
        )
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

# --- Khối Demo (Cập nhật để hiển thị 2 cảm biến) ---
if __name__ == "__main__":
    print("Đang chạy demo môi trường với Agent NGẪU NHIÊN...")
    print("Logic mới: Agent chỉ 'thấy' vật cản/jam zone ở vòng XANH LAM (tầm ngắn).")
    print("Agent chỉ 'thấy' mục tiêu ở vòng VÀNG (tầm xa).")
    print("Hình phạt hết giờ (-1500) giờ NẶNG HƠN hình phạt chết (-1000).")
    
    env = DronePursuitEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while True:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("evader_in_sensor"):
            print("!!! Agent: ĐÃ PHÁT HIỆN MỤC TIÊU TRONG CẢM BIẾN (VÀNG) !!!")

        if terminated or truncated:
            print(f"Kết thúc lượt. Lý do: {info}")
            obs, info = env.reset()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()