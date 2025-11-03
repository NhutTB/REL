# Enviroment.py
# Phiên bản 3.0:
# - Vật cản (CNV) và Jam Zone là HÌNH CHỮ NHẬT.
# - Kích thước (W, H) của CNV/Jam Zone là ngẫu nhiên.
# - Thêm logic "Mật độ vật cản" (Obstacle Density) để đảm bảo map luôn có đường đi.
# - Cập nhật logic va chạm Circle-vs-Rect.
# - Cập nhật Observation Space để chứa thông tin (x, y, w, h) của vật cản.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class DronePursuitEnv(gym.Env):
    """
    Môi trường Drone Pursuit 2D.
    Logic:
    - CNV & Jam Zone là HÌNH CHỮ NHẬT với kích thước ngẫu nhiên.
    - Logic va chạm Rect-vs-Circle (chính xác).
    - Logic 2 cảm biến (Evader tầm xa, Va chạm tầm ngắn).
    - Phạt nặng hành vi "đứng yên" để agent luôn khám phá.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, n_obstacles=6, obstacle_density=0.6, world_size=(800, 600)):
        super().__init__()
        
        self.width, self.height = world_size
        self.map_area = self.width * self.height
        
        # === MỚI: Logic Mật độ & Kích thước ===
        self.n_obstacles = n_obstacles # Số lượng CNV cố định
        self.obstacle_density_limit = obstacle_density # Giới hạn % map bị che (ví dụ: 0.6 = 60%)
        self.OBSTACLE_MIN_SIZE = 30
        self.OBSTACLE_MAX_SIZE = 100
        self.JAM_ZONE_MIN_SIZE = 80
        self.JAM_ZONE_MAX_SIZE = 150
        
        self.max_steps = 1000 # Giới hạn thời gian mỗi lượt
        # Hằng số vật lý
        self.CHASER_SPEED = 4.0
        self.EVADER_SPEED = 3.0
        self.TURN_RATE = 0.2 # Tốc độ quay nhanh
        self.CAPTURE_RADIUS = 25.0
        self.DRONE_RADIUS = 10

        # Hằng số cảm biến
        self.EVADER_SENSOR_RADIUS = 500.0
        self.COLLISION_SENSOR_RADIUS = 80.0 # Cảm biến va chạm tầm ngắn

        # Không gian hành động
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # === MỚI: Observation Space (đã thay đổi) ===
        # Agent cần biết (x, y, w, h) của mỗi vật thể
        # 4: Thông tin bản thân (vận tốc, hướng)
        # 4: Evader (rel_pos x, y, rel_vel x, y)
        # 4: Jam Zone (rel_pos x, y, w, h)
        # (n_obstacles * 4): CNV (rel_pos x, y, w, h)
        # 1: Cờ báo "Evader trong cảm biến"
        obs_shape = 4 + 4 + 4 + (self.n_obstacles * 4) + 1
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
        
        # === MỚI: Lưu trữ CNV/Jam Zone dạng pygame.Rect ===
        self.obstacles = [] # List các pygame.Rect
        self.jam_zone = pygame.Rect(0, 0, 0, 0)
        
        self.current_step = 0
        self.prev_dist_to_evader = np.inf

        # Pygame
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
    def _check_rect_circle_collision(self, rect, circle_center, circle_radius):
        """Hàm kiểm tra va chạm giữa HÌNH CHỮ NHẬT (AABB) và HÌNH TRÒN (Circle)."""
        # Tìm điểm gần nhất trên hình chữ nhật so với tâm hình tròn
        closest_x = np.clip(circle_center[0], rect.left, rect.right)
        closest_y = np.clip(circle_center[1], rect.top, rect.bottom)
        
        # Tính khoảng cách từ tâm hình tròn đến điểm gần nhất đó
        dist_x = circle_center[0] - closest_x
        dist_y = circle_center[1] - closest_y
        
        # So sánh với bán kính (bình phương để tránh dùng sqrt)
        distance_sq = (dist_x ** 2) + (dist_y ** 2)
        return distance_sq < (circle_radius ** 2)

    def _get_obs(self):
        """Hàm lấy quan sát. Cập nhật để gửi (x, y, w, h) của CNV/Jam Zone."""
        obs = []
        
        # 1. Chaser info (4)
        obs.extend(self.chaser_vel)
        obs.extend([math.cos(self.chaser_heading), math.sin(self.chaser_heading)])
        
        # 2. Evader info (4) - Logic Cảm biến Tầm xa
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

        # 3. Jam zone info (4) - Logic Cảm biến Tầm ngắn
        rel_center_x = self.jam_zone.centerx - self.chaser_pos[0]
        rel_center_y = self.jam_zone.centery - self.chaser_pos[1]
        dist_to_jam = np.linalg.norm([rel_center_x, rel_center_y])
        
        if dist_to_jam < self.COLLISION_SENSOR_RADIUS:
            obs.extend([rel_center_x, rel_center_y, self.jam_zone.width, self.jam_zone.height])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0]) # Không thấy
            
        # 4. Obstacles info (n_obstacles * 4) - Logic Cảm biến Tầm ngắn
        for obs_rect in self.obstacles:
            rel_center_x = obs_rect.centerx - self.chaser_pos[0]
            rel_center_y = obs_rect.centery - self.chaser_pos[1]
            dist_to_obs = np.linalg.norm([rel_center_x, rel_center_y])
            
            if dist_to_obs < self.COLLISION_SENSOR_RADIUS:
                obs.extend([rel_center_x, rel_center_y, obs_rect.width, obs_rect.height])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0]) # Không thấy

        # 5. Cờ báo Evader (1)
        obs.append(evader_in_sensor)
            
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Hàm lấy thông tin. Cập nhật để dùng va chạm Rect-Circle."""
        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        return {
            "distance": dist,
            "captured": dist < self.CAPTURE_RADIUS,
            "jammed": self._check_rect_circle_collision(self.jam_zone, self.chaser_pos, self.DRONE_RADIUS),
            "collision": self._check_chaser_obstacle_collision(),
            "evader_in_sensor": dist < self.EVADER_SENSOR_RADIUS
        }
        
    def reset(self, seed=None, options=None):
        """Hàm reset. Cập nhật để tạo CNV hình chữ nhật và kiểm tra mật độ."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Vị trí ban đầu (Agent & Evader)
        self.chaser_pos = np.array([50.0, self.height / 2.0])
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        self.evader_pos = np.array([self.width - 50.0, self.height / 2.0])
        self.evader_vel = np.array([-self.EVADER_SPEED, 0.0])
        self.evader_turn_timer = 0
        
        # Vùng an toàn để đặt CNV
        min_x, max_x = 150, self.width - 150
        min_y, max_y = 100, self.height - 100
        max_obstacle_area = self.map_area * self.obstacle_density_limit
        
        # === MỚI: Vòng lặp Tạo Map (Đảm bảo mật độ < 60%) ===
        while True: 
            self.obstacles = []
            total_obstacle_area = 0
            
            # 1. Tạo Jam Zone (Hình chữ nhật)
            jz_w = self.np_random.uniform(self.JAM_ZONE_MIN_SIZE, self.JAM_ZONE_MAX_SIZE)
            jz_h = self.np_random.uniform(self.JAM_ZONE_MIN_SIZE, self.JAM_ZONE_MAX_SIZE)
            jz_x = self.np_random.uniform(min_x, max_x - jz_w)
            jz_y = self.np_random.uniform(min_y, max_y - jz_h)
            self.jam_zone = pygame.Rect(jz_x, jz_y, jz_w, jz_h)
            total_obstacle_area += self.jam_zone.width * self.jam_zone.height
            
            # 2. Tạo Obstacles (Hình chữ nhật)
            for _ in range(self.n_obstacles):
                placement_tries = 0
                while placement_tries < 100: # Thử đặt 100 lần
                    obs_w = self.np_random.uniform(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
                    obs_h = self.np_random.uniform(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
                    obs_x = self.np_random.uniform(min_x, max_x - obs_w)
                    obs_y = self.np_random.uniform(min_y, max_y - obs_h)
                    new_rect = pygame.Rect(obs_x, obs_y, obs_w, obs_h)
                    
                    # Kiểm tra chồng chéo với Jam Zone
                    if self.jam_zone.colliderect(new_rect):
                        placement_tries += 1
                        continue
                        
                    # Kiểm tra chồng chéo với các CNV khác
                    if any(obs.colliderect(new_rect) for obs in self.obstacles):
                        placement_tries += 1
                        continue
                        
                    # Nếu không chồng chéo, thêm vào list và thoát vòng lặp
                    self.obstacles.append(new_rect)
                    total_obstacle_area += new_rect.width * new_rect.height
                    break
            
            # 3. Kiểm tra Mật độ
            current_density = total_obstacle_area / self.map_area
            if current_density <= self.obstacle_density_limit:
                # print(f"Map generated. Density: {current_density*100:.1f}%") # Dùng để debug
                break # Map hợp lệ, thoát khỏi vòng lặp tạo map
            # Nếu không, vòng lặp 'while True' sẽ tự động chạy lại, tạo map mới
            
        self.prev_dist_to_evader = np.linalg.norm(self.chaser_pos - self.evader_pos)
        if self.render_mode == "human":
            self._init_render()
            
        return self._get_obs(), self._get_info()

    def _check_chaser_obstacle_collision(self):
        """Kiểm tra Chaser (Circle) với tất cả CNV (Rects)."""
        for obs_rect in self.obstacles:
            if self._check_rect_circle_collision(obs_rect, self.chaser_pos, self.DRONE_RADIUS):
                return True
        return False
        
    def _update_evader(self):
        """Cập nhật Evader để né CNV hình chữ nhật."""
        # 1. Thay đổi hướng ngẫu nhiên
        if self.evader_turn_timer <= 0:
            turn = self.np_random.uniform(-0.5, 0.5)
            self.evader_vel = self._rotate(self.evader_vel, turn)
            self.evader_turn_timer = self.np_random.integers(30, 90)
        self.evader_turn_timer -= 1
        
        # 2. Vị trí mới
        new_pos = self.evader_pos + self.evader_vel

        # 3. Kiểm tra va chạm CNV (Rect-vs-Circle)
        collided = False
        for obs_rect in self.obstacles:
            if self._check_rect_circle_collision(obs_rect, new_pos, self.DRONE_RADIUS):
                collided = True
                break
        
        # 4. Kiểm tra va chạm biên
        if not (self.DRONE_RADIUS < new_pos[0] < self.width - self.DRONE_RADIUS):
            collided = True
        if not (self.DRONE_RADIUS < new_pos[1] < self.height - self.DRONE_RADIUS):
            collided = True
        
        # 5. Cập nhật vị trí
        if collided:
            self.evader_vel = self._rotate(self.evader_vel, self.np_random.uniform(-math.pi, math.pi))
        else:
            self.evader_pos = new_pos

    def _rotate(self, vec, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        x = vec[0] * cos - vec[1] * sin
        y = vec[0] * sin + vec[1] * cos
        return np.array([x, y])

    def step(self, action):
        """Hàm step: Giữ nguyên logic "phạt đứng yên" để agent hung hăng."""
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
        info = self._get_info() # Hàm này đã được cập nhật để dùng va chạm Rect
        curr_dist = info["distance"]
        
        # === CHIẾN LƯỢC REWARD: CHỐNG "ĐỨNG YÊN" ===
        CAPTURE_REWARD = 1000.0
        DEATH_PENALTY = -1000.0
        
        # 1. Hình phạt khi không làm gì (Throttle thấp)
        if throttle_normalized < 0.2:
            INACTIVITY_PENALTY = -5.0
            reward += INACTIVITY_PENALTY
        else:
            MOVEMENT_PENALTY = -0.5
            reward += MOVEMENT_PENALTY

        # 2. Thưởng khi "Khóa mục tiêu"
        SENSOR_LOCK_BONUS = 2.0
        
        # 3. Thưởng khi "Tiến lại gần"
        DISTANCE_REWARD_SCALE = 0.5 

        # (A) Thắng
        if info["captured"]:
            terminated = True
            reward = CAPTURE_REWARD
        
        # (B) Thua (Hàm info["jammed"] và info["collision"] đã dùng logic Rect)
        elif info["jammed"] or info["collision"] or \
             not (0 < self.chaser_pos[0] < self.width and 0 < self.chaser_pos[1] < self.height):
            terminated = True
            reward = DEATH_PENALTY
            
        # 4. Kiểm tra hết giờ
        truncated = self.current_step >= self.max_steps
        
        # 5. Tính reward shaping
        if not terminated:
            reward += (self.prev_dist_to_evader - curr_dist) * DISTANCE_REWARD_SCALE
            if info["evader_in_sensor"]:
                reward += SENSOR_LOCK_BONUS
                
        self.prev_dist_to_evader = curr_dist
        
        if self.render_mode == "human":
            self.render()

        # QUAN TRỌNG: _get_obs() phải được gọi Ở CUỐI CÙNG
        return self._get_obs(), reward, terminated, truncated, info

    # --- HÀM RENDER ---
    
    def _init_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Drone Pursuit Demo (Rect Obstacles)")
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

    def render(self):
        """Hàm render. Cập nhật để vẽ HÌNH CHỮ NHẬT."""
        if self.render_mode not in ["human", "rgb_array"]:
            return
            
        self._init_render()
        self.screen.fill((20, 20, 30))  # Nền tối

        # Vẽ Jam Zone (Màu đỏ nguy hiểm) - Giờ là hình chữ nhật
        pygame.draw.rect(self.screen, (100, 0, 0), self.jam_zone)
        pygame.draw.rect(self.screen, (255, 0, 0), self.jam_zone, 2)
        
        # Vẽ Obstacles (Màu xám) - Giờ là hình chữ nhật
        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, (80, 80, 90), obs_rect)
            pygame.draw.rect(self.screen, (150, 150, 150), obs_rect, 2)
            
        # Vẽ Drones (vẫn là hình tròn)
        self._draw_drone(self.evader_pos, math.atan2(self.evader_vel[1], self.evader_vel[0]), (0, 150, 255))
        self._draw_drone(self.chaser_pos, self.chaser_heading, (255, 150, 0))
        
        # Vẽ 2 Vòng Cảm Biến (vẫn là hình tròn)
        pygame.draw.circle(self.screen, (200, 200, 0), self.chaser_pos, self.EVADER_SENSOR_RADIUS, 1) # Cảm biến Evader (Vàng)
        pygame.draw.circle(self.screen, (0, 100, 200), self.chaser_pos, self.COLLISION_SENSOR_RADIUS, 1) # Cảm biến Va chạm (Xanh lam)
        
        # Vẽ vòng capture (vẫn là hình tròn)
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

# --- KHỐI DEMO MÔI TRƯỜNG ---
if __name__ == "__main__":
    print("Đang chạy demo môi trường với Agent NGẪU NHIÊN...")
    print("Logic mới: CNV và Jam Zone là HÌNH CHỮ NHẬT, kích thước ngẫu nhiên.")
    print(f"Mật độ map được giới hạn dưới 60%.")
    print("Agent bị PHẠT NẶNG nếu đứng yên.")
    
    # Bạn có thể thay đổi tham số ở đây để test:
    # ví dụ: n_obstacles=10, obstacle_density=0.8
    env = DronePursuitEnv(render_mode="human", n_obstacles=6, obstacle_density=0.6)
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while True:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("evader_in_sensor"):
            pass # Tắt bớt log spam

        if terminated or truncated:
            print(f"Kết thúc lượt. Lý do: {info}")
            obs, info = env.reset()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()