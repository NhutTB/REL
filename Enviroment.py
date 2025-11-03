import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class DronePursuitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, difficulty="hard", n_obstacles_hard=16, obstacle_density=0.8, world_size=(800, 600)):
        super().__init__()
        
        self.width, self.height = world_size
        self.map_area = self.width * self.height
        
        self.difficulty = difficulty
        self.n_obstacles_hard = n_obstacles_hard
        self.n_obstacles_medium = 3
        self.n_obstacles_easy = 0
        
        self.n_obstacles_current = 0
        
        self.obstacle_density_limit = obstacle_density
        self.OBSTACLE_MIN_SIZE = 30
        self.OBSTACLE_MAX_SIZE = 100
        self.JAM_ZONE_MIN_SIZE = 80
        self.JAM_ZONE_MAX_SIZE = 150
        
        self.max_steps = 1000
        
        self.CHASER_SPEED = 5.0
        self.EVADER_SPEED = 3.5
        self.TURN_RATE = 0.3
        self.CAPTURE_RADIUS = 25.0
        self.DRONE_RADIUS = 10

        self.EVADER_SENSOR_RADIUS = 250.0 
        self.COLLISION_SENSOR_RADIUS = 80.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        self._update_obs_space()
        self.chaser_pos = np.zeros(2)
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        self.evader_pos = np.zeros(2)
        self.evader_vel = np.zeros(2)
        self.evader_turn_timer = 0
        
        self.obstacles = []
        self.jam_zone = pygame.Rect(0, 0, 0, 0)
        
        self.current_step = 0
        self.prev_dist_to_evader = np.inf
        self.min_distance_achieved = np.inf
        self.consecutive_visible_steps = 0
        
        self.grid_size = 50
        self.visited_grid = np.zeros((1,1))

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
    def _update_obs_space(self):
        max_obstacles_possible = self.n_obstacles_hard
        obs_shape = 4 + 4 + 4 + (max_obstacles_possible * 4) + 1
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        
    def set_n_obstacles(self, n):
        print(f"Warning: set_n_obstacles() called. Overriding difficulty='{self.difficulty}' with {n} obstacles.")
        self.difficulty = "custom"
        self.n_obstacles_current = n
        
    def _check_rect_circle_collision(self, rect, circle_center, circle_radius):
        closest_x = np.clip(circle_center[0], rect.left, rect.right)
        closest_y = np.clip(circle_center[1], rect.top, rect.bottom)
        dist_x = circle_center[0] - closest_x
        dist_y = circle_center[1] - closest_y
        distance_sq = (dist_x ** 2) + (dist_y ** 2)
        return distance_sq < (circle_radius ** 2)

    def _get_obs(self):
        obs = []
        
        obs.extend(self.chaser_vel)
        obs.extend([math.cos(self.chaser_heading), math.sin(self.chaser_heading)])
        
        _real_rel_pos_evader = self.evader_pos - self.chaser_pos
        _real_rel_vel_evader = self.evader_vel - self.chaser_vel
        dist_to_evader = np.linalg.norm(_real_rel_pos_evader)
        evader_in_sensor = 1.0 if dist_to_evader < self.EVADER_SENSOR_RADIUS else 0.0
        
        if evader_in_sensor:
            obs.extend(_real_rel_pos_evader)
            obs.extend(_real_rel_vel_evader)
        else:
            obs.extend([0.0, 0.0])
            obs.extend([0.0, 0.0])

        rel_center_x = self.jam_zone.centerx - self.chaser_pos[0]
        rel_center_y = self.jam_zone.centery - self.chaser_pos[1]
        dist_to_jam = np.linalg.norm([rel_center_x, rel_center_y])
        
        if dist_to_jam < self.COLLISION_SENSOR_RADIUS:
            obs.extend([rel_center_x, rel_center_y, self.jam_zone.width, self.jam_zone.height])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
            
        max_obs_obstacles = self.n_obstacles_hard
        
        for i in range(max_obs_obstacles):
            if i < len(self.obstacles):
                obs_rect = self.obstacles[i]
                rel_center_x = obs_rect.centerx - self.chaser_pos[0]
                rel_center_y = obs_rect.centery - self.chaser_pos[1]
                dist_to_obs = np.linalg.norm([rel_center_x, rel_center_y])
                
                if dist_to_obs < self.COLLISION_SENSOR_RADIUS:
                    obs.extend([rel_center_x, rel_center_y, obs_rect.width, obs_rect.height])
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        obs.append(evader_in_sensor)
            
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        return {
            "distance": dist,
            "captured": dist < self.CAPTURE_RADIUS,
            "jammed": self._check_rect_circle_collision(self.jam_zone, self.chaser_pos, self.DRONE_RADIUS),
            "collision": self._check_chaser_obstacle_collision(),
            "evader_in_sensor": dist < self.EVADER_SENSOR_RADIUS
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.consecutive_visible_steps = 0
        
        disable_jam_zone = False
        if self.difficulty == "easy":
            self.n_obstacles_current = self.n_obstacles_easy
            disable_jam_zone = True
        elif self.difficulty == "medium":
            self.n_obstacles_current = self.n_obstacles_medium
            disable_jam_zone = False
        else:
            self.n_obstacles_current = self.n_obstacles_hard
            disable_jam_zone = False
        
        self.chaser_pos = np.array([50.0, self.height / 2.0])
        self.chaser_vel = np.zeros(2)
        self.chaser_heading = 0.0
        self.evader_pos = np.array([self.width - 50.0, self.height / 2.0])
        self.evader_vel = np.array([-self.EVADER_SPEED, 0.0])
        self.evader_turn_timer = 0
        
        min_x, max_x = 150, self.width - 150
        min_y, max_y = 100, self.height - 100
        max_obstacle_area = self.map_area * self.obstacle_density_limit
        
        while True: 
            self.obstacles = []
            total_obstacle_area = 0
            
            if disable_jam_zone:
                self.jam_zone = pygame.Rect(-100, -100, 0, 0)
            else:
                jz_w = self.np_random.uniform(self.JAM_ZONE_MIN_SIZE, self.JAM_ZONE_MAX_SIZE)
                jz_h = self.np_random.uniform(self.JAM_ZONE_MIN_SIZE, self.JAM_ZONE_MAX_SIZE)
                jz_x = self.np_random.uniform(min_x, max_x - jz_w)
                jz_y = self.np_random.uniform(min_y, max_y - jz_h)
                self.jam_zone = pygame.Rect(jz_x, jz_y, jz_w, jz_h)
                total_obstacle_area += self.jam_zone.width * self.jam_zone.height
            
            for _ in range(self.n_obstacles_current):
                placement_tries = 0
                while placement_tries < 100:
                    obs_w = self.np_random.uniform(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
                    obs_h = self.np_random.uniform(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
                    obs_x = self.np_random.uniform(min_x, max_x - obs_w)
                    obs_y = self.np_random.uniform(min_y, max_y - obs_h)
                    new_rect = pygame.Rect(obs_x, obs_y, obs_w, obs_h)
                    
                    if self.jam_zone.colliderect(new_rect):
                        placement_tries += 1
                        continue
                    if any(obs.colliderect(new_rect) for obs in self.obstacles):
                        placement_tries += 1
                        continue
                    self.obstacles.append(new_rect)
                    total_obstacle_area += new_rect.width * new_rect.height
                    break
            
            current_density = total_obstacle_area / self.map_area
            if current_density <= self.obstacle_density_limit:
                break
            
        self.prev_dist_to_evader = np.linalg.norm(self.chaser_pos - self.evader_pos)
        self.min_distance_achieved = self.prev_dist_to_evader
        self.visited_grid = np.zeros((int(self.width // self.grid_size) + 1, int(self.height // self.grid_size) + 1))
        
        if self.render_mode == "human":
            self._init_render()
            
        return self._get_obs(), self._get_info()

    def _check_chaser_obstacle_collision(self):
        for obs_rect in self.obstacles:
            if self._check_rect_circle_collision(obs_rect, self.chaser_pos, self.DRONE_RADIUS):
                return True
        return False
        
    def _update_evader(self):
        if self.evader_turn_timer <= 0:
            turn = self.np_random.uniform(-0.5, 0.5)
            self.evader_vel = self._rotate(self.evader_vel, turn)
            self.evader_turn_timer = self.np_random.integers(30, 90)
        self.evader_turn_timer -= 1
        
        new_pos = self.evader_pos + self.evader_vel
        collided = False
        for obs_rect in self.obstacles:
            if self._check_rect_circle_collision(obs_rect, new_pos, self.DRONE_RADIUS):
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

    def step(self, action):
        self.current_step += 1
        self._update_evader()
        
        throttle, turn = action
        throttle_normalized = (throttle + 1) / 2
        
        self.chaser_heading += turn * self.TURN_RATE
        self.chaser_vel[0] = math.cos(self.chaser_heading) * throttle_normalized * self.CHASER_SPEED
        self.chaser_vel[1] = math.sin(self.chaser_heading) * throttle_normalized * self.CHASER_SPEED
        self.chaser_pos += self.chaser_vel
        
        terminated = False
        reward = 0.0
        info = self._get_info()
        curr_dist = info["distance"]
        
        CAPTURE_REWARD = 5000.0
        DEATH_PENALTY = -1000.0
        TIMEOUT_PENALTY = -500.0
        
        if info["captured"]:
            terminated = True
            reward = CAPTURE_REWARD
        
        elif info["jammed"] or info["collision"] or \
             not (0 < self.chaser_pos[0] < self.width and 0 < self.chaser_pos[1] < self.height):
            terminated = True
            reward = DEATH_PENALTY
            
        else:
            if info["evader_in_sensor"]:
                self.consecutive_visible_steps += 1
                
                distance_improvement = self.prev_dist_to_evader - curr_dist
                reward += distance_improvement * 5.0
                
                if distance_improvement > 0:
                    reward += 2.0 * throttle_normalized
                
                reward += 0.1 * self.consecutive_visible_steps
                
                if curr_dist < self.min_distance_achieved:
                    reward += 3.0
                    self.min_distance_achieved = curr_dist
                
                if distance_improvement < -2.0:
                    reward -= 10.0
                    
                reward -= 0.1
                
            else:
                self.consecutive_visible_steps = 0
                
                cell_x = int(self.chaser_pos[0] // self.grid_size)
                cell_y = int(self.chaser_pos[1] // self.grid_size)
                if 0 <= cell_x < self.visited_grid.shape[0] and 0 <= cell_y < self.visited_grid.shape[1]:
                    if self.visited_grid[cell_x, cell_y] == 0:
                        reward += 1.0  # Tăng từ 0.5 lên 1.0
                        self.visited_grid[cell_x, cell_y] = 1
                
                if throttle_normalized < 0.2:
                    reward -= 2.0
                elif throttle_normalized < 0.5:
                    reward -= 0.5
                
                if self.consecutive_visible_steps == 0 and self.prev_dist_to_evader < self.EVADER_SENSOR_RADIUS * 1.5:
                    last_known_direction = self.evader_pos - self.chaser_pos
                    current_direction = np.array([math.cos(self.chaser_heading), math.sin(self.chaser_heading)])
                    dot_product = np.dot(last_known_direction, current_direction)
                    if dot_product > 0:
                        reward += 0.5
                
                reward -= 0.2
        
        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            reward = TIMEOUT_PENALTY
            
        self.prev_dist_to_evader = curr_dist
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Drone Pursuit")
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
            
        self._init_render()
        self.screen.fill((20, 20, 30))

        pygame.draw.rect(self.screen, (100, 0, 0), self.jam_zone)
        pygame.draw.rect(self.screen, (255, 0, 0), self.jam_zone, 2)
        
        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, (80, 80, 90), obs_rect)
            pygame.draw.rect(self.screen, (150, 150, 150), obs_rect, 2)
            
        self._draw_drone(self.evader_pos, math.atan2(self.evader_vel[1], self.evader_vel[0]), (0, 150, 255))
        self._draw_drone(self.chaser_pos, self.chaser_heading, (255, 150, 0))
        
        pygame.draw.circle(self.screen, (200, 200, 0), self.chaser_pos.astype(int), int(self.EVADER_SENSOR_RADIUS), 1)
        pygame.draw.circle(self.screen, (0, 100, 200), self.chaser_pos.astype(int), int(self.COLLISION_SENSOR_RADIUS), 1)
        pygame.draw.circle(self.screen, (0, 255, 0), self.evader_pos.astype(int), int(self.CAPTURE_RADIUS), 1)
        pygame.draw.line(self.screen, (100, 100, 100), self.chaser_pos, self.evader_pos, 1)

        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        text1 = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
        text2 = self.font.render(f"Distance: {dist:.1f}", True, (255, 255, 255))
        text3 = self.font.render(f"Visible Steps: {self.consecutive_visible_steps}", True, (255, 255, 255))
        text4 = self.font.render(f"Difficulty: {self.difficulty}", True, (255, 255, 255))
        self.screen.blit(text1, (10, 10))
        self.screen.blit(text2, (10, 35))
        self.screen.blit(text3, (10, 60))
        self.screen.blit(text4, (10, 85))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
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


if __name__ == "__main__":
    print("Running environment demo...")
    
    DIFFICULTY_TO_TEST = "hard"
    
    env = DronePursuitEnv(render_mode="human", difficulty=DIFFICULTY_TO_TEST) 
    
    obs, info = env.reset()
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended. Final reward: {reward:.2f}. Info: {info}")
            obs, info = env.reset()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
