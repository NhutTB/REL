"""
BEAUTIFUL Drone Pursuit Environment - Enhanced Visual Design
Features:
- Gradient backgrounds with parallax effect
- Particle systems for engine trails
- Glow effects and shadows
- Smooth animations
- Modern UI with glassmorphism
- Mini-map
- Performance graph
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Optional, Tuple, List
import math
import collections
from dataclasses import dataclass

@dataclass
class Particle:
    """Particle for trail effects"""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float

class ParticleSystem:
    """Manages particle effects"""
    def __init__(self):
        self.particles = []
    
    def emit(self, x, y, vx, vy, color, count=3):
        """Emit new particles"""
        for _ in range(count):
            # Add randomness
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            speed = np.random.uniform(0.5, 1.5)
            pvx = vx * speed * np.cos(angle) + np.random.uniform(-0.5, 0.5)
            pvy = vy * speed * np.sin(angle) + np.random.uniform(-0.5, 0.5)
            
            life = np.random.uniform(20, 40)
            self.particles.append(Particle(
                x=x, y=y,
                vx=pvx, vy=pvy,
                life=life,
                max_life=life,
                color=color,
                size=np.random.uniform(2, 5)
            ))
    
    def update(self):
        """Update all particles"""
        for p in self.particles[:]:
            p.x += p.vx
            p.y += p.vy
            p.vx *= 0.97  # Friction
            p.vy *= 0.97
            p.life -= 1
            
            if p.life <= 0:
                self.particles.remove(p)
    
    def draw(self, surface):
        """Draw all particles"""
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            size = int(p.size * (p.life / p.max_life))
            if size > 0:
                # Create glow effect
                for i in range(3):
                    glow_size = size + (3-i)*2
                    glow_alpha = alpha // (i+2)
                    s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*p.color, glow_alpha), 
                                     (glow_size, glow_size), glow_size)
                    surface.blit(s, (int(p.x)-glow_size, int(p.y)-glow_size))

class DronePhysics:
    """Enhanced drone physics"""
    def __init__(self, max_speed=5.0, max_accel=0.4, max_turn_rate=0.2):
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.heading = 0.0
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_turn_rate = max_turn_rate
        self.last_action = np.array([0.0, 0.0])
        self.pos_history = collections.deque(maxlen=5)
        self.vel_history = collections.deque(maxlen=5)

    def step(self, action: np.ndarray):
        throttle, turn_rate = action
        self.last_action = action
        
        self.pos_history.append(self.pos.copy())
        self.vel_history.append(self.vel.copy())
        
        self.heading += turn_rate * self.max_turn_rate
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        accel = throttle * self.max_accel
        ax = np.cos(self.heading) * accel
        ay = np.sin(self.heading) * accel
        
        self.vel += np.array([ax, ay])
        self.vel *= 0.98
        
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        
        self.pos += self.vel
    
    def predict_future_pos(self, steps_ahead=10):
        if len(self.vel_history) < 2:
            return self.pos + self.vel * steps_ahead
        recent_vel = np.mean(list(self.vel_history)[-3:], axis=0)
        return self.pos + recent_vel * steps_ahead

    def set_position(self, x: float, y: float):
        self.pos = np.array([x, y])

    def set_velocity(self, vx: float, vy: float):
        self.vel = np.array([vx, vy])

class Obstacle:
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def collides_with(self, pos: np.ndarray, margin: float = 0) -> bool:
        dist = np.linalg.norm(pos - np.array([self.x, self.y]))
        return dist < (self.radius + margin)

class JamZone:
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.active = True

    def contains(self, pos: np.ndarray) -> bool:
        if not self.active:
            return False
        dist = np.linalg.norm(pos - np.array([self.x, self.y]))
        return dist < self.radius

class SmartEvaderController:
    def __init__(self, waypoints: List[Tuple[float, float]], obstacles: List[Obstacle]):
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 30.0
        self.panic_radius = 200.0
        
    def get_action(self, evader_drone: DronePhysics, chaser_pos: np.ndarray) -> np.ndarray:
        if len(self.waypoints) == 0:
            return np.array([0.0, 0.0])
        
        dist_to_chaser = np.linalg.norm(evader_drone.pos - chaser_pos)
        
        if dist_to_chaser < self.panic_radius:
            panic_intensity = 1.0 - (dist_to_chaser / self.panic_radius)
            flee_vector = evader_drone.pos - chaser_pos
            noise_angle = np.random.uniform(-0.3, 0.3) * panic_intensity
            desired_heading = np.arctan2(flee_vector[1], flee_vector[0]) + noise_angle
            throttle = min(1.0, 0.7 + 0.3 * panic_intensity)
        else:
            target = np.array(self.waypoints[self.current_waypoint_idx])
            if np.linalg.norm(evader_drone.pos - target) < self.waypoint_threshold:
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
                target = np.array(self.waypoints[self.current_waypoint_idx])
            to_target = target - evader_drone.pos
            desired_heading = np.arctan2(to_target[1], to_target[0])
            throttle = 0.7
        
        heading_error = desired_heading - evader_drone.heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        turn_rate = np.clip(heading_error * 2.0, -1.0, 1.0)
        
        avoidance_force = np.array([0.0, 0.0])
        for obs in self.obstacles:
            obs_pos = np.array([obs.x, obs.y])
            to_obs = obs_pos - evader_drone.pos
            dist = np.linalg.norm(to_obs)
            if dist < obs.radius + 80:
                if dist > 0.1:
                    avoidance_force -= to_obs / (dist ** 2) * 2000
        
        if np.linalg.norm(avoidance_force) > 0.1:
            avoidance_heading = np.arctan2(avoidance_force[1], avoidance_force[0])
            avoidance_error = avoidance_heading - evader_drone.heading
            avoidance_error = (avoidance_error + np.pi) % (2 * np.pi) - np.pi
            turn_rate = np.clip(turn_rate + avoidance_error * 0.6, -1.0, 1.0)
        
        return np.array([throttle, turn_rate])

class DronePursuitEnv(gym.Env):
    """Beautiful Drone Pursuit Environment"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, config: dict = None):
        super().__init__()
        
        self.width = 1400
        self.height = 900
        self.max_steps = 2000
        self.capture_radius = 24.0
        self.drone_size = 32
        
        default_config = {
            'num_obstacles': 8,
            'jam_zone_radius': 100,
            'random_start': True,
        }
        self.config = {**default_config, **(config or {})}
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float32
        )
        
        self.chaser = None
        self.evader = None
        self.evader_controller = None
        self.obstacles = []
        self.jam_zone = None
        
        self.current_step = 0
        self.episode_reward = 0
        self.total_episodes = 0
        self.captures = 0
        self.jams = 0
        
        self.prev_distance = None
        self.distance_history = collections.deque(maxlen=10)
        self.closing_speed_history = collections.deque(maxlen=5)
        self.reward_history = collections.deque(maxlen=100)
        
        # Visual effects
        self.particle_system = ParticleSystem()
        self.background_stars = []
        self.camera_shake = 0
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.fonts = {}
        
        if self.render_mode == "human":
            self._init_pygame()
    
    def _generate_obstacles(self) -> List[Obstacle]:
        obstacles = []
        max_attempts = 100
        
        for _ in range(self.config['num_obstacles']):
            for attempt in range(max_attempts):
                radius = np.random.uniform(40, 80)
                x = np.random.uniform(radius + 100, self.width - radius - 100)
                y = np.random.uniform(radius + 100, self.height - radius - 100)
                
                valid = True
                for obs in obstacles:
                    if np.linalg.norm(np.array([x, y]) - np.array([obs.x, obs.y])) < radius + obs.radius + 30:
                        valid = False
                        break
                
                if valid:
                    obstacles.append(Obstacle(x, y, radius))
                    break
        return obstacles
    
    def _generate_waypoints(self, num_waypoints: int = 5) -> List[Tuple[float, float]]:
        waypoints = []
        for _ in range(num_waypoints):
            for attempt in range(100):
                x = np.random.uniform(100, self.width - 100)
                y = np.random.uniform(100, self.height - 100)
                pos = np.array([x, y])
                
                valid = True
                for obs in self.obstacles:
                    if obs.collides_with(pos, margin=40):
                        valid = False
                        break
                if self.jam_zone and self.jam_zone.contains(pos):
                    valid = False
                
                if valid:
                    waypoints.append((x, y))
                    break
        return waypoints
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0
        self.total_episodes += 1
        
        self.distance_history.clear()
        self.closing_speed_history.clear()
        self.particle_system.particles.clear()
        
        if self.render_mode == "human" and self.screen is None:
            self._init_pygame()
        
        self.obstacles = self._generate_obstacles()
        
        jam_x = self.width / 2 + np.random.uniform(-self.width * 0.2, self.width * 0.2)
        jam_y = self.height / 2 + np.random.uniform(-self.height * 0.2, self.height * 0.2)
        self.jam_zone = JamZone(jam_x, jam_y, self.config['jam_zone_radius'])
        
        self.chaser = DronePhysics(max_speed=5.0, max_accel=0.4, max_turn_rate=0.2)
        self.evader = DronePhysics(max_speed=3.5, max_accel=0.3, max_turn_rate=0.15)
        
        if self.config['random_start']:
            self.chaser.set_position(np.random.uniform(50, 200), np.random.uniform(100, self.height - 100))
            self.evader.set_position(np.random.uniform(self.width - 200, self.width - 50), np.random.uniform(100, self.height - 100))
        else:
            self.chaser.set_position(100, self.height / 2)
            self.evader.set_position(self.width - 100, self.height / 2)
        
        waypoints = self._generate_waypoints(6)
        self.evader_controller = SmartEvaderController(waypoints, self.obstacles)
        
        self.prev_distance = np.linalg.norm(self.evader.pos - self.chaser.pos)
        
        # Generate background stars
        self._generate_stars()
        
        return self._get_obs(), {}
    
    def _generate_stars(self):
        """Generate parallax background stars"""
        self.background_stars = []
        for _ in range(150):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            size = np.random.uniform(1, 3)
            layer = np.random.uniform(0.3, 1.0)  # Parallax layer
            self.background_stars.append([x, y, size, layer])
    
    def _get_obs(self) -> np.ndarray:
        """Enhanced observation"""
        rel_pos = self.evader.pos - self.chaser.pos
        rel_vel = self.evader.vel - self.chaser.vel
        dist_to_evader = np.linalg.norm(rel_pos)
        
        if np.linalg.norm(self.evader.vel) > 0.1:
            evader_future = self.evader.predict_future_pos(steps_ahead=10)
            rel_future = evader_future - self.chaser.pos
        else:
            rel_future = rel_pos
        
        closing_speed = -np.dot(rel_pos, rel_vel) / (dist_to_evader + 1e-6)
        
        dist_to_jam = np.linalg.norm(
            self.chaser.pos - np.array([self.jam_zone.x, self.jam_zone.y])
        ) - self.jam_zone.radius
        
        min_obs_dist = float('inf')
        for obs in self.obstacles:
            d = np.linalg.norm(self.chaser.pos - np.array([obs.x, obs.y])) - obs.radius
            min_obs_dist = min(min_obs_dist, d)
        
        ray_distances = []
        num_rays = 8
        ray_length = 200
        
        for i in range(num_rays):
            angle = self.chaser.heading + (2 * np.pi * i / num_rays)
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            min_dist = ray_length
            for obs in self.obstacles:
                to_obs = np.array([obs.x, obs.y]) - self.chaser.pos
                proj = np.dot(to_obs, ray_dir)
                if proj > 0:
                    closest_point = self.chaser.pos + ray_dir * proj
                    dist_to_center = np.linalg.norm(closest_point - np.array([obs.x, obs.y]))
                    if dist_to_center < obs.radius:
                        dist = proj - np.sqrt(obs.radius**2 - dist_to_center**2)
                        min_dist = min(min_dist, max(0, dist))
            ray_distances.append(min_dist / ray_length)
        
        angle_to_target = np.arctan2(rel_pos[1], rel_pos[0]) - self.chaser.heading
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        
        angle_to_future = np.arctan2(rel_future[1], rel_future[0]) - self.chaser.heading
        angle_to_future = (angle_to_future + np.pi) % (2 * np.pi) - np.pi
        
        rel_heading = self.evader.heading - self.chaser.heading
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
        
        if closing_speed > 0.1:
            time_to_intercept = dist_to_evader / closing_speed / 100.0
        else:
            time_to_intercept = 10.0
        
        obs = np.concatenate([
            rel_pos / max(self.width, self.height),
            rel_vel / 10.0,
            [dist_to_evader / max(self.width, self.height)],
            [dist_to_jam / 200.0],
            [self.chaser.heading / np.pi],
            self.chaser.vel / 10.0,
            [min_obs_dist / 200.0],
            ray_distances,
            [angle_to_target / np.pi],
            [angle_to_future / np.pi],
            [rel_heading / np.pi],
            [closing_speed / 10.0],
            [time_to_intercept],
            [self.current_step / self.max_steps],
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, prev_dist: float, curr_dist: float, 
                         terminated: bool, info: dict) -> float:
        """Optimized reward function"""
        reward = 0.0
        
        distance_change = prev_dist - curr_dist
        self.distance_history.append(distance_change)
        
        if distance_change > 0:
            reward += distance_change * 2.0
            if len(self.distance_history) >= 3:
                if all(d > 0 for d in list(self.distance_history)[-3:]):
                    reward += 0.5
        else:
            reward += distance_change * 3.0
        
        closing_speed = -np.dot(
            self.evader.pos - self.chaser.pos,
            self.evader.vel - self.chaser.vel
        ) / (curr_dist + 1e-6)
        
        if closing_speed > 0:
            reward += closing_speed * 0.3
        
        if curr_dist < 200:
            proximity_bonus = (200 - curr_dist) / 200 * 0.5
            reward += proximity_bonus
        
        to_target = self.evader.pos - self.chaser.pos
        target_heading = np.arctan2(to_target[1], to_target[0])
        heading_error = abs(target_heading - self.chaser.heading)
        heading_error = min(heading_error, 2*np.pi - heading_error)
        heading_alignment = 1.0 - (heading_error / np.pi)
        reward += heading_alignment * 0.2
        
        reward -= 0.02
        
        dist_to_jam = np.linalg.norm(
            self.chaser.pos - np.array([self.jam_zone.x, self.jam_zone.y])
        )
        if dist_to_jam < self.jam_zone.radius + 80:
            danger = (self.jam_zone.radius + 80 - dist_to_jam) / 80
            reward -= danger * 1.0
        
        for obs in self.obstacles:
            d = np.linalg.norm(self.chaser.pos - np.array([obs.x, obs.y])) - obs.radius
            if d < 50:
                reward -= (50 - d) / 50 * 0.5
        
        if terminated:
            if info.get('captured', False):
                base_reward = 200.0
                time_bonus = (self.max_steps - self.current_step) / self.max_steps * 100
                speed_multiplier = 1.0 + (time_bonus / 100)
                reward += base_reward * speed_multiplier
                self.camera_shake = 15  # Shake effect on capture
            elif info.get('jammed', False):
                reward -= 100.0
            elif info.get('collision', False):
                reward -= 80.0
        
        return reward
    
    def step(self, action: np.ndarray):
        self.current_step += 1
        
        prev_dist = np.linalg.norm(self.evader.pos - self.chaser.pos)
        
        self.chaser.step(action)
        evader_action = self.evader_controller.get_action(self.evader, self.chaser.pos)
        self.evader.step(evader_action)
        
        # Emit particles
        if self.chaser.last_action[0] > 0.1:
            angle = self.chaser.heading + np.pi
            self.particle_system.emit(
                self.chaser.pos[0] + np.cos(angle) * 15,
                self.chaser.pos[1] + np.sin(angle) * 15,
                -np.cos(angle) * 2,
                -np.sin(angle) * 2,
                (255, 150, 80),
                count=2
            )
        
        if self.evader.last_action[0] > 0.1:
            angle = self.evader.heading + np.pi
            self.particle_system.emit(
                self.evader.pos[0] + np.cos(angle) * 15,
                self.evader.pos[1] + np.sin(angle) * 15,
                -np.cos(angle) * 2,
                -np.sin(angle) * 2,
                (100, 150, 255),
                count=2
            )
        
        curr_dist = np.linalg.norm(self.evader.pos - self.chaser.pos)
        
        terminated = False
        info = {}
        
        if curr_dist < self.capture_radius:
            terminated = True
            info['captured'] = True
            self.captures += 1
        
        if self.jam_zone.contains(self.chaser.pos):
            terminated = True
            info['jammed'] = True
            self.jams += 1
        
        for obs in self.obstacles:
            if obs.collides_with(self.chaser.pos, margin=self.drone_size / 2):
                terminated = True
                info['collision'] = True
                break
        
        truncated = self.current_step >= self.max_steps
        
        reward = self._calculate_reward(prev_dist, curr_dist, terminated, info)
        self.episode_reward += reward
        self.reward_history.append(reward)
        
        obs = self._get_obs()
        info['distance'] = curr_dist
        info['episode_reward'] = self.episode_reward
        info['step'] = self.current_step
        
        self.prev_distance = curr_dist
        
        return obs, reward, terminated, truncated, info
    
    def _init_pygame(self):
        """Initialize Pygame with beautiful setup"""
        pygame.init()
        pygame.display.set_caption("🚁 Drone Pursuit - Enhanced Graphics")
        
        try:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.clock = pygame.time.Clock()
        
        # Load fonts
        try:
            self.fonts = {
                'title': pygame.font.Font(None, 48),
                'large': pygame.font.Font(None, 36),
                'normal': pygame.font.Font(None, 28),
                'small': pygame.font.Font(None, 22),
            }
        except:
            default_font = pygame.font.Font(None, 30)
            self.fonts = {
                'title': default_font,
                'large': default_font,
                'normal': default_font,
                'small': default_font,
            }

    def _draw_gradient_background(self):
        """Draw animated gradient background"""
        # Create gradient from dark blue to purple
        time_factor = self.current_step * 0.01
        
        for y in range(self.height):
            ratio = y / self.height
            
            # Animated colors
            r = 10 + 10 * (1 + math.sin(time_factor + ratio * 2))  # Range [10, 30]
            g = 15 + 15 * (1 + math.sin(time_factor * 0.7 + ratio * 3)) # Range [15, 45]
            b = 35 + 25 * (1 + math.sin(time_factor * 0.5 + ratio))   # Range [35, 85]
            
            pygame.draw.line(
                self.screen,
                (int(r), int(g), int(b)),
                (0, y),
                (self.width, y)
            )

        
        # Draw stars with parallax
        for star in self.background_stars:
            x, y, size, layer = star
            # Parallax effect based on camera shake
            offset_x = self.camera_shake * layer * 0.5
            offset_y = self.camera_shake * layer * 0.3
            
            alpha = int(150 * layer)
            s = pygame.Surface((int(size)*2, int(size)*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (200, 200, 255, alpha), (int(size), int(size)), int(size))
            self.screen.blit(s, (int(x + offset_x), int(y + offset_y)))

    def _draw_obstacles_beautiful(self):
        """Draw obstacles with glow and 3D effect"""
        for obs in self.obstacles:
            x, y, r = int(obs.x), int(obs.y), int(obs.radius)
            
            # Shadow
            shadow_surf = pygame.Surface((r*2+20, r*2+20), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 50), (r+10, r+10), r+5)
            self.screen.blit(shadow_surf, (x-r-10, y-r-10))
            
            # Outer glow
            for i in range(4):
                glow_r = r + (4-i)*5
                alpha = 30 // (i+1)
                glow_surf = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (100, 120, 180, alpha), (glow_r, glow_r), glow_r)
                self.screen.blit(glow_surf, (x-glow_r, y-glow_r))
            
            # Main obstacle with gradient
            for i in range(r, 0, -2):
                ratio = i / r
                color_val = int(60 + 100 * ratio)
                pygame.draw.circle(self.screen, (color_val, color_val+10, color_val+30), (x, y), i)
            
            # Highlight
            highlight_offset = r // 3
            pygame.draw.circle(self.screen, (180, 190, 220), 
                             (x - highlight_offset, y - highlight_offset), r // 4)
            
            # Border
            pygame.draw.circle(self.screen, (150, 170, 200), (x, y), r, 2)

    def _draw_jam_zone_beautiful(self):
        """Draw jam zone with animated effects"""
        if not self.jam_zone:
            return
        
        pulse = math.sin(self.current_step * 0.08) * 0.3 + 0.7
        pulse2 = math.sin(self.current_step * 0.12) * 0.2 + 0.8
        
        x, y = int(self.jam_zone.x), int(self.jam_zone.y)
        r = int(self.jam_zone.radius)
        
        # Multiple pulsing circles
        for i in range(5):
            radius_offset = (self.current_step * 2 + i * 30) % 150
            alpha = int(100 * (1 - radius_offset / 150))
            
            circle_surf = pygame.Surface(((r + radius_offset)*2, (r + radius_offset)*2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (255, 50, 50, alpha), 
                             (r + radius_offset, r + radius_offset), r + radius_offset, 3)
            self.screen.blit(circle_surf, (x - r - radius_offset, y - r - radius_offset))
        
        # Main jam zone with gradient
        jam_surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        for i in range(r, 0, -5):
            ratio = i / r
            alpha = int(40 * pulse * ratio)
            pygame.draw.circle(jam_surf, (255, 80, 80, alpha), (r, r), i)
        self.screen.blit(jam_surf, (x - r, y - r))
        
        # Border
        border_alpha = int(255 * pulse2)
        pygame.draw.circle(self.screen, (255, 100, 100, border_alpha), (x, y), r, 4)
        
        # Warning text with glow
        text = self.fonts['large'].render("⚠ JAM ZONE ⚠", True, (255, 200, 200))
        text_glow = self.fonts['large'].render("⚠ JAM ZONE ⚠", True, (255, 50, 50))
        
        glow_offset = int(3 * pulse)
        for dx, dy in [(-glow_offset, 0), (glow_offset, 0), (0, -glow_offset), (0, glow_offset)]:
            self.screen.blit(text_glow, (x - text.get_width()//2 + dx, y - 15 + dy))
        self.screen.blit(text, (x - text.get_width()//2, y - 15))

    def _draw_drones_beautiful(self):
        """Draw drones with modern design"""
        def draw_drone(drone, color, is_chaser=False):
            x, y = drone.pos[0], drone.pos[1]
            
            # Add camera shake offset
            shake_x = np.random.uniform(-self.camera_shake, self.camera_shake) if self.camera_shake > 0 else 0
            shake_y = np.random.uniform(-self.camera_shake, self.camera_shake) if self.camera_shake > 0 else 0
            x += shake_x
            y += shake_y
            
            heading = drone.heading
            size = self.drone_size / 2
            
            # Shadow
            shadow_points = []
            for angle_offset in [0, -2.5, 2.5]:
                angle = heading + angle_offset
                shadow_points.append((
                    x + (size+3) * math.cos(angle) + 5,
                    y + (size+3) * math.sin(angle) + 5
                ))
            
            shadow_surf = pygame.Surface((self.drone_size*3, self.drone_size*3), pygame.SRCALPHA)
            adjusted_shadow = [(p[0] - x + self.drone_size*1.5, p[1] - y + self.drone_size*1.5) 
                              for p in shadow_points]
            pygame.draw.polygon(shadow_surf, (0, 0, 0, 60), adjusted_shadow)
            self.screen.blit(shadow_surf, (x - self.drone_size*1.5, y - self.drone_size*1.5))
            
            # Glow effect
            glow_size = self.drone_size * 1.5
            glow_surf = pygame.Surface((int(glow_size*2), int(glow_size*2)), pygame.SRCALPHA)
            glow_color = (*color, 40)
            pygame.draw.circle(glow_surf, glow_color, (int(glow_size), int(glow_size)), int(glow_size))
            self.screen.blit(glow_surf, (int(x - glow_size), int(y - glow_size)))
            
            # Main body
            points = []
            for angle_offset in [0, -2.5, 2.5]:
                angle = heading + angle_offset
                points.append((
                    x + size * math.cos(angle),
                    y + size * math.sin(angle)
                ))
            
            # Draw with gradient
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.polygon(self.screen, dark_color, points)
            
            # Inner highlight
            inner_points = [(p[0] * 0.7 + x * 0.3, p[1] * 0.7 + y * 0.3) for p in points]
            light_color = tuple(min(255, c + 60) for c in color)
            pygame.draw.polygon(self.screen, light_color, inner_points)
            
            # Border with anti-aliasing
            pygame.draw.aalines(self.screen, (255, 255, 255), True, points, 2)
            
            # Direction indicator
            nose_x = x + size * 1.2 * math.cos(heading)
            nose_y = y + size * 1.2 * math.sin(heading)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(nose_x), int(nose_y)), 4)
            pygame.draw.circle(self.screen, color, (int(nose_x), int(nose_y)), 2)
            
            # Capture radius for evader
            if not is_chaser:
                radius_surf = pygame.Surface((self.capture_radius*4, self.capture_radius*4), pygame.SRCALPHA)
                pulse = math.sin(self.current_step * 0.1) * 0.3 + 0.7
                
                for i in range(3):
                    r = int(self.capture_radius * (1 + i * 0.1))
                    alpha = int(50 * pulse / (i + 1))
                    pygame.draw.circle(radius_surf, (*color, alpha), 
                                     (self.capture_radius*2, self.capture_radius*2), r, 2)
                
                self.screen.blit(radius_surf, (int(x - self.capture_radius*2), int(y - self.capture_radius*2)))
            
            # Velocity vector
            if np.linalg.norm(drone.vel) > 0.5:
                vel_scale = 20
                vel_end_x = x + drone.vel[0] * vel_scale
                vel_end_y = y + drone.vel[1] * vel_scale
                
                # Draw arrow
                pygame.draw.line(self.screen, (*color, 180), 
                               (int(x), int(y)), (int(vel_end_x), int(vel_end_y)), 2)
                
                # Arrow head
                arrow_angle = math.atan2(drone.vel[1], drone.vel[0])
                arrow_size = 8
                arrow_points = [
                    (vel_end_x, vel_end_y),
                    (vel_end_x - arrow_size * math.cos(arrow_angle - 0.5),
                     vel_end_y - arrow_size * math.sin(arrow_angle - 0.5)),
                    (vel_end_x - arrow_size * math.cos(arrow_angle + 0.5),
                     vel_end_y - arrow_size * math.sin(arrow_angle + 0.5))
                ]
                pygame.draw.polygon(self.screen, color, arrow_points)
        
        # Draw evader first (behind)
        draw_drone(self.evader, (80, 150, 255), False)
        
        # Draw chaser on top
        draw_drone(self.chaser, (255, 80, 80), True)
        
        # Draw connection line
        dist = np.linalg.norm(self.evader.pos - self.chaser.pos)
        if dist < 300:
            alpha = int(100 * (1 - dist / 300))
            line_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(line_surf, (255, 255, 100, alpha),
                           self.chaser.pos.astype(int), 
                           self.evader.pos.astype(int), 2)
            self.screen.blit(line_surf, (0, 0))
            
            # Distance text
            mid_x = (self.chaser.pos[0] + self.evader.pos[0]) / 2
            mid_y = (self.chaser.pos[1] + self.evader.pos[1]) / 2
            dist_text = self.fonts['small'].render(f"{dist:.0f}m", True, (255, 255, 150))
            self.screen.blit(dist_text, (int(mid_x) - 20, int(mid_y) - 10))

    def _draw_modern_hud(self):
        """Draw modern HUD with glassmorphism"""
        # Semi-transparent panel
        hud_width = 320
        hud_height = 280
        hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        
        # Glassmorphism background
        pygame.draw.rect(hud_surf, (20, 30, 50, 180), (0, 0, hud_width, hud_height), border_radius=15)
        pygame.draw.rect(hud_surf, (100, 150, 200, 100), (0, 0, hud_width, hud_height), 2, border_radius=15)
        
        self.screen.blit(hud_surf, (20, 20))
        
        # HUD content
        y_offset = 35
        
        # Title
        title = self.fonts['normal'].render("🎯 Mission Status", True, (200, 220, 255))
        self.screen.blit(title, (35, y_offset))
        y_offset += 40
        
        # Stats
        stats = [
            (f"Episode: #{self.total_episodes}", (180, 220, 255)),
            (f"Step: {self.current_step}/{self.max_steps}", (255, 255, 255)),
            (f"Distance: {np.linalg.norm(self.evader.pos - self.chaser.pos):.1f}m", 
             (255, 200, 100)),
            (f"Reward: {self.episode_reward:.1f}", 
             (100, 255, 150) if self.episode_reward > 0 else (255, 100, 100)),
        ]
        
        for text, color in stats:
            surface = self.fonts['small'].render(text, True, color)
            self.screen.blit(surface, (35, y_offset))
            y_offset += 28
        
        # Progress bar
        y_offset += 10
        bar_width = 260
        bar_height = 20
        progress = self.current_step / self.max_steps
        
        # Background
        pygame.draw.rect(self.screen, (40, 50, 70), (35, y_offset, bar_width, bar_height), border_radius=10)
        
        # Progress
        if progress > 0:
            progress_width = int(bar_width * progress)
            color = (100, 255, 150) if progress < 0.8 else (255, 200, 100)
            pygame.draw.rect(self.screen, color, (35, y_offset, progress_width, bar_height), border_radius=10)
        
        # Border
        pygame.draw.rect(self.screen, (150, 170, 200), (35, y_offset, bar_width, bar_height), 2, border_radius=10)
        
        y_offset += 30
        
        # Statistics
        stats_text = self.fonts['small'].render(f"✓ Captures: {self.captures}  ✗ Jammed: {self.jams}", 
                                                True, (200, 200, 200))
        self.screen.blit(stats_text, (35, y_offset))

    def _draw_mini_map(self):
        """Draw mini-map in corner"""
        map_size = 180
        map_x = self.width - map_size - 20
        map_y = 20
        
        # Background
        map_surf = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
        pygame.draw.rect(map_surf, (20, 30, 50, 180), (0, 0, map_size, map_size), border_radius=10)
        pygame.draw.rect(map_surf, (100, 150, 200, 100), (0, 0, map_size, map_size), 2, border_radius=10)
        
        # Scale factor
        scale_x = map_size / self.width
        scale_y = map_size / self.height
        
        # Draw obstacles
        for obs in self.obstacles:
            x = int(obs.x * scale_x)
            y = int(obs.y * scale_y)
            r = max(2, int(obs.radius * scale_x))
            pygame.draw.circle(map_surf, (100, 120, 150), (x, y), r)
        
        # Draw jam zone
        if self.jam_zone:
            jx = int(self.jam_zone.x * scale_x)
            jy = int(self.jam_zone.y * scale_y)
            jr = max(3, int(self.jam_zone.radius * scale_x))
            pygame.draw.circle(map_surf, (255, 100, 100, 100), (jx, jy), jr)
            pygame.draw.circle(map_surf, (255, 100, 100), (jx, jy), jr, 1)
        
        # Draw drones
        chaser_x = int(self.chaser.pos[0] * scale_x)
        chaser_y = int(self.chaser.pos[1] * scale_y)
        pygame.draw.circle(map_surf, (255, 100, 100), (chaser_x, chaser_y), 5)
        pygame.draw.circle(map_surf, (255, 255, 255), (chaser_x, chaser_y), 5, 1)
        
        evader_x = int(self.evader.pos[0] * scale_x)
        evader_y = int(self.evader.pos[1] * scale_y)
        pygame.draw.circle(map_surf, (100, 150, 255), (evader_x, evader_y), 5)
        pygame.draw.circle(map_surf, (255, 255, 255), (evader_x, evader_y), 5, 1)
        
        # Connection line
        pygame.draw.line(map_surf, (255, 255, 100, 100), 
                        (chaser_x, chaser_y), (evader_x, evader_y), 1)
        
        self.screen.blit(map_surf, (map_x, map_y))
        
        # Mini-map title
        title = self.fonts['small'].render("MAP", True, (200, 220, 255))
        self.screen.blit(title, (map_x + 10, map_y + 5))

    def _draw_performance_graph(self):
        """Draw real-time reward graph"""
        if len(self.reward_history) < 2:
            return
        
        graph_width = 300
        graph_height = 100
        graph_x = self.width - graph_width - 20
        graph_y = self.height - graph_height - 20
        
        # Background
        graph_surf = pygame.Surface((graph_width, graph_height), pygame.SRCALPHA)
        pygame.draw.rect(graph_surf, (20, 30, 50, 180), (0, 0, graph_width, graph_height), border_radius=10)
        pygame.draw.rect(graph_surf, (100, 150, 200, 100), (0, 0, graph_width, graph_height), 2, border_radius=10)
        
        # Draw graph
        rewards = list(self.reward_history)
        if len(rewards) > 1:
            max_reward = max(max(rewards), 1)
            min_reward = min(min(rewards), -1)
            range_reward = max_reward - min_reward
            
            if range_reward > 0:
                points = []
                for i, r in enumerate(rewards):
                    x = int((i / len(rewards)) * (graph_width - 20)) + 10
                    y = int(graph_height - 10 - ((r - min_reward) / range_reward) * (graph_height - 20))
                    points.append((x, y))
                
                # Draw line
                if len(points) > 1:
                    pygame.draw.lines(graph_surf, (100, 255, 150), False, points, 2)
                
                # Draw zero line
                zero_y = int(graph_height - 10 - ((0 - min_reward) / range_reward) * (graph_height - 20))
                pygame.draw.line(graph_surf, (200, 200, 200, 100), (10, zero_y), (graph_width - 10, zero_y), 1)
        
        self.screen.blit(graph_surf, (graph_x, graph_y))
        
        # Title
        title = self.fonts['small'].render("Reward History", True, (200, 220, 255))
        self.screen.blit(title, (graph_x + 10, graph_y + 5))

    def render(self):
        if self.render_mode != "human":
            return
        
        if self.screen is None:
            self._init_pygame()
        
        # Update effects
        self.particle_system.update()
        if self.camera_shake > 0:
            self.camera_shake *= 0.9
            if self.camera_shake < 0.5:
                self.camera_shake = 0
        
        # Draw everything
        self._draw_gradient_background()
        self._draw_obstacles_beautiful()
        self._draw_jam_zone_beautiful()
        
        # Draw particles
        self.particle_system.draw(self.screen)
        
        self._draw_drones_beautiful()
        self._draw_modern_hud()
        self._draw_mini_map()
        self._draw_performance_graph()
        
        # FPS counter
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps > 50 else (255, 200, 100) if fps > 30 else (255, 100, 100)
        fps_text = self.fonts['small'].render(f"FPS: {fps}", True, fps_color)
        self.screen.blit(fps_text, (self.width - 100, self.height - 30))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    print("🚁 Beautiful Drone Pursuit Environment")
    print("=" * 60)
    print("Press ESC to exit")
    print("=" * 60)
    
    env = DronePursuitEnv(render_mode="human")
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            try:
                env.render()
            except pygame.error:
                print("Pygame error - window closed")
                done = True
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
        
        if not env.screen:
            break
        
        print(f"Episode {episode + 1}: {info}")
    
    env.close()
    print("\n✅ Demo completed!")