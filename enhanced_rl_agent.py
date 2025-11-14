"""
Enhanced RL Farming Agent with Full Perception System (Phase 2 Complete)
This file integrates all perception modules into the RL agent

DEPENDENCIES - Run these commands first:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python numpy mss pyautogui pillow
    pip install easyocr pytesseract
    
Or simply run:
    pip install torch torchvision opencv-python numpy mss pyautogui pillow easyocr pytesseract
"""

import sys
import subprocess

# Auto-install dependencies if missing
def check_and_install_dependencies():
    """Check and install required packages"""
    required = {
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'mss': 'mss',
        'pyautogui': 'pyautogui',
        'PIL': 'pillow',
        'easyocr': 'easyocr'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✓ Dependencies installed!")

# Uncomment to auto-install (disabled by default)
# check_and_install_dependencies()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import mss
import pyautogui
import time
from dataclasses import dataclass
from collections import deque
from typing import Tuple, List, Optional, Dict
import json
import pickle
from pathlib import Path
import heapq

# Import perception modules
from perception.health_detection import HealthDetector, HealthManaState
from perception.enemy_detection import EnemyDetector, EnemyState
from perception.reward_detection import RewardDetector, RewardEvent, RewardState


# ============================================================================
# SPATIAL MEMORY & NAVIGATION SYSTEM
# ============================================================================

class SpatialMemory:
    """
    Persistent spatial memory for building an internal map of the game world
    
    Features:
    - Occupancy grid for obstacles and traversable areas
    - Visited location tracking
    - Enemy spawn point memory
    - Safe zone identification
    - Death location recording
    """
    
    def __init__(self, map_size: Tuple[int, int] = (500, 500), cell_size: float = 5.0):
        """
        Initialize spatial memory
        
        Args:
            map_size: Size of the map in cells (width, height)
            cell_size: Size of each cell in game units (affects resolution)
        """
        self.map_size = map_size
        self.cell_size = cell_size
        
        # Occupancy grid: 0 = unknown, 1 = free, 2 = obstacle, 3 = dangerous
        self.occupancy_grid = np.zeros(map_size, dtype=np.uint8)
        
        # Visited locations (heatmap - higher = more visits)
        self.visit_heatmap = np.zeros(map_size, dtype=np.float32)
        
        # Enemy spawn locations (position -> count)
        self.enemy_spawns = {}  # {(x, y): spawn_count}
        
        # Safe zones (areas with no enemies for extended periods)
        self.safe_zones = []  # List of (x, y, radius)
        
        # Death locations (to avoid)
        self.death_locations = []  # List of (x, y, timestamp)
        
        # Loot locations (high-value areas)
        self.loot_locations = {}  # {(x, y): loot_count}
        
        # Center of map (starting position)
        self.center = (map_size[0] // 2, map_size[1] // 2)
        
        # Mark center as free initially
        self.occupancy_grid[self.center[1], self.center[0]] = 1
        
        # Statistics
        self.total_updates = 0
        self.obstacles_found = 0
        self.safe_zones_found = 0
    
    def world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_pos[0] / self.cell_size) + self.center[0]
        grid_y = int(world_pos[1] / self.cell_size) + self.center[1]
        
        # Clamp to map bounds
        grid_x = max(0, min(self.map_size[0] - 1, grid_x))
        grid_y = max(0, min(self.map_size[1] - 1, grid_y))
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = (grid_pos[0] - self.center[0]) * self.cell_size
        world_y = (grid_pos[1] - self.center[1]) * self.cell_size
        return (world_x, world_y)
    
    def update_position(self, position: Tuple[float, float], is_obstacle: bool = False):
        """
        Update map with current position
        
        Args:
            position: World position (x, y)
            is_obstacle: True if position is blocked
        """
        grid_pos = self.world_to_grid(position)
        
        if is_obstacle:
            self.occupancy_grid[grid_pos[1], grid_pos[0]] = 2
            self.obstacles_found += 1
        else:
            # Mark as free if previously unknown
            if self.occupancy_grid[grid_pos[1], grid_pos[0]] == 0:
                self.occupancy_grid[grid_pos[1], grid_pos[0]] = 1
            
            # Update visit heatmap
            self.visit_heatmap[grid_pos[1], grid_pos[0]] += 1
        
        self.total_updates += 1
    
    def record_enemy_spawn(self, position: Tuple[float, float]):
        """Record enemy spawn location"""
        grid_pos = self.world_to_grid(position)
        
        if grid_pos in self.enemy_spawns:
            self.enemy_spawns[grid_pos] += 1
        else:
            self.enemy_spawns[grid_pos] = 1
    
    def record_death(self, position: Tuple[float, float]):
        """Record death location"""
        grid_pos = self.world_to_grid(position)
        self.death_locations.append((grid_pos[0], grid_pos[1], time.time()))
        
        # Mark surrounding area as dangerous
        radius = 3
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = grid_pos[0] + dx, grid_pos[1] + dy
                if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                    if self.occupancy_grid[ny, nx] != 2:  # Don't override obstacles
                        self.occupancy_grid[ny, nx] = 3  # Dangerous
    
    def record_loot(self, position: Tuple[float, float]):
        """Record loot collection location"""
        grid_pos = self.world_to_grid(position)
        
        if grid_pos in self.loot_locations:
            self.loot_locations[grid_pos] += 1
        else:
            self.loot_locations[grid_pos] = 1
    
    def is_safe(self, position: Tuple[float, float], radius: float = 10.0) -> bool:
        """
        Check if position is in a safe zone (no recent deaths nearby)
        
        Args:
            position: World position
            radius: Safety check radius
            
        Returns:
            True if position is safe
        """
        grid_pos = self.world_to_grid(position)
        current_time = time.time()
        
        # Check recent deaths (last 5 minutes)
        for death_x, death_y, death_time in self.death_locations:
            if current_time - death_time < 300:  # 5 minutes
                distance = ((grid_pos[0] - death_x)**2 + (grid_pos[1] - death_y)**2)**0.5
                if distance < radius / self.cell_size:
                    return False
        
        return True
    
    def is_obstacle(self, position: Tuple[float, float]) -> bool:
        """Check if position is an obstacle"""
        grid_pos = self.world_to_grid(position)
        return self.occupancy_grid[grid_pos[1], grid_pos[0]] == 2
    
    def is_explored(self, position: Tuple[float, float]) -> bool:
        """Check if position has been visited"""
        grid_pos = self.world_to_grid(position)
        return self.visit_heatmap[grid_pos[1], grid_pos[0]] > 0
    
    def get_unexplored_nearby(self, position: Tuple[float, float], radius: float = 50.0) -> List[Tuple[float, float]]:
        """
        Find unexplored areas near position
        
        Args:
            position: Current world position
            radius: Search radius
            
        Returns:
            List of unexplored world positions
        """
        grid_pos = self.world_to_grid(position)
        grid_radius = int(radius / self.cell_size)
        
        unexplored = []
        for dy in range(-grid_radius, grid_radius + 1, 5):  # Sample every 5 cells
            for dx in range(-grid_radius, grid_radius + 1, 5):
                nx, ny = grid_pos[0] + dx, grid_pos[1] + dy
                
                if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                    # Check if unexplored and not obstacle
                    if self.visit_heatmap[ny, nx] == 0 and self.occupancy_grid[ny, nx] != 2:
                        world_pos = self.grid_to_world((nx, ny))
                        unexplored.append(world_pos)
        
        return unexplored
    
    def get_high_value_areas(self, position: Tuple[float, float], top_n: int = 5) -> List[Tuple[float, float]]:
        """
        Get high-value farming areas (enemy spawns + loot)
        
        Args:
            position: Current position
            top_n: Number of areas to return
            
        Returns:
            List of world positions sorted by value
        """
        # Combine enemy spawns and loot locations
        value_map = {}
        
        for grid_pos, count in self.enemy_spawns.items():
            value_map[grid_pos] = value_map.get(grid_pos, 0) + count * 2  # Enemies worth 2x
        
        for grid_pos, count in self.loot_locations.items():
            value_map[grid_pos] = value_map.get(grid_pos, 0) + count
        
        # Sort by value
        sorted_areas = sorted(value_map.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to world coordinates and filter by safety
        high_value = []
        for grid_pos, value in sorted_areas[:top_n * 2]:  # Get extra in case some are unsafe
            world_pos = self.grid_to_world(grid_pos)
            if self.is_safe(world_pos):
                high_value.append(world_pos)
                if len(high_value) >= top_n:
                    break
        
        return high_value
    
    def save(self, filepath: str = "spatial_memory.pkl"):
        """Save spatial memory to disk"""
        data = {
            'occupancy_grid': self.occupancy_grid,
            'visit_heatmap': self.visit_heatmap,
            'enemy_spawns': self.enemy_spawns,
            'safe_zones': self.safe_zones,
            'death_locations': self.death_locations,
            'loot_locations': self.loot_locations,
            'map_size': self.map_size,
            'cell_size': self.cell_size,
            'total_updates': self.total_updates,
            'obstacles_found': self.obstacles_found,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[SpatialMemory] Saved to {filepath}")
    
    def load(self, filepath: str = "spatial_memory.pkl"):
        """Load spatial memory from disk"""
        if not Path(filepath).exists():
            print(f"[SpatialMemory] File not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.occupancy_grid = data['occupancy_grid']
        self.visit_heatmap = data['visit_heatmap']
        self.enemy_spawns = data['enemy_spawns']
        self.safe_zones = data['safe_zones']
        self.death_locations = data['death_locations']
        self.loot_locations = data['loot_locations']
        self.map_size = data['map_size']
        self.cell_size = data['cell_size']
        self.total_updates = data['total_updates']
        self.obstacles_found = data['obstacles_found']
        
        print(f"[SpatialMemory] Loaded from {filepath}")
        print(f"  Total updates: {self.total_updates}")
        print(f"  Obstacles found: {self.obstacles_found}")
        print(f"  Enemy spawns: {len(self.enemy_spawns)}")
        print(f"  Death locations: {len(self.death_locations)}")
        
        return True
    
    def get_visualization(self) -> np.ndarray:
        """
        Generate visualization of the map
        
        Returns:
            RGB image of the map
        """
        vis = np.zeros((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        
        # Draw occupancy grid
        # Unknown = black, Free = white, Obstacle = red, Dangerous = orange
        vis[self.occupancy_grid == 0] = [0, 0, 0]      # Unknown - black
        vis[self.occupancy_grid == 1] = [255, 255, 255]  # Free - white
        vis[self.occupancy_grid == 2] = [0, 0, 255]    # Obstacle - red
        vis[self.occupancy_grid == 3] = [0, 165, 255]  # Dangerous - orange
        
        # Overlay visit heatmap (green gradient)
        visit_normalized = np.clip(self.visit_heatmap / max(1, np.max(self.visit_heatmap)), 0, 1)
        green_channel = (visit_normalized * 255).astype(np.uint8)
        vis[:, :, 1] = np.maximum(vis[:, :, 1], green_channel)
        
        # Mark enemy spawns (yellow)
        for (gx, gy), count in self.enemy_spawns.items():
            intensity = min(255, count * 50)
            cv2.circle(vis, (gx, gy), 3, (0, intensity, intensity), -1)
        
        # Mark death locations (purple)
        for gx, gy, _ in self.death_locations:
            cv2.circle(vis, (gx, gy), 5, (255, 0, 255), 2)
        
        # Mark center
        cv2.circle(vis, self.center, 7, (255, 255, 0), -1)
        
        return vis


class Navigator:
    """
    Intelligent navigation system with pathfinding and obstacle avoidance
    
    Features:
    - A* pathfinding
    - Smooth path following
    - Obstacle avoidance
    - Position tracking and odometry
    """
    
    def __init__(self, spatial_memory: SpatialMemory):
        """
        Initialize navigator
        
        Args:
            spatial_memory: Reference to spatial memory
        """
        self.memory = spatial_memory
        
        # Current position estimate (world coordinates)
        self.current_position = (0.0, 0.0)
        
        # Movement history for odometry
        self.movement_history = deque(maxlen=100)
        
        # Current path (list of waypoints)
        self.current_path = []
        self.current_waypoint_idx = 0
        
        # Navigation state
        self.is_navigating = False
        self.navigation_target = None
        
        # Odometry parameters
        self.movement_speed = 5.0  # units per action (estimated)
        self.last_update_time = time.time()
    
    def update_position_from_minimap(self, minimap: np.ndarray) -> Tuple[float, float]:
        """
        Estimate position from minimap analysis
        
        Args:
            minimap: Minimap image
            
        Returns:
            Estimated world position (x, y)
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Look for player indicator (usually bright/white dot on minimap)
            # This is game-specific - adjust thresholds based on Evil Lands minimap
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours (player position indicators)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (likely player indicator)
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                
                if M['m00'] > 0:
                    # Calculate center of contour
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Convert minimap pixel coordinates to world coordinates
                    # Minimap usually shows relative position, so we need to estimate scale
                    minimap_center_x = minimap.shape[1] // 2
                    minimap_center_y = minimap.shape[0] // 2
                    
                    # Estimate world offset (this is approximate - may need tuning)
                    scale_factor = 2.0  # pixels to world units (adjust based on game)
                    world_dx = (cx - minimap_center_x) * scale_factor
                    world_dy = (cy - minimap_center_y) * scale_factor
                    
                    # Update position relative to current estimate
                    estimated_pos = (
                        self.current_position[0] + world_dx,
                        self.current_position[1] + world_dy
                    )
                    
                    # Smooth position updates to avoid jitter
                    smoothing_factor = 0.3
                    self.current_position = (
                        self.current_position[0] * (1 - smoothing_factor) + estimated_pos[0] * smoothing_factor,
                        self.current_position[1] * (1 - smoothing_factor) + estimated_pos[1] * smoothing_factor
                    )
                    
                    return self.current_position
            
        except Exception as e:
            # If minimap analysis fails, fall back to odometry
            pass
        
        # Fallback to odometry-based estimation
        return self.current_position
    
    def update_odometry(self, action_id: int, actions_dict: Dict):
        """
        Update position estimate based on movement action
        
        Args:
            action_id: Action that was taken
            actions_dict: Dictionary of all actions
        """
        action = actions_dict.get(action_id, {})
        action_name = action.get('name', '')
        
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Estimate movement based on action
        dx, dy = 0.0, 0.0
        speed = self.movement_speed * dt
        
        if 'forward' in action_name:
            dy -= speed
        if 'backward' in action_name:
            dy += speed
        if 'left' in action_name:
            dx -= speed
        if 'right' in action_name:
            dx += speed
        
        # Update position
        self.current_position = (
            self.current_position[0] + dx,
            self.current_position[1] + dy
        )
        
        # Record movement
        self.movement_history.append((dx, dy, current_time))
        
        # Update spatial memory
        self.memory.update_position(self.current_position)
    
    def detect_collision(self, screen: np.ndarray, prev_screen: Optional[np.ndarray] = None) -> bool:
        """
        Detect if agent collided with obstacle (screen didn't change much after movement)
        
        Args:
            screen: Current screen
            prev_screen: Previous screen
            
        Returns:
            True if collision detected
        """
        if prev_screen is None:
            return False
        
        # Calculate frame difference
        diff = cv2.absdiff(screen, prev_screen)
        diff_score = np.mean(diff)
        
        # If screen barely changed, we likely hit an obstacle
        if diff_score < 5.0:  # Threshold for "no movement"
            # Mark current position as obstacle
            self.memory.update_position(self.current_position, is_obstacle=True)
            return True
        
        return False
    
    def plan_path(self, target: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Plan path from current position to target using A*
        
        Args:
            target: Target world position
            
        Returns:
            List of waypoints (world coordinates)
        """
        start_grid = self.memory.world_to_grid(self.current_position)
        goal_grid = self.memory.world_to_grid(target)
        
        # A* pathfinding
        path = self._astar(start_grid, goal_grid)
        
        if path:
            # Convert to world coordinates
            world_path = [self.memory.grid_to_world(p) for p in path]
            
            # Smooth path (remove redundant waypoints)
            world_path = self._smooth_path(world_path)
            
            return world_path
        
        return []
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm
        
        Args:
            start: Start grid position
            goal: Goal grid position
            
        Returns:
            List of grid positions (path)
        """
        # Priority queue: (f_score, counter, node)
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Check neighbors (8-directional)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.memory.map_size[0] and 
                       0 <= neighbor[1] < self.memory.map_size[1]):
                    continue
                
                # Check if obstacle
                if self.memory.occupancy_grid[neighbor[1], neighbor[0]] == 2:
                    continue
                
                # Calculate cost (diagonal moves cost more)
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                
                # Penalize dangerous areas
                if self.memory.occupancy_grid[neighbor[1], neighbor[0]] == 3:
                    move_cost *= 2.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        return []
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Smooth path by removing redundant waypoints
        
        Args:
            path: Original path
            
        Returns:
            Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Try to find furthest visible point
            for j in range(len(path) - 1, i, -1):
                if self._is_line_of_sight(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
        
        return smoothed
    
    def _is_line_of_sight(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """
        Check if there's a clear line of sight between two points
        
        Args:
            a: Start world position
            b: End world position
            
        Returns:
            True if clear line of sight
        """
        grid_a = self.memory.world_to_grid(a)
        grid_b = self.memory.world_to_grid(b)
        
        # Bresenham's line algorithm
        dx = abs(grid_b[0] - grid_a[0])
        dy = abs(grid_b[1] - grid_a[1])
        sx = 1 if grid_a[0] < grid_b[0] else -1
        sy = 1 if grid_a[1] < grid_b[1] else -1
        err = dx - dy
        
        x, y = grid_a
        
        while True:
            # Check if this cell is obstacle
            if self.memory.occupancy_grid[y, x] == 2:
                return False
            
            if (x, y) == grid_b:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def get_next_action(self, current_position: Tuple[float, float], 
                       target: Tuple[float, float]) -> Optional[int]:
        """
        Get next movement action to follow path
        
        Args:
            current_position: Current world position
            target: Target waypoint
            
        Returns:
            Action ID for movement (0-7) or None
        """
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]
        
        # Calculate distance
        distance = (dx**2 + dy**2)**0.5
        
        # If very close, we've reached waypoint
        if distance < 2.0:
            return None
        
        # Normalize direction
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # Map to action (8 directions)
        # Forward = -Y, Backward = +Y, Left = -X, Right = +X
        
        # Determine primary and secondary directions
        threshold = 0.3  # For diagonal movement
        
        action = None
        
        if abs(dy) > abs(dx):
            # Vertical movement dominant
            if dy < -threshold:
                # Forward
                if dx > threshold:
                    action = 5  # forward+right
                elif dx < -threshold:
                    action = 4  # forward+left
                else:
                    action = 0  # forward
            elif dy > threshold:
                # Backward
                if dx > threshold:
                    action = 7  # backward+right
                elif dx < -threshold:
                    action = 6  # backward+left
                else:
                    action = 1  # backward
        else:
            # Horizontal movement dominant
            if dx > threshold:
                # Right
                if dy < -threshold:
                    action = 5  # forward+right
                elif dy > threshold:
                    action = 7  # backward+right
                else:
                    action = 3  # right
            elif dx < -threshold:
                # Left
                if dy < -threshold:
                    action = 4  # forward+left
                elif dy > threshold:
                    action = 6  # backward+left
                else:
                    action = 2  # left
        
        return action
    
    def start_navigation(self, target: Tuple[float, float]) -> bool:
        """
        Start navigation to target
        
        Args:
            target: Target world position
            
        Returns:
            True if path found
        """
        path = self.plan_path(target)
        
        if path:
            self.current_path = path
            self.current_waypoint_idx = 0
            self.is_navigating = True
            self.navigation_target = target
            return True
        
        return False
    
    def update_navigation(self) -> Optional[int]:
        """
        Update navigation and get next action
        
        Returns:
            Next movement action or None if navigation complete
        """
        if not self.is_navigating or not self.current_path:
            return None
        
        # Get current waypoint
        if self.current_waypoint_idx >= len(self.current_path):
            # Reached destination
            self.is_navigating = False
            return None
        
        waypoint = self.current_path[self.current_waypoint_idx]
        
        # Get action to reach waypoint
        action = self.get_next_action(self.current_position, waypoint)
        
        if action is None:
            # Reached waypoint, move to next
            self.current_waypoint_idx += 1
            return self.update_navigation()
        
        return action
    
    def cancel_navigation(self):
        """Cancel current navigation"""
        self.is_navigating = False
        self.current_path = []
        self.current_waypoint_idx = 0
        self.navigation_target = None


@dataclass
class EnhancedGameState:
    """
    Enhanced game state with full perception + spatial awareness
    """
    # Visual input
    screen: np.ndarray
    minimap: np.ndarray
    
    # Health/XP (from HealthDetector)
    health_percentage: float
    health_current: int
    health_max: int
    xp_percentage: float
    xp_current: int
    xp_max: int
    is_low_health: bool
    is_critical: bool
    
    # Enemies (from EnemyDetector)
    enemy_count: int
    has_target: bool
    nearest_enemy_distance: float  # 0-1
    enemies_visible: List  # Full enemy list
    
    # Rewards (from RewardDetector)
    recent_kills: int  # In last 5 seconds
    recent_loot: int
    recent_damage_dealt: float
    recent_damage_taken: float
    
    # Combat state
    is_in_combat: bool
    
    # Timestamp
    timestamp: float
    
    # Spatial awareness (NEW)
    current_position: Tuple[float, float] = (0.0, 0.0)
    is_navigating: bool = False
    has_path: bool = False
    distance_to_goal: float = 0.0
    is_stuck: bool = False
    nearby_obstacles: int = 0
    exploration_progress: float = 0.0  # 0-1, how much of map explored
    is_in_safe_zone: bool = True
    
    def to_state_vector(self) -> np.ndarray:
        """
        Convert to fixed-size state vector for neural network
        
        Returns:
            State vector [health, xp_progress, enemy_count, spatial features, ...]
        """
        return np.array([
            self.health_percentage / 100.0,
            self.xp_percentage / 100.0,  # XP progress (not mana)
            float(self.is_low_health),
            float(self.is_critical),
            min(self.enemy_count / 10.0, 1.0),  # Normalize to 0-1
            float(self.has_target),
            self.nearest_enemy_distance,
            float(self.is_in_combat),
            min(self.recent_kills / 5.0, 1.0),
            min(self.recent_loot / 5.0, 1.0),
            # Spatial features (NEW)
            float(self.is_navigating),
            float(self.has_path),
            min(self.distance_to_goal / 100.0, 1.0),  # Normalize
            float(self.is_stuck),
            min(self.nearby_obstacles / 20.0, 1.0),
            self.exploration_progress,
            float(self.is_in_safe_zone),
        ], dtype=np.float32)


class EnhancedDQNetwork(nn.Module):
    """
    Enhanced DQN with dual-input architecture + spatial awareness
    
    Input 1: Visual (screen + minimap) → CNN
    Input 2: Game state vector (now with spatial features) → FC layers
    Combined → Q-values
    """
    
    def __init__(self, num_actions=16, state_size=17):  # 17 state features (10 original + 7 spatial)
        super(EnhancedDQNetwork, self).__init__()
        
        # Visual stream (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size (for 84x84 input)
        self.conv_output_size = 3136
        
        # State vector stream (FC layers) - expanded for spatial features
        self.state_fc1 = nn.Linear(state_size, 128)  # 17 state features (health, enemies, spatial)
        self.state_fc2 = nn.Linear(128, 256)
        
        # Combined stream
        self.fc1 = nn.Linear(self.conv_output_size + 256, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        
    def forward(self, visual_input, state_vector):
        """
        Forward pass
        
        Args:
            visual_input: (batch, 3, 84, 84) screen image
            state_vector: (batch, 17) game state features (health, xp, enemies, combat, spatial, etc.)
        """
        # Visual stream
        x_vis = self.relu(self.conv1(visual_input))
        x_vis = self.relu(self.conv2(x_vis))
        x_vis = self.relu(self.conv3(x_vis))
        x_vis = x_vis.view(x_vis.size(0), -1)
        
        # State stream
        x_state = self.relu(self.state_fc1(state_vector))
        x_state = self.relu(self.state_fc2(x_state))
        
        # Combine
        x = torch.cat([x_vis, x_state], dim=1)
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


class EnhancedFarmingAgent:
    """
    State-of-the-art RL farming agent with full perception
    """
    
    # Action space (same as before)
    ACTIONS = {
        # Movement (8 directions)
        0: {'name': 'forward', 'keys': ['w']},
        1: {'name': 'backward', 'keys': ['s']},
        2: {'name': 'left', 'keys': ['a']},
        3: {'name': 'right', 'keys': ['d']},
        4: {'name': 'forward_left', 'keys': ['w', 'a']},
        5: {'name': 'forward_right', 'keys': ['w', 'd']},
        6: {'name': 'backward_left', 'keys': ['s', 'a']},
        7: {'name': 'backward_right', 'keys': ['s', 'd']},
        
        # Combat (Evil Lands specific)
        8: {'name': 'attack', 'keys': ['space']},          # Spam until red icon disappears
        9: {'name': 'collect_loot', 'keys': ['b']},        # Press B after kill
        
        # Camera control
        10: {'name': 'look_left', 'keys': ['num4']},
        11: {'name': 'look_right', 'keys': ['num6']},
        12: {'name': 'look_up', 'keys': ['num8']},
        13: {'name': 'look_down', 'keys': ['num5']},
        
        # Combined actions for efficiency
        14: {'name': 'attack_move_forward', 'keys': ['space', 'w']},  # Attack while moving
        15: {'name': 'attack_collect', 'keys': ['space', 'b']},       # Attack then loot
    }
    
    def __init__(self, config_path: Optional[str] = "config_rl.json"):
        """Initialize enhanced agent with spatial memory"""
        print("[EnhancedFarmingAgent] Initializing state-of-the-art RL agent with spatial awareness...")
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")
        
        # Initialize perception modules
        print("  Initializing perception systems...")
        
        # Ensure perception modules have access to game_region and minimap_region
        enemy_config = self.config.get('enemy_detection', {})
        if 'game_region' not in enemy_config:
            enemy_config['game_region'] = self.config.get('game_region', [0, 0, 1920, 1080])
        if 'minimap_region' not in enemy_config:
            enemy_config['minimap_region'] = self.config.get('minimap_region', [1670, 50, 200, 200])
        
        self.health_detector = HealthDetector(self.config.get('health_detection'))
        self.enemy_detector = EnemyDetector(enemy_config)
        self.reward_detector = RewardDetector(self.config.get('reward_detection'))
        
        # Initialize spatial memory and navigation (NEW)
        print("  Initializing spatial memory and navigation...")
        map_size = self.config.get('map_size', (500, 500))
        cell_size = self.config.get('cell_size', 5.0)
        self.spatial_memory = SpatialMemory(map_size=map_size, cell_size=cell_size)
        self.navigator = Navigator(self.spatial_memory)
        
        # Try loading existing spatial memory
        memory_file = self.config.get('spatial_memory_file', 'spatial_memory.pkl')
        if Path(memory_file).exists():
            self.spatial_memory.load(memory_file)
        
        # Initialize neural networks
        print("  Building neural networks...")
        self.policy_net = EnhancedDQNetwork(num_actions=len(self.ACTIONS)).to(self.device)
        self.target_net = EnhancedDQNetwork(num_actions=len(self.ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.get('learning_rate', 1e-4))
        
        # Experience replay
        self.memory = deque(maxlen=self.config.get('memory_size', 10000))
        
        # Training parameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Screen capture
        self.sct = mss.mss()
        self.game_region = self.config.get('game_region', [0, 0, 1920, 1080])
        self.minimap_region = self.config.get('minimap_region', [1670, 50, 200, 200])
        
        # State tracking
        self.current_state = None
        self.prev_screen = None
        self.episode_rewards = []
        self.episode_steps = 0
        
        # Navigation mode (NEW)
        self.exploration_mode = 'explore'  # 'explore', 'farm', 'patrol'
        self.exploration_target = None
        self.patrol_points = []  # High-value patrol points
        self.last_position_check = time.time()
        
        # TensorBoard (if available)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f'runs/farming_agent_{int(time.time())}')
            self.use_tensorboard = True
        except ImportError:
            print("  TensorBoard not available (install: pip install tensorboard)")
            self.use_tensorboard = False
        
        print("[EnhancedFarmingAgent] Initialization complete!\n")
        print(f"  Actions: {len(self.ACTIONS)}")
        print(f"  Memory size: {len(self.memory)}")
        print(f"  Epsilon: {self.epsilon}")
        print(f"  Spatial memory: {self.spatial_memory.total_updates} updates")
        print(f"  Map exploration: {self.spatial_memory.obstacles_found} obstacles found")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with spatial memory defaults"""
        default_config = {
            'game_region': [0, 0, 1920, 1080],
            'minimap_region': [1670, 50, 200, 200],
            'learning_rate': 1e-4,
            'memory_size': 10000,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            # Spatial memory configuration
            'map_size': (500, 500),  # Map dimensions in cells
            'cell_size': 5.0,        # Size of each cell in world units
            'spatial_memory_file': 'spatial_memory.pkl',
            # Navigation parameters
            'movement_speed': 5.0,   # Units per action (estimated)
            # Perception module configs
            'health_detection': {},
            'enemy_detection': {},
            'reward_detection': {},
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Deep merge: update top-level and nested configs
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                        # Merge nested dictionaries
                        default_config[key].update(value)
                    else:
                        # Replace top-level values
                        default_config[key] = value
                print(f"  Config loaded from {config_path}")
        except FileNotFoundError:
            print(f"  Config not found, using defaults (spatial memory enabled)")
        
        return default_config
    
    def capture_enhanced_state(self) -> EnhancedGameState:
        """
        Capture full game state with all perception systems + spatial awareness
        """
        # Capture screen
        game_region_dict = {
            "left": self.game_region[0],
            "top": self.game_region[1],
            "width": self.game_region[2],
            "height": self.game_region[3]
        }
        screenshot = self.sct.grab(game_region_dict)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Extract minimap
        mx, my, mw, mh = self.minimap_region
        minimap = screen[my:my+mh, mx:mx+mw]
        
        # Run perception systems
        health_state = self.health_detector.detect(screen)
        enemy_state = self.enemy_detector.detect(screen, minimap)
        reward_events = self.reward_detector.detect(screen, self.prev_screen)
        
        # Count recent events (last 5 seconds)
        current_time = time.time()
        recent_kills = sum(1 for e in self.reward_detector.event_history 
                          if e.event_type == 'kill' and current_time - e.timestamp < 5.0)
        recent_loot = sum(1 for e in self.reward_detector.event_history 
                         if e.event_type == 'loot' and current_time - e.timestamp < 5.0)
        recent_damage_dealt = sum(e.value for e in self.reward_detector.event_history 
                                  if e.event_type == 'damage_dealt' and current_time - e.timestamp < 2.0)
        recent_damage_taken = sum(e.value for e in self.reward_detector.event_history 
                                  if e.event_type == 'damage_taken' and current_time - e.timestamp < 2.0)
        
        # Determine combat state (has enemies or taking damage)
        is_in_combat = enemy_state.enemy_count > 0 or recent_damage_taken > 0
        
        # Update spatial information (NEW)
        # Update position from minimap (with fallback to odometry)
        self.navigator.update_position_from_minimap(minimap)
        current_position = self.navigator.current_position
        
        # Check for collision/obstacle detection
        if self.prev_screen is not None:
            collision = self.navigator.detect_collision(screen, self.prev_screen)
        
        # Record enemy spawns
        for enemy in enemy_state.enemies:
            # Convert screen position to world position (approximate)
            enemy_world_pos = self._screen_to_world_position(enemy.position, current_position)
            self.spatial_memory.record_enemy_spawn(enemy_world_pos)
        
        # Record loot if collected recently
        if recent_loot > 0:
            self.spatial_memory.record_loot(current_position)
        
        # Calculate spatial features
        is_navigating = self.navigator.is_navigating
        has_path = len(self.navigator.current_path) > 0
        distance_to_goal = 0.0
        if self.navigator.navigation_target:
            dx = self.navigator.navigation_target[0] - current_position[0]
            dy = self.navigator.navigation_target[1] - current_position[1]
            distance_to_goal = (dx**2 + dy**2)**0.5
        
        # Count nearby obstacles
        nearby_obstacles = self._count_nearby_obstacles(current_position, radius=20.0)
        
        # Calculate exploration progress (visited cells / total cells)
        explored_cells = np.sum(self.spatial_memory.visit_heatmap > 0)
        total_cells = self.spatial_memory.map_size[0] * self.spatial_memory.map_size[1]
        exploration_progress = min(explored_cells / max(total_cells * 0.1, 1), 1.0)  # Cap at 10% of map
        
        # Check if in safe zone
        is_in_safe_zone = self.spatial_memory.is_safe(current_position)
        
        # Build state
        state = EnhancedGameState(
            screen=screen,
            minimap=minimap,
            health_percentage=health_state.health_percentage,
            health_current=health_state.health_current,
            health_max=health_state.health_max,
            xp_percentage=health_state.xp_percentage,
            xp_current=health_state.xp_current,
            xp_max=health_state.xp_max,
            is_low_health=health_state.is_low_health,
            is_critical=health_state.is_critical,
            enemy_count=enemy_state.enemy_count,
            has_target=enemy_state.has_target,
            nearest_enemy_distance=enemy_state.nearest_enemy.distance_score if enemy_state.nearest_enemy else 0.0,
            enemies_visible=enemy_state.enemies,
            recent_kills=recent_kills,
            recent_loot=recent_loot,
            recent_damage_dealt=recent_damage_dealt,
            recent_damage_taken=recent_damage_taken,
            is_in_combat=is_in_combat,
            timestamp=current_time,
            # Spatial features (NEW)
            current_position=current_position,
            is_navigating=is_navigating,
            has_path=has_path,
            distance_to_goal=distance_to_goal,
            is_stuck=False,  # Will be updated by stuck detection
            nearby_obstacles=nearby_obstacles,
            exploration_progress=exploration_progress,
            is_in_safe_zone=is_in_safe_zone,
        )
        
        # Save for next frame
        self.prev_screen = screen.copy()
        
        return state
    
    def _screen_to_world_position(self, screen_pos: Tuple[int, int], 
                                  agent_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert screen position to approximate world position
        
        Args:
            screen_pos: Position on screen (x, y)
            agent_pos: Agent's current world position
            
        Returns:
            Approximate world position
        """
        # Simple approximation: assume screen center is agent position
        screen_center = (self.game_region[2] // 2, self.game_region[3] // 2)
        
        # Offset from center
        dx = (screen_pos[0] - screen_center[0]) * 0.1  # Scale factor
        dy = (screen_pos[1] - screen_center[1]) * 0.1
        
        world_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        return world_pos
    
    def _count_nearby_obstacles(self, position: Tuple[float, float], radius: float = 20.0) -> int:
        """
        Count obstacles near position
        
        Args:
            position: World position
            radius: Search radius
            
        Returns:
            Number of obstacle cells nearby
        """
        grid_pos = self.spatial_memory.world_to_grid(position)
        grid_radius = int(radius / self.spatial_memory.cell_size)
        
        count = 0
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                nx = grid_pos[0] + dx
                ny = grid_pos[1] + dy
                
                if 0 <= nx < self.spatial_memory.map_size[0] and 0 <= ny < self.spatial_memory.map_size[1]:
                    if self.spatial_memory.occupancy_grid[ny, nx] == 2:  # Obstacle
                        count += 1
        
        return count
    
    def calculate_enhanced_reward(self, state: EnhancedGameState, prev_state: Optional[EnhancedGameState], 
                                   reward_events: List[RewardEvent]) -> float:
        """
        Calculate reward with full perception data
        
        Reward structure:
        - Kill: +10.0
        - Loot (common): +5.0, (rare): +10.0, (epic): +20.0
        - XP gain: +0.1 per point
        - Damage dealt: +0.1 per point
        - Damage taken: -0.5 per point
        - Death: -50.0
        - Moving toward enemy: +0.01
        - Low health survival: +0.05 per step
        - Idle (no action): -0.5
        """
        reward = 0.0
        
        # Immediate rewards from events
        for event in reward_events:
            if event.event_type == 'kill':
                reward += 10.0 * event.confidence
            elif event.event_type == 'loot':
                reward += event.value * 5.0  # Value = rarity (1-5)
            elif event.event_type == 'xp':
                reward += event.value * 0.1
            elif event.event_type == 'death':
                reward -= 50.0
            elif event.event_type == 'level_up':
                reward += 25.0
            elif event.event_type == 'damage_dealt':
                reward += 0.1
            elif event.event_type == 'damage_taken':
                reward -= 0.5
        
        # Shaped rewards (guide learning)
        if prev_state:
            # Reward for moving toward enemies
            if state.nearest_enemy_distance > prev_state.nearest_enemy_distance:
                reward += 0.01
            
            # Reward for staying alive with low health
            if state.is_low_health and not state.is_critical:
                reward += 0.05
            
            # Penalty for health loss
            health_loss = prev_state.health_percentage - state.health_percentage
            if health_loss > 0:
                reward -= health_loss * 0.1
        
        # Combat engagement bonus
        if state.is_in_combat and state.has_target:
            reward += 0.02
        
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def select_action(self, state: EnhancedGameState) -> int:
        """
        Select action using epsilon-greedy policy with smart heuristics + spatial navigation
        
        This adds game-specific logic on top of RL with intelligent pathfinding
        """
        # === PRIORITY 1: Safety and Combat (unchanged) ===
        
        # CRITICAL: If very low health, run away using pathfinding!
        if state.is_critical:
            # Try to navigate to safe zone
            safe_zones = self.spatial_memory.get_high_value_areas(state.current_position, top_n=1)
            if safe_zones and not self.navigator.is_navigating:
                self.navigator.start_navigation(safe_zones[0])
            
            # If navigating, use path; otherwise retreat
            if self.navigator.is_navigating:
                nav_action = self.navigator.update_navigation()
                if nav_action is not None:
                    return nav_action
            return 1  # backward (fallback)
        
        # If low health and no enemies, navigate to safe area
        if state.is_low_health and state.enemy_count == 0:
            if not self.navigator.is_navigating and np.random.random() < 0.3:
                # Find safe zone
                safe_zones = self.spatial_memory.get_high_value_areas(state.current_position, top_n=1)
                if safe_zones:
                    self.navigator.start_navigation(safe_zones[0])
        
        # === PRIORITY 2: Combat Engagement ===
        
        # If in combat with target, attack!
        if state.has_target and state.is_in_combat:
            # 80% attack, 20% RL decision
            if np.random.random() < 0.8:
                return 8  # attack
        
        # If enemy nearby but not targeted, use intelligent pathfinding to approach
        if state.enemy_count > 0 and not state.has_target:
            # Use pathfinding to approach enemy (avoids obstacles)
            return self._navigate_to_enemy(state)
        
        # If just killed (no enemies, was in combat), collect loot!
        if state.enemy_count == 0 and state.recent_kills > 0:
            return 9  # collect_loot
        
        # === PRIORITY 3: Intelligent Exploration and Navigation ===
        
        # Dynamically switch exploration modes based on progress
        self._update_exploration_mode(state)
        
        # Check if currently navigating
        if self.navigator.is_navigating:
            nav_action = self.navigator.update_navigation()
            if nav_action is not None:
                return nav_action
            else:
                # Reached destination
                self.navigator.cancel_navigation()
        
        # Decide navigation strategy based on current mode
        if not self.navigator.is_navigating:
            target = self._get_navigation_target(state)
            if target:
                success = self.navigator.start_navigation(target)
                if success:
                    # If navigation started, get first action
                    nav_action = self.navigator.update_navigation()
                    if nav_action is not None:
                        return nav_action
        
        # === PRIORITY 4: Exploration/RL Policy ===
        
        # Use RL policy for exploration/movement
        if np.random.random() < self.epsilon:
            # Exploration: smart random with spatial awareness
            if state.enemy_count > 0:
                # 60% move towards enemy, 30% attack, 10% camera
                rand = np.random.random()
                if rand < 0.6:
                    return self._navigate_to_enemy(state)  # Smart pathfinding
                elif rand < 0.9:
                    return 8  # attack
                else:
                    return np.random.randint(10, 14)  # camera
            else:
                # No enemies: intelligent exploration
                # Check if we've been here before
                if self.spatial_memory.is_explored(state.current_position):
                    # Explore new areas
                    unexplored = self.spatial_memory.get_unexplored_nearby(state.current_position, radius=50.0)
                    if unexplored and not self.navigator.is_navigating:
                        # Navigate to closest unexplored area
                        closest = min(unexplored, key=lambda p: (p[0]-state.current_position[0])**2 + (p[1]-state.current_position[1])**2)
                        self.navigator.start_navigation(closest)
                
                # 40% movement (avoid obstacles), 30% camera (look around), 30% attack
                rand = np.random.random()
                if rand < 0.4:
                    # Random movement but avoid known obstacles
                    return self._get_safe_movement_action(state)
                elif rand < 0.7:
                    return np.random.randint(10, 14)  # look around
                else:
                    return 8  # try attacking (might be enemies we can't see)
        
        # Exploitation: use neural network
        with torch.no_grad():
            # Preprocess visual
            visual = self._preprocess_visual(state.screen)
            visual = visual.unsqueeze(0).to(self.device)
            
            # Get state vector
            state_vector = torch.FloatTensor(state.to_state_vector()).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values = self.policy_net(visual, state_vector)
            return q_values.argmax().item()
    
    def _get_navigation_target(self, state: EnhancedGameState) -> Optional[Tuple[float, float]]:
        """
        Get intelligent navigation target based on current mode and state
        
        Args:
            state: Current game state
            
        Returns:
            Target world position or None
        """
        if self.exploration_mode == 'explore':
            # Find unexplored areas, but prioritize areas near known enemy spawns
            unexplored = self.spatial_memory.get_unexplored_nearby(state.current_position, radius=100.0)
            
            if unexplored:
                # If we have enemy spawn data, explore near known farming spots
                high_value = self.spatial_memory.get_high_value_areas(state.current_position, top_n=3)
                if high_value and np.random.random() < 0.7:  # 70% chance to explore near high-value areas
                    # Find unexplored areas near high-value spots
                    for hv_pos in high_value:
                        nearby_unexplored = [pos for pos in unexplored 
                                           if (pos[0] - hv_pos[0])**2 + (pos[1] - hv_pos[1])**2 < 400]  # Within 20 units
                        if nearby_unexplored:
                            return min(nearby_unexplored, key=lambda p: (p[0]-state.current_position[0])**2 + (p[1]-state.current_position[1])**2)
                
                # Otherwise, explore closest unexplored area
                return min(unexplored, key=lambda p: (p[0]-state.current_position[0])**2 + (p[1]-state.current_position[1])**2)
        
        elif self.exploration_mode == 'farm':
            # Target known high-value farming areas
            high_value = self.spatial_memory.get_high_value_areas(state.current_position, top_n=5)
            if high_value:
                # Choose based on distance and safety
                safe_targets = [pos for pos in high_value if self.spatial_memory.is_safe(pos)]
                if safe_targets:
                    return min(safe_targets, key=lambda p: (p[0]-state.current_position[0])**2 + (p[1]-state.current_position[1])**2)
                else:
                    # If no safe targets, take the closest
                    return min(high_value, key=lambda p: (p[0]-state.current_position[0])**2 + (p[1]-state.current_position[1])**2)
        
        elif self.exploration_mode == 'patrol':
            # Patrol between known high-value points
            if not self.patrol_points:
                # Initialize patrol points with top farming locations
                self.patrol_points = self.spatial_memory.get_high_value_areas(state.current_position, top_n=5)
            
            if self.patrol_points:
                # Cycle through patrol points
                target = self.patrol_points[0]
                self.patrol_points = self.patrol_points[1:] + [self.patrol_points[0]]  # Rotate
                return target
        
        return None
    
    def _update_exploration_mode(self, state: EnhancedGameState):
        """
        Dynamically switch exploration modes based on current state and progress
        
        Args:
            state: Current game state
        """
        # Get exploration progress
        explored_cells = np.sum(self.spatial_memory.visit_heatmap > 0)
        total_cells = self.spatial_memory.map_size[0] * self.spatial_memory.map_size[1]
        exploration_percent = (explored_cells / total_cells) * 100.0
        
        # Get farming potential (known enemy spawns)
        num_enemy_spawns = len(self.spatial_memory.enemy_spawns)
        
        # Decision logic for mode switching
        if exploration_percent < 30.0 and num_enemy_spawns < 10:
            # Early game: focus on exploration to build map
            self.exploration_mode = 'explore'
        
        elif num_enemy_spawns >= 5 and state.exploration_progress > 0.5:
            # Mid game: we have farming spots, switch to farming
            if np.random.random() < 0.8:  # 80% farming, 20% exploration
                self.exploration_mode = 'farm'
            else:
                self.exploration_mode = 'explore'
        
        elif num_enemy_spawns >= 10:
            # Late game: focus on systematic farming with occasional exploration
            rand = np.random.random()
            if rand < 0.6:
                self.exploration_mode = 'farm'
            elif rand < 0.8:
                self.exploration_mode = 'patrol'
            else:
                self.exploration_mode = 'explore'
        
        # Emergency exploration if stuck
        if state.is_stuck and self.exploration_mode != 'explore':
            self.exploration_mode = 'explore'
            self.navigator.cancel_navigation()  # Cancel current path
    
    def _navigate_to_enemy(self, state: EnhancedGameState) -> int:
        """
        Navigate to nearest enemy using pathfinding
        
        Args:
            state: Current game state
            
        Returns:
            Movement action
        """
        if not state.enemies_visible or len(state.enemies_visible) == 0:
            return 0  # Default: move forward
        
        # Find nearest enemy
        nearest = None
        min_dist = float('inf')
        
        for enemy in state.enemies_visible:
            enemy_world_pos = self._screen_to_world_position(enemy.position, state.current_position)
            dist = ((enemy_world_pos[0] - state.current_position[0])**2 + 
                   (enemy_world_pos[1] - state.current_position[1])**2)**0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        
        if not nearest:
            return 0
        
        # Convert enemy position to world coordinates
        enemy_world_pos = self._screen_to_world_position(nearest.position, state.current_position)
        
        # Use navigator to get action
        action = self.navigator.get_next_action(state.current_position, enemy_world_pos)
        
        if action is not None:
            return action
        
        # Fallback to direct approach
        return self._move_towards_enemy(state)
    
    def _get_safe_movement_action(self, state: EnhancedGameState) -> int:
        """
        Get random movement action that avoids known obstacles
        
        Args:
            state: Current game state
            
        Returns:
            Safe movement action
        """
        # Try several random movements and pick one that doesn't hit obstacles
        for _ in range(5):
            action = np.random.randint(0, 8)  # Random movement
            
            # Estimate where this action would take us
            next_pos = self._estimate_next_position(state.current_position, action)
            
            # Check if safe
            if not self.spatial_memory.is_obstacle(next_pos):
                return action
        
        # If all random movements hit obstacles, move forward
        return 0
    
    def _estimate_next_position(self, current_pos: Tuple[float, float], action: int) -> Tuple[float, float]:
        """
        Estimate next position after taking action
        
        Args:
            current_pos: Current position
            action: Action to take
            
        Returns:
            Estimated next position
        """
        dx, dy = 0.0, 0.0
        speed = self.navigator.movement_speed
        
        action_name = self.ACTIONS[action]['name']
        
        if 'forward' in action_name:
            dy -= speed
        if 'backward' in action_name:
            dy += speed
        if 'left' in action_name:
            dx -= speed
        if 'right' in action_name:
            dx += speed
        
        return (current_pos[0] + dx, current_pos[1] + dy)
    
    def _move_towards_enemy(self, state: EnhancedGameState) -> int:
        """
        Calculate best movement action to approach nearest enemy
        
        Args:
            state: Current game state with enemy positions
            
        Returns:
            Action ID for movement towards enemy
        """
        if not state.enemies_visible or state.enemy_count == 0:
            return 0  # default: forward
        
        # Find nearest enemy
        nearest_enemy = None
        for enemy in state.enemies_visible:
            if nearest_enemy is None or enemy.distance_score > nearest_enemy.distance_score:
                nearest_enemy = enemy
        
        if not nearest_enemy:
            return 0  # forward
        
        # Get enemy position relative to screen center
        screen_center_x = self.game_region[2] // 2
        screen_center_y = self.game_region[3] // 2
        
        enemy_x, enemy_y = nearest_enemy.position
        
        # Calculate direction vector
        dx = enemy_x - screen_center_x
        dy = enemy_y - screen_center_y
        
        # Calculate distance
        distance = (dx**2 + dy**2)**0.5
        
        # Determine movement direction
        # Map screen positions to movement actions
        
        # Determine primary direction
        move_horizontal = abs(dx) > abs(dy)
        
        if move_horizontal:
            # Enemy is more to left/right
            if dx > 50:  # Enemy on right
                if dy > 50:  # Enemy below-right
                    return 7  # down+right
                elif dy < -50:  # Enemy above-right
                    return 5  # up+right
                else:
                    return 3  # right
            elif dx < -50:  # Enemy on left
                if dy > 50:  # Enemy below-left
                    return 6  # down+left
                elif dy < -50:  # Enemy above-left
                    return 4  # up+left
                else:
                    return 2  # left
            else:
                # Enemy centered horizontally, move vertically
                return 0 if dy < 0 else 1  # up or down
        else:
            # Enemy is more up/down
            if dy > 50:  # Enemy below
                if dx > 50:  # Enemy below-right
                    return 7  # down+right
                elif dx < -50:  # Enemy below-left
                    return 6  # down+left
                else:
                    return 1  # down
            elif dy < -50:  # Enemy above
                if dx > 50:  # Enemy above-right
                    return 5  # up+right
                elif dx < -50:  # Enemy above-left
                    return 4  # up+left
                else:
                    return 0  # up
            else:
                # Enemy centered vertically, move horizontally
                return 3 if dx > 0 else 2  # right or left
    
    def _preprocess_visual(self, screen: np.ndarray) -> torch.Tensor:
        """Preprocess screen for neural network"""
        # Resize
        resized = cv2.resize(screen, (84, 84))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        return torch.FloatTensor(transposed)
    
    def execute_action(self, action_id: int, state: Optional[EnhancedGameState] = None):
        """
        Execute action in game (Evil Lands specific with proper mechanics + odometry tracking)
        
        Args:
            action_id: Action to execute
            state: Current game state (for smart execution)
        """
        action = self.ACTIONS[action_id]
        action_name = action['name']
        
        # Special handling for Evil Lands combat
        if action_name == 'attack':
            # Spam spacebar until red icon disappears (up to 2 seconds)
            self._attack_until_hit(max_duration=2.0)
            
        elif action_name == 'collect_loot':
            # Press B to collect loot
            pyautogui.press('b')
            time.sleep(0.1)
            
        elif action_name == 'attack_collect':
            # Attack then immediately loot
            self._attack_until_hit(max_duration=1.5)
            time.sleep(0.1)
            pyautogui.press('b')
            time.sleep(0.1)
            
        elif action_name == 'attack_move_forward':
            # Attack while moving forward (hold W, spam space)
            pyautogui.keyDown('w')
            self._attack_until_hit(max_duration=1.0)
            pyautogui.keyUp('w')
            
        elif 'forward' in action_name or 'backward' in action_name or 'left' in action_name or 'right' in action_name:
            # Movement - hold for longer duration
            for key in action['keys']:
                pyautogui.keyDown(key)
            time.sleep(0.3)  # Hold movement longer
            for key in action['keys']:
                pyautogui.keyUp(key)
            time.sleep(0.05)
            
        elif 'look' in action_name:
            # Camera control - press briefly
            for key in action['keys']:
                # Convert numpad keys
                if key == 'num4':
                    pyautogui.press('4')
                elif key == 'num6':
                    pyautogui.press('6')
                elif key == 'num8':
                    pyautogui.press('8')
                elif key == 'num5':
                    pyautogui.press('5')
            time.sleep(0.1)
        else:
            # Default: press keys briefly
            for key in action['keys']:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in action['keys']:
                pyautogui.keyUp(key)
            time.sleep(0.05)
        
        # Update odometry after movement action (NEW)
        self.navigator.update_odometry(action_id, self.ACTIONS)
    
    def _attack_until_hit(self, max_duration: float = 2.0, state: Optional[EnhancedGameState] = None):
        """
        Spam spacebar until red attack icon appears or timeout
        (Evil Lands specific attack mechanic)
        
        Args:
            max_duration: Max time to spam spacebar (seconds)
            state: Current game state to check for red icon
            
        Returns:
            True if hit detected, False if timeout
        """
        start_time = time.time()
        hit_detected = False
        
        while time.time() - start_time < max_duration:
            # Press spacebar
            pyautogui.press('space')
            time.sleep(0.05)  # Brief delay between presses
            
            # Check for red attack icon (real-time detection)
            if state and time.time() - start_time > 0.3:  # Check after 0.3s
                # Capture current screen quickly
                game_region_dict = {
                    "left": self.game_region[0],
                    "top": self.game_region[1],
                    "width": self.game_region[2],
                    "height": self.game_region[3]
                }
                screenshot = self.sct.grab(game_region_dict)
                screen = np.array(screenshot)
                screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                
                # Quick check for red attack icon
                if self._detect_red_attack_icon(screen):
                    hit_detected = True
                    # Keep attacking until icon disappears
                    while self._detect_red_attack_icon(screen) and time.time() - start_time < max_duration:
                        pyautogui.press('space')
                        time.sleep(0.05)
                        screenshot = self.sct.grab(game_region_dict)
                        screen = np.array(screenshot)
                        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                    break
            
        return hit_detected
    
    def _detect_red_attack_icon(self, screen: np.ndarray) -> bool:
        """
        Detect red attack icon on screen (fast check)
        
        Args:
            screen: Game screen image
            
        Returns:
            True if red attack icon detected
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Red color range (attack icon)
            lower_red1 = np.array([0, 150, 150])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 150, 150])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Focus on center area (where attack icon appears above enemy)
            h, w = red_mask.shape
            center_region = red_mask[h//4:3*h//4, w//4:3*w//4]
            
            # Count red pixels
            red_pixels = np.sum(center_region > 0)
            
            # If significant red pixels, attack icon is present
            return red_pixels > 200  # Threshold for icon presence
            
        except Exception as e:
            return False
    
    def _move_towards_enemy(self, state: EnhancedGameState) -> int:
        """
        Calculate and execute movement towards nearest enemy
        
        Args:
            state: Current game state
            
        Returns:
            Action ID for movement direction
        """
        if not state.enemies_visible or len(state.enemies_visible) == 0:
            return 0  # Default: move forward
        
        # Find nearest enemy
        nearest = None
        min_dist = float('inf')
        screen_center_x = self.game_region[2] // 2
        screen_center_y = self.game_region[3] // 2
        
        for enemy in state.enemies_visible:
            ex, ey = enemy.position
            dist = ((ex - screen_center_x) ** 2 + (ey - screen_center_y) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        
        if not nearest:
            return 0  # Forward
        
        # Calculate direction to enemy
        ex, ey = nearest.position
        dx = ex - screen_center_x
        dy = ey - screen_center_y
        
        # Determine movement direction based on enemy position
        # Enemy positions relative to screen center:
        # Top-Left: forward+left (4)
        # Top-Right: forward+right (5)
        # Bottom-Left: backward+left (6)
        # Bottom-Right: backward+right (7)
        # Top: forward (0)
        # Bottom: backward (1)
        # Left: left (2)
        # Right: right (3)
        
        # Thresholds for diagonal movement
        threshold = 100  # pixels
        
        if abs(dx) < threshold and abs(dy) < threshold:
            # Enemy is very close, attack!
            return 8  # attack
        
        # Determine primary direction
        if abs(dy) > abs(dx):
            # Vertical movement dominant
            if dy < -threshold:
                # Enemy above
                if abs(dx) > 50:
                    return 5 if dx > 0 else 4  # forward+right or forward+left
                else:
                    return 0  # forward
            else:
                # Enemy below
                if abs(dx) > 50:
                    return 7 if dx > 0 else 6  # backward+right or backward+left
                else:
                    return 1  # backward
        else:
            # Horizontal movement dominant
            if dx > threshold:
                # Enemy to the right
                if abs(dy) > 50:
                    return 5 if dy < 0 else 7  # forward+right or backward+right
                else:
                    return 3  # right
            else:
                # Enemy to the left
                if abs(dy) > 50:
                    return 4 if dy < 0 else 6  # forward+left or backward+left
                else:
                    return 2  # left
        """
        Try to target the nearest enemy by clicking on it
        
        Args:
            state: Current game state with enemy positions
        """
        if state.nearest_enemy_distance > 0 and state.enemies_visible:
            # Find nearest enemy
            nearest = None
            min_dist = float('inf')
            
            for enemy in state.enemies_visible:
                # Calculate distance from screen center
                screen_center_x = self.game_region[2] // 2
                screen_center_y = self.game_region[3] // 2
                
                ex, ey = enemy.position
                dist = ((ex - screen_center_x) ** 2 + (ey - screen_center_y) ** 2) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    nearest = enemy
            
            if nearest:
                # Click on enemy to target
                click_x = nearest.position[0] + self.game_region[0]
                click_y = nearest.position[1] + self.game_region[1]
                pyautogui.click(click_x, click_y)
                time.sleep(0.1)
                return True
        
        return False
    
    def _combat_sequence(self, state: EnhancedGameState) -> int:
        """
        Execute a full combat sequence: target → attack → loot
        Returns number of actions taken
        """
        actions_taken = 0
        
        # Step 1: Target enemy if not targeted
        if state.enemy_count > 0 and not state.has_target:
            self._target_nearest_enemy(state)
            time.sleep(0.2)
            actions_taken += 1
        
        # Step 2: Attack until enemy dies (check for red icon disappearance)
        if state.has_target or state.enemy_count > 0:
            # Attack for up to 3 seconds
            self._attack_until_hit(max_duration=3.0)
            actions_taken += 1
            time.sleep(0.3)
        
        # Step 3: Collect loot
        pyautogui.press('b')
        time.sleep(0.2)
        actions_taken += 1
        
        return actions_taken
    
    def _detect_death_screen(self, screen: np.ndarray) -> bool:
        """
        Detect if we're on death/respawn screen
        
        Checks for:
        - Black/gray screen (death overlay)
        - "Respawn" button
        - HP at 0%
        
        Args:
            screen: Game screen image
            
        Returns:
            True if death screen detected
        """
        try:
            # Method 1: Check if screen is mostly dark (death overlay)
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Death screens are usually darker
            if mean_brightness < 50:  # Very dark
                return True
            
            # Method 2: Check for large dark overlay (center area)
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            center_brightness = np.mean(center_region)
            
            if center_brightness < 40:
                return True
            
            # Method 3: Check for "Respawn" or "You Died" text (OCR if available)
            if hasattr(self, 'reward_detector') and self.reward_detector.use_ocr:
                # Check center area for death text
                center_img = screen[h//3:2*h//3, w//3:2*w//3]
                text = self.reward_detector._run_ocr(center_img).lower()
                
                death_keywords = ['respawn', 'you died', 'dead', 'defeat', 'revive']
                if any(keyword in text for keyword in death_keywords):
                    return True
            
            # Method 4: Check if HP is 0 (from health detector)
            if hasattr(self, 'health_detector'):
                health_state = self.health_detector.detect(screen)
                if health_state.health_percentage <= 0 or health_state.health_current <= 0:
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _recover_from_death(self):
        """
        Handle death recovery (respawn, etc.)
        """
        print("  [DEATH DETECTED] Attempting recovery...")
        
        # Wait for death animation
        time.sleep(2.0)
        
        # Try clicking respawn button (multiple common locations)
        respawn_locations = [
            # Center of screen (most common)
            (self.game_region[2] // 2, self.game_region[3] // 2),
            # Bottom center
            (self.game_region[2] // 2, self.game_region[3] * 3 // 4),
            # Slightly above center
            (self.game_region[2] // 2, self.game_region[3] * 2 // 5),
        ]
        
        for x, y in respawn_locations:
            click_x = x + self.game_region[0]
            click_y = y + self.game_region[1]
            
            # Click multiple times
            for _ in range(2):
                pyautogui.click(click_x, click_y)
                time.sleep(0.3)
        
        # Wait for respawn to complete
        time.sleep(3.0)
        
        # Verify we respawned (check if HP is back)
        state = self.capture_enhanced_state()
        if state.health_percentage > 0:
            print("  [RECOVERY] Respawned successfully!")
            return True
        else:
            print("  [RECOVERY] Failed to respawn, retrying...")
            # Try pressing ESC and clicking again
            pyautogui.press('esc')
            time.sleep(1.0)
            for x, y in respawn_locations:
                pyautogui.click(x + self.game_region[0], y + self.game_region[1])
                time.sleep(0.5)
            time.sleep(2.0)
            return False
    
    def _check_stuck(self, states_history: List[EnhancedGameState], window: int = 20) -> bool:
        """
        Check if agent is stuck (not moving, no progress) using spatial awareness
        
        Args:
            states_history: Recent states
            window: Number of states to check
            
        Returns:
            True if stuck
        """
        if len(states_history) < window:
            return False
        
        recent = states_history[-window:]
        
        # Check multiple indicators of being stuck
        
        # 1. Position hasn't changed (NEW - spatial awareness)
        positions = [s.current_position for s in recent]
        position_variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
        position_stuck = position_variance < 1.0  # Very little movement
        
        # 2. Visiting same locations repeatedly (looping)
        grid_positions = [self.spatial_memory.world_to_grid(p) for p in positions]
        unique_positions = len(set(grid_positions))
        looping = unique_positions < window * 0.3  # Visiting fewer than 30% unique cells
        
        # 3. HP, XP haven't changed
        hp_values = [s.health_percentage for s in recent]
        xp_values = [s.xp_percentage for s in recent]
        enemy_counts = [s.enemy_count for s in recent]
        
        # Check variance - if too low, likely stuck
        hp_variance = np.var(hp_values)
        xp_variance = np.var(xp_values)
        
        # 4. No enemies for extended period
        no_enemies_count = sum(1 for e in enemy_counts if e == 0)
        
        # 5. No kills or loot
        no_kills = all(s.recent_kills == 0 for s in recent)
        no_loot = all(s.recent_loot == 0 for s in recent)
        
        # 6. High obstacle density nearby
        recent_obstacles = [s.nearby_obstacles for s in recent]
        high_obstacles = np.mean(recent_obstacles) > 10
        
        # Stuck conditions (multiple criteria):
        # - Position hasn't changed OR looping
        # - HP and XP haven't changed (variance < 0.1)
        # - No enemies for most of window
        # - No progress (kills/loot)
        hp_stuck = hp_variance < 0.1
        xp_stuck = xp_variance < 0.01
        no_enemies_stuck = no_enemies_count > (window * 0.8)
        no_progress = no_kills and no_loot
        
        is_stuck = (position_stuck or looping) and hp_stuck and xp_stuck and no_enemies_stuck and no_progress
        
        return is_stuck
    
    def _unstuck_maneuver(self, state: Optional[EnhancedGameState] = None):
        """
        Perform intelligent maneuvers to get unstuck using spatial memory
        
        Args:
            state: Current game state (for spatial context)
        """
        print("  [STUCK DETECTED] Using spatial awareness to escape...")
        
        if state:
            # Mark current position as problematic obstacle area
            self.spatial_memory.update_position(state.current_position, is_obstacle=True)
            
            # Try to find a clear path to nearby free space
            current_grid = self.spatial_memory.world_to_grid(state.current_position)
            
            # Find nearest free space
            best_direction = None
            min_obstacles = float('inf')
            
            # Check 8 directions
            for direction in range(8):
                # Simulate movement in this direction
                test_pos = self._estimate_next_position(state.current_position, direction)
                
                # Count obstacles in that direction
                obstacles = self._count_nearby_obstacles(test_pos, radius=10.0)
                
                if obstacles < min_obstacles and not self.spatial_memory.is_obstacle(test_pos):
                    min_obstacles = obstacles
                    best_direction = direction
            
            if best_direction is not None:
                print(f"  [UNSTUCK] Found escape direction: {self.ACTIONS[best_direction]['name']}")
                
                # Execute movement in best direction multiple times
                for _ in range(3):
                    self.execute_action(best_direction)
                    time.sleep(0.2)
                
                # Cancel current navigation and replan
                self.navigator.cancel_navigation()
                
                print("  [UNSTUCK] Escaped! Replanning navigation...")
                return
        
        # Fallback: Random movement sequence to escape (old method)
        print("  [UNSTUCK] Using random escape sequence...")
        unstuck_sequence = [
            1,  # backward
            1,  # backward
            2,  # left
            3,  # right
            0,  # forward
            10, # look left
            11, # look right
            0,  # forward
        ]
        
        for action_id in unstuck_sequence:
            action = self.ACTIONS[action_id]
            
            # Execute movement
            for key in action['keys']:
                if 'num' in key:
                    # Handle numpad keys
                    pyautogui.press(key.replace('num', ''))
                else:
                    pyautogui.keyDown(key)
            
            time.sleep(0.3)
            
            for key in action['keys']:
                if 'num' not in key:
                    pyautogui.keyUp(key)
            
            time.sleep(0.1)
        
        # Cancel navigation after unstuck
        self.navigator.cancel_navigation()
        print("  [UNSTUCK] Maneuvers complete, resuming...")
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        visual_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_visual_batch = []
        next_state_batch = []
        done_batch = []
        
        for idx in batch:
            s, a, r, s_next, done = self.memory[idx]
            visual_batch.append(self._preprocess_visual(s.screen))
            state_batch.append(s.to_state_vector())
            action_batch.append(a)
            reward_batch.append(r)
            next_visual_batch.append(self._preprocess_visual(s_next.screen))
            next_state_batch.append(s_next.to_state_vector())
            done_batch.append(done)
        
        # Convert to tensors
        visual_batch = torch.stack(visual_batch).to(self.device)
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_visual_batch = torch.stack(next_visual_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(visual_batch, state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
        # Compute target
        with torch.no_grad():
            next_q_values = self.target_net(next_visual_batch, next_state_batch).max(1)[0]
            target = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Loss
        loss = nn.functional.smooth_l1_loss(q_values, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str = "enhanced_farming_agent.pth"):
        """Save model checkpoint + spatial memory"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }, path)
        print(f"[EnhancedFarmingAgent] Model saved to {path}")
        
        # Save spatial memory separately
        memory_path = path.replace('.pth', '_memory.pkl')
        self.spatial_memory.save(memory_path)
    
    def load_model(self, path: str = "enhanced_farming_agent.pth"):
        """Load model checkpoint + spatial memory"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"[EnhancedFarmingAgent] Model loaded from {path}")
        
        # Load spatial memory if exists
        memory_path = path.replace('.pth', '_memory.pkl')
        if Path(memory_path).exists():
            self.spatial_memory.load(memory_path)
        else:
            print(f"  [Warning] Spatial memory not found at {memory_path}")
    
    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 500):
        """
        Main training loop - This makes the agent learn!
        
        Args:
            num_episodes: How many episodes to train
            max_steps_per_episode: Max actions per episode
        """
        print("\n" + "=" * 70)
        print("  ENHANCED RL FARMING AGENT - TRAINING MODE")
        print("=" * 70)
        print(f"\nTraining for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Starting epsilon: {self.epsilon}")
        print(f"Batch size: {self.batch_size}")
        print("\nMake sure Evil Lands is visible and ready!")
        print("Starting in 5 seconds...\n")
        time.sleep(5)
        
        for episode in range(num_episodes):
            print(f"\n[Episode {episode + 1}/{num_episodes}]")
            
            # Reset episode
            episode_reward = 0
            episode_kills = 0
            episode_deaths = 0
            step_count = 0
            states_history = deque(maxlen=20)  # For stuck detection
            
            # Get initial state
            state = self.capture_enhanced_state()
            
            for step in range(max_steps_per_episode):
                # Select action (with smart heuristics + navigation)
                action = self.select_action(state)
                action_name = self.ACTIONS[action]['name']
                
                # Execute action with state context
                self.execute_action(action, state)
                
                # Wait for game to respond
                time.sleep(0.15)  # Slightly longer for game to process
                
                # Get new state
                next_state = self.capture_enhanced_state()
                
                # Get reward events
                reward_events = self.reward_detector.detect(next_state.screen, state.screen)
                
                # Calculate reward
                reward = self.calculate_enhanced_reward(next_state, state, reward_events)
                episode_reward += reward
                
                # Count kills/deaths
                for event in reward_events:
                    if event.event_type == 'kill':
                        episode_kills += 1
                    elif event.event_type == 'death':
                        episode_deaths += 1
                        # Record death location in spatial memory (NEW)
                        self.spatial_memory.record_death(next_state.current_position)
                
                # Check if episode done
                done = False
                if next_state.is_critical:  # Very low health
                    done = True
                    reward -= 10.0  # Penalty for being near death
                
                # Check for death
                if self._detect_death_screen(next_state.screen):
                    done = True
                    reward -= 50.0
                    self._recover_from_death()
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Train
                if len(self.memory) >= self.batch_size and step % 4 == 0:
                    loss = self.train_step()
                
                # Update state and history
                states_history.append(next_state)
                state = next_state
                step_count += 1
                
                # Check for stuck (NEW - with spatial awareness)
                if step > 0 and step % 20 == 0:
                    if self._check_stuck(list(states_history)):
                        print(f"  [STUCK] Agent is stuck at position {state.current_position}")
                        self._unstuck_maneuver(state)
                        # Clear history after unstuck
                        states_history.clear()
                
                # Print progress (enhanced with spatial info)
                if step % 50 == 0:
                    print(f"  Step {step}/{max_steps_per_episode} | "
                          f"Action: {action_name:20s} | "
                          f"Reward: {reward:+6.2f} | "
                          f"HP: {state.health_percentage:5.1f}% | "
                          f"Pos: ({state.current_position[0]:.1f}, {state.current_position[1]:.1f}) | "
                          f"Enemies: {state.enemy_count} | "
                          f"Kills: {episode_kills}")
                
                if done:
                    print(f"  Episode ended at step {step} (critical health or death)")
                    break
            
            # Episode complete
            self.episode_rewards.append(episode_reward)
            
            # Update epsilon (decay exploration)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if (episode + 1) % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"  → Target network updated")
            
            # Save checkpoint
            if (episode + 1) % 50 == 0:
                self.save_model(f"enhanced_agent_ep{episode + 1}.pth")
            
            # Episode summary (enhanced with spatial stats)
            reward_state = self.reward_detector.get_state()
            explored_cells = np.sum(self.spatial_memory.visit_heatmap > 0)
            total_cells = self.spatial_memory.map_size[0] * self.spatial_memory.map_size[1]
            exploration_percent = (explored_cells / total_cells) * 100.0
            
            print(f"\n[Episode {episode + 1} Summary]")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Kills: {episode_kills}")
            print(f"  Deaths: {episode_deaths}")
            print(f"  Total XP: {reward_state.total_xp:.0f}")
            print(f"  Total Loot: {reward_state.total_loot}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Avg Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
            # Spatial statistics (NEW)
            print(f"  Map Exploration: {exploration_percent:.1f}% ({explored_cells}/{total_cells} cells)")
            print(f"  Obstacles Found: {self.spatial_memory.obstacles_found}")
            print(f"  Enemy Spawns Known: {len(self.spatial_memory.enemy_spawns)}")
            print(f"  Death Locations: {len(self.spatial_memory.death_locations)}")
            
            # TensorBoard logging (enhanced)
            if self.use_tensorboard:
                self.writer.add_scalar('Reward/Episode', episode_reward, episode)
                self.writer.add_scalar('Stats/Kills', episode_kills, episode)
                self.writer.add_scalar('Stats/Deaths', episode_deaths, episode)
                self.writer.add_scalar('Stats/Epsilon', self.epsilon, episode)
                self.writer.add_scalar('Stats/Steps', step_count, episode)
                # Spatial metrics (NEW)
                self.writer.add_scalar('Spatial/Exploration', exploration_percent, episode)
                self.writer.add_scalar('Spatial/Obstacles', self.spatial_memory.obstacles_found, episode)
                self.writer.add_scalar('Spatial/EnemySpawns', len(self.spatial_memory.enemy_spawns), episode)
        
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE!")
        print("=" * 70)
        self.save_model("enhanced_agent_final.pth")
        print(f"\nFinal Stats:")
        print(f"  Total Episodes: {num_episodes}")
        print(f"  Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"  Best Episode: {max(self.episode_rewards):.2f}")
        print(f"  Final Epsilon: {self.epsilon:.4f}")
        
        # Save final map visualization (NEW)
        self._save_map_visualization("final_map.png")
    
    def play(self, num_episodes: int = 10, max_steps: int = 1000):
        """
        Play using trained agent (no learning, pure exploitation)
        
        Args:
            num_episodes: How many episodes to play
            max_steps: Max actions per episode
        """
        print("\n" + "=" * 70)
        print("  ENHANCED RL FARMING AGENT - PLAY MODE")
        print("=" * 70)
        print(f"\nPlaying for {num_episodes} episodes...")
        print("No exploration - using learned policy only")
        print("\nMake sure Evil Lands is visible!")
        print("Starting in 3 seconds...\n")
        time.sleep(3)
        
        # Disable exploration
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        for episode in range(num_episodes):
            print(f"\n[Play Episode {episode + 1}/{num_episodes}]")
            
            episode_reward = 0
            episode_kills = 0
            state = self.capture_enhanced_state()
            
            for step in range(max_steps):
                # Select best action (no exploration)
                action = self.select_action(state)
                action_name = self.ACTIONS[action]['name']
                
                # Execute
                self.execute_action(action)
                time.sleep(0.1)
                
                # Get new state
                next_state = self.capture_enhanced_state()
                reward_events = self.reward_detector.detect(next_state.screen, state.screen)
                reward = self.calculate_enhanced_reward(next_state, state, reward_events)
                
                episode_reward += reward
                
                # Count kills
                for event in reward_events:
                    if event.event_type == 'kill':
                        episode_kills += 1
                
                # Progress
                if step % 50 == 0:
                    print(f"  Step {step} | Action: {action_name:20s} | "
                          f"HP: {next_state.health_percentage:5.1f}% | "
                          f"Enemies: {next_state.enemy_count} | "
                          f"Kills: {episode_kills}")
                
                state = next_state
                
                # Stop if critical
                if state.is_critical:
                    print(f"  Stopping - critical health!")
                    break
            
            print(f"\n[Play Episode {episode + 1} Complete]")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Kills: {episode_kills}")
        
        # Restore epsilon
        self.epsilon = old_epsilon
        print("\n" + "=" * 70)
        print("  PLAY MODE COMPLETE!")
        print("=" * 70)
    
    def _save_map_visualization(self, filename: str = "spatial_map.png"):
        """
        Save visualization of learned spatial map
        
        Args:
            filename: Output filename
        """
        print(f"\n[Saving Map] Generating spatial map visualization...")
        
        vis = self.spatial_memory.get_visualization()
        
        # Add current position marker
        current_grid = self.spatial_memory.world_to_grid(self.navigator.current_position)
        cv2.circle(vis, (current_grid[0], current_grid[1]), 10, (0, 255, 255), -1)  # Cyan = agent
        
        # Add text overlay with stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        stats_text = [
            f"Explored: {np.sum(self.spatial_memory.visit_heatmap > 0)} cells",
            f"Obstacles: {self.spatial_memory.obstacles_found}",
            f"Enemy Spawns: {len(self.spatial_memory.enemy_spawns)}",
            f"Deaths: {len(self.spatial_memory.death_locations)}",
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(vis, text, (10, y_offset), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Save
        cv2.imwrite(filename, vis)
        print(f"[Saved Map] {filename}")
    
    def visualize_map_live(self):
        """
        Show live visualization of spatial map while agent is running
        """
        print("\n[Live Map] Press 'q' to close window")
        
        cv2.namedWindow("Spatial Memory Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Spatial Memory Map", 800, 800)
        
        while True:
            vis = self.spatial_memory.get_visualization()
            
            # Add current position
            current_grid = self.spatial_memory.world_to_grid(self.navigator.current_position)
            cv2.circle(vis, (current_grid[0], current_grid[1]), 5, (0, 255, 255), -1)
            
            # Add navigation path if active
            if self.navigator.current_path:
                for i, waypoint in enumerate(self.navigator.current_path):
                    wp_grid = self.spatial_memory.world_to_grid(waypoint)
                    cv2.circle(vis, (wp_grid[0], wp_grid[1]), 3, (255, 255, 0), -1)
                    
                    if i > 0:
                        prev_wp = self.navigator.current_path[i-1]
                        prev_grid = self.spatial_memory.world_to_grid(prev_wp)
                        cv2.line(vis, prev_grid, (wp_grid[0], wp_grid[1]), (255, 255, 0), 1)
            
            cv2.imshow("Spatial Memory Map", vis)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()


def main():
    """Main entry point for enhanced farming agent"""
    print("\n" + "=" * 70)
    print("  ENHANCED RL FARMING AGENT")
    print("  With Full Perception System (HP, Enemies, Rewards)")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Train new agent")
    print("  2. Continue training existing agent")
    print("  3. Play with trained agent")
    print("  4. Test perception systems")
    print("  5. Visualize learned spatial map")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Train from scratch
        config_file = input("Config file (default 'config_spatial.json'): ") or "config_spatial.json"
        episodes = int(input("Number of episodes (default 500): ") or "500")
        agent = EnhancedFarmingAgent(config_path=config_file)
        agent.train(num_episodes=episodes)
        
    elif choice == '2':
        # Continue training
        model_file = input("Model file (default 'enhanced_farming_agent.pth'): ") or "enhanced_farming_agent.pth"
        config_file = input("Config file (default 'config_spatial.json'): ") or "config_spatial.json"
        agent = EnhancedFarmingAgent(config_path=config_file)
        try:
            agent.load_model(model_file)
            episodes = int(input("Additional episodes (default 200): ") or "200")
            agent.train(num_episodes=episodes)
        except FileNotFoundError:
            print(f"Error: Model file '{model_file}' not found!")
            
    elif choice == '3':
        # Play with trained agent
        model_file = input("Model file (default 'enhanced_farming_agent.pth'): ") or "enhanced_farming_agent.pth"
        config_file = input("Config file (default 'config_spatial.json'): ") or "config_spatial.json"
        agent = EnhancedFarmingAgent(config_path=config_file)
        try:
            agent.load_model(model_file)
            episodes = int(input("Number of episodes (default 5): ") or "5")
            agent.play(num_episodes=episodes)
        except FileNotFoundError:
            print(f"Error: Model file '{model_file}' not found!")
            
    elif choice == '4':
        # Test perception
        print("\nTesting perception systems...")
        print("Press 'q' to quit\n")
        
        config_file = input("Config file (default 'config_spatial.json'): ") or "config_spatial.json"
        agent = EnhancedFarmingAgent(config_path=config_file)
        
        try:
            while True:
                state = agent.capture_enhanced_state()
                
                print(f"\r[Perception Test] "
                      f"HP: {state.health_percentage:5.1f}% ({state.health_current}/{state.health_max}) | "
                      f"XP: {state.xp_percentage:5.1f}% ({state.xp_current}/{state.xp_max}) | "
                      f"Pos: ({state.current_position[0]:.1f}, {state.current_position[1]:.1f}) | "
                      f"Enemies: {state.enemy_count} | "
                      f"Combat: {state.is_in_combat} | "
                      f"Recent Kills: {state.recent_kills}", 
                      end='', flush=True)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\nPerception test stopped.")
    
    elif choice == '5':
        # Visualize spatial map
        print("\nVisualizing learned spatial map...")
        config_file = input("Config file (default 'config_spatial.json'): ") or "config_spatial.json"
        memory_file = input("Memory file (default 'spatial_memory.pkl'): ") or "spatial_memory.pkl"
        
        if not Path(memory_file).exists():
            print(f"Error: Memory file '{memory_file}' not found!")
            print("Train an agent first to build spatial memory.")
        else:
            agent = EnhancedFarmingAgent(config_path=config_file)
            agent.spatial_memory.load(memory_file)
            
            # Save static visualization
            agent._save_map_visualization("spatial_map_view.png")
            print("\nStatic map saved as 'spatial_map_view.png'")
            
            # Ask if user wants live view
            live = input("Show live map view? (y/n): ").strip().lower()
            if live == 'y':
                print("\nNote: This requires the agent to be running.")
                print("Starting live map visualization (press 'q' to close)...")
                agent.visualize_map_live()
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
