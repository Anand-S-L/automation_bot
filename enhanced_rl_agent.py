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

# Import perception modules
from perception.health_detection import HealthDetector, HealthManaState
from perception.enemy_detection import EnemyDetector, EnemyState
from perception.reward_detection import RewardDetector, RewardEvent, RewardState


@dataclass
class EnhancedGameState:
    """
    Enhanced game state with full perception
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
    
    def to_state_vector(self) -> np.ndarray:
        """
        Convert to fixed-size state vector for neural network
        
        Returns:
            State vector [health, xp_progress, enemy_count, has_target, distance, ...]
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
        ], dtype=np.float32)


class EnhancedDQNetwork(nn.Module):
    """
    Enhanced DQN with dual-input architecture
    
    Input 1: Visual (screen + minimap) → CNN
    Input 2: Game state vector → FC layers
    Combined → Q-values
    """
    
    def __init__(self, num_actions=16):  # 16 actions (0-15)
        super(EnhancedDQNetwork, self).__init__()
        
        # Visual stream (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size (for 84x84 input)
        self.conv_output_size = 3136
        
        # State vector stream (FC layers)
        self.state_fc1 = nn.Linear(10, 128)  # 10 state features (removed is_low_mana)
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
            state_vector: (batch, 10) game state features (health, xp, enemies, combat, etc.)
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
        """Initialize enhanced agent"""
        print("[EnhancedFarmingAgent] Initializing state-of-the-art RL agent...")
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")
        
        # Initialize perception modules
        print("  Initializing perception systems...")
        self.health_detector = HealthDetector(self.config.get('health_detection'))
        self.enemy_detector = EnemyDetector(self.config.get('enemy_detection'))
        self.reward_detector = RewardDetector(self.config.get('reward_detection'))
        
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
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"  Config not found, using defaults")
            return {
                'game_region': [0, 0, 1920, 1080],
                'minimap_region': [1670, 50, 200, 200],
                'learning_rate': 1e-4,
                'memory_size': 10000,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
            }
    
    def capture_enhanced_state(self) -> EnhancedGameState:
        """
        Capture full game state with all perception systems
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
            timestamp=current_time
        )
        
        # Save for next frame
        self.prev_screen = screen.copy()
        
        return state
    
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
        Select action using epsilon-greedy policy with smart heuristics
        
        This adds game-specific logic on top of RL to make it actually work
        """
        # Smart heuristics (override RL if needed)
        
        # CRITICAL: If very low health, run away!
        if state.is_critical:
            return 1  # backward
        
        # If low health and no enemies, do nothing risky
        if state.is_low_health and state.enemy_count == 0:
            return 0  # forward (explore carefully)
        
        # If in combat with target, attack!
        if state.has_target and state.is_in_combat:
            # 80% attack, 20% RL decision
            if np.random.random() < 0.8:
                return 8  # attack
        
        # If enemy nearby but not targeted, MOVE TOWARDS ENEMY
        if state.enemy_count > 0 and not state.has_target:
            # Use smart pathfinding to move towards enemy
            return self._move_towards_enemy(state)
        
        # If just killed (no enemies, was in combat), collect loot!
        if state.enemy_count == 0 and state.recent_kills > 0:
            return 9  # collect_loot
        
        # Use RL policy for exploration/movement
        if np.random.random() < self.epsilon:
            # Exploration: smart random
            # If enemies visible, bias towards movement/attack
            if state.enemy_count > 0:
                # 60% move towards enemy, 30% attack, 10% camera
                rand = np.random.random()
                if rand < 0.6:
                    return self._move_towards_enemy(state)  # Smart movement
                elif rand < 0.9:
                    return 8  # attack
                else:
                    return np.random.randint(10, 14)  # camera
            else:
                # No enemies: explore
                # 50% movement, 30% camera (look around), 20% attack
                rand = np.random.random()
                if rand < 0.5:
                    return np.random.randint(0, 8)  # random movement
                elif rand < 0.8:
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
        Execute action in game (Evil Lands specific with proper mechanics)
        
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
        Check if agent is stuck (not moving, no progress)
        
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
        
        # 1. HP, XP, position haven't changed
        hp_values = [s.health_percentage for s in recent]
        xp_values = [s.xp_percentage for s in recent]
        enemy_counts = [s.enemy_count for s in recent]
        
        # Check variance - if too low, likely stuck
        hp_variance = np.var(hp_values)
        xp_variance = np.var(xp_values)
        
        # 2. No enemies for extended period
        no_enemies_count = sum(1 for e in enemy_counts if e == 0)
        
        # 3. No kills or loot
        no_kills = all(s.recent_kills == 0 for s in recent)
        no_loot = all(s.recent_loot == 0 for s in recent)
        
        # Stuck conditions:
        # - HP and XP haven't changed (variance < 0.1)
        # - No enemies for most of window
        # - No progress (kills/loot)
        hp_stuck = hp_variance < 0.1
        xp_stuck = xp_variance < 0.01
        no_enemies_stuck = no_enemies_count > (window * 0.8)
        no_progress = no_kills and no_loot
        
        is_stuck = hp_stuck and xp_stuck and no_enemies_stuck and no_progress
        
        return is_stuck
    
    def _unstuck_maneuver(self):
        """
        Perform maneuvers to get unstuck from walls/obstacles
        """
        print("  [STUCK DETECTED] Performing unstuck maneuvers...")
        
        # Strategy: Random movement sequence to escape
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
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }, path)
        print(f"[EnhancedFarmingAgent] Model saved to {path}")
    
    def load_model(self, path: str = "enhanced_farming_agent.pth"):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"[EnhancedFarmingAgent] Model loaded from {path}")
    
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
            
            # Get initial state
            state = self.capture_enhanced_state()
            
            for step in range(max_steps_per_episode):
                # Select action (with smart heuristics)
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
                
                # Check if episode done
                done = False
                if next_state.is_critical:  # Very low health
                    done = True
                    reward -= 10.0  # Penalty for being near death
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Train
                if len(self.memory) >= self.batch_size and step % 4 == 0:
                    loss = self.train_step()
                
                # Update state
                state = next_state
                step_count += 1
                
                # Print progress
                if step % 50 == 0:
                    print(f"  Step {step}/{max_steps_per_episode} | "
                          f"Action: {action_name:20s} | "
                          f"Reward: {reward:+6.2f} | "
                          f"HP: {state.health_percentage:5.1f}% | "
                          f"Enemies: {state.enemy_count} | "
                          f"Kills: {episode_kills}")
                
                if done:
                    print(f"  Episode ended at step {step} (critical health)")
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
            
            # Episode summary
            reward_state = self.reward_detector.get_state()
            print(f"\n[Episode {episode + 1} Summary]")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Kills: {episode_kills}")
            print(f"  Deaths: {episode_deaths}")
            print(f"  Total XP: {reward_state.total_xp:.0f}")
            print(f"  Total Loot: {reward_state.total_loot}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Avg Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
            
            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalar('Reward/Episode', episode_reward, episode)
                self.writer.add_scalar('Stats/Kills', episode_kills, episode)
                self.writer.add_scalar('Stats/Deaths', episode_deaths, episode)
                self.writer.add_scalar('Stats/Epsilon', self.epsilon, episode)
                self.writer.add_scalar('Stats/Steps', step_count, episode)
        
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE!")
        print("=" * 70)
        self.save_model("enhanced_agent_final.pth")
        print(f"\nFinal Stats:")
        print(f"  Total Episodes: {num_episodes}")
        print(f"  Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"  Best Episode: {max(self.episode_rewards):.2f}")
        print(f"  Final Epsilon: {self.epsilon:.4f}")
    
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
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Train from scratch
        episodes = int(input("Number of episodes (default 500): ") or "500")
        agent = EnhancedFarmingAgent()
        agent.train(num_episodes=episodes)
        
    elif choice == '2':
        # Continue training
        model_file = input("Model file (default 'enhanced_farming_agent.pth'): ") or "enhanced_farming_agent.pth"
        agent = EnhancedFarmingAgent()
        try:
            agent.load_model(model_file)
            episodes = int(input("Additional episodes (default 200): ") or "200")
            agent.train(num_episodes=episodes)
        except FileNotFoundError:
            print(f"Error: Model file '{model_file}' not found!")
            
    elif choice == '3':
        # Play with trained agent
        model_file = input("Model file (default 'enhanced_farming_agent.pth'): ") or "enhanced_farming_agent.pth"
        agent = EnhancedFarmingAgent()
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
        
        agent = EnhancedFarmingAgent()
        
        try:
            while True:
                state = agent.capture_enhanced_state()
                
                print(f"\r[Perception Test] "
                      f"HP: {state.health_percentage:5.1f}% ({state.health_current}/{state.health_max}) | "
                      f"XP: {state.xp_percentage:5.1f}% ({state.xp_current}/{state.xp_max}) | "
                      f"Enemies: {state.enemy_count} | "
                      f"Combat: {state.is_in_combat} | "
                      f"Recent Kills: {state.recent_kills}", 
                      end='', flush=True)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\nPerception test stopped.")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
